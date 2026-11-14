"""Build signed unadjusted-significance DEET rank Parquet shards and metadata.

Reads:
  - DEET_short_final_Aug2022.rds (from https://www.wilsonlab.org/public/DEET_data/intermediary_files/DEET_short_final_Aug2022.rds)
    containing per-comparison DESeq2 tables (baseMean, log2FoldChange, padj, stat)
    for all 3,162 DEET comparisons.
  - DEET_metadata_table_2023-01-28.txt
  - data/protein_coding_genes.txt.gz

Why signed DESeq2 test statistic (`stat`) instead of `padj`:
  - `padj` (Benjamini-Hochberg FDR) clamps the majority of genes in each comparison
    to `padj = 1` (and creates step-function plateaus elsewhere), causing massive ties.
  - Within every comparison, DESeq2 computes unadjusted p-values directly from `stat`
    via a strictly monotonic 1-to-1 survival function:
      * Wald comparisons: p_unadj = 2 * Phi(-|stat|)
      * LRT comparisons:  p_unadj = P(chi^2_df > max(stat, 0))  (fixed df per comparison)
  - Ranking by `sign(log2FoldChange) * stat_magnitude` (with `log2FoldChange` breaking
    any remaining ties, e.g. undetected genes with stat = 0) produces the EXACT same
    within-comparison gene ordering as `sign(log2FoldChange) * -log10(p_unadjusted)`
    without BH `padj = 1` ties and without float64 underflow at p < 1e-308.

Output shards (lossless doubled uint16 average ranks, Zstandard-compressed):
  - data/deet_ranks_sra.parquet  (SRA, SRA-manual, RecountBrain; 611 comparisons)
  - data/deet_ranks_tcga.parquet (TCGA; 957 comparisons)
  - data/deet_ranks_gtex.parquet (GTEx; 1,594 comparisons)
  - data/deet_metadata.parquet   (3,162 comparisons)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time
import unicodedata

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SHARD_SOURCES: dict[str, tuple[str, ...]] = {
    "sra": ("SRA", "SRA-manual", "RecountBrain"),
    "tcga": ("TCGA",),
    "gtex": ("GTEx",),
}

R_EXTRACT_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
rdata_path <- args[1]
protein_coding_path <- args[2]
out_genes_path <- args[3]
out_profiles_path <- args[4]
out_bin_path <- args[5]
out_stats_path <- args[6]

t0 <- proc.time()[["elapsed"]]
pc_allowed <- unique(trimws(readLines(protein_coding_path, warn = FALSE)))
pc_allowed <- pc_allowed[nzchar(pc_allowed)]

cat(sprintf("Loading %s ...\n", rdata_path))
env <- new.env(parent = emptyenv())
loaded_names <- load(rdata_path, envir = env)
obj <- env[[loaded_names[1]]]
rm(env)

n_cols <- length(obj)
profile_names <- names(obj)
if (is.null(profile_names) || any(!nzchar(profile_names))) {
  stop("DEET_short_final is missing comparison names!")
}
cat(sprintf("Loaded %d comparisons in %.1fs.\n", n_cols, proc.time()[["elapsed"]] - t0))

# Check column structure across all comparisons and collect ordered union of protein-coding genes
col_sigs <- unique(lapply(obj, function(x) colnames(x$DE)))
cat("Distinct DE column signatures:", length(col_sigs), "\n")
for (sig in col_sigs) {
  cat("  cols:", paste(sig, collapse = ", "), "\n")
}

all_genes <- character(0)
seen_env <- new.env(hash = TRUE, parent = emptyenv(), size = 30000L)
for (j in seq_len(n_cols)) {
  rn <- rownames(obj[[j]]$DE)
  rn_pc <- rn[rn %in% pc_allowed]
  new_g <- rn_pc[!vapply(rn_pc, exists, logical(1), envir = seen_env, inherits = FALSE)]
  if (length(new_g) > 0L) {
    for (g in new_g) assign(g, TRUE, envir = seen_env)
    all_genes <- c(all_genes, new_g)
  }
}
all_genes <- sort(all_genes)
n_genes <- length(all_genes)
cat(sprintf("Protein-coding gene universe across all comparisons: %d genes.\n", n_genes))
if (2L * n_genes > 65535L) {
  stop("2 * n_genes exceeds uint16 max!")
}

doubled_mat <- matrix(0L, nrow = n_genes, ncol = n_cols)
n_wald <- 0L
n_lrt <- 0L
unique_rank_counts <- integer(n_cols)
detected_counts <- integer(n_cols)
monotonic_violations <- 0L

for (j in seq_len(n_cols)) {
  df <- obj[[j]]$DE
  obj[[j]] <- list(NULL)  # release memory progressively

  idx <- match(all_genes, rownames(df))
  detected <- !is.na(idx)
  detected_counts[j] <- sum(detected)

  lfc <- numeric(n_genes)
  stat <- numeric(n_genes)
  lfc[detected] <- df$log2FoldChange[idx[detected]]
  stat[detected] <- df$stat[idx[detected]]

  lfc[!is.finite(lfc)] <- 0.0
  stat[!is.finite(stat)] <- 0.0

  # Distinguish Wald (signed z-statistic) vs LRT (non-negative deviance G^2)
  is_wald <- (sum(stat < -0.5) > 10L) && (min(stat) < -1.0)
  if (is_wald) {
    n_wald <- n_wald + 1L
    stat_mag <- abs(stat)
  } else {
    n_lrt <- n_lrt + 1L
    stat_mag <- pmax(stat, 0.0)
  }

  # Check monotonicity against non-clamped padj (< 0.999)
  if ("padj" %in% colnames(df)) {
    padj_det <- df$padj[idx[detected]]
    sm_det <- stat_mag[detected]
    ok <- is.finite(padj_det) & (padj_det < 0.999) & (sm_det > 0)
    if (sum(ok) > 2L) {
      ord_check <- order(sm_det[ok], decreasing = TRUE)
      if (any(diff(padj_det[ok][ord_check]) < -1e-9)) {
        monotonic_violations <- monotonic_violations + 1L
      }
    }
  }

  primary <- sign(lfc) * stat_mag
  secondary <- lfc

  ord <- order(primary, secondary, method = "radix")
  p_s <- primary[ord]
  s_s <- secondary[ord]

  is_new <- c(TRUE, (p_s[-1L] != p_s[-n_genes]) | (s_s[-1L] != s_s[-n_genes]))
  starts <- which(is_new)
  ends <- c(starts[-1L] - 1L, n_genes)
  counts <- ends - starts + 1L
  # starts and ends are 1-based in R, so doubled average rank is (starts + ends)
  group_doubled <- starts + ends

  col_doubled <- integer(n_genes)
  col_doubled[ord] <- rep.int(group_doubled, counts)
  doubled_mat[, j] <- col_doubled
  unique_rank_counts[j] <- length(starts)
}

writeLines(all_genes, out_genes_path)
writeLines(profile_names, out_profiles_path)

con <- file(out_bin_path, "wb")
# Write column-major uint16 (size = 2 bytes, little-endian)
writeBin(as.integer(doubled_mat), con, size = 2L, endian = "little")
close(con)

stats_json <- sprintf(
  '{"n_genes": %d, "n_cols": %d, "n_wald": %d, "n_lrt": %d, "monotonic_violations": %d, "median_detected": %.1f, "median_unique_ranks": %.1f}',
  n_genes, n_cols, n_wald, n_lrt, monotonic_violations,
  median(detected_counts), median(unique_rank_counts)
)
writeLines(stats_json, out_stats_path)
cat("Extraction stats:", stats_json, "\n")
"""


def extract_ranks_via_r(
    rdata_path: Path, protein_coding_path: Path
) -> tuple[np.ndarray, list[str], np.ndarray, dict]:
    with tempfile.TemporaryDirectory(prefix="deet_extract_") as tmp:
        tmp_dir = Path(tmp)
        r_script = tmp_dir / "extract.R"
        genes_txt = tmp_dir / "genes.txt"
        profiles_txt = tmp_dir / "profiles.txt"
        ranks_bin = tmp_dir / "doubled_ranks.bin"
        stats_json = tmp_dir / "stats.json"

        r_script.write_text(R_EXTRACT_SCRIPT, encoding="utf-8")
        subprocess.run(
            [
                "Rscript",
                str(r_script),
                str(rdata_path),
                str(protein_coding_path),
                str(genes_txt),
                str(profiles_txt),
                str(ranks_bin),
                str(stats_json),
            ],
            check=True,
        )

        genes = np.asarray(
            genes_txt.read_text(encoding="utf-8").splitlines(), dtype=str
        )
        profiles = profiles_txt.read_text(encoding="utf-8").splitlines()
        stats = json.loads(stats_json.read_text(encoding="utf-8"))

        raw_u16 = np.fromfile(ranks_bin, dtype="<u2")
        expected_len = len(genes) * len(profiles)
        if len(raw_u16) != expected_len:
            raise ValueError(
                f"Binary rank buffer length mismatch: {len(raw_u16)} vs {expected_len}"
            )
        # R wrote column-major (Fortran order)
        doubled_ranks = np.ndarray(
            shape=(len(genes), len(profiles)),
            dtype=np.uint16,
            buffer=raw_u16.tobytes(),
            order="F",
        ).copy(order="C")

    return genes, profiles, doubled_ranks, stats


def verify_directions(
    genes: np.ndarray,
    profiles: list[str],
    doubled_ranks: np.ndarray,
    meta: pd.DataFrame,
) -> None:
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    col_to_idx = {c: j for j, c in enumerate(profiles)}
    midpoint = len(genes) + 1  # in doubled rank units, midpoint is (N + 1)

    up_ok = 0
    up_total = 0
    down_ok = 0
    down_total = 0

    for row in meta.itertuples(index=False):
        j = col_to_idx[row.DEET_ID]
        col = doubled_ranks[:, j].astype(np.float64)

        if isinstance(row.DEGs_Up, str) and row.DEGs_Up:
            u_idx = [gene_to_idx[g] for g in row.DEGs_Up.split(";") if g in gene_to_idx]
            if u_idx:
                up_total += 1
                if col[u_idx].mean() > midpoint:
                    up_ok += 1

        if isinstance(row.DEGs_Down, str) and row.DEGs_Down:
            d_idx = [gene_to_idx[g] for g in row.DEGs_Down.split(";") if g in gene_to_idx]
            if d_idx:
                down_total += 1
                if col[d_idx].mean() < midpoint:
                    down_ok += 1

    print(
        f"Direction verification across protein-coding top DEGs:\n"
        f"  DEGs.Up   mean rank > midpoint: {up_ok:,} / {up_total:,} "
        f"({100 * up_ok / max(1, up_total):.2f}%)\n"
        f"  DEGs.Down mean rank < midpoint: {down_ok:,} / {down_total:,} "
        f"({100 * down_ok / max(1, down_total):.2f}%)"
    )


def write_rank_shard(
    out_path: Path,
    genes: np.ndarray,
    profile_names: list[str],
    doubled_sub: np.ndarray,
) -> None:
    arrays = [pa.array(genes, type=pa.string())] + [
        pa.array(doubled_sub[:, j], type=pa.uint16())
        for j in range(doubled_sub.shape[1])
    ]
    schema = pa.schema(
        [pa.field("gene_symbol", pa.string())]
        + [pa.field(name, pa.uint16()) for name in profile_names],
        metadata={
            b"rank_encoding": b"doubled_uint16",
            b"ranking_metric": (
                b"sign(log2FC) * DESeq2_stat_magnitude "
                b"(unadjusted p-value order), tie-broken by log2FC"
            ),
        },
    )
    table = pa.Table.from_arrays(arrays, schema=schema)
    pq.write_table(
        table,
        out_path,
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(
        f"Wrote {out_path.name}: {doubled_sub.shape[0]:,} genes x "
        f"{doubled_sub.shape[1]:,} profiles ({size_mb:.2f} MB)"
    )


# Three SRA comparisons in DEET_metadata_eachcomparison_SRA.rds where upstream
# DESeq2 factor levels were ordered manually rather than via ICU English collation:
SRA_EXPLICIT_DOWN_UP: dict[str, tuple[str, str]] = {
    "SRP066956_13": (
        "pc9 derivative: PC9-GR2",
        "pc9 derivative: PC9 WZ4002-tolerant",
    ),
    "SRP066956_15": (
        "pc9 derivative: PC9-GR3",
        "pc9 derivative: PC9 gefitinib-tolerant",
    ),
    "SRP066956_16": (
        "pc9 derivative: PC9-GR2",
        "pc9 derivative: PC9 gefitinib-tolerant",
    ),
}

SRP056733_AGENT_MAP: dict[str, str] = {
    "agent: none": "uninfected control",
    "agent: Rv": "Mycobacterium tuberculosis (Rv)",
    "agent: Rv+": "heat-inactivated M. tuberculosis H37Rv (Rv+)",
    "agent: GC": "Mycobacterium tuberculosis GC1237",
    "agent: BCG": "bacillus Calmette-Guérin (BCG)",
    "agent: Staph": "Staphylococcus epidermidis",
    "agent: Smeg": "Mycobacterium smegmatis",
    "agent: Yers": "Yersinia pseudotuberculosis",
    "agent: Salm": "Salmonella typhimurium",
}

# Official NCI GDC TCGA study abbreviations:
TCGA_TISSUE_MAP: dict[str, str] = {
    "ACC": "Adrenocortical Carcinoma",
    "BLCA": "Bladder Urothelial Carcinoma",
    "BRCA": "Breast Invasive Carcinoma",
    "CESC": "Cervical Squamous Cell Carcinoma and Endocervical Adenocarcinoma",
    "CHOL": "Cholangiocarcinoma",
    "COAD": "Colon Adenocarcinoma",
    "DLBC": "Lymphoid Neoplasm Diffuse Large B-cell Lymphoma",
    "ESCA": "Esophageal Carcinoma",
    "GBM": "Glioblastoma Multiforme",
    "HNSC": "Head and Neck Squamous Cell Carcinoma",
    "KICH": "Kidney Chromophobe",
    "KIRC": "Kidney Renal Clear Cell Carcinoma",
    "KIRP": "Kidney Renal Papillary Cell Carcinoma",
    "LAML": "Acute Myeloid Leukemia",
    "LGG": "Brain Lower Grade Glioma",
    "LIHC": "Liver Hepatocellular Carcinoma",
    "LUAD": "Lung Adenocarcinoma",
    "LUSC": "Lung Squamous Cell Carcinoma",
    "MESO": "Mesothelioma",
    "OV": "Ovarian Serous Cystadenocarcinoma",
    "PAAD": "Pancreatic Adenocarcinoma",
    "PCPG": "Pheochromocytoma and Paraganglioma",
    "PRAD": "Prostate Adenocarcinoma",
    "READ": "Rectum Adenocarcinoma",
    "SARC": "Sarcoma",
    "SKCM": "Skin Cutaneous Melanoma",
    "STAD": "Stomach Adenocarcinoma",
    "TGCT": "Testicular Germ Cell Tumors",
    "THCA": "Thyroid Carcinoma",
    "THYM": "Thymoma",
    "UCEC": "Uterine Corpus Endometrial Carcinoma",
    "UCS": "Uterine Carcinosarcoma",
    "UVM": "Uveal Melanoma",
}


def format_tissue_label(tissue: str, source: str) -> str:
    cleaned = _clean_text(tissue)
    if source == "TCGA":
        return TCGA_TISSUE_MAP.get(cleaned, cleaned)
    if source == "GTEx":
        return cleaned.replace("_", " ")
    return cleaned


def _clean_text(text: str) -> str:
    text = (
        text.replace("Î²", "β")
        .replace("Ã©", "é")
        .replace("\xc2\xa0", " ")
        .replace("\xa0", " ")
        .replace("Â", "")
    )
    return re.sub(r"\s+", " ", text).strip()


def _icu_char_primary(ch: str) -> tuple[int, str]:
    """Primary ICU root/English collation bucket matching Linux R factor() order."""
    if ch.isspace():
        return (0, " ")
    cat = unicodedata.category(ch)
    if cat.startswith("P"):
        return (1, ch)
    if cat.startswith("S"):
        return (2, ch)
    if ch.isdigit():
        return (3, ch)
    if ch == "ß":
        return (4, "ss")
    if ch == "µ":
        return (4, "μ")
    base = "".join(
        c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c)
    ).lower()
    return (4, base or ch.lower())


def _icu_en_key(text: str) -> tuple:
    cleaned = _clean_text(text)
    primary = tuple(_icu_char_primary(ch) for ch in cleaned)
    tertiary = tuple(
        0 if ch.islower() else (1 if ch.isupper() else 0) for ch in cleaned
    )
    return (primary, tertiary, cleaned)


def _parse_sample_entry(entry: str) -> tuple[str, int]:
    match = re.match(r"^(.*):(\d+)$", entry.strip())
    if not match:
        raise ValueError(f"Cannot parse sample entry: {entry!r}")
    return match.group(1).strip(), int(match.group(2))


def resolve_up_down_groups(
    deet_id: str, source: str, samples_up: str, samples_down: str
) -> tuple[str, int, str, int]:
    """Return (raw_down, n_down, raw_up, n_up) matching DESeq2 log2FC sign.

    In DEET's raw metadata table, Samples.Up and Samples.Down were ordered by
    sample size (n_1 <= n_2) rather than fold-change sign. DESeq2 sets the
    first factor level as the reference denominator (Group.Down, log2FC < 0)
    and the second level as the numerator (Group.Up, log2FC > 0):
      - GTEx, TCGA, RecountBrain: encoded directly in DEET.ID as <Down>_vs_<Up>.
      - SRA, SRA-manual: R factor() order under Linux ICU English collation
        (plus 3 explicit factor orders in SRP066956).
    """
    g_a, n_a = _parse_sample_entry(samples_up)
    g_b, n_b = _parse_sample_entry(samples_down)

    if source in {"GTEx", "TCGA", "RecountBrain"}:
        stem = (
            re.sub(r"_strat_.*$", "", deet_id) if source == "RecountBrain" else deet_id
        )
        ab = stem.endswith(f"_{g_a}_vs_{g_b}")
        ba = stem.endswith(f"_{g_b}_vs_{g_a}")
        if ab and not ba:
            return g_a, n_a, g_b, n_b
        if ba and not ab:
            return g_b, n_b, g_a, n_a

        na_str = re.sub(r"[^0-9A-Za-z]+", "", g_a)
        nb_str = re.sub(r"[^0-9A-Za-z]+", "", g_b)
        for idx in [m.start() for m in re.finditer(r"_vs_", stem)]:
            right_raw = stem[idx + 4 :]
            left_prefix = stem[:idx]
            r_norm = re.sub(r"[^0-9A-Za-z]+", "", right_raw)
            l_norm = re.sub(r"[^0-9A-Za-z]+", "", left_prefix)
            ab_norm = (r_norm == nb_str) and l_norm.endswith(na_str)
            ba_norm = (r_norm == na_str) and l_norm.endswith(nb_str)
            if ab_norm and not ba_norm:
                return g_a, n_a, g_b, n_b
            if ba_norm and not ab_norm:
                return g_b, n_b, g_a, n_a

        raise ValueError(f"Could not match DEET.ID {deet_id!r} to {g_a!r} vs {g_b!r}")

    if source in {"SRA", "SRA-manual"}:
        if deet_id in SRA_EXPLICIT_DOWN_UP:
            exp_down, exp_up = SRA_EXPLICIT_DOWN_UP[deet_id]
            if (g_a, g_b) == (exp_down, exp_up):
                return g_a, n_a, g_b, n_b
            if (g_b, g_a) == (exp_down, exp_up):
                return g_b, n_b, g_a, n_a
            raise ValueError(
                f"Explicit override mismatch for {deet_id}: {g_a!r}, {g_b!r}"
            )

        if _icu_en_key(g_a) <= _icu_en_key(g_b):
            return g_a, n_a, g_b, n_b
        return g_b, n_b, g_a, n_a

    raise ValueError(f"Unknown Source {source!r} for {deet_id!r}")


GTEX_RACE_DISPLAY_MAP: dict[str, str] = {
    "asian": "Asian",
    "black": "Black",
    "white": "White",
}

SRA_CATEGORY_OVERRIDES: dict[str, str] = {
    # Asthma vs Control donor comparisons (stratified by treatment) copy-pasted as "treatment":
    "SRP043162_7": "disease",
    "SRP043162_8": "disease",
    "SRP043162_9": "disease",
    # FSHD muscular dystrophy patient biopsies/myotubes vs control labeled "treatment":
    "SRP043221_6": "disease",
    "SRP043221_27": "disease",
    # Glioblastoma MRI-localized biopsies: tumor margin vs core ("disease stage") and tumor vs non-neoplastic brain ("disease"):
    "SRP044668_1": "disease stage",
    "SRP044668_2": "disease",
    "SRP044668_3": "disease",
    # Senescent cells expressing ZFP36L1wt vs senescent cells (matches sibling SRP046254_4,5):
    "SRP046254_6": "cell type + genetic modification",
    # Leiomyosarcoma tumor vs normal smooth muscle copy-pasted from subtype rows as "disease stage":
    "SRP049097_3": "disease",
    # Bone-marrow erythroleukemia vs normal labeled "cell type" (sibling SRP050493_1 MDS vs normal is "disease"):
    "SRP050493_3": "disease",
    # Human primordial germ cells across gestational weeks (4-17 wk gestation) labeled "timepoint":
    "SRP050499_11": "developmental stage",
    "SRP050499_12": "developmental stage",
    "SRP050499_16": "developmental stage",
    "SRP050499_27": "developmental stage",
    "SRP050499_30": "developmental stage",
    # ADARB1 long-isoform transfection vs empty vector (matches sibling SRP055105_1,2):
    "SRP055105_3": "treatment + genetic modification",
    # Whole-blood brucellosis and leishmaniasis patient cohorts vs healthy controls labeled "treatment":
    "SRP059172_1": "disease",
    "SRP059172_3": "disease",
    "SRP059172_22": "disease",
    "SRP059172_32": "disease",
    "SRP059172_34": "disease",
    "SRP059172_37": "disease",
    "SRP059172_39": "disease",
    "SRP059172_43": "disease",
    "SRP059172_44": "disease",
    "SRP059172_47": "disease",
    "SRP059172_49": "disease",
    # Normal hematopoietic FACS populations and mesoderm-differentiated iPSCs labeled "disease":
    "SRP059205_1": "cell type + differentiation",
    "SRP059205_21": "cell type",
    "SRP059205_28": "cell type",
    "SRP059205_49": "cell type + differentiation",
    # Small-molecule treatments (SpliceostatinA, VWR #BDH1135-4LP vs DMSO) copy-pasted from siRNA rows:
    "SRP062389_42": "treatment",
    "SRP062389_44": "treatment",
    # Fixed-timepoint (24 hr MOCK) chemical TPA vs DMSO comparison matching sibling SRP064561 rows:
    "SRP064561_2": "treatment",
}


def format_deet_name(deet_name: str, source: str) -> str:
    cleaned = _clean_text(deet_name)
    if source == "GTEx" and "race." in cleaned:
        cleaned = re.sub(
            r"(race\.\s*)(asian|black|white)(\s+vs\s+)(asian|black|white)\b",
            lambda m: (
                f"{m.group(1)}{GTEX_RACE_DISPLAY_MAP[m.group(2)]}"
                f"{m.group(3)}{GTEX_RACE_DISPLAY_MAP[m.group(4)]}"
            ),
            cleaned,
        )
    return cleaned


def normalize_category(
    deet_id: str, deet_name: str, source: str, raw_category: str
) -> str:
    """Fix upstream DEET category misclassifications and split-variable inconsistencies."""
    cat = _clean_text(raw_category)

    if source == "GTEx":
        if "_sex_sex_female_vs_male" in deet_id:
            return "sex"
        if any(
            tok in deet_id
            for tok in ("_MHDLYSIS_", "_MHORGNTP_", "_MHPLLABS_", "_MHTXCEXP_")
        ):
            return "treatment"
        if any(tok in deet_id for tok in ("_MHPNMIAB_", "_MHWTLSUA_", "_fever_")):
            return "disease"
        if any(tok in deet_id for tok in ("_MHSRGHM_", "_MHOPNWND_", "_MHNPHYS4W_")):
            return "other"
        return cat

    if source == "TCGA":
        parts = deet_name.split(". ")
        var_raw = parts[1].strip() if len(parts) >= 2 else ""
        if var_raw in {
            "xml targeted molecular therapy",
            "xml preoperative antiseizure meds",
        }:
            return "treatment"
        if var_raw in {
            "xml neoplasm histologic grade",
            "cgc case new tumor event after initial treatment",
        }:
            return "disease stage"
        if var_raw == "gdc cases.samples.sample type":
            return "disease" if "Solid Tissue Normal" in deet_name else "disease stage"
        if var_raw == "cgc case histological diagnosis":
            return "tissue type"
        return cat

    if source in {"SRA", "SRA-manual"} and deet_id in SRA_CATEGORY_OVERRIDES:
        return SRA_CATEGORY_OVERRIDES[deet_id]

    return cat


def format_group_display_label(
    raw_group: str,
    other_raw_group: str,
    deet_id: str,
    deet_name: str,
    source: str,
) -> str:
    """Format a comparison group into a concise, unambiguous display label."""
    g = _clean_text(raw_group)
    other = _clean_text(other_raw_group)
    name = _clean_text(deet_name)

    if source == "GTEx":
        if g in GTEX_RACE_DISPLAY_MAP:
            return GTEX_RACE_DISPLAY_MAP[g]
        if g in {"20_39", "40_59", "60_79"}:
            return f"age {g.replace('_', '-')}"
        if g in {"yes", "no"}:
            parts = name.split(". ")
            var_desc = parts[1] if len(parts) >= 3 else name
            var_desc = re.sub(r"^(?:MH|DTH|TR|SM)[A-Z0-9_]*\s+", "", var_desc).strip()
            var_desc = re.sub(r"\s+(?:A|DLRD)$", "", var_desc).strip()
            if g == "yes":
                return var_desc
            if var_desc.startswith("ever "):
                return "n" + var_desc
            if var_desc.startswith("bed bound"):
                return f"not {var_desc}"
            return f"no {var_desc}"
        if g == "normal" and "body temperature" in name:
            return "normal body temperature"
        return g

    if source == "TCGA":
        parts = name.split(". ")
        var_raw = parts[1].strip() if len(parts) >= 2 else ""
        var_clean = re.sub(r"^(?:cgc case |xml |gdc cases )", "", var_raw).strip()
        var_clean = re.sub(
            r"\s+indicator\s+(?:ABSENT|NO)\s+vs\s+(?:PRESENT|YES)$",
            "",
            var_clean,
            flags=re.IGNORECASE,
        ).strip()
        var_clean = re.sub(r"\s+(?:present|found)$", "", var_clean).strip()

        if var_raw == "drugFinal" and g.lower() == "none":
            return f"no {other}"
        if g.upper() in {"YES", "PRESENT"} and var_clean:
            return var_clean
        if g.upper() in {"NO", "ABSENT"} and var_clean:
            return f"no {var_clean}"
        if var_clean == "immunohistochemistry positive cell score" and g in {
            "1+",
            "2+",
            "3+",
            "4+",
        }:
            return f"IHC {g}"
        if var_clean == "neoplasm histologic grade" and re.fullmatch(r"G[1-4BX]", g):
            return f"histologic grade {g}"
        if var_clean == "stage event pathologic stage" and g == "IS":
            return "Stage IS"
        if var_clean == "anatomic neoplasm subdivision" and g in {
            "Left",
            "Right",
            "Dome",
            "Neck",
            "Body",
            "Lower",
            "Upper",
        }:
            return f"{g} ({var_clean})"
        if g in {"Low", "Normal", "Elevated"} and var_clean:
            return f"{g} ({var_clean})"
        if ("receptor level cell percent" in var_clean) and "%" in g:
            short_rec = var_clean.replace(" level cell percent category", "")
            return f"{g} ({short_rec})"
        return g

    if source == "RecountBrain":
        if g in {"Disease", "Control"}:
            match = re.search(rf"\b{g}\s*\(([^)]+)\)", name)
            if match:
                inner = match.group(1).strip().replace("Rhetts Syndome", "Rett syndrome")
                return f"{g} ({inner})"
        return g

    # SRA and SRA-manual
    if deet_id.startswith("SRP056733_") and g in SRP056733_AGENT_MAP:
        return SRP056733_AGENT_MAP[g]
    if deet_id.startswith("SRP042228_"):
        if g.endswith(": CD"):
            return "Crohn's disease (CD)"
        if g.endswith(": UC"):
            return "ulcerative colitis (UC)"

    if ": " in g:
        prefix, val = g.split(": ", 1)
        prefix_l = prefix.strip().lower()
        val_s = val.strip()

        if val_s == "none (untreated control)":
            return "untreated control"
        if val_s.upper() == "TRUE":
            return prefix.strip()
        if val_s.upper() == "FALSE":
            return f"no {prefix.strip()}"

        keep_prefixes = {
            "time",
            "reprogramming time point (day)",
            "number of days in vitro in terminal differentiation conditions",
            "braak stage",
            "grade",
            "subtypes",
            "sample type",
            "imortalization",
            "immortalization",
            "passage",
            "age",
            "stage",
            "culture",
        }
        if (
            prefix_l in keep_prefixes
            or re.fullmatch(r"[\d.\-+]+", val_s)
            or val_s.upper() in {"YES", "NO", "I", "II", "III", "IV", "V", "VI", "E"}
        ):
            return f"{prefix.strip()}: {val_s}"

        if val_s.lower() == "none":
            if prefix_l in {"agent", "bacterial infection", "virus", "infection"}:
                return "uninfected control"
            if prefix_l in {
                "treatment",
                "treated with",
                "screening chemicals",
                "chemical",
                "compound",
                "drug",
            }:
                return "untreated control"
            return f"no {prefix.strip()}"

        return val_s

    return g


def build_metadata(meta_txt_path: Path, profile_set: set[str]) -> pd.DataFrame:
    meta = pd.read_csv(meta_txt_path, sep="\t", dtype=str, keep_default_na=False)
    missing = profile_set.difference(meta["DEET.ID"])
    if missing:
        raise ValueError(f"Profiles missing from metadata: {sorted(missing)[:5]}")

    meta["DEET.Name"] = [
        format_deet_name(name, source)
        for name, source in zip(meta["DEET.Name"], meta["Source"])
    ]
    meta["Category"] = [
        normalize_category(deet_id, name, source, cat)
        for deet_id, name, source, cat in zip(
            meta["DEET.ID"], meta["DEET.Name"], meta["Source"], meta["Category"]
        )
    ]
    meta["Tissue"] = [
        format_tissue_label(tissue, source)
        for tissue, source in zip(meta["Tissue"], meta["Source"])
    ]

    groups_down: list[str] = []
    groups_up: list[str] = []
    sample_groups: list[str] = []

    for deet_id, name, source, s_up, s_dn in zip(
        meta["DEET.ID"],
        meta["DEET.Name"],
        meta["Source"],
        meta["Samples.Up"],
        meta["Samples.Down"],
    ):
        raw_down, n_down, raw_up, n_up = resolve_up_down_groups(
            deet_id, source, s_up, s_dn
        )
        lbl_down = format_group_display_label(raw_down, raw_up, deet_id, name, source)
        lbl_up = format_group_display_label(raw_up, raw_down, deet_id, name, source)
        groups_down.append(lbl_down)
        groups_up.append(lbl_up)
        sample_groups.append(f"{lbl_down} (n={n_down}) vs {lbl_up} (n={n_up})")

    meta["Group.Down"] = groups_down
    meta["Group.Up"] = groups_up
    meta["Sample.Groups"] = sample_groups

    for int_col in ("Samples", "numDEs", "numDEup", "numDEdown", "clusterID"):
        meta[int_col] = pd.to_numeric(meta[int_col], errors="coerce").astype("Int64")

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--short-final",
        type=Path,
        default=Path("/tmp/DEET_short_final_Aug2022.rds"),
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("/tmp/DEET_metadata_table_2023-01-28.txt"),
    )
    parser.add_argument(
        "--protein-coding",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "protein_coding_genes.txt.gz",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    genes, profiles, doubled_ranks, stats = extract_ranks_via_r(
        args.short_final, args.protein_coding
    )
    if stats["monotonic_violations"] != 0:
        raise ValueError(
            f"Detected {stats['monotonic_violations']} monotonicity violations vs padj!"
        )

    meta = build_metadata(args.meta, set(profiles))
    verify_view = meta.rename(
        columns={
            "DEET.ID": "DEET_ID",
            "DEGs.Up": "DEGs_Up",
            "DEGs.Down": "DEGs_Down",
        }
    )
    verify_directions(genes, profiles, doubled_ranks, verify_view)

    col_to_idx = {c: j for j, c in enumerate(profiles)}
    total_written = 0
    for shard_key, source_labels in SHARD_SOURCES.items():
        shard_ids = meta.loc[
            meta["Source"].isin(source_labels), "DEET.ID"
        ].tolist()
        shard_indices = [col_to_idx[cid] for cid in shard_ids]
        write_rank_shard(
            args.out_dir / f"deet_ranks_{shard_key}.parquet",
            genes,
            shard_ids,
            doubled_ranks[:, shard_indices],
        )
        total_written += len(shard_ids)

    if total_written != len(profiles):
        raise ValueError(
            f"Shard count mismatch: wrote {total_written}, expected {len(profiles)}"
        )

    meta_out = args.out_dir / "deet_metadata.parquet"
    meta.to_parquet(meta_out, index=False, compression="zstd", compression_level=9)
    print(
        f"Wrote {meta_out.name}: {len(meta):,} rows "
        f"({meta_out.stat().st_size / 1e6:.2f} MB) in {time.perf_counter() - t0:.1f}s total."
    )


if __name__ == "__main__":
    main()
