"""Polygenic differential expression gene-set tester for the DEET human atlas."""

from __future__ import annotations

from functools import partial
import hashlib
from io import BytesIO
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "deet_streamlit_matplotlib"),
)

import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
import streamlit as st

from analysis import (
    RankMatrix,
    convert_to_human,
    load_metadata,
    load_ortholog_map,
    load_rank_matrix,
    parse_gene_list,
    run_auroc_analysis,
)


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

DATASETS: dict[str, tuple[tuple[str, ...], str]] = {
    "All comparisons (3,162)": (("sra", "tcga", "gtex"), "all"),
    "SRA / Experimental (611)": (("sra",), "sra"),
    "TCGA (957)": (("tcga",), "tcga"),
    "GTEx (1,594)": (("gtex",), "gtex"),
}

CATEGORIES: dict[str, tuple[str | None, str]] = {
    "All categories (3,162)": (None, "all"),
    "disease (1,103)": ("disease", "disease"),
    "disease stage (518)": ("disease stage", "disease_stage"),
    "treatment (510)": ("treatment", "treatment"),
    "tissue type (217)": ("tissue type", "tissue_type"),
    "other (206)": ("other", "other"),
    "developmental stage (110)": ("developmental stage", "developmental_stage"),
    "phenotype (107)": ("phenotype", "phenotype"),
    "ancestry (93)": ("ancestry", "ancestry"),
}

PAPER_URL = "https://doi.org/10.1093/nargab/lqad003"
ZENODO_URL = "https://zenodo.org/records/7321664"

DEFAULT_GENES = """ADAR
BST2
CMPK2
CXCL10
DDX58
EIF2AK2
EPSTI1
GBP1
HERC5
IFI27
IFI35
IFI44
IFI44L
IFI6
IFIH1
IFIT1
IFIT2
IFIT3
IFITM1
IFITM3
IRF7
IRF9
ISG15
LGALS3BP
MX1
MX2
OAS1
OAS2
OAS3
OASL
PARP9
RSAD2
RTP4
SAMD9L
SP110
STAT1
STAT2
TRIM22
USP18
XAF1"""


@st.cache_resource(show_spinner="Loading DEET signed rank matrix...")
def get_rank_shard(shard_key: str, file_version: tuple[int, int]) -> RankMatrix:
    return load_rank_matrix(DATA_DIR / f"deet_ranks_{shard_key}.parquet")


@st.cache_resource(show_spinner=False)
def get_metadata(file_version: tuple[int, int]) -> pd.DataFrame:
    return load_metadata(DATA_DIR / "deet_metadata.parquet")


@st.cache_resource(show_spinner=False)
def get_ortholog_map():
    return load_ortholog_map(DATA_DIR / "homologene_to_human.csv.gz")


def load_selected_shards(dataset_label: str) -> tuple[RankMatrix, ...]:
    shard_keys, _ = DATASETS[dataset_label]
    shards: list[RankMatrix] = []
    for key in shard_keys:
        shard_path = DATA_DIR / f"deet_ranks_{key}.parquet"
        stat = shard_path.stat()
        shards.append(get_rank_shard(key, (stat.st_mtime_ns, stat.st_size)))
    return tuple(shards)


def filter_shards_by_profiles(
    shards: tuple[RankMatrix, ...],
    allowed_ids: set[str] | None,
) -> tuple[RankMatrix, ...]:
    if allowed_ids is None:
        return shards
    filtered: list[RankMatrix] = []
    for shard in shards:
        keep_idx = [
            idx for idx, profile_id in enumerate(shard.profiles) if profile_id in allowed_ids
        ]
        if keep_idx:
            filtered.append(
                RankMatrix(
                    genes=shard.genes,
                    profiles=tuple(shard.profiles[idx] for idx in keep_idx),
                    ranks=shard.ranks[:, keep_idx],
                )
            )
    return tuple(filtered)


def format_results_table(table: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    merged = table.merge(
        metadata,
        left_on="DEET ID",
        right_on="DEET.ID",
        how="inner",
        sort=False,
    )
    if {"Group.Up", "Group.Down"}.issubset(merged.columns):
        is_up = merged["Direction"] == "Upregulation of input genes"
        is_down = merged["Direction"] == "Downregulation of input genes"
        higher_group = pd.Series("—", index=merged.index, dtype=object)
        lower_group = pd.Series("—", index=merged.index, dtype=object)
        higher_group.loc[is_up] = merged.loc[is_up, "Group.Up"]
        lower_group.loc[is_up] = merged.loc[is_up, "Group.Down"]
        higher_group.loc[is_down] = merged.loc[is_down, "Group.Down"]
        lower_group.loc[is_down] = merged.loc[is_down, "Group.Up"]
        merged["Higher Group"] = higher_group
        merged["Lower Group"] = lower_group

    preferred_columns = [
        "DEET.Name",
        "Higher Group",
        "Lower Group",
        "Source",
        "Tissue",
        "AUROC",
        "pValue",
        "FDR",
        "Samples",
        "Description",
        "Category",
        "DEGs.Up",
        "DEGs.Down",
        "numDEs",
        "numDEup",
        "numDEdown",
        "Sample.Groups",
        "Age.Mean.SD",
        "Sexes",
        "Study.ID",
        "PMID",
        "DEET ID",
    ]
    ordered = [col for col in preferred_columns if col in merged.columns]
    return merged.loc[:, ordered]


@st.cache_data(show_spinner=False)
def create_heatmap_pdf(
    heatmap: pd.DataFrame,
    dataset_label: str,
    label_map: dict[str, str],
) -> bytes:
    clustered_heatmap = heatmap
    if heatmap.shape[0] > 1:
        row_order = leaves_list(
            linkage(heatmap.to_numpy(), method="complete", metric="euclidean")
        )
        clustered_heatmap = clustered_heatmap.iloc[row_order, :]
    if heatmap.shape[1] > 1:
        column_order = leaves_list(
            linkage(heatmap.to_numpy().T, method="complete", metric="euclidean")
        )
        clustered_heatmap = clustered_heatmap.iloc[:, column_order]

    width = min(110, max(10, 3 + clustered_heatmap.shape[1] * 0.12))
    height = max(6, 3 + clustered_heatmap.shape[0] * 0.2)
    figure, axis = plt.subplots(figsize=(width, height))
    image = axis.imshow(
        clustered_heatmap.to_numpy(),
        aspect="auto",
        cmap="viridis",
        interpolation="none",
    )
    axis.set_xticks(range(clustered_heatmap.shape[1]))
    axis.set_xticklabels(
        [label_map.get(col, col)[:90] for col in clustered_heatmap.columns],
        rotation=90,
        fontsize=4,
    )
    axis.set_yticks(range(clustered_heatmap.shape[0]))
    axis.set_yticklabels(clustered_heatmap.index, fontsize=6)
    axis.set_xlabel("DEET comparison")
    axis.set_ylabel("Gene")
    axis.set_title(
        f"DEET human differential expression — {dataset_label}: "
        f"{heatmap.shape[0]} genes across {heatmap.shape[1]:,} comparisons\n"
        "hierarchically clustered; higher ranks indicate stronger upregulation (log2FC > 0)"
    )
    figure.colorbar(image, ax=axis, label="Signed differential expression rank")
    figure.tight_layout()

    output = BytesIO()
    figure.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(figure)
    return output.getvalue()


def format_p_value(value: float) -> str:
    if pd.isna(value):
        return ""
    if 0 < value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3g}"


st.set_page_config(
    page_title="DEET polygenic differential expression tester",
    layout="wide",
)

with st.container(key="page_styles"):
    st.html("""
        <style>
        .st-key-page_styles,
        [data-testid="stLayoutWrapper"]:has(.st-key-page_styles) {
            display: none !important;
        }
        .stMainBlockContainer {
            padding-top: 3.25rem !important;
            padding-bottom: 2rem !important;
        }
        .stMainBlockContainer > div > .stVerticalBlock {
            gap: 0.75rem;
        }
        .st-key-top_header {
            gap: 0.65rem !important;
        }
        .st-key-top_header h1 {
            font-size: 1.85rem !important;
            padding: 0 0 0.25rem 0 !important;
            margin: 0 !important;
            line-height: 1.25 !important;
        }
        .st-key-summary_left {
            gap: 0.45rem !important;
        }
        .st-key-result_downloads {
            gap: 1rem !important;
        }
        .st-key-result_downloads button {
            color: #0068c9;
            text-decoration: underline;
            padding: 0;
            min-height: 0;
        }
        </style>
    """)

with st.sidebar:
    st.markdown(
        "**Test whether a gene set is upregulated or downregulated across "
        "human RNA-seq differential expression results from the Differential "
        "Expression Enrichment Tool (DEET) atlas by Sokolowski et al.**"
    )
    dataset = st.selectbox("DEET source:", options=list(DATASETS), index=0)
    category_label = st.selectbox("Category:", options=list(CATEGORIES), index=0)
    species = st.selectbox(
        "Species of input genes:", options=["Human", "Mouse", "Rhesus Macaque"]
    )

    gene_text = st.text_area(
        "Input your gene list:",
        value=DEFAULT_GENES,
        height=220,
    )
    background_text = st.text_area(
        "Background gene list (optional):",
        value="",
        height=90,
    )
    st.divider()
    st.markdown("**Source data:**")
    st.markdown(
        f"[Differential Expression Enrichment Tool (DEET)]({PAPER_URL})  \n"
        "Sokolowski et al. (*NAR Genomics and Bioinformatics*, 2023) · "
        f"[Zenodo dataset]({ZENODO_URL})"
    )
    st.caption(
        "Uniformly reprocessed human RNA-seq differential expression profiles "
        "from recount2 across SRA, TCGA, and GTEx."
    )
    st.markdown("**Example gene set:**")
    st.markdown(
        "Default genes are 40 canonical human Type I interferon-stimulated genes "
        "([Schoggins et al., *Nature*, 2011](https://doi.org/10.1038/nature09907); "
        "[MSigDB `HALLMARK_INTERFERON_ALPHA_RESPONSE`]"
        "(https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/HALLMARK_INTERFERON_ALPHA_RESPONSE.html))."
    )


saved = None
error_message = None
try:
    parsed_targets = parse_gene_list(gene_text)
    parsed_background = parse_gene_list(background_text)
    if not parsed_targets:
        raise ValueError("Enter at least one target gene.")

    ortholog_map = None if species == "Human" else get_ortholog_map()
    conversion_unmapped = (
        []
        if species == "Human"
        else [
            gene
            for gene in parsed_targets
            if not ortholog_map.get(species, {}).get(gene)
        ]
    )
    target_genes = convert_to_human(parsed_targets, species, ortholog_map)
    background_genes = (
        convert_to_human(parsed_background, species, ortholog_map)
        if parsed_background
        else None
    )
    if not target_genes:
        raise ValueError("None of the input genes could be converted to human symbols.")

    shards = load_selected_shards(dataset)
    meta_stat = (DATA_DIR / "deet_metadata.parquet").stat()
    metadata = get_metadata((meta_stat.st_mtime_ns, meta_stat.st_size))

    selected_category, category_slug = CATEGORIES[category_label]
    allowed_ids = (
        None
        if selected_category is None
        else set(
            metadata.loc[metadata["Category"] == selected_category, "DEET.ID"]
        )
    )
    filtered_shards = filter_shards_by_profiles(shards, allowed_ids)
    if not filtered_shards:
        raise ValueError(
            f"No comparisons in '{dataset}' have category '{selected_category}'."
        )

    result = run_auroc_analysis(
        matrix=filtered_shards,
        target_genes=target_genes,
        background_genes=background_genes,
    )
    total_profiles = sum(len(shard.profiles) for shard in filtered_shards)
    label_map = dict(zip(metadata["DEET.ID"], metadata["DEET.Name"]))
    table_key = "results_table_" + hashlib.blake2b(
        repr(
            (
                dataset,
                selected_category,
                species,
                tuple(target_genes),
                tuple(background_genes or ()),
            )
        ).encode("utf-8"),
        digest_size=8,
    ).hexdigest()
    saved = {
        "table": format_results_table(result.table, metadata),
        "table_key": table_key,
        "heatmap": result.heatmap,
        "label_map": label_map,
        "matched": result.matched_gene_count,
        "input": result.input_gene_count,
        "background": result.background_gene_count,
        "profile_count": total_profiles,
        "selected_category": selected_category,
        "category_slug": category_slug,
        "conversion_unmapped": conversion_unmapped,
        "matrix_unmatched": list(result.unmatched_genes),
    }
except (OSError, ValueError, pd.errors.ParserError) as error:
    error_message = str(error)


with st.container(key="top_header"):
    st.title("DEET polygenic differential expression tester")
    if error_message is not None:
        st.error(error_message)
    if saved is not None:
        results_table = saved["table"]
        unmapped_count = len(saved["conversion_unmapped"]) + len(
            saved["matrix_unmatched"]
        )
        dataset_short = dataset.split(" (", 1)[0]
        scope_parts = [dataset_short]
        if saved["selected_category"] is not None:
            scope_parts.append(f"Category: {saved['selected_category']}")
        scope_parts.append(f"{saved['profile_count']:,} comparisons")
        if species != "Human":
            scope_parts.append("unique converted human symbols")

        summary_columns = st.columns([4, 1], vertical_alignment="center")
        with summary_columns[0]:
            with st.container(key="summary_left"):
                st.markdown(
                    f"Genes found in data: {saved['matched']} of {saved['input']} "
                    f"&nbsp;·&nbsp; Background genes: {saved['background']:,}"
                )
                st.caption(" · ".join(scope_parts))
        with summary_columns[1]:
            with st.popover(f"Unmapped genes ({unmapped_count})"):
                if saved["conversion_unmapped"]:
                    st.markdown("**Input genes without a human ortholog:**")
                    st.code("\n".join(saved["conversion_unmapped"]), language=None)
                if saved["matrix_unmatched"]:
                    label = (
                        "Converted human symbols not found in the selected data/background:"
                        if species != "Human"
                        else "Input genes not found in the selected data/background:"
                    )
                    st.markdown(f"**{label}**")
                    st.code("\n".join(saved["matrix_unmatched"]), language=None)
                if unmapped_count == 0:
                    st.success(
                        "All input genes were mapped and found in the selected data."
                    )

if saved is not None:
    results_table = saved["table"]

    display_table = results_table.style.format(
        {
            "AUROC": lambda value: f"{value:.3g}",
            "pValue": format_p_value,
            "FDR": format_p_value,
        }
    )
    with st.container(key=saved["table_key"]):
        st.dataframe(
            display_table,
            column_config={
                "DEET.Name": st.column_config.TextColumn("DEET Name", width=280),
                "Higher Group": st.column_config.TextColumn("Higher Group", width="medium"),
                "Lower Group": st.column_config.TextColumn("Lower Group", width="medium"),
                "Tissue": st.column_config.TextColumn("Tissue", width="medium"),
                "Description": st.column_config.TextColumn("Description", width="medium"),
                "DEGs.Up": st.column_config.TextColumn("Top DEGs (Up)", width="medium"),
                "DEGs.Down": st.column_config.TextColumn("Top DEGs (Down)", width="medium"),
                "Sample.Groups": st.column_config.TextColumn("Sample Groups", width="medium"),
            },
            hide_index=True,
            width="stretch",
            height=600,
        )

    download_prefix = f"deet_human_{DATASETS[dataset][1]}"
    if saved["selected_category"] is not None:
        download_prefix = f"{download_prefix}_{saved['category_slug']}"
    heatmap_scope = (
        dataset
        if saved["selected_category"] is None
        else f"{dataset} ({saved['selected_category']})"
    )
    with st.container(key="result_downloads", horizontal=True):
        st.download_button(
            "Download as CSV",
            data=results_table.to_csv(index=False).encode("utf-8"),
            file_name=f"{download_prefix}_results.csv",
            mime="text/csv",
            type="tertiary",
            on_click="ignore",
        )
        st.download_button(
            "Download heatmap (PDF)",
            data=partial(
                create_heatmap_pdf,
                saved["heatmap"],
                heatmap_scope,
                saved["label_map"],
            ),
            file_name=f"{download_prefix}_heatmap.pdf",
            mime="application/pdf",
            type="tertiary",
            on_click="ignore",
        )
    st.caption(
        "Genes are ranked within each comparison from 1 (most significantly "
        "downregulated, log2FC < 0) to N (most significantly upregulated, log2FC > 0) "
        "using sign(log2FC) × DESeq2 test-statistic magnitude (equivalent to "
        "sign(log2FC) × −log10(p) using unadjusted p-values, avoiding padj = 1 ties), "
        "with log2FC breaking any remaining ties. AUROC compares matched target genes "
        "against non-target background genes via within-comparison rank sums; two-sided "
        "p-values use the continuity-corrected normal approximation, and FDR uses "
        "Benjamini–Hochberg across the filtered comparisons tested."
    )
    st.caption(
        "Direction convention: 'Higher Group' and 'Lower Group' list the two sample "
        "groups in each comparison ordered by higher vs. lower expression of the input "
        "gene set. When AUROC > 0.5, Higher Group is DEET's positive log2FC / Top DEGs "
        "(Up) group; when AUROC < 0.5, Higher Group is DEET's negative log2FC / Top DEGs "
        "(Down) reference group."
    )
