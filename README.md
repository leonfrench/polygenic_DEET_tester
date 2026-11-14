# DEET human polygenic differential expression app

This Streamlit app tests whether a gene set is coordinately upregulated or
downregulated across **3,162 human RNA-seq differential expression comparisons**
from the **Differential Expression Enrichment Tool (DEET)** atlas. Filter by
source (**All comparisons (3,162)** by default, **SRA / Experimental (611)**,
**TCGA (957)**, or **GTEx (1,594)**) and optionally by comparison **Category**
(**All categories** by default, or any major DEET category from `disease` down
to `ancestry`); Benjamini–Hochberg `FDR` is computed strictly within the
filtered comparison set.

## Data and attribution

Source: [Differential Expression Enrichment Tool (DEET)](https://doi.org/10.1093/nargab/lqad003)
— Sokolowski et al., *NAR Genomics and Bioinformatics* (2023), 5(1), lqad003.
DEET datasets are distributed via [Zenodo (record 7321664)](https://zenodo.org/records/7321664)
and the [Wilson Lab DEET repository](https://www.wilsonlab.org/public/DEET_data/).
DEET aggregates uniformly reprocessed human RNA-seq studies from recount2 across
SRA (`SRA`, `SRA-manual`, and `RecountBrain`), TCGA, and GTEx.

Keep these files in `data/` (all under GitHub's 100 MB file limit; no manual
decompression is needed):

| Grouping / Resource | File | Genes / Rows | Profiles | Size |
| --- | --- | ---: | ---: | ---: |
| SRA / Experimental (signed) | `deet_ranks_sra.parquet` | 19,444 | 611 | 25.3 MB |
| TCGA (signed) | `deet_ranks_tcga.parquet` | 19,444 | 957 | 42.9 MB |
| GTEx (signed) | `deet_ranks_gtex.parquet` | 19,444 | 1,594 | 74.2 MB |
| SRA / Experimental (unsigned) | `deet_unsigned_ranks_sra.parquet` | 19,444 | 611 | 25.3 MB |
| TCGA (unsigned) | `deet_unsigned_ranks_tcga.parquet` | 19,444 | 957 | 42.8 MB |
| GTEx (unsigned) | `deet_unsigned_ranks_gtex.parquet` | 19,444 | 1,594 | 74.1 MB |
| Comparison metadata | `deet_metadata.parquet` | 3,162 | — | 0.5 MB |
| Ortholog mapping | `homologene_to_human.csv.gz` | 35,983 | — | 0.2 MB |
| Protein-coding gene list | `protein_coding_genes.txt.gz` | 20,341 | — | 0.05 MB |

Each rank shard starts with `gene_symbol` (19,444 human protein-coding genes),
followed by Zstandard-compressed `uint16` columns storing exact doubled average
ranks (`2 × rank`), decoded on load to `float32` ranks in `[1, 19444]`.
Replacing a data file invalidates its cached object via modification time and
byte size.

## Signed ranking and directionality conventions

Within every comparison, protein-coding genes are ranked from `1` (most
significantly downregulated, `log2FC < 0`) to `19,444` (most significantly
upregulated, `log2FC > 0`) by:

1. **Primary score**: `sign(log2FC) × DESeq2 test-statistic magnitude` (`|stat|` for two-group Wald comparisons; `max(stat, 0)` for likelihood-ratio deviance `G²` comparisons). Because DESeq2 computes unadjusted *p*-values within each comparison directly from `stat` via a strictly monotonic survival function (`2 × Φ(−|stat|)` for Wald; `P(χ²_df > G²)` with fixed `df` per comparison for LRT), this yields the **exact same within-comparison ordering as `sign(log2FC) × −log10(p_unadjusted)`**, avoiding both `padj = 1` Benjamini–Hochberg ties and `float64` *p*-value underflow at extreme significance.
2. **Tie-breaker**: `log2FC` (resolving any identical primary scores, e.g. `stat = 0`).

This signed-significance ranking resolves three directionality pitfalls present
in the earlier R Shiny prototype:

- **Signed vs. unsigned test statistics**: A large fraction of DEET comparisons
  (`1,257` of `3,162` in `DEET_short_final_Aug2022.rds`, and >93% in the earlier
  June 2022 `Short_Mat_DEG_stat` snapshot) were fit with DESeq2's likelihood
  ratio test (`test = "LRT"`), whose raw `$stat` column is an unsigned deviance
  difference (`G² ≥ 0`) that places *both* upregulated and downregulated genes
  at the top. Multiplying the continuous DESeq2 test statistic magnitude from
  `DEET_short_final_Aug2022.rds` by `sign(log2FC)` restores true directionality
  (`100%` concordance with DEET's `DEGs.Up` and `DEGs.Down` gene lists across
  all comparisons) without introducing `padj = 1` ties (yielding a median of
  18,525 unique rank levels across 18,532 detected protein-coding genes per
  comparison).
- **Sample-count ordering vs. fold-change sign in metadata**: In DEET's raw
  metadata table, `Samples.Up` and `Samples.Down` were ordered by group sample
  size (`n_1 ≤ n_2`) rather than fold-change sign. `build_data.py` resolves the
  exact DESeq2 reference (`Group.Down`, `log2FC < 0`) and numerator (`Group.Up`,
  `log2FC > 0`) condition for all 3,162 comparisons—matching `<Down>_vs_<Up>` in
  `DEET.ID` for `GTEx`, `TCGA`, and `RecountBrain`, and R's factor level
  collation in `DEET_metadata_eachcomparison_SRA.rds` for `SRA` and
  `SRA-manual`—and formats `Sample.Groups` consistently as
  `Group.Down (n=...) vs Group.Up (n=...)`.
- **Explicit `Higher Group` and `Lower Group` columns**: Instead of a generic
  `"Upregulation"` / `"Downregulation"` label (which can look inverted when
  DESeq2 sets the alphabetically first condition as the reference denominator),
  the results table displays **`Higher Group`** and **`Lower Group`** side by
  side right after `DEET Name`, showing both sample groups of the comparison
  ordered by expression of the input gene set:
  - **`AUROC > 0.5`** → `Higher Group = Group.Up` (`log2FC > 0` / `Top DEGs (Up)`),
    `Lower Group = Group.Down` (e.g., `male` vs `female`, or
    `Mycobacterium tuberculosis (Rv)` vs `uninfected control`).
  - **`AUROC < 0.5`** → `Higher Group = Group.Down` (`log2FC < 0` / `Top DEGs (Down)`),
    `Lower Group = Group.Up` (e.g., `INTERFERON` vs `no INTERFERON`,
    `no idiopathic heart disease` vs `idiopathic heart disease`, or
    `Mycobacterium tuberculosis GC1237` vs `uninfected control`).
- **Optional unsigned significance mode (`Ignore direction (rank by significance)`)**:
  A toggle at the bottom of the sidebar (off by default) switches to
  `deet_unsigned_ranks_*.parquet`, where genes within each comparison are ranked
  from `1` (least significant) to `19,444` (most significantly differentially
  expressed in either direction) using `DESeq2 test-statistic magnitude`
  (equivalent to `−log10(p_unadjusted)`), tie-broken by `|log2FC|`. This tests
  whether input genes are enriched for significant differential expression
  regardless of whether individual genes are upregulated or downregulated.

## Run locally

```bash
git clone https://github.com/leonfrench/polygenic_DEET_tester.git
cd polygenic_DEET_tester
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Enter gene symbols separated by whitespace, commas, or semicolons, and select
**Human**, **Mouse**, or **Rhesus Macaque**. Both target and background lists
use the selected species. Mouse and rhesus macaque inputs are converted to human
symbols using `data/homologene_to_human.csv.gz` (all mapped human orthologs are
retained and deduplicated). An optional background list restricts and re-ranks
the comparison universe; without it, the full 19,444-gene protein-coding
universe is used. The default example gene list comprises 40 canonical human
Type I interferon-stimulated genes
([Schoggins et al., *Nature*, 2011](https://doi.org/10.1038/nature09907);
[MSigDB `HALLMARK_INTERFERON_ALPHA_RESPONSE`](https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/HALLMARK_INTERFERON_ALPHA_RESPONSE.html)).

## Analysis and downloads

For each comparison, AUROC compares the within-comparison ranks of matched
target genes against non-target background genes (`n_negative = n_background − n_target`)
using the analytical Wilcoxon rank-sum identity (Ballouz et al., *EGAD*).
Two-sided *p*-values use the continuity-corrected normal approximation, and
Benjamini–Hochberg `FDR` is computed from full-precision *p*-values across all
comparisons in the selected `(DEET source, Category)` filter. Upstream metadata
is also normalized in `build_data.py` (expanding TCGA study acronyms, formatting
GTEx tissue and race labels, and harmonizing 117 upstream `Category`
misclassifications such as GTEx `sex` comparisons previously split across
`phenotype`, `ancestry`, and `developmental stage`).

Results update immediately when inputs change. Download the full table as CSV or
generate a hierarchically clustered PDF heatmap (complete-linkage Euclidean
clustering; computed on demand and cached for identical inputs).

## Deploy on Streamlit Community Cloud (`streamlit.io`)

- **Repository**: `leonfrench/polygenic_DEET_tester`
- **Main file path**: `app.py`
- **Python version**: Select **Python 3.12** in Advanced Settings on Streamlit
  Community Cloud. `.streamlit/config.toml`, `static/`, `data/`, and
  `requirements.txt` are picked up automatically from the repository root.

## Rebuilding the data shards (optional)

To regenerate `data/deet_ranks_*.parquet`, `data/deet_unsigned_ranks_*.parquet`,
and `data/deet_metadata.parquet` from the upstream DEET source archive
(`DEET_short_final_Aug2022.rds` from the
[Wilson Lab DEET repository](https://www.wilsonlab.org/public/DEET_data/) and
the metadata table exported from [DEET-shiny](https://wilsonlab-sickkids-uoft.shinyapps.io/DEET-shiny/)
or [Zenodo (record 7321664)](https://zenodo.org/records/7321664); requires R):

```bash
curl -fL -o /tmp/DEET_short_final_Aug2022.rds "https://www.wilsonlab.org/public/DEET_data/intermediary_files/DEET_short_final_Aug2022.rds"
python build_data.py \
  --short-final /tmp/DEET_short_final_Aug2022.rds \
  --meta /path/to/DEET_metadata_table_2023-01-28.txt \
  --protein-coding data/protein_coding_genes.txt.gz \
  --out-dir data
```

The main app and dataframe use the bundled Roboto Condensed font
(`static/RobotoCondensed.ttf`, licensed under the bundled SIL Open Font License
`static/RobotoCondensed-OFL.txt`).
