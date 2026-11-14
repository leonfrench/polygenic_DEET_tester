"""Core functions for DEET differential expression gene-set AUROC analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import ndtr
from scipy.stats import rankdata


@dataclass(frozen=True)
class RankMatrix:
    genes: np.ndarray
    profiles: tuple[str, ...]
    ranks: np.ndarray


@dataclass(frozen=True)
class AnalysisResult:
    table: pd.DataFrame
    heatmap: pd.DataFrame
    unmatched_genes: tuple[str, ...]
    input_gene_count: int
    matched_gene_count: int
    background_gene_count: int


def unique_in_order(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def parse_gene_list(text: str) -> list[str]:
    """Split gene symbols on whitespace, commas, or semicolons."""
    return unique_in_order(
        [value for value in re.split(r"[\s,;]+", text.strip()) if value]
    )


def load_rank_matrix(path: Path) -> RankMatrix:
    """Load a DEET rank Parquet shard (or CSV/CSV.GZ fallback)."""
    if path.suffix == ".parquet":
        table = pq.read_table(path)
        names = table.column_names
        if not names or names[0] != "gene_symbol":
            raise ValueError(f"{path.name} does not start with a gene_symbol column.")

        genes = table.column(0).to_numpy(zero_copy_only=False).astype(str)
        if pd.Index(genes).has_duplicates:
            raise ValueError(f"{path.name} contains duplicate gene symbols.")

        n_rows = table.num_rows
        n_profiles = table.num_columns - 1
        ranks = np.empty((n_rows, n_profiles), dtype=np.float32)
        for j in range(n_profiles):
            ranks[:, j] = table.column(j + 1).to_numpy(zero_copy_only=False)

        schema_meta = table.schema.metadata or {}
        if schema_meta.get(b"rank_encoding") == b"doubled_uint16":
            ranks *= np.float32(0.5)
    else:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if not header or header[0] != "gene_symbol":
            raise ValueError(f"{path.name} does not start with a gene_symbol column.")
        dtypes = {column: np.float32 for column in header[1:]}
        frame = pd.read_csv(path, dtype=dtypes)
        genes = frame["gene_symbol"].astype(str).to_numpy()
        if pd.Index(genes).has_duplicates:
            raise ValueError(f"{path.name} contains duplicate gene symbols.")
        names = header
        ranks = frame.iloc[:, 1:].to_numpy(dtype=np.float32, copy=False)

    if not np.isfinite(ranks).all():
        raise ValueError(f"{path.name} contains missing or non-finite rank values.")
    genes.setflags(write=False)
    ranks.setflags(write=False)
    return RankMatrix(genes=genes, profiles=tuple(names[1:]), ranks=ranks)


def load_metadata(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "DEET.ID" not in frame.columns:
        raise ValueError(f"{path.name} is missing the required DEET.ID column.")
    return frame


def load_ortholog_map(path: Path) -> dict[str, dict[str, tuple[str, ...]]]:
    frame = pd.read_csv(path, dtype=str)
    required = {"species", "source_symbol", "human_symbol"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path.name} is missing required ortholog columns.")

    output: dict[str, dict[str, tuple[str, ...]]] = {}
    for (species, source_symbol), group in frame.groupby(
        ["species", "source_symbol"], sort=False
    ):
        human_symbols = tuple(unique_in_order(group["human_symbol"].tolist()))
        output.setdefault(species, {})[source_symbol] = human_symbols
    return output


def convert_to_human(
    genes: Sequence[str],
    species: str,
    ortholog_map: dict[str, dict[str, tuple[str, ...]]] | None = None,
) -> list[str]:
    if species == "Human":
        return unique_in_order(genes)
    if ortholog_map is None:
        raise ValueError("The ortholog map is required for non-human input genes.")

    species_map = ortholog_map.get(species, {})
    converted: list[str] = []
    for gene in genes:
        converted.extend(species_map.get(gene, ()))
    return unique_in_order(converted)


def auroc_analytic_ranked_profiles(ranked_profiles, target_indices) -> np.ndarray:
    """Vectorized EGAD rank-sum AUROC (Sara Ballouz).

    Rows are genes and columns are profiles. Inputs must already be ascending
    within-column ranks (average ranks for ties), with aligned binary labels.
    Returns one AUROC per column; does not mutate either input.
    Source: https://github.com/sarbal/EGAD/blob/master/EGAD/R/auroc_analytic.R
    """
    ranks = np.asarray(ranked_profiles)
    labels = np.asarray(target_indices)
    if ranks.ndim != 2 or labels.ndim != 1:
        raise ValueError("Expected a 2D rank matrix and a 1D label vector.")
    if len(labels) != ranks.shape[0]:
        raise ValueError("Index length does not equal input table row count!")
    if labels.dtype != bool and not np.isin(labels, [0, 1]).all():
        raise ValueError("Target labels must be binary (0 or 1).")

    mask = labels.astype(bool, copy=False)
    n_positive = float(mask.sum())
    n_negative = float(len(mask)) - n_positive
    if n_positive == 0 or n_negative == 0:
        raise ValueError("At least one positive and one negative are required.")

    summed_ranks = ranks[mask, :].sum(axis=0, dtype=np.float64)
    return (summed_ranks / n_positive - (n_positive + 1.0) / 2.0) / n_negative


def auroc_analytic(scores, labels) -> float:
    """Single-profile version; scores must already be ascending ranks."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError("Expected a 1D rank vector.")
    return float(auroc_analytic_ranked_profiles(scores[:, None], labels)[0])


def get_p_from_auc(n_x, n_y, auc, alternative: str = "two.sided", correct: bool = True):
    """Vectorized normal-approximation Wilcoxon p-value from AUROC.

    n_x is the positive (matched target) count; n_y is the negative
    (non-target background) count, not total background size.
    """
    n_x, n_y = float(n_x), float(n_y)
    if not np.isfinite([n_x, n_y]).all() or n_x <= 0 or n_y <= 0:
        raise ValueError("Both group sizes must be positive and finite.")
    auc = np.asarray(auc, dtype=np.float64)
    if not np.isfinite(auc).all() or ((auc < 0) | (auc > 1)).any():
        raise ValueError("AUROC must be finite and between 0 and 1.")
    centered = auc * (n_x * n_y) - n_x * n_y / 2.0
    sigma = np.sqrt((n_x * n_y / 12.0) * (n_x + n_y + 1.0))
    if alternative == "two.sided":
        correction = np.sign(centered) * 0.5
    elif alternative == "greater":
        correction = 0.5
    elif alternative == "less":
        correction = -0.5
    else:
        raise ValueError("alternative must be 'two.sided', 'greater', or 'less'.")
    z = (centered - (correction if correct else 0.0)) / sigma
    if alternative == "less":
        p = ndtr(z)
    elif alternative == "greater":
        p = ndtr(-z)
    else:
        p = 2.0 * ndtr(-np.abs(z))
    return float(p) if p.ndim == 0 else p


def _significant(value: float, digits: int = 3) -> float:
    if value == 0 or not np.isfinite(value):
        return float(value)
    return float(format(value, f".{digits}g"))


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment using full-precision p-values."""
    number_of_tests = len(p_values)
    order = np.argsort(p_values)
    ordered_p_values = p_values[order]
    adjusted_ordered = ordered_p_values * number_of_tests / np.arange(
        1, number_of_tests + 1
    )
    adjusted_ordered = np.minimum.accumulate(adjusted_ordered[::-1])[::-1]
    adjusted = np.empty(number_of_tests, dtype=np.float64)
    adjusted[order] = np.clip(adjusted_ordered, 0.0, 1.0)
    return adjusted


def run_auroc_analysis(
    matrix: RankMatrix | Sequence[RankMatrix],
    target_genes: Sequence[str],
    background_genes: Sequence[str] | None = None,
    group_labels: Mapping[str, tuple[str, str]] | None = None,
) -> AnalysisResult:
    shards: tuple[RankMatrix, ...] = (
        (matrix,) if isinstance(matrix, RankMatrix) else tuple(matrix)
    )
    if not shards:
        raise ValueError("At least one RankMatrix shard is required.")

    base_genes = shards[0].genes
    for shard in shards[1:]:
        if not np.array_equal(base_genes, shard.genes):
            raise ValueError("All RankMatrix shards must share identical gene ordering.")

    input_gene_count = len(target_genes)
    matrix_gene_set = set(base_genes)

    if background_genes is None:
        selected_mask: np.ndarray | None = None
        selected_genes = base_genes
    else:
        background_set = set(background_genes).intersection(matrix_gene_set)
        selected_mask = np.fromiter(
            (gene in background_set for gene in base_genes),
            dtype=bool,
            count=len(base_genes),
        )
        selected_genes = base_genes[selected_mask]
        if len(selected_genes) == 0:
            raise ValueError("None of the background genes were found in the matrix.")

    selected_gene_set = set(selected_genes)
    unmatched_targets = tuple(
        gene for gene in target_genes if gene not in selected_gene_set
    )
    matched_targets = selected_gene_set.intersection(target_genes)
    target_mask = np.fromiter(
        (gene in matched_targets for gene in selected_genes),
        dtype=bool,
        count=len(selected_genes),
    )
    n_target = int(target_mask.sum())
    n_background = int(len(selected_genes))
    n_negative = n_background - n_target

    if n_target == 0:
        raise ValueError(
            "None of the submitted genes were found in the selected rank-matrix background."
        )
    if n_negative == 0:
        raise ValueError("The background must contain at least one non-target gene.")

    all_profiles: list[str] = []
    auc_chunks: list[np.ndarray] = []
    heatmap_chunks: list[np.ndarray] = []

    for shard in shards:
        all_profiles.extend(shard.profiles)
        shard_ranks = (
            shard.ranks if selected_mask is None else shard.ranks[selected_mask, :]
        )
        # Re-rank within a custom background universe; full universe shards already
        # store exact within-column average ranks.
        analysis_ranks = (
            shard_ranks
            if selected_mask is None
            else rankdata(shard_ranks, axis=0, method="average")
        )
        auc_chunks.append(auroc_analytic_ranked_profiles(analysis_ranks, target_mask))
        heatmap_chunks.append(shard_ranks[target_mask, :])

    aucs = np.concatenate(auc_chunks)
    p_values = get_p_from_auc(n_target, n_negative, aucs)
    fdr_values = _benjamini_hochberg(p_values)

    if group_labels is not None:
        directions_list: list[str] = []
        for profile_id, auc in zip(all_profiles, aucs):
            pair = group_labels.get(profile_id)
            if auc > 0.5:
                directions_list.append(
                    f"Higher in {pair[1]}" if pair is not None else "Upregulation of input genes"
                )
            elif auc < 0.5:
                directions_list.append(
                    f"Higher in {pair[0]}" if pair is not None else "Downregulation of input genes"
                )
            else:
                directions_list.append("Neutral")
        directions = np.asarray(directions_list, dtype=object)
    else:
        directions = np.where(
            aucs > 0.5,
            "Upregulation of input genes",
            np.where(aucs < 0.5, "Downregulation of input genes", "Neutral"),
        )

    rounded_aucs = np.asarray([_significant(value) for value in aucs])
    rounded_p_values = np.asarray([_significant(value) for value in p_values])
    rounded_fdr = np.asarray([_significant(value) for value in fdr_values])

    # Sort by full-precision two-sided p-value ascending, breaking ties by |AUROC - 0.5| descending
    sort_order = np.lexsort((-np.abs(aucs - 0.5), p_values))

    table = pd.DataFrame(
        {
            "DEET ID": np.asarray(all_profiles, dtype=object)[sort_order],
            "Direction": directions[sort_order],
            "AUROC": rounded_aucs[sort_order],
            "pValue": rounded_p_values[sort_order],
            "FDR": rounded_fdr[sort_order],
        }
    )

    heatmap_matrix = np.hstack(heatmap_chunks)
    heatmap = pd.DataFrame(
        heatmap_matrix,
        index=selected_genes[target_mask],
        columns=all_profiles,
    )
    heatmap.index.name = "gene_symbol"

    return AnalysisResult(
        table=table,
        heatmap=heatmap,
        unmatched_genes=unmatched_targets,
        input_gene_count=input_gene_count,
        matched_gene_count=n_target,
        background_gene_count=n_background,
    )
