"""Held-out readout: per-arm means and treatment-minus-control deltas with bootstrap intervals, per split and pooled.

Episodes are paired across arms by (split, seed index): every arm ran the same stage seeds in the same order.
Per split the bootstrap resamples episodes; the pooled row resamples whole splits (stats.py cluster bootstrap).
The differences are sorted first: paired_differences returns rows in no fixed order and the seeded draws depend on it.
"""

import argparse
import math
from pathlib import Path

import polars as pl
from arena_evaluation.experiments.stats import paired_differences, summarize_family

METRICS = {
    "zc_success": "zero-contact success",
    "reached": "goal reached",
    "contacts": "pedestrian contacts / episode",
    "time_to_goal": "time to goal [s]",
    "ne": "final geodesic distance to goal [m]",
    "stuck": "stuck",
    "velocity_mean": "mean speed [m/s]",
    "path_length": "path length [m]",
    "min_pedestrian_clearance": "min pedestrian clearance [m]",
}


def load(run_dir: Path, arm: str) -> pl.DataFrame:
    df = pl.read_parquet(run_dir / "combined_metrics.parquet")
    if "is_reference" in df.columns:
        df = df.filter(~pl.col("is_reference").fill_null(False))
    zc = pl.col("success").fill_null(False).cast(pl.Float64)
    df = df.select(
        pl.lit(arm).alias("arm"),
        pl.col("stage").alias("split"),
        "episode",
        zc.alias("zc_success"),
        pl.col("collision_amount_pedestrian").alias("contacts"),
        pl.when(zc > 0).then(pl.col("time_to_goal")).alias("time_to_goal"),
        "ne",
        pl.col("stuck").cast(pl.Float64),
        "velocity_mean",
        "path_length",
        "min_pedestrian_clearance",
    )
    df = df.sort("split", "episode").with_columns(pl.col("episode").rank("ordinal").over("split").alias("seed"), pl.lit(0).alias("rep"))
    progress = pl.read_csv(run_dir / "progress.csv", infer_schema_length=0)
    stage_col = next(c for c in ("stage", "stage_name") if c in progress.columns)
    reached = progress.select(pl.col(stage_col).alias("split"), (pl.col("outcome_state") == "2").cast(pl.Float64).alias("reached"))
    reached = reached.with_columns(pl.int_range(1, pl.len() + 1).over("split").alias("seed"))
    return df.join(reached, on=["split", "seed"], how="left")


def fmt(r: dict) -> str:
    if r is None or r["estimate"] is None or math.isnan(r["estimate"]):
        return "–"
    star = " *" if r["excludes_zero"] else ""
    return f"{r['estimate']:+.2f} [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]{star}"


def deltas(df: pl.DataFrame, metric: str, treatment: str, control: str) -> tuple[dict, dict]:
    d = paired_differences(df, cell=["split"], metric=metric, arm="arm", treatment=treatment, control=control, scenario="seed", seed="rep").sort("split", "seed")
    per_split = summarize_family(d, cell=["split"], scenario="seed") if d.height else pl.DataFrame()
    pooled = summarize_family(d.with_columns(pl.lit("all").alias("cell")), cell=["cell"], scenario="split") if d.height else pl.DataFrame()
    return ({r["split"]: r for r in per_split.to_dicts()}, pooled.to_dicts()[0] if pooled.height else None)


def verdict(per_split: dict) -> str:
    above = any(r["ci_low"] > 0 for r in per_split.values())
    below = any(r["ci_high"] < 0 for r in per_split.values())
    if above and not below:
        return "chase"
    if not above:
        return "drop"
    return "inconclusive"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench", type=Path, default=Path("/opt/arena_ws/data/benchmarks"))
    p.add_argument("--runs", nargs="+", default=["agent5=heldout_agent5", "labels=heldout_labels", "clusters=heldout_clusters"])
    p.add_argument("--treatment", default="labels")
    p.add_argument("--out", type=Path, default=Path("/opt/arena_ws/data/benchmarks/readout.md"))
    args = p.parse_args()

    runs = dict(r.split("=") for r in args.runs)
    df = pl.concat([load(args.bench / run, arm) for arm, run in runs.items()], how="diagonal_relaxed")
    controls = [a for a in runs if a != args.treatment]
    splits = sorted(df["split"].unique().to_list())
    lines = [
        f"# Held-out readout ({', '.join(f'{a}: {r}' for a, r in runs.items())})",
        "",
        f"Deltas are {args.treatment} minus control, 95 % bootstrap interval; * = excludes zero after Holm over the splits.",
        "",
    ]

    means = df.group_by("arm").agg([pl.col(m).mean() for m in METRICS]).sort("arm")
    lines += ["## Means per arm (all splits)", "", "| arm | " + " | ".join(METRICS.values()) + " |", "|---" * (len(METRICS) + 1) + "|"]
    lines += [f"| {r['arm']} | " + " | ".join("–" if r[m] is None else f"{r[m]:.3f}" for m in METRICS) + " |" for r in means.to_dicts()]

    verdicts = {}
    for metric, title in METRICS.items():
        lines += ["", f"## {title}", "", "| split | " + " | ".join(f"{args.treatment} − {c}" for c in controls) + " |", "|---" * (len(controls) + 1) + "|"]
        results = {c: deltas(df, metric, args.treatment, c) for c in controls}
        for s in splits:
            lines.append(f"| {s} | " + " | ".join(fmt(results[c][0].get(s)) for c in controls) + " |")
        lines.append("| **pooled** | " + " | ".join(fmt(results[c][1]) for c in controls) + " |")
        if metric == "zc_success":
            verdicts = {c: verdict(results[c][0]) for c in controls}

    lines += ["", "## Verdict (rule on zero-contact success, per control)", ""] + [f"- vs {c}: **{v}**" for c, v in verdicts.items()]
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
