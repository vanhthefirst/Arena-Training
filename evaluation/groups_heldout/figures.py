"""Report figures for the pedestrian-groups spike, drawn from the recorded held-out runs, training logs and episodes.

Every figure is written as <name>.pdf, <name>.png and <name>.csv (the plotted numbers) under --out.
Intervals are 95 % cluster bootstraps over splits, the resampling used for the pooled rows of readout.py.
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

from readout import load

DATA = Path("/opt/arena_ws/data")
ARMS = {
    "agent5": ("AGENT_5", "#0072B2", "heldout_agent5"),
    "labels": ("labels", "#D55E00", "heldout_labels"),
    "clusters": ("clusters", "#009E73", "heldout_clusters"),
    "default": ("planner", "#8C8C8C", "heldout_default"),
}
STAGE_STARTS = [60_000, 119_000, 178_000, 255_000]
EVALUATED_TICK = 700_000
TRAJECTORY_SPLITS = ["socialgail_junction_15", "socialgail_sbend_15", "socialgail_reception_15", "nsp_junction_15", "nsp_sbend_15", "nsp_reception_15"]
WIDTH = 6.3
OFFLINE_MODES = [("real", "real last action", "#0072B2"), ("delayed", "delayed one step", "#56B4E9"), ("zero", "set to zero (diagnostic)", "#E69F00")]


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )


def save(fig: plt.Figure, table: pl.DataFrame, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    table.write_csv(out / f"{name}.csv")
    plt.close(fig)
    print(f"wrote {out / name}.{{pdf,png,csv}}")
    print(table)


def heldout_frame(bench: Path) -> pl.DataFrame:
    return pl.concat([load(bench / run, arm) for arm, (_, _, run) in ARMS.items()], how="diagonal_relaxed")


def cluster_ci(df: pl.DataFrame, metric: str, n: int = 10_000, seed: int = 0) -> tuple[float, float, float, int, float]:
    g = df.group_by("split").agg(pl.col(metric).sum().alias("s"), pl.col(metric).count().alias("n")).sort("split")
    s, c = g["s"].to_numpy().astype(float), g["n"].to_numpy().astype(float)
    idx = np.random.default_rng(seed).integers(0, len(s), (n, len(s)))
    boots = s[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return s.sum() / c.sum(), lo, hi, int(c.sum()), s.sum()


def fig_heldout(args: argparse.Namespace) -> None:
    df = heldout_frame(args.bench)
    panels = [
        ("reached", "(a) goal\nreached"),
        ("contacts", "(b) contacts\nper episode"),
        ("ne", "(c) final distance\nto goal [m]"),
        ("stuck", "(d) stuck\n(fraction)"),
        ("velocity_mean", "(e) mean\nspeed [m/s]"),
    ]
    rows = []
    fig, axes = plt.subplots(1, len(panels), figsize=(WIDTH, 2.0))
    for ax, (metric, title) in zip(axes, panels):
        for i, (arm, (label, color, _)) in enumerate(ARMS.items()):
            mean, lo, hi, n, total = cluster_ci(df.filter(pl.col("arm") == arm), metric)
            rows.append({"metric": metric, "arm": label, "mean": mean, "ci_low": lo, "ci_high": hi, "episodes": n})
            planner = arm == "default"
            ax.bar(i, mean, color="white" if planner else color, edgecolor=color, hatch="////" if planner else None, linewidth=0.8, width=0.75)
            ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], color="#333333", lw=0.8, capsize=2)
            if metric == "reached":
                ax.text(i, max(hi, mean) + 0.03, f"{int(round(total))}/{n}", ha="center", va="bottom", rotation=90, fontsize=6)
        ax.set_title(title)
        ax.set_xticks([])
        ax.margins(y=0.15)
    axes[0].set_ylim(0, 1.05)
    handles = [plt.Rectangle((0, 0), 1, 1, fc="white" if a == "default" else c, ec=c, hatch="////" if a == "default" else None) for a, (_, c, _) in ARMS.items()]
    fig.legend(handles, [v[0] for v in ARMS.values()], loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(w_pad=0.8)
    save(fig, pl.DataFrame(rows), args.out, "fig2_heldout")


def episode_bundle(ep_dir: Path):
    from arena_evaluation.processing.pipeline import ProcessingPipeline
    from arena_evaluation.storage.folder_manager import FolderManager

    fm = FolderManager(data_root=DATA)
    desc = fm.describe_episode(ep_dir)
    if desc is None:
        raise FileNotFoundError(f"{ep_dir} is not a readable episode")
    bundles = ProcessingPipeline(fm, workers=1).extract_episode(desc)
    if not bundles:
        raise FileNotFoundError(f"no topics for {ep_dir}")
    return desc, next(iter(bundles.values()))


def collect(frame) -> pl.DataFrame | None:
    return frame.collect() if isinstance(frame, pl.LazyFrame) else frame


def robot_track(bundle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from arena_evaluation.processing.pose_anchor import resolve_pose_source

    pose, source = resolve_pose_source(bundle, (None, None))
    if pose is None:
        odom = collect(bundle.odom).sort("time_ns")
        print(f"warning: no map-frame pose ({source.kind}); robot drawn in the odom frame")
        return odom["time_ns"].to_numpy(), odom["pos_x"].to_numpy(), odom["pos_y"].to_numpy()
    pose = pose.sort("time_ns")
    return pose["time_ns"].to_numpy(), pose["pos_x_gt"].to_numpy(), pose["pos_y_gt"].to_numpy()


def pedestrian_tracks(bundle) -> list[np.ndarray]:
    peds = collect(bundle.peds)
    if peds is None or peds.is_empty():
        return []
    long = (
        peds.sort("time_ns")
        .select("peds_ids", "peds_positions")
        .with_columns(pl.col("peds_positions").list.gather_every(3).alias("x"), pl.col("peds_positions").list.gather_every(3, offset=1).alias("y"))
        .select("peds_ids", "x", "y")
        .explode("peds_ids", "x", "y")
    )
    return [g.select("x", "y").to_numpy() for _, g in long.group_by("peds_ids", maintain_order=True)]


def draw_map(ax: plt.Axes, map_name: str, run_dir: Path) -> bool:
    from arena_evaluation.processing.map_registry import MapRegistry

    meta = MapRegistry.get_map(map_name, run_dir=run_dir)
    if not meta:
        return False
    img = plt.imread(meta["png_path"])
    ox, oy = meta["origin"][:2]
    res = meta["resolution"]
    ax.imshow(img, extent=[ox, ox + meta["width"] * res, oy, oy + meta["height"] * res], origin="upper", cmap="gray", alpha=0.45, zorder=0)
    return True


def pick_episode(df: pl.DataFrame) -> tuple[str, int]:
    planner = df.filter((pl.col("arm") == "default") & (pl.col("reached") > 0))
    for split in TRAJECTORY_SPLITS + sorted(planner["split"].unique().to_list()):
        hits = planner.filter(pl.col("split") == split).sort("seed")
        if hits.height:
            return split, int(hits["seed"][0])
    raise RuntimeError("the planner reached no goal in any split")


def fig_trajectories(args: argparse.Namespace) -> None:
    df = heldout_frame(args.bench)
    split, seed = (args.split, args.seed) if args.split else pick_episode(df)
    print(f"episode: split {split}, seed index {seed}")
    order = ["default", "agent5", "labels", "clusters"]
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 2.0), sharex=True, sharey=True)
    rows, extent = [], []
    norm = plt.Normalize(0, 60)
    for ax, arm in zip(axes, order):
        label, color, run = ARMS[arm]
        row = df.filter((pl.col("arm") == arm) & (pl.col("split") == split) & (pl.col("seed") == seed)).row(0, named=True)
        meta = pl.read_parquet(args.bench / run / "combined_metrics.parquet").filter(pl.col("episode") == row["episode"]).row(0, named=True)
        desc, bundle = episode_bundle(args.bench / run / "episodes" / f"episode_{row['episode']:03d}")
        has_map = draw_map(ax, desc.map, args.bench / run)
        for track in pedestrian_tracks(bundle):
            ax.plot(track[:, 0], track[:, 1], color="#BBBBBB", lw=0.4, zorder=1)
        t, x, y = robot_track(bundle)
        t = (t - t[0]) / 1e9
        seg = np.stack([np.column_stack([x[:-1], y[:-1]]), np.column_stack([x[1:], y[1:]])], axis=1)
        ax.add_collection(LineCollection(seg, cmap="viridis", norm=norm, array=t[:-1], lw=1.4, zorder=3))
        start, goal = meta["start"][:2], meta["goal"][:2]
        ax.plot(*start, "o", color="black", ms=4, zorder=4)
        ax.plot(*goal, "*", color="black", ms=7, zorder=4)
        ax.add_patch(Circle(goal, 0.8, fill=False, ls="--", lw=0.6, ec="black", zorder=4))
        outcome = "timeout" if not row["reached"] else f"goal in {meta['time_to_goal']:.1f} s" if meta["time_to_goal"] else "goal"
        ax.set_title(f"{label}\n{outcome}, {int(row['contacts'] or 0)} contacts", fontsize=6.5)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.tick_params(labelsize=6.5)
        extent.append(np.array([[x.min(), y.min()], [x.max(), y.max()], start, goal]))
        rows.append({"arm": label, "split": split, "seed": seed, "episode": row["episode"], "reached": row["reached"], "time_to_goal": meta["time_to_goal"], "contacts": row["contacts"], "duration_s": float(t[-1]), "map_drawn": has_map})
    pts = np.vstack([np.vstack(e) for e in extent])
    lo, hi = pts.min(0) - 1.5, pts.max(0) + 1.5
    axes[0].set_xlim(lo[0], hi[0])
    axes[0].set_ylim(lo[1], hi[1])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=axes, fraction=0.015, pad=0.01, shrink=0.75)
    cbar.set_label("time [s]")
    save(fig, pl.DataFrame(rows), args.out, "fig3_trajectories")


TABLE_ROW = re.compile(r"\|\s*([^|]+?)\s*\|\s*([^|]*?)\s*\|\s*$")
TABLE_EDGE = re.compile(r"-{20,}\s*$")
EVAL_STEP = re.compile(r"Eval num_timesteps=(\d+)")
EVAL_SUCCESS = re.compile(r"Success rate:?\s*([\d.]+)\s*%")
EPISODE_END = re.compile(r"Episode \d+ ended, reason: DONE_REASONS\.(\w+)")


def parse_training_log(path: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """SB3 tables, in-training evaluations and training episode ends (tagged with the last logged step) of the last run in the log."""
    tables, evals, episodes, block, section, pending = [], [], [], {}, "", None
    for line in path.read_text(errors="replace").splitlines():
        if TABLE_EDGE.search(line):
            if "time/total_timesteps" in block:
                step = int(float(block["time/total_timesteps"]))
                if tables and step < tables[-1]["step"]:
                    tables, evals, episodes = [], [], []
                tables.append({"step": step, "std": block.get("train/std"), "rollout_success": block.get("rollout/success_rate")})
            block, section = {}, ""
        elif m := TABLE_ROW.search(line):
            key, value = m.groups()
            if not value:
                section = key
            else:
                try:
                    block[section + key] = float(value)
                except ValueError:
                    pass
        elif m := EPISODE_END.search(line):
            episodes.append({"step": tables[-1]["step"] if tables else 0, "reason": m.group(1)})
        elif m := EVAL_STEP.search(line):
            pending = int(m.group(1))
        elif (m := EVAL_SUCCESS.search(line)) and pending is not None:
            evals.append({"step": pending, "eval_success": float(m.group(1)) / 100})
            pending = None
    schema_t = {"step": pl.Int64, "std": pl.Float64, "rollout_success": pl.Float64}
    return (
        pl.DataFrame(tables, schema=schema_t),
        pl.DataFrame(evals, schema={"step": pl.Int64, "eval_success": pl.Float64}),
        pl.DataFrame(episodes, schema={"step": pl.Int64, "reason": pl.String}),
    )


def stage_marks(ax: plt.Axes, end: int) -> None:
    for s in STAGE_STARTS:
        ax.axvline(s / 1e3, color="#999999", ls="--", lw=0.6)
    ax.axvspan(STAGE_STARTS[-1] / 1e3, end / 1e3, color="#F2F2F2", zorder=0)
    ax.axvline(EVALUATED_TICK / 1e3, color="#333333", ls=":", lw=0.9)


def fig_training(args: argparse.Namespace) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.1))
    rows, end = [], 0
    for arm in ("agent5", "labels", "clusters"):
        label, color, _ = ARMS[arm]
        table, evals, episodes = parse_training_log(args.logs / f"train_groups_{arm}.log")
        if table.is_empty():
            raise RuntimeError(f"no SB3 tables in train_groups_{arm}.log")
        end = max(end, table["step"].max())
        if table["rollout_success"].drop_nulls().len():
            a.plot(table["step"] / 1e3, table["rollout_success"], color=color, lw=0.6, alpha=0.45)
        a.plot(evals["step"] / 1e3, evals["eval_success"], "o-", color=color, ms=3.5, lw=1.1, label=label)
        b.plot(table["step"] / 1e3, table["std"], color=color, lw=1.1, label=label)
        stage4 = episodes.filter(pl.col("step") >= STAGE_STARTS[-1])["reason"].value_counts().sort("reason")
        print(f"{label}: evals {evals.to_dicts()}; final std {table['std'].drop_nulls()[-1]:.3f}; walking-group stage episode ends {dict(stage4.iter_rows())}")
        rows += [{"arm": label, "kind": f"walking_stage_episodes_{r}", "step": STAGE_STARTS[-1], "value": float(n)} for r, n in stage4.iter_rows()]
        rows += [{"arm": label, "kind": "eval", "step": r["step"], "value": r["eval_success"]} for r in evals.to_dicts()]
        rows += [{"arm": label, "kind": "std", "step": r["step"], "value": r["std"]} for r in table.to_dicts()]
        rows += [{"arm": label, "kind": "rollout_success", "step": r["step"], "value": r["rollout_success"]} for r in table.to_dicts()]
    for ax in (a, b):
        stage_marks(ax, end)
        ax.set_xlabel("training steps [k]")
        ax.set_xlim(0, end / 1e3)
    a.text((STAGE_STARTS[-1] + end) / 2e3, 1.0, "walking-group stage", ha="center", va="top", fontsize=6.5, color="#555555")
    a.set_ylim(-0.03, 1.05)
    a.set_title("(a) success rate (dots: evaluation, line: rollouts)")
    b.axhline(0.5, color="#333333", ls=(0, (1, 2)), lw=0.8)
    b.text(10, 0.52, "action range ±0.5 m/s", fontsize=6.5, va="bottom")
    b.set_ylim(0, None)
    b.set_title("(b) policy action std")
    handles, labels = b.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    save(fig, pl.DataFrame(rows), args.out, "fig4_training")


def command_trace(episode_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, bundle = episode_bundle(episode_dir)
    if bundle.cmd_vel is None:
        raise FileNotFoundError(f"no cmd_vel recorded in {episode_dir}")
    cmd = collect(bundle.cmd_vel).sort("time_ns")
    odom = collect(bundle.odom)
    t = (cmd["time_ns"].to_numpy() - cmd["time_ns"][0]) / 1e9
    return t, cmd["linear_x"].to_numpy(), odom["vel_linear"].to_numpy()


def flip_rate(v: np.ndarray) -> float:
    s = np.sign(v[v != 0])
    return float((s[1:] != s[:-1]).mean()) if len(s) > 1 else 0.0


def fig_oscillation(args: argparse.Namespace) -> None:
    panels = 2 if args.offline_csv else 1
    fig, axes = plt.subplots(1, panels, figsize=(WIDTH if panels == 2 else WIDTH * 0.55, 2.1), squeeze=False)
    a = axes[0, 0]
    rows = []
    for ep, (label, color) in zip(args.cmd_episodes, [("AGENT_5 (200k)", ARMS["agent5"][1]), ("planner", ARMS["default"][1])]):
        t, v, measured = command_trace(ep)
        keep = t <= args.window
        a.step(t[keep], v[keep], where="post", color=color, lw=0.8, label=label, zorder=3 if "AGENT" in label else 2)
        rows.append({"panel": "a", "series": label, "episode_dir": str(ep), "sign_flip_rate": flip_rate(v), "mean_command": float(v.mean()), "net_measured_speed": float(measured.mean()), "mean_abs_measured_speed": float(np.abs(measured).mean())})
    a.set_xlabel("time [s]")
    a.set_ylabel("linear command [m/s]")
    a.set_ylim(-0.6, 0.6)
    a.set_title("(a) commands on the empty map")
    a.legend(loc="upper right", fontsize=6.5, frameon=True, framealpha=0.9, edgecolor="none")
    if args.offline_csv:
        b = axes[0, 1]
        off = pl.read_csv(args.offline_csv).filter(pl.col("task") == "open")
        for mode, label, color in OFFLINE_MODES:
            g = off.filter(pl.col("mode") == mode).sort("t")
            b.plot(g["t"], g["distance"], lw=1.1, color=color, label=label)
            rows.append({"panel": "b", "series": label, "sign_flip_rate": flip_rate(g["linear_cmd"].to_numpy()), "final_distance": g["distance"][-1], "duration_s": g["t"][-1]})
        b.set_xlabel("time [s]")
        b.set_ylabel("distance to goal [m]")
        b.set_title("(b) offline test: last-action input")
        b.legend(fontsize=6.5)
    fig.tight_layout()
    save(fig, pl.DataFrame(rows, strict=False), args.out, "fig5_oscillation")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fig", choices=["heldout", "trajectories", "training", "oscillation"])
    p.add_argument("--bench", type=Path, default=DATA / "benchmarks")
    p.add_argument("--logs", type=Path, default=DATA / "logs")
    p.add_argument("--out", type=Path, default=DATA / "figures")
    p.add_argument("--split", help="trajectories: split to show (default: first where the planner reached the goal)")
    p.add_argument("--seed", type=int, help="trajectories: seed index within the split, 1-based")
    p.add_argument("--cmd-episodes", type=Path, nargs=2, metavar=("AGENT_EP", "PLANNER_EP"), help="oscillation: episode_NNN directories")
    p.add_argument("--window", type=float, default=20.0)
    p.add_argument("--offline-csv", type=Path, help="oscillation: output of offline_last_action.py")
    args = p.parse_args()
    style()
    {"heldout": fig_heldout, "trajectories": fig_trajectories, "training": fig_training, "oscillation": fig_oscillation}[args.fig](args)


if __name__ == "__main__":
    main()
