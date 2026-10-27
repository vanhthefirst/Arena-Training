"""Offline last-action test: a checkpoint's mean action in a kinematic unicycle, with rosnav-rl's own encoding.

The agent is loaded and queried as in the inference node (training_config.yaml, RL_Agent.get_action on the raw
observation dict), with the same straight-line subgoal rule. No pedestrians; the lidar sees open space or the walls
of a rectangular room. The linear velocity follows the command under the jackal's 2.5 m/s² limit, the angular
velocity follows it directly. Only the last-action input differs between modes:
real = previous command (as in training and evaluation), delayed = command two steps back, zero = diagnostic.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import polars as pl
from arena_training.arena_rosnav_rl.cfg.train import TrainingCfg
from rosnav_rl.observations.utils.semantic import straight_line_subgoal
from rosnav_rl.rl_agent import RL_Agent
from rosnav_rl.utils.utils import load_yaml

DT = 0.1
ACCEL = 2.5
LOOKAHEAD = (0.5, 2.5, 1.5)
MODES = ("real", "delayed", "zero")
TASKS = {
    "open": (None, [((0.0, 0.0, 0.0), (8.0, 0.0))]),
    "room": ((-5.0, 5.0, -4.0, 4.0), [((-3.5, -2.5, 0.0), (3.5, 2.5)), ((3.5, -2.5, math.pi), (-3.5, 2.5))]),
}
NO_PEDESTRIANS = {
    "pedestrian_relative_locations": np.array([], dtype=np.float32),
    "pedestrian_vel_x": np.array([]),
    "pedestrian_vel_y": np.array([]),
    "pedestrian_social_states": np.array([], dtype=np.uint8),
    "pedestrian_types": np.array([], dtype=np.int32),
}


def load_agent(agent_dir: Path, checkpoint: str) -> RL_Agent:
    spec = TrainingCfg.model_validate(load_yaml(agent_dir / "training_config.yaml")).agent_config
    agent = RL_Agent(spec)
    agent.load_model(path=agent_dir / checkpoint)
    return agent


def lidar(x: float, y: float, yaw: float, room: tuple | None, p) -> np.ndarray:
    n = p.laser_num_beams
    inc = p.laser_angle_increment or 2 * math.pi / n
    ranges = np.full(n, p.laser_max_range, dtype=np.float32)
    if room is None:
        return ranges
    xmin, xmax, ymin, ymax = room
    a = yaw + p.laser_angle_min + inc * np.arange(n)
    c, s = np.cos(a), np.sin(a)
    with np.errstate(divide="ignore"):
        tx = np.where(c > 0, (xmax - x) / c, np.where(c < 0, (xmin - x) / c, np.inf))
        ty = np.where(s > 0, (ymax - y) / s, np.where(s < 0, (ymin - y) / s, np.inf))
    return np.minimum(ranges, np.minimum(tx, ty)).astype(np.float32)


def run(agent: RL_Agent, room: tuple | None, start: tuple, goal: tuple, mode: str, timeout: float) -> list[dict]:
    p = agent.spec.parameters
    agent.reset()
    x, y, yaw = start
    v = 0.0
    history = [np.zeros(3), np.zeros(3)]
    rows = []
    for k in range(int(timeout / DT)):
        speed = math.hypot(history[-1][0], history[-1][1])
        sx, sy = straight_line_subgoal((x, y), goal, speed, *LOOKAHEAD)
        dx, dy = sx - x, sy - y
        rx, ry = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy
        scan = lidar(x, y, yaw, room, p)
        obs = {
            "front_laser": scan,
            "is_terminal": False,
            **NO_PEDESTRIANS,
            "dist_angle_to_subgoal": np.array([math.hypot(rx, ry), math.atan2(ry, rx)]),
            "last_action": {"real": history[-1], "delayed": history[-2], "zero": np.zeros(3)}[mode],
        }
        cmd = np.asarray(agent.get_action(obs, deterministic=True), dtype=float).reshape(-1)
        history.append(cmd)
        v += float(np.clip(cmd[0] - v, -ACCEL * DT, ACCEL * DT))
        yaw = math.atan2(math.sin(yaw + cmd[-1] * DT), math.cos(yaw + cmd[-1] * DT))
        x, y = x + v * math.cos(yaw) * DT, y + v * math.sin(yaw) * DT
        dist = math.hypot(goal[0] - x, goal[1] - y)
        crashed = room is not None and float(lidar(x, y, yaw, room, p).min()) < p.robot_radius
        outcome = "goal" if dist < p.goal_radius else "collision" if crashed else None
        rows.append({"t": (k + 1) * DT, "x": x, "y": y, "distance": dist, "linear_cmd": cmd[0], "angular_cmd": cmd[-1], "linear_vel": v, "outcome": outcome})
        if outcome:
            break
    rows[-1]["outcome"] = rows[-1]["outcome"] or "timeout"
    return rows


def flip_rate(v: np.ndarray) -> float:
    s = np.sign(v[v != 0])
    return float((s[1:] != s[:-1]).mean()) if len(s) > 1 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("agent_dir", type=Path)
    ap.add_argument("--checkpoint", default="best_model.zip")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--out", type=Path, default=Path("/opt/arena_ws/data/figures/offline_last_action.csv"))
    args = ap.parse_args()

    agent = load_agent(args.agent_dir, args.checkpoint)
    frames, summary = [], []
    for task, (room, episodes) in TASKS.items():
        for i, (start, goal) in enumerate(episodes):
            for mode in MODES:
                run_df = pl.DataFrame(run(agent, room, start, goal, mode, args.timeout), infer_schema_length=None)
                frames.append(run_df.with_columns(pl.lit(task).alias("task"), pl.lit(i).alias("episode"), pl.lit(mode).alias("mode")))
                summary.append(
                    {
                        "task": task,
                        "episode": i,
                        "mode": mode,
                        "outcome": run_df["outcome"][-1],
                        "duration_s": run_df["t"][-1],
                        "sign_flip_rate": flip_rate(run_df["linear_cmd"].to_numpy()),
                        "mean_linear_cmd": run_df["linear_cmd"].mean(),
                        "final_distance": run_df["distance"][-1],
                    }
                )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pl.concat(frames).write_csv(args.out)
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(pl.DataFrame(summary))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
