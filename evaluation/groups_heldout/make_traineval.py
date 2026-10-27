"""Evaluate a checkpoint through the training env: resume it, run one deterministic eval at the second step, stop.

The in-training eval (SharedEnvEvalCallback) is the code path that produced the logged success rates, so this
reproduces it outside training. Weights never change: the eval fires on the second env step and the stop
callback ends learning before any PPO update.
"""

import argparse
import shutil
from pathlib import Path

import yaml

HELDOUT = {"junction": "hospital_1_bare", "sbend": "sbend_corridor", "reception": "reception"}


def heldout_envs(group: str) -> list[list[str]]:
    driver, n_peds = group.split("_")
    return [[f"world:={w}", f"task.scenario.file:=heldout_{f}_{n_peds}", f"humansim.local_planner:={driver}"] for f, w in HELDOUT.items()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True, help="the arm's training config")
    p.add_argument("--agents", type=Path, default=Path("/opt/arena_ws/data/agents"))
    p.add_argument("--configs", type=Path, default=Path("/opt/arena_ws/data/configs"))
    p.add_argument("--tick", type=int, required=True)
    p.add_argument("--name", required=True, help="name of the eval run (agent copy and config)")
    p.add_argument("--episodes", type=int, required=True, help="eval episodes in total, spread evenly over the envs")
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--heldout", help="socialgail_15 | socialgail_20 | nsp_15 | nsp_20; omit to keep the base envs")
    args = p.parse_args()

    cfg = yaml.safe_load(args.base.read_text())
    src = args.agents / cfg["agent_config"]["name"]
    dst = args.agents / args.name
    if dst.exists():
        raise SystemExit(f"{dst} exists")
    dst.mkdir()
    shutil.copy(src / "training_config.yaml", dst)
    shutil.copy(src / f"tick_{args.tick}.zip", dst)

    cfg["resume"] = True
    agent = cfg["agent_config"]
    agent["name"] = args.name
    algo = agent["framework"]["algorithm"]
    algo["checkpoint"] = f"tick_{args.tick}"
    algo["parameters"].update(total_timesteps=args.tick + 100_000, device="cpu")
    general = cfg["arena_cfg"]["general"]
    general["no_gpu"] = True
    if args.heldout:
        general["env_launch_args"] = heldout_envs(args.heldout)
        general["n_envs"] = len(general["env_launch_args"])
    callbacks = cfg["arena_cfg"]["callbacks"]
    # eval_freq 2: the tick curriculum restarts at stage 0 and only moves to the checkpoint's stage in its first
    # on_step, which runs after the eval callback; the eval at call 2 resets the envs into that stage.
    callbacks["periodic_evaluation"].update(eval_freq=2, n_eval_episodes=args.episodes, max_num_moves_per_eps=args.max_steps)
    callbacks["stop_training_on_threshold"] = {"threshold_type": "rew", "threshold": -1.0e9, "verbose": 1}
    callbacks["checkpoint_ticks"] = []

    out = args.configs / f"{args.name}.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(out)


if __name__ == "__main__":
    main()
