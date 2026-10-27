"""Write the held-out suites (with and without references) and one contest per arm."""

import argparse
from pathlib import Path

import yaml

DRIVERS = ("socialgail", "nsp")
FAMILIES = {"junction": "hospital_1_bare", "sbend": "sbend_corridor", "reception": "reception"}
COUNTS = (15, 20)
EPISODES = 20
SEED0 = 27000
TIMEOUT = "60s"  # training episodes end at 600 steps at 10 Hz

ARMS = {"agent5": "groups_agent5", "labels": "groups_labels", "clusters": "groups_clusters"}


def stage(driver: str, family: str, n_peds: int, seed: int, episodes: int) -> dict:
    return {
        "name": f"{driver}_{family}_{n_peds}",
        "map": FAMILIES[family],
        "robot": "jackal",
        "tm_robots": "scenario",
        "tm_obstacles": "scenario",
        "episodes": episodes,
        "seed": seed,
        "timeout": TIMEOUT,
        "config": {"scenario": {"file": f"heldout_{family}_{n_peds}"}, "humansim": {"local_planner": driver}},
    }


def suite(stages: list[dict]) -> dict:
    return {"launch": {"lockstep": True}, "metrics": {"max_collisions": 1}, "references": True, "stages": stages}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bench", type=Path, help="configs/benchmark dir of arena_evaluation")
    p.add_argument("--tick", type=int, required=True, help="common checkpoint, e.g. 700000")
    args = p.parse_args()

    splits = [(d, f, n) for d in DRIVERS for f in FAMILIES for n in COUNTS]
    stages = [stage(d, f, n, SEED0 + 100 * i, EPISODES) for i, (d, f, n) in enumerate(splits)]

    (args.bench / "suites").mkdir(parents=True, exist_ok=True)
    (args.bench / "contests").mkdir(parents=True, exist_ok=True)
    for name, data in {"rss27_heldout": suite(stages), "rss27_heldout_norefs": suite(stages) | {"references": False}}.items():
        (args.bench / "suites" / f"{name}.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    for arm, agent in ARMS.items():
        entry = [{"name": arm, "mobile": {"driver": "rosnav_rl", "agent": f"{agent}_t{args.tick // 1000}k"}}]
        (args.bench / "contests" / f"rss27_{arm}.yaml").write_text(yaml.safe_dump(entry, sort_keys=False))
    print("\n".join(sorted(str(f) for f in args.bench.glob("*/rss27_*.yaml"))))


if __name__ == "__main__":
    main()
