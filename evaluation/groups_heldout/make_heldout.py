"""Write the held-out group scenarios (junction, S-bend, reception) at 15 and 20 pedestrians."""

import argparse
import math
from pathlib import Path

import yaml

COUNTS = (15, 20)
MODELS = (
    "Common/Human/arenian",
    "Common/Human/casual_female_african_young",
    "Common/Human/worker_male_asian_middleage",
    "Hospital/Human/nurse_female_caucasian_young",
)
TOLERANCE = 0.8  # last training stage's goto_pose tolerance

FAMILIES = {
    "junction": {
        "world": "hospital_1_bare",
        "start": (10.0, 6.0, 1.5708),
        "goal": (16.0, 11.5, 0.0),
        "walks": [
            {"leader": (10.6, 17.0, -1.5708), "followers": [(9.8, 17.8), (11.4, 17.8)], "legs": [(10.6, 3.0), (10.6, 22.0)]},
            {"leader": (21.0, 10.8, 3.1416), "followers": [(21.9, 11.0)], "legs": [(9.0, 10.8), (23.0, 10.8)]},
        ],
        "talks": [
            [(14.0, 12.55, -1.5708), (13.57, 11.8, 0.5236), (14.43, 11.8, 2.618)],
            [(8.7, 8.3, 1.5708), (8.7, 9.7, -1.5708)],
        ],
        "zones": ["central_hallway", "sub_hallway"],
    },
    "sbend": {
        "world": "sbend_corridor",
        "start": (1.2, 6.5, 0.0),
        "goal": (8.5, 11.5, 1.5708),
        "walks": [
            {
                "leader": (6.0, 11.5, 0.0),
                "followers": [(5.2, 11.0), (5.2, 12.0)],
                "legs": [(8.5, 11.5), (8.5, 6.6), (1.8, 6.6), (8.5, 6.6), (8.5, 11.5), (3.5, 11.5)],
            },
            {
                "leader": (3.4, 6.0, 0.0),
                "followers": [(2.6, 5.7)],
                "legs": [(8.6, 6.0), (8.6, 11.8), (3.5, 11.8), (8.6, 11.8), (8.6, 6.0), (1.5, 6.0)],
                "speed": 0.6,
            },
        ],
        "talks": [
            [(6.6, 7.6, -1.5708), (6.21, 6.925, 0.5236), (6.99, 6.925, 2.618)],
            [(1.5, 12.3, 0.0), (2.9, 12.3, 3.1416)],
        ],
        "zones": ["corridor_middle", "corridor_north", "corridor_south"],
    },
    "reception": {
        "world": "reception",
        "start": (26.0, 18.0, -2.29),
        "goal": (19.0, 10.0, -2.29),
        "walks": [
            {"leader": (18.5, 16.0, 0.0), "followers": [(17.7, 15.2), (17.7, 16.8)], "legs": [(29.0, 16.0), (17.0, 16.0)]},
            {"leader": (21.0, 10.5, 0.852), "followers": [(20.3, 9.9)], "legs": [(27.0, 17.4), (20.0, 9.4)]},
        ],
        "talks": [
            [(22.15, 14.15, -1.5708), (21.67, 13.33, 0.5236), (22.63, 13.33, 2.618)],
            [(27.5, 8.0, 1.5708), (27.5, 9.4, -1.5708)],
        ],
        "zones": ["entrance"],
    },
}

FOLLOWER = {
    "name": "follower",
    "mode": "behavior_tree",
    "extends": "adult",
    "initial_sequence": "follow",
    "sequences": {"follow": {"steps": {"settle": {"duration": 2.0}, "join": {"interaction": "GROUP_WALK", "duration": 10000.0}}}},
}
TALKER = {
    "name": "talker",
    "mode": "behavior_tree",
    "extends": "adult",
    "initial_sequence": "talk",
    "sequences": {
        "talk": {
            "steps": {
                "settle": {"duration": 2.0},
                "talk": {"interaction": "GROUP_CONVERSATION", "attention": {"gaze": "partner"}, "duration": 10000.0},
            }
        }
    },
}


def leader_type(name: str, start: tuple[float, float], legs: list[tuple[float, float]], n_legs: int = 12) -> dict:
    steps: dict = {"form": {"interaction": "GROUP_WALK"}}
    prev = start
    for i in range(n_legs):
        x, y = legs[i % len(legs)]
        theta = round(math.atan2(y - prev[1], x - prev[0]), 4)
        steps[f"leg_{i + 1}"] = {"kind": "go_to", "target_pose": {"x": x, "y": y, "theta": theta}}
        prev = (x, y)
    return {"name": name, "mode": "behavior_tree", "extends": "adult", "initial_sequence": "lead", "sequences": {"lead": {"steps": steps}}}


def human(name: str, i: int, pose: list | str, agent_type: str, speed: float | None = None) -> dict:
    agent = {"agent_type": agent_type} | ({"desired_velocity": speed} if speed else {})
    return {"name": name, "model": MODELS[i % len(MODELS)], "pose": pose, "agent": agent}


def scenario(fam: dict, n_peds: int) -> tuple[dict, dict[str, dict]]:
    humans, types = [], {"follower.yaml": FOLLOWER, "talker.yaml": TALKER}
    for g, walk in enumerate(fam["walks"]):
        leader = f"leader_{g}"
        x0, y0, heading = walk["leader"]
        types[f"{leader}.yaml"] = leader_type(leader, (x0, y0), walk["legs"])
        humans.append(human(f"walk{g}_leader", len(humans), [x0, y0, heading], f"./{leader}.yaml", walk.get("speed")))
        for k, (x, y) in enumerate(walk["followers"]):
            humans.append(human(f"walk{g}_follower_{k}", len(humans), [x, y, heading], "./follower.yaml", walk.get("speed")))
    for g, members in enumerate(fam["talks"]):
        for k, pose in enumerate(members):
            humans.append(human(f"talk{g}_member_{k}", len(humans), list(pose), "./talker.yaml"))
    zones = fam["zones"]
    for k in range(n_peds - len(humans)):
        zone = zones[k % len(zones)]
        humans.append(human(f"single_{k}", len(humans), zone, "adult") | {"waypoint_mode": "random", "waypoints": [zone] * 3})
    assert len(humans) == n_peds
    robot = {"start": list(fam["start"]), "phases": [{"goto": list(fam["goal"]), "tolerance_radius": TOLERANCE}]}
    return {"format": "arena-scenario/2", "robots": [robot], "humans": humans}, types


def dump(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=None, width=200))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("worlds", type=Path, help="worlds root, e.g. src/Arena/arena_simulation_setup/worlds")
    args = p.parse_args()
    for family, fam in FAMILIES.items():
        for n_peds in COUNTS:
            out = args.worlds / fam["world"] / "scenarios" / f"heldout_{family}_{n_peds}"
            out.mkdir(parents=True, exist_ok=True)
            scen, types = scenario(fam, n_peds)
            dump(out / "scenario.yaml", scen)
            for name, data in types.items():
                dump(out / name, data)
            print(out)


if __name__ == "__main__":
    main()
