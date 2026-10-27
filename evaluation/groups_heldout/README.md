# Pedestrian groups: held-out evaluation kit

Tools behind the held-out comparison of AGENT_5, AGENT_5_GROUPS with simulator labels (`labels`) and AGENT_5_GROUPS with proximity clusters (`clusters`). Paths below are the ones inside the Arena container (`/opt/arena_ws`).

| File | What it does |
|---|---|
| `make_heldout.py <worlds_root>` | Writes the scenarios `heldout_{junction,sbend,reception}_{15,20}` (also committed in the Arena fork). |
| `setup_eval_worlds.sh <arena_src> <out>` | Copies the three held-out worlds to `<out>` and writes the scenarios into the copies. Run with `ARENA_WORLD_PATH=<out>`, so the training workspace needs no rebuild. |
| `make_suite.py <bench_dir> --tick T` | Suites `rss27_heldout` (as specified, references on) and `rss27_heldout_norefs` (as run); contests `rss27_{agent5,labels,clusters}`. |
| `eval_agent.sh <agents_dir> <agent> <tick>` | Makes `<agent>_t<tick/1000>k/` with `tick_<tick>.zip` copied as `best_model.zip` (the rosnav_rl driver always loads `best_model.zip`). |
| `make_traineval.py` | Resumes a checkpoint in the training env and runs one deterministic in-training eval with no PPO update. Used to rule out the benchmark driver. Run with `arena train` on its own ROS domain, never with `train_arm.sh`. |
| `readout.py` | Means per arm and labels-minus-control deltas with bootstrap intervals (arena_evaluation `experiments/stats.py`), per split and pooled, plus the verdict. |
| `configs/rss27_sanity.yaml` | Sanity suite (empty map and stage-3-like tasks) for the checks; its contests and the default-planner contest were written by hand. |
| `heldout_layouts.png` | Start, goal, groups and walking routes of the three families. |

## Design

- Families not seen in training: T-junction of `hospital_1_bare`, `sbend_corridor`, `reception`. Each has a walking trio, a walking pair, a talking trio and a talking pair, plus 5 or 10 singles with random waypoints, so 15 or 20 pedestrians. Group anchors are at least 3 m apart (humansim's interaction radius).
- Goals 8.1 / 8.8 / 10.6 m straight-line, tolerance 0.8 m (last training stage), timeout 60 s (training: 600 steps at 10 Hz).
- Splits: {socialgail, nsp} × 3 families × {15, 20}. Stage seeds 27000, 27100, …, 28100; episode seed = stage seed + episode index; the same for every arm.
- `lockstep: true`, `max_collisions: 1`. Mean actions (`deterministic_inference` default).

## Run (as done for the hand-in)

```bash
K=/opt/arena_ws/data/rss27_eval
bash $K/setup_eval_worlds.sh /opt/arena_ws/src/Arena $K/worlds
"$ARENA_VENV_DIR/bin/python" $K/make_suite.py /opt/arena_ws/install/arena_evaluation/share/arena_evaluation/configs/benchmark --tick 700000
export ARENA_WORLD_PATH=$K/worlds ROS_DOMAIN_ID=88 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for arm in agent5 labels clusters; do
  bash $K/eval_agent.sh /opt/arena_ws/data/agents groups_$arm 700000
  arena evaluation benchmark --suite rss27_heldout_norefs --contest rss27_$arm --scale-episodes 0.6 --run-id heldout_$arm sim:=flatsim headless:=true
  arena evaluation process --benchmark-dir /opt/arena_ws/data/benchmarks/heldout_$arm
done
"$ARENA_VENV_DIR/bin/python" $K/readout.py
```

`--scale-episodes 0.6` gives episodes 0–11 of each stage (12 per split). References were off to fit the time budget, so there is no pedestrian-deviation metric. Two arms can run side by side on different `ROS_DOMAIN_ID`s. After a crash, use `--resume heldout_<arm>` in place of `--run-id`.

## Python packages added to the locked venv

`numba` 0.68.0, `torch_geometric` 2.8.0.post1, `polars`, `dtaidistance` 2.5.1, `mcap` 1.6.0, `mcap-ros2-support` 0.6.0, `narwhals` 2.27.1, `plotly` 7.1.0, `seaborn` 0.13.2, installed with every existing version held fixed.
