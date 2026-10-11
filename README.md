# Wi-Fi/Cellular UAV Mobility Prediction and Trajectory Optimization

**[Live demo: 3D drone coverage viewer](https://sahilsingh069.github.io/wifi-uav-project/)**

College project implementing a full RF-signal-to-UAV-control research pipeline:

1. Simulate Wi-Fi/cellular RF measurements for moving users (3GPP TR 38.901 UMi/UMa path loss,
   spatially correlated shadowing and LoS state, distance-dependent multipath CSI)
2. Extract time-series signal features and split **by episode** (no leakage between train and test)
3. Train an LSTM-Transformer that infers each user's position 5 steps ahead from RF alone
4. Build a multi-UAV coverage environment where **UAVs observe the predicted user positions**
   (coverage is still scored on the true positions), altitude trades beam footprint against link
   range, and a UAV with an empty battery lands
5. Train cooperative MADDPG agents on that environment: one shared actor and a centralised critic
   (parameter sharing), egocentric observations (each UAV sees its 6 nearest users and its
   teammates), and per-UAV difference rewards (the users only that UAV covers)
6. Evaluate the trained actor without exploration noise against centralised k-means greedy,
   decentralised local greedy (same information as one MADDPG actor), random-walk, and
   static-hover baselines on identical held-out trajectories

Every phase accepts `--positions oracle` (where relevant) to rerun with ground-truth user positions,
which gives an upper bound for how much prediction error costs in coverage.

## Web Viewer

`web/` is a static 3D replay viewer (Three.js, no build step). Drones fly over the area with their
coverage cones, users turn green or red, and yellow rings show where the RF predictor thinks each
user is. You can switch policy, position source, and episode, and compare coverage over time.

```bash
python scripts/export_replays.py --preset medium --steps 80   # writes web/data/replays.json
python -m http.server 8000 -d web                             # open http://localhost:8000
```

Links can open a specific moment: `?policy=Greedy&mode=oracle&episode=2&t=40`.
Pushing changes under `web/` deploys it to GitHub Pages via `.github/workflows/pages.yml`
(one-time setup: repo Settings -> Pages -> Source: GitHub Actions).

## Recommended Runtime

Use Google Colab with GPU enabled for training. Local execution is intended for smoke tests and development.

## Quick Local Smoke Test

```bash
python -m pip install -e ".[dev]"
pytest -q
```

## Phase Scripts

```bash
python -m pip install -e ".[dev]"     # scripts import the installed wifi_uav package
python scripts/phase1_collect_data.py --preset smoke
python scripts/phase2_build_features.py --preset smoke
python scripts/phase3_train_predictor.py --preset smoke      # epochs default to the preset
python scripts/phase4_sanity_env.py --preset smoke
python scripts/phase5_train_maddpg.py --preset smoke         # --positions predicted|oracle
python scripts/phase6_evaluate.py --preset smoke             # --episodes N per policy
```

Presets: `smoke` (minutes, checks wiring only, numbers are meaningless), `medium`, `full` (Colab GPU).

## Outputs

- `results/predictor_summary.txt`: test RMSE on held-out episodes plus the centre-guess baseline it must beat
- `results/final_summary.txt`: MADDPG vs baselines, mean coverage over every evaluation step
- `results/training_curves.png`, `results/baseline_comparison.png`, `results/coverage_heatmap.png`
- `checkpoints/predictor_best.pt`, `checkpoints/maddpg_actor_*.pt`

## Results

Medium preset (18 users, 5 UAVs, 500 m x 500 m), UAVs observing **RF-predicted** user positions,
50 held-out evaluation episodes per policy, coverage averaged over every step:

| Policy | Coverage | Flown per UAV per step | Information used |
|---|---|---|---|
| Greedy (centralised k-means) | 97.8% | 4.8 m | every user's position, plans for all UAVs at once |
| Local greedy | 96.2% | 6.4 m | same as one MADDPG actor |
| **MADDPG (600 episodes)** | **89.7%** | 10.1 m | 6 nearest users + teammates, learned policy |
| Static hover | 66.2% | 0 m | none |
| Random walk | 58.9% | 18.5 m | none |

- Position predictor: test RMSE `59.3 m` on held-out episodes (validation `49.5 m`) vs `202.7 m`
  for always guessing the area centre; mean position error inside the coverage env `47.8 m`.
- The predictor reads the last 60 RF measurements (30 s). Shadowing decorrelates over about 20 m of
  walking, so a short window sees one shadowing draw and cannot average it out. Window length was
  chosen on the validation set only (10 steps 67.0 m, 30 steps 59.4 m, 60 steps 53.0 m); more
  epochs, a lower learning rate, and 4x more training episodes did not help, because validation
  loss plateaus after a few epochs while training loss keeps falling (the remaining error is
  per-episode shadowing that no model can learn). Test RMSE went from `75.3 m` to `59.3 m`.
- Better prediction also saves energy: with the 10-step predictor greedy flew 9.7 m per step
  chasing prediction noise; with the 60-step one it flies 4.8 m for the same coverage.
- UAVs get 30 s of RF history before launch so the predictor has a full window from step one.
- An earlier version reported 21.76 m RMSE. That number was inflated: overlapping windows were
  shuffled before the train/test split, so near-duplicates of test windows were in training, and
  shadowing was redrawn independently every step, which made it easy to average away.
- MADDPG beats static hover by 23 points and random by 31, but still trails both hand-written
  heuristics by 6-8 points. In this environment coverage is close to a clustering problem,
  which greedy solves almost exactly; RL would earn its keep with objectives a heuristic can't
  express (obstacles, link quality, battery-aware hand-offs).
- How MADDPG got from 47% to about 90%: the first version gave each UAV only distances (no direction)
  to its nearest users, so its actors saturated and pinned 39% of UAVs to the area edge. Egocentric
  relative observations, a pre-tanh action penalty, a slower actor learning rate, parameter sharing,
  difference rewards, and 600 training episodes fixed that (edge-pinning down to under 5%).
  Doubling gradient updates per step did not help further.

Reproduce (scripts use CUDA, Apple-silicon MPS, or CPU automatically):

```bash
python scripts/phase1_collect_data.py --preset medium
python scripts/phase2_build_features.py --preset medium
python scripts/phase3_train_predictor.py --preset medium   # ~7 min on an M2 GPU (MPS)
python scripts/phase5_train_maddpg.py --preset medium            # 600 episodes
python scripts/phase6_evaluate.py --preset medium --episodes 50
```

`phase5_train_maddpg.py` and `phase6_evaluate.py` accept `--set key=value` to override any config
field (for example `--set maddpg_gamma=0.9`).

## Watching Training Live

Phase 3 prints one line per epoch and phase 5 one line per episode:

```text
epoch  train_loss   val_loss   val_RMSE        lr   time
    1     0.00357    0.00279     52.8 m   1.0e-03    19s  * best, saved
    2     0.00090    0.00304     55.1 m   7.5e-04    19s  (no improvement 1/15)

episode  noise   reward  coverage  last-10 avg  critic_loss  actor_loss  replay   time
     14   0.05     1.92     30.1%        31.4%       0.0061     -1.0091     840   1.6s
```

Full runs also save `results/predictor_history.csv`, `results/predictor_training.png` and
`results/maddpg_history.csv`. Add `--no-save` to phases 3, 5 and 6 to run a short live demo without
overwriting trained models or results:

```bash
python scripts/phase3_train_predictor.py --preset medium --epochs 3 --no-save    # ~1 min on M2
python scripts/phase5_train_maddpg.py --preset medium --episodes 15 --no-save    # ~30 s
python scripts/phase6_evaluate.py --preset medium --episodes 5 --no-save         # trained models vs baselines
```

## Tests

`30 passed` (`pytest -q`); the smoke pipeline runs all 6 phases end to end.
