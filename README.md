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

| Policy | Coverage | Information used |
|---|---|---|
| Greedy (centralised k-means) | 97.7% | every user's position, plans for all UAVs at once |
| Local greedy | 97.1% | same as one MADDPG actor |
| **MADDPG (600 episodes)** | **90.5%** | 6 nearest users + teammates, learned policy |
| Static hover | 64.7% | none |
| Random walk | 58.9% | none |

- Position predictor: test RMSE `75.3 m` on held-out episodes vs `202.7 m` for always guessing the
  area centre. Greedy loses only about 1 point of coverage when it uses predicted instead of true
  positions, so RF-only localisation is good enough for coverage control.
- MADDPG beats static hover by 26 points and random by 32, but still trails both hand-written
  heuristics by about 7 points. In this environment coverage is close to a clustering problem,
  which greedy solves almost exactly; RL would earn its keep with objectives a heuristic can't
  express (obstacles, link quality, battery-aware hand-offs).
- How MADDPG got from 47% to 90.5%: the first version gave each UAV only distances (no direction)
  to its nearest users, so its actors saturated and pinned 39% of UAVs to the area edge. Egocentric
  relative observations, a pre-tanh action penalty, a slower actor learning rate, parameter sharing,
  difference rewards, and 600 training episodes fixed that (edge-pinning down to under 5%).
  Doubling gradient updates per step did not help further.

Reproduce (about 10 minutes on a laptop CPU for MADDPG):

```bash
python scripts/phase1_collect_data.py --preset medium
python scripts/phase2_build_features.py --preset medium
python scripts/phase3_train_predictor.py --preset medium --epochs 8
python scripts/phase5_train_maddpg.py --preset medium            # 600 episodes
python scripts/phase6_evaluate.py --preset medium --episodes 50
```

`phase5_train_maddpg.py` and `phase6_evaluate.py` accept `--set key=value` to override any config
field (for example `--set maddpg_gamma=0.9`).

## Tests

`30 passed` (`pytest -q`); the smoke pipeline runs all 6 phases end to end.
