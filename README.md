# Wi-Fi/Cellular UAV Mobility Prediction and Trajectory Optimization

College project implementing a full RF-signal-to-UAV-control research pipeline:

1. Simulate Wi-Fi/cellular RF measurements for moving users (3GPP TR 38.901 UMi/UMa path loss,
   spatially correlated shadowing and LoS state, distance-dependent multipath CSI)
2. Extract time-series signal features and split **by episode** (no leakage between train and test)
3. Train an LSTM-Transformer that infers each user's position 5 steps ahead from RF alone
4. Build a multi-UAV coverage environment where **UAVs observe the predicted user positions**
   (coverage is still scored on the true positions), altitude trades beam footprint against link
   range, and a UAV with an empty battery lands
5. Train cooperative MADDPG agents on that environment
6. Evaluate the trained actors without exploration noise against k-means greedy, random-walk, and
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

## Verified Runs

- Tests: `25 passed`
- Smoke pipeline runs end to end (all 6 phases)
- Medium preset, predictor after 6 CPU epochs: test RMSE `73.5 m` on held-out episodes vs `202.7 m`
  for always guessing the centre of the 500 m x 500 m area
- Medium preset, 80 MADDPG episodes, predicted positions, 10 eval episodes per policy (mean coverage):
  k-means greedy `98.3%`, static `74.7%`, random `59.1%`, MADDPG `40.7%`. With this little training
  the actors saturate and push UAVs to the area edges. Greedy with ground-truth positions reaches
  `99.3%`, so RF prediction error (`57.5 m` mean in-env) costs only about 1 point of coverage.
  MADDPG needs the full preset (500 episodes) and/or tuning before it can beat greedy.
