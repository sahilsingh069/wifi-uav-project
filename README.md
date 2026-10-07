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
