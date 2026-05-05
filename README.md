# Wi-Fi/Cellular UAV Mobility Prediction and Trajectory Optimization

College project implementing a full RF-signal-to-UAV-control research pipeline:

1. Generate Wi-Fi/cellular RF mobility data
2. Extract time-series signal features
3. Train an LSTM-Transformer mobility predictor
4. Build a multi-UAV coverage environment
5. Train MADDPG agents
6. Evaluate against greedy, random, and static baselines

## Recommended Runtime

Use Google Colab with GPU enabled for training. Local execution is intended for smoke tests and development.

## Quick Local Smoke Test

```bash
python -m pip install -e ".[dev]"
pytest -q
```

## Phase Scripts

```bash
python scripts/phase1_collect_data.py --preset smoke
python scripts/phase2_build_features.py --preset smoke
python scripts/phase3_train_predictor.py --preset smoke --epochs 2
python scripts/phase4_sanity_env.py --preset smoke
python scripts/phase5_train_maddpg.py --preset smoke
python scripts/phase6_evaluate.py --preset smoke
```

## Verified Smoke Run

The local smoke pipeline verifies that every phase executes before moving to Colab full training.

Latest smoke verification:

- Tests: `17 passed`
- RF dataset: 450 rows, 28 columns
- Feature tensor: `X_train` shape `(138, 10, 33)`
- Predictor smoke RMSE: `74.20 m`
- MADDPG smoke coverage: `83.3%`

Generated artifacts:

- `data/rf_dataset.csv`
- `data/X_train.npy`
- `checkpoints/predictor_best.pt`
- `results/predictor_summary.txt`
- `results/maddpg_rewards.npy`
- `results/training_curves.png`
- `results/baseline_comparison.png`
- `results/final_summary.txt`
