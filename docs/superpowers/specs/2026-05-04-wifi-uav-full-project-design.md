# Wi-Fi/Cellular UAV Mobility Prediction and Trajectory Optimization Design

## Purpose

Build a complete college project for AI-driven user mobility prediction and intelligent UAV trajectory optimization using Wi-Fi/cellular signal data. The final project must include runnable code, Colab execution flow, trained/evaluated models, plots/results, a written report, and presentation material.

The implementation follows the provided build manual's technical phases:

1. RF signal and mobility data collection
2. Signal feature extraction
3. LSTM-Transformer user mobility prediction
4. Multi-UAV coverage simulation environment
5. MADDPG multi-agent reinforcement learning
6. Evaluation, plots, baselines, and final reporting

## Runtime Strategy

The project will be built in this local workspace, but heavy execution will target Google Colab GPU.

Primary runtime:

- Google Colab with GPU enabled
- Python notebook cells organized by the manual's phases
- Checkpoint and result files saved under project folders

Development/runtime split:

- Local workspace: source code, notebook generation, report/PPT assets, version control
- Colab: dataset generation, model training, reinforcement learning training, evaluation plots

## Technical Approach

Use a hybrid Sionna-ready implementation.

The first complete version will use a physics-based wireless simulator that follows the manual's requirements: 3GPP-style UMa LoS/NLoS path loss, log-normal shadowing, Rayleigh multipath CSI, Wi-Fi APs, cellular base stations, Gauss-Markov users, and configurable simulation settings.

The code will be structured so that a real Sionna RT-backed implementation can replace or extend the signal model later if Colab supports the required `sionna-rt` install and APIs cleanly. This avoids blocking the full project on external package instability while preserving the academic signal-modeling intent.

## Project Structure

```text
wifi_uav_project/
  notebooks/
    WiFi_UAV_Full_Project_Colab.ipynb
  src/
    config.py
    signal_model.py
    mobility.py
    features.py
    predictor.py
    uav_env.py
    maddpg.py
    evaluate.py
  scripts/
    phase1_collect_data.py
    phase2_build_features.py
    phase3_train_predictor.py
    phase5_train_maddpg.py
    phase6_evaluate.py
  data/
  models/
  checkpoints/
  results/
  reports/
```

## Components

### Configuration

`src/config.py` will define the manual's core constants: area size, number of users, APs, base stations, sequence length, prediction horizon, UAV count, UAV speed, training batch size, and episode counts.

It will also define a smaller smoke-test configuration so every phase can be verified quickly before full overnight training.

### Signal and Mobility Simulation

`src/signal_model.py` will generate Wi-Fi and cellular RSSI plus CSI summary data. It will model:

- 8 Wi-Fi APs over a 500 m x 500 m area
- 3 cellular base stations
- 2.4 GHz Wi-Fi and 3.5 GHz cellular assumptions
- LoS/NLoS probability
- path loss, shadowing, and multipath fading
- CSI amplitude/phase features

`src/mobility.py` will implement Gauss-Markov mobile user trajectories with boundary handling.

### Feature Extraction

`src/features.py` will convert raw RF time series into model-ready sequences:

- raw Wi-Fi RSSI
- delta RSSI
- rolling RSSI variance
- cellular RSSI
- CSI summary statistics
- sliding windows of 10 steps
- position labels 5 steps ahead
- train/validation/test split
- normalization metadata saved for reproducibility

### Mobility Predictor

`src/predictor.py` will implement the LSTM-Transformer model:

- input projection to hidden size 128
- LayerNorm and ReLU
- 2-layer LSTM
- 2-layer Transformer encoder
- pooled LSTM and Transformer features
- MLP head with sigmoid output for normalized `(x, y)`

Training will use AdamW, Huber loss, cosine learning rate schedule, gradient clipping, early stopping, and checkpoint saving.

### UAV Environment

`src/uav_env.py` will implement a Gym-style multi-UAV coverage environment:

- 5 UAVs
- 30 mobile users
- normalized observations per UAV
- continuous 3D actions
- altitude constraints from 50 m to 150 m
- reward terms for coverage, coverage improvement, energy, and overlap penalty
- random-action sanity check

### MADDPG

`src/maddpg.py` will implement:

- replay buffer
- actor networks per UAV
- centralized critic networks
- target networks
- Gaussian exploration noise with annealing
- soft target updates
- gradient clipping
- checkpoint saving every fixed number of episodes

### Evaluation

`src/evaluate.py` and `scripts/phase6_evaluate.py` will generate:

- predictor RMSE in metres
- MADDPG coverage percentage
- training curves
- baseline comparison against greedy, random walk, and static hover policies
- coverage heatmap
- final summary text
- report/PPT-ready figures in `results/`

## Notebook Flow

`notebooks/WiFi_UAV_Full_Project_Colab.ipynb` will mirror the manual:

1. Install dependencies and verify GPU
2. Load configuration
3. Collect/generate RF dataset
4. Build features and splits
5. Train LSTM-Transformer predictor
6. Build and sanity-check UAV environment
7. Train MADDPG agents
8. Evaluate baselines and generate plots
9. Zip outputs for download

## Validation Strategy

Every phase will have a smoke-test path before full runs.

Minimum checks:

- imports succeed
- dataset CSV is non-empty and has expected columns
- feature arrays have expected dimensions
- predictor trains for a few epochs without NaN loss
- UAV environment reset/step works
- MADDPG replay buffer samples correctly
- one short RL training run produces non-crashing metrics
- evaluation generates plot files

Full-run checks:

- predictor RMSE reported in metres
- MADDPG coverage reported as a percentage
- baseline comparison generated
- result files saved under `results/`

## Deliverables

Final deliverables will include:

- source code
- Colab notebook
- generated dataset or reproducible generation script
- trained model checkpoints when available
- evaluation plots
- summary report material
- PPT/presentation material
- zipped project outputs for submission

## Risks and Mitigations

Sionna RT may be difficult to install or run reliably in Colab. The mitigation is to deliver the full system first with a physics-based 3GPP-style signal model, then add Sionna RT as an optional backend if installation succeeds.

Long RL training may exceed practical notebook time. The mitigation is to support smoke, medium, and full configurations, with checkpoints and resumable training.

Coverage/RMSE targets may not be reached in the first full run. The mitigation is to preserve reproducible metrics, compare against baselines, tune training settings, and honestly report results.

## Out of Scope for First Build

- Real drone hardware integration
- Real-world RF measurement collection
- Live map import from an external city model
- Production deployment
- Safety-critical UAV flight control

## Approval

The user approved the Colab-first hybrid Sionna-ready design on 2026-05-04.
