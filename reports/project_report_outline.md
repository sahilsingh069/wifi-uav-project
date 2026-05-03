# Project Report Outline

## Title

AI-Driven User Mobility Prediction and Intelligent UAV Trajectory Optimization Using Wi-Fi/Cellular Signals

## Abstract

Summarize RF data generation, LSTM-Transformer mobility prediction, MADDPG UAV coordination, and baseline evaluation.

## Problem Statement

Mobile users change position over time, causing wireless coverage demand to shift. The project predicts near-future user locations from RF signals and optimizes UAV placement for improved coverage.

## Methodology

1. RF signal simulation with path loss, shadowing, fading, RSSI, and CSI summaries
2. Feature extraction using RSSI deltas and rolling variance
3. LSTM-Transformer future position prediction
4. Gym-style UAV coverage environment
5. MADDPG cooperative control
6. Baseline comparison

## Results

Insert predictor RMSE, MADDPG coverage, baseline chart, training curves, and coverage heatmap.

## Conclusion

Discuss achieved coverage, prediction accuracy, limitations, and future work with real Sionna RT or real RF data.
