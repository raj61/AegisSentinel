# ML Models Directory

This directory contains trained ML models for Aegis Sentinel.

## Directory Structure

- `anomaly/`: Contains anomaly detection models
  - `metrics_model.npy`: Time series anomaly detection model for metrics
  - `logs_model.npy`: Log anomaly detection model
  - `ml_model.npy`: Deep learning-based anomaly detection model

- `remediation/`: Contains remediation learning models
  - `rules_model.json`: Rule-based remediation model
  - `rl_model.json`: Reinforcement learning-based remediation model
  - `action_library.json`: Library of remediation actions
  - `experience_history.json`: History of remediation experiences

## Training Models

Models are automatically trained when running the system with the `--enable-ml` flag. You can also manually train models using the following command:

```bash
python examples/aegis_sentinel_demo.py --save-models
```

## Loading Models

Models are automatically loaded when running the system with the `--enable-ml` flag. You can also manually load models using the `--load-models` flag:

```bash
python examples/aegis_sentinel_demo.py --load-models
```

## Model Formats

- `.npy`: NumPy array format for statistical models
- `.json`: JSON format for rule-based models and configuration
- `.h5`: HDF5 format for deep learning models (when using TensorFlow)
- `.pt`: PyTorch format for deep learning models (when using PyTorch)

## Notes

- The models in this directory are placeholders and will be replaced with actual trained models when the system is run with ML capabilities enabled.
- In a production environment, you would want to version your models and keep track of their performance metrics.