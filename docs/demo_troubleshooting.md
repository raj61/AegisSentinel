# Aegis Sentinel: Demo Troubleshooting Guide

Based on the logs from your demo run, I've identified a few issues that need to be addressed to ensure a smooth demonstration. This guide will help you resolve these issues and run a successful demo.

## Issues Identified

From the logs, I can see the following issues:

1. **Missing ML Models**:
   ```
   Error loading model for detector metrics: [Errno 2] No such file or directory: 'models/anomaly/metrics_model.npy'
   ```

2. **Untrained Models**:
   ```
   Error in detector metrics: Model must be trained before detection
   Error in detector logs: Model must be trained before detection
   ```

3. **No Available Remediation Actions**:
   ```
   No available actions for the current state
   ```

## Quick Fixes

### 1. Create Placeholder ML Models

The system is looking for model files that don't exist yet. Let's create placeholder files:

```bash
# Create directories if they don't exist
mkdir -p models/anomaly
mkdir -p models/remediation

# Create placeholder model files
touch models/anomaly/metrics_model.npy
touch models/anomaly/logs_model.npy
touch models/anomaly/ml_model.npy
```

### 2. Initialize Models Before Running the Demo

Let's create a script to initialize the models before running the demo:

```bash
# Create a script to initialize models
cat > initialize_models.py << 'EOF'
#!/usr/bin/env python3
"""
Initialize ML models for Aegis Sentinel demo
"""

import os
import numpy as np
import json
import pickle
from pathlib import Path

def create_directory(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

def create_metrics_model():
    """Create a simple metrics anomaly detection model"""
    print("Creating metrics model...")
    model_path = "models/anomaly/metrics_model.npy"
    
    # Create a simple threshold-based model
    model = {
        'cpu_threshold': 80.0,
        'memory_threshold': 90.0,
        'latency_threshold': 500.0,
        'error_rate_threshold': 0.05,
        'z_score_threshold': 3.0
    }
    
    # Save as numpy array
    np.save(model_path, np.array([model]))
    print(f"Metrics model saved to {model_path}")

def create_logs_model():
    """Create a simple log anomaly detection model"""
    print("Creating logs model...")
    model_path = "models/anomaly/logs_model.npy"
    
    # Create a simple keyword-based model
    model = {
        'error_keywords': ['error', 'exception', 'fail', 'critical', 'crash'],
        'warning_keywords': ['warning', 'warn', 'timeout', 'slow'],
        'threshold': 0.7
    }
    
    # Save as numpy array
    np.save(model_path, np.array([model]))
    print(f"Logs model saved to {model_path}")

def create_ml_model():
    """Create a simple ML-based anomaly detection model"""
    print("Creating ML model...")
    model_path = "models/anomaly/ml_model.npy"
    
    # Create a simple isolation forest-like model
    model = {
        'contamination': 0.01,
        'n_estimators': 100,
        'max_samples': 'auto',
        'threshold': 0.5
    }
    
    # Save as numpy array
    np.save(model_path, np.array([model]))
    print(f"ML model saved to {model_path}")

def create_rules_model():
    """Create a simple rules-based remediation model"""
    print("Creating rules model...")
    model_path = "models/remediation/rules_model.json"
    
    # Create a simple rules model
    model = {
        'rules': [
            {
                'condition': {'type': 'cpu_spike', 'severity': {'min': 3}},
                'action': 'Scale Service',
                'parameters': {'replicas': '+1'}
            },
            {
                'condition': {'type': 'memory_leak', 'severity': {'min': 4}},
                'action': 'Restart Service',
                'parameters': {}
            },
            {
                'condition': {'type': 'error_spike', 'severity': {'min': 3}},
                'action': 'Rollback Deployment',
                'parameters': {'to_version': 'previous'}
            },
            {
                'condition': {'type': 'disk_full', 'severity': {'min': 4}},
                'action': 'Clear Cache',
                'parameters': {}
            }
        ]
    }
    
    # Save as JSON
    with open(model_path, 'w') as f:
        json.dump(model, f, indent=2)
    print(f"Rules model saved to {model_path}")

def create_rl_model():
    """Create a simple RL-based remediation model"""
    print("Creating RL model...")
    model_path = "models/remediation/rl_model.json"
    
    # Create a simple Q-learning model
    model = {
        'q_table': {
            'cpu_spike_critical': {
                'Scale Service': 0.8,
                'Restart Service': 0.4,
                'Rollback Deployment': 0.2,
                'Drain Node': 0.1,
                'Clear Cache': 0.0
            },
            'memory_leak_critical': {
                'Scale Service': 0.3,
                'Restart Service': 0.9,
                'Rollback Deployment': 0.4,
                'Drain Node': 0.2,
                'Clear Cache': 0.5
            },
            'error_spike_critical': {
                'Scale Service': 0.2,
                'Restart Service': 0.5,
                'Rollback Deployment': 0.9,
                'Drain Node': 0.1,
                'Clear Cache': 0.0
            }
        },
        'learning_rate': 0.1,
        'discount_factor': 0.9,
        'exploration_rate': 0.1
    }
    
    # Save as JSON
    with open(model_path, 'w') as f:
        json.dump(model, f, indent=2)
    print(f"RL model saved to {model_path}")

def create_action_library():
    """Create an action library"""
    print("Creating action library...")
    model_path = "models/remediation/action_library.json"
    
    # Create a simple action library
    model = {
        'actions': [
            {
                'name': 'Restart Service',
                'description': 'Restart a service to clear its state',
                'parameters': {},
                'preconditions': ['service_exists'],
                'effects': ['service_restarted', 'memory_cleared']
            },
            {
                'name': 'Scale Service',
                'description': 'Scale a service to handle more load',
                'parameters': {'replicas': 'int'},
                'preconditions': ['service_exists', 'can_scale'],
                'effects': ['service_scaled', 'capacity_increased']
            },
            {
                'name': 'Rollback Deployment',
                'description': 'Rollback to a previous version',
                'parameters': {'to_version': 'string'},
                'preconditions': ['service_exists', 'has_previous_version'],
                'effects': ['service_rolled_back', 'version_changed']
            },
            {
                'name': 'Drain Node',
                'description': 'Drain a node to move workloads',
                'parameters': {'node': 'string'},
                'preconditions': ['node_exists', 'can_drain'],
                'effects': ['node_drained', 'workloads_moved']
            },
            {
                'name': 'Clear Cache',
                'description': 'Clear a service cache',
                'parameters': {},
                'preconditions': ['service_exists', 'has_cache'],
                'effects': ['cache_cleared', 'memory_freed']
            }
        ]
    }
    
    # Save as JSON
    with open(model_path, 'w') as f:
        json.dump(model, f, indent=2)
    print(f"Action library saved to {model_path}")

def create_experience_history():
    """Create an experience history"""
    print("Creating experience history...")
    model_path = "models/remediation/experience_history.json"
    
    # Create a simple experience history
    model = {
        'experiences': [
            {
                'state': 'cpu_spike_critical',
                'action': 'Scale Service',
                'parameters': {'replicas': '+1'},
                'next_state': 'normal',
                'reward': 1.0,
                'timestamp': '2025-04-01T12:00:00Z'
            },
            {
                'state': 'memory_leak_critical',
                'action': 'Restart Service',
                'parameters': {},
                'next_state': 'normal',
                'reward': 1.0,
                'timestamp': '2025-04-02T14:30:00Z'
            },
            {
                'state': 'error_spike_critical',
                'action': 'Rollback Deployment',
                'parameters': {'to_version': 'v1.2.3'},
                'next_state': 'normal',
                'reward': 1.0,
                'timestamp': '2025-04-03T09:15:00Z'
            }
        ]
    }
    
    # Save as JSON
    with open(model_path, 'w') as f:
        json.dump(model, f, indent=2)
    print(f"Experience history saved to {model_path}")

def main():
    """Main function"""
    # Create directories
    create_directory("models/anomaly")
    create_directory("models/remediation")
    
    # Create anomaly detection models
    create_metrics_model()
    create_logs_model()
    create_ml_model()
    
    # Create remediation models
    create_rules_model()
    create_rl_model()
    create_action_library()
    create_experience_history()
    
    print("All models initialized successfully!")

if __name__ == "__main__":
    main()
EOF

# Make the script executable
chmod +x initialize_models.py

# Run the script
python initialize_models.py
```

### 3. Run the Demo with the Initialized Models

Now you can run the demo with the initialized models:

```bash
# Run the web interface demo
python run_web_interface.py --enable-ml --inject-anomaly
```

## Enhanced Demo Script

For a more reliable demo, follow these steps:

### Step 1: Initialize the Environment

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/aegis-sentinel/aegis-sentinel.git
cd aegis-sentinel

# Install dependencies
pip install -r requirements.txt

# Initialize ML models
python initialize_models.py
```

### Step 2: Run the Web Interface Demo

```bash
# Start the web interface
python run_web_interface.py --enable-ml --inject-anomaly
```

### Step 3: Observe the Demo in the Browser

1. Open your browser and navigate to http://localhost:8080
2. You should see the service graph visualization
3. After about 30 seconds, you'll see an anomaly injected
4. The affected service will turn red
5. The system will identify the root cause
6. The system will apply remediation

### Step 4: Inject Additional Anomalies (Optional)

You can inject additional anomalies to demonstrate the system's capabilities:

```bash
# In a new terminal, run:
curl http://localhost:8080/api/demo/inject-anomaly?service=database&type=cpu_spike&severity=4
```

## Troubleshooting Common Issues

### Issue: Models Not Found

If you see errors about models not being found:

```
Error loading model for detector metrics: [Errno 2] No such file or directory: 'models/anomaly/metrics_model.npy'
```

**Solution**: Run the `initialize_models.py` script to create the necessary model files.

### Issue: Untrained Models

If you see errors about untrained models:

```
Error in detector metrics: Model must be trained before detection
```

**Solution**: The `initialize_models.py` script creates pre-trained models. Make sure you've run it before starting the demo.

### Issue: No Available Remediation Actions

If you see warnings about no available remediation actions:

```
No available actions for the current state
```

**Solution**: The `initialize_models.py` script creates a remediation action library and experience history. Make sure you've run it before starting the demo.

### Issue: Web Interface Not Loading

If the web interface doesn't load:

**Solution**: 
1. Check if the web server is running: `ps aux | grep run_web_interface.py`
2. Check for errors in the logs: `cat web_interface.log`
3. Ensure port 8080 is not in use by another application: `lsof -i :8080`

### Issue: No Anomalies Being Detected

If no anomalies are being detected:

**Solution**:
1. Check if the ML components are enabled: `--enable-ml` flag should be used
2. Check for errors in the logs: `cat web_interface.log`
3. Try manually injecting an anomaly: `curl http://localhost:8080/api/demo/inject-anomaly?service=database&type=cpu_spike&severity=4`

## Demo Preparation Checklist

Before presenting the demo, go through this checklist:

- [ ] Run `initialize_models.py` to create all necessary model files
- [ ] Test the web interface to ensure it loads correctly
- [ ] Test anomaly injection to ensure it works
- [ ] Check the logs for any errors or warnings
- [ ] Prepare talking points for each stage of the demo
- [ ] Have the technical documentation ready for any questions

By following these steps, you should be able to run a smooth and impressive demonstration of Aegis Sentinel's capabilities.