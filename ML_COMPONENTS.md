# ML Components for Aegis Sentinel

This document describes the machine learning components added to the Aegis Sentinel system to enhance its capabilities for autonomous reliability engineering.

## Overview

The ML components enhance the existing Aegis Sentinel system with advanced capabilities for:

1. **Anomaly Detection** - Using machine learning to detect anomalies in logs and metrics
2. **Root Cause Analysis** - Using graph algorithms and ML to identify the root causes of issues
3. **Autonomous Remediation** - Using reinforcement learning to select and execute remediation actions
4. **Learning from Manual Fixes** - Learning from human operators to improve remediation over time

## Components

### 1. ML-based Log Anomaly Detection

The `MLLogAnomalyDetector` class extends the pattern-based log anomaly detection with machine learning capabilities:

- **Feature Extraction** - Extracts features from log lines using TF-IDF vectorization
- **Anomaly Detection Models** - Uses multiple ML models (Isolation Forest, Local Outlier Factor, Autoencoder) to detect anomalies
- **Service Identification** - Automatically identifies the service associated with log lines
- **Model Training** - Learns from normal log patterns to detect anomalies

```python
from src.ml.anomaly import MLLogAnomalyDetector

# Create a detector with default patterns
detector = MLLogAnomalyDetector.create_with_default_patterns()

# Process logs
anomalies = detector.process_log("ERROR: Connection refused")

# Train models on collected logs
detector.train_models()
```

### 2. Metric Anomaly Detection

The `MetricAnomalyDetector` class provides ML-based anomaly detection for metrics:

- **Statistical Models** - Uses statistical models to detect anomalies in metrics
- **Deep Learning Models** - Uses autoencoders and VAEs for complex anomaly detection
- **Prometheus Integration** - Collects metrics from Prometheus for analysis

```python
from src.ml.anomaly import MetricAnomalyDetector, MetricDataPoint

# Create a detector
detector = MetricAnomalyDetector()

# Add a metric
metric = MetricDataPoint(
    timestamp=datetime.now(),
    value=95.0,
    metric_name="cpu_usage",
    metric_type="cpu",
    service="frontend",
    dimensions={"instance": "pod-1"}
)

# Detect anomalies
anomaly = detector.add_metric(metric)
```

### 3. ML-based Root Cause Analysis

The `RootCauseAnalyzer` class uses graph algorithms and machine learning to identify the root causes of issues:

- **Graph-based Analysis** - Uses graph algorithms to trace the propagation of failures
- **ML-based Prediction** - Uses machine learning to predict the likelihood of a node being the root cause
- **Health Scoring** - Assigns health scores to nodes based on anomalies and issues

```python
from src.ml.root_cause import RootCauseAnalyzer

# Create an analyzer
analyzer = RootCauseAnalyzer(service_graph)

# Add anomalies and issues
analyzer.add_node_anomaly(node_id, anomaly)
analyzer.add_node_issue(node_id, issue)

# Analyze root causes
root_causes = analyzer.analyze_root_cause()
```

### 4. Reinforcement Learning-based Remediation

The `RLRemediationEngine` class uses reinforcement learning to select and execute remediation actions:

- **Environment Simulation** - Simulates the effects of remediation actions
- **RL Models** - Uses PPO (Proximal Policy Optimization) to learn optimal remediation strategies
- **Action Selection** - Selects the best action based on the current state

```python
from src.ml.remediation import RLRemediationEngine

# Create an engine with available actions
engine = RLRemediationEngine(remediation_actions)

# Train the model
engine.train(total_timesteps=10000)

# Select an action
action = engine.select_action(issues, anomalies, metric_anomalies)
```

### 5. Remediation Learning

The `RemediationLearner` class learns from manual fixes to improve remediation over time:

- **Signature Capture** - Captures the context and actions of manual fixes
- **Similarity Matching** - Finds similar past incidents to suggest remediation actions
- **ML-based Prediction** - Uses machine learning to predict the best remediation action

```python
from src.ml.learning import RemediationLearner

# Create a learner
learner = RemediationLearner()

# Capture a manual fix
learner.capture_manual_fix(
    command="kubectl restart deployment/frontend",
    issues=issues,
    anomalies=anomalies,
    success=True
)

# Suggest an action for a new issue
action = learner.suggest_action(issues, anomalies)
```

## Integration

The ML components are integrated with the existing Aegis Sentinel system:

1. **Service Graph** - The ML components use the service graph to understand the system topology
2. **Detection Engine** - The ML components enhance the detection engine with ML-based anomaly detection
3. **Resolution Engine** - The ML components provide advanced remediation capabilities
4. **API and Web Interface** - The ML components expose their functionality through the API and web interface

## Demo

The `examples/ml_demo.py` script demonstrates the ML components:

```bash
# Run the demo with Kubernetes manifests
python examples/ml_demo.py --source examples/kubernetes/microservices-demo.yaml --simulate

# Run with log analysis
python examples/ml_demo.py --source examples/kubernetes/microservices-demo.yaml --log-file /var/log/application.log

# Train ML models
python examples/ml_demo.py --source examples/kubernetes/microservices-demo.yaml --train
```

## Future Enhancements

1. **Predictive Analytics** - Predict future issues based on current trends
2. **Causal Inference** - Use causal inference to better understand the relationships between issues
3. **Transfer Learning** - Apply knowledge from one system to another
4. **Explainable AI** - Provide explanations for ML-based decisions
5. **Federated Learning** - Learn from multiple deployments while preserving privacy