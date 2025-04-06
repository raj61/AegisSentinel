# Aegis Sentinel: ML-Enhanced Service Graph Analysis

Aegis Sentinel is an advanced platform for service graph creation, anomaly detection, and automated remediation. It uses machine learning to provide intelligent monitoring and self-healing capabilities for Kubernetes and cloud infrastructure.

## Features

- **Service Graph Generation**: Automatically build service dependency graphs from Kubernetes, Terraform, or CloudFormation configurations
- **ML-Based Anomaly Detection**: Detect anomalies in service metrics and logs using machine learning
- **Automated Remediation**: Use reinforcement learning to automatically remediate detected issues
- **Interactive Visualization**: Visualize service dependencies and health in a modern web interface
- **Real-time Monitoring**: Monitor services in real-time and get alerts for anomalies

## Architecture

Aegis Sentinel consists of several components:

- **Parsers**: Parse Kubernetes, Terraform, and CloudFormation configurations
- **Service Graph Builder**: Build service dependency graphs from parsed configurations
- **Anomaly Detection Engine**: Detect anomalies in service metrics and logs
- **Remediation Engine**: Automatically remediate detected issues
- **Visualization Engine**: Visualize service dependencies and health

For a detailed architecture diagram, see [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md).

## ML Capabilities

Aegis Sentinel uses several machine learning techniques:

### Anomaly Detection

- **Time Series Anomaly Detection**: Detect anomalies in service metrics using statistical methods and LSTM neural networks
- **Log Anomaly Detection**: Detect anomalies in service logs using NLP techniques
- **Relationship Inference**: Infer service relationships using graph neural networks

### Automated Remediation

- **Reinforcement Learning**: Learn optimal remediation strategies from past incidents
- **Decision Trees**: Make remediation decisions based on issue type and severity
- **Knowledge Base**: Build a knowledge base of remediation strategies

## Getting Started

### Prerequisites

- Python 3.8+
- Kubernetes cluster (for Kubernetes parsing)
- Terraform (for Terraform parsing)
- CloudFormation (for CloudFormation parsing)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/aegis-sentinel.git
cd aegis-sentinel
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Usage

#### Building a Service Graph

```python
from src.parsers.kubernetes_parser import KubernetesParser
from src.graph.service_graph import ServiceGraph

# Parse Kubernetes manifest
parser = KubernetesParser()
parser.parse_file('path/to/kubernetes/manifest.yaml')

# Build service graph
graph = ServiceGraph()
parser.build_graph(graph)

# Infer relationships
graph.infer_relationships()

# Visualize the graph
graph.visualize('service_graph.png')
```

#### Running the ML Demo

```bash
python examples/aegis_sentinel_demo.py --inject-anomaly
```

#### Starting the Web Interface

```bash
python run_web_interface.py
```

## Examples

The `examples` directory contains several examples:

- `analyze_k8s_cluster.py`: Analyze a Kubernetes cluster
- `analyze_terraform.py`: Analyze Terraform configurations
- `aegis_sentinel_demo.py`: Demonstrate ML capabilities

## ML Components

### Anomaly Detection

The anomaly detection engine uses several techniques:

- **Statistical Methods**: Z-score analysis for simple anomaly detection
- **LSTM Neural Networks**: For complex time series anomaly detection
- **NLP Techniques**: For log anomaly detection

### Remediation Learning

The remediation learning engine uses:

- **Rule-Based Learning**: Learn rules from past incidents
- **Reinforcement Learning**: Learn optimal remediation strategies
- **Knowledge Base**: Build a knowledge base of remediation strategies

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.