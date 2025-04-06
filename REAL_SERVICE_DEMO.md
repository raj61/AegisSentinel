# Monitoring Real Kubernetes Services with Aegis Sentinel

This guide explains how to set up and run Aegis Sentinel with real Kubernetes services instead of the simulated demo.

## Prerequisites

- Kubernetes cluster (local or remote)
- kubectl configured to access your cluster
- Prometheus monitoring set up in your cluster
- Python 3.8+ with required dependencies installed

## Quick Setup

We've provided several setup scripts that can help you quickly set up a local demo environment:

### Basic Setup

```bash
# Check available options
./setup_demo_cluster.py --help

# Set up everything (minikube, services, and Prometheus)
./setup_demo_cluster.py --all

# Or set up components individually
./setup_demo_cluster.py --start-minikube
./setup_demo_cluster.py --deploy-services
./setup_demo_cluster.py --setup-prometheus
```

### Alternative Prometheus Setup (No Helm Required)

If you're experiencing issues with Helm, you can use our alternative Prometheus setup script:

```bash
# Set up Prometheus without Helm
./setup_prometheus.py --port-forward
```

This script deploys Prometheus directly using kubectl and YAML configurations, without requiring Helm.

### Generating Traffic

To demonstrate real-time monitoring, you can generate traffic to your services:

```bash
# Generate traffic to services with port forwarding
./generate_traffic.py --port-forward --duration 300 --rate 10

# Generate some error traffic too
./generate_traffic.py --port-forward --duration 300 --rate 10 --error-rate 0.2
```

This will create real traffic patterns that Aegis Sentinel can monitor and detect.

## Running with Real Services

Once you have a Kubernetes cluster with services running, you can run Aegis Sentinel with:

### With Prometheus Metrics (recommended)

If you have Prometheus set up and running:

```bash
python3 run_k8s_demo.py --use-real-k8s --namespace your-namespace --enable-metrics --prometheus-url http://localhost:9090
```

### Without Metrics

If you don't have Prometheus set up or are experiencing issues with Helm:

```bash
python3 run_k8s_demo.py --use-real-k8s --namespace your-namespace
```

This will still show the service graph and relationships but without real-time metrics.

Replace `your-namespace` with the namespace where your services are deployed (e.g., `aegis-demo` if you used our setup script).

### Command Line Options

- `--use-real-k8s`: Connect to a real Kubernetes cluster instead of using the demo YAML
- `--kubeconfig`: Path to kubeconfig file (optional, uses default if not specified)
- `--namespace`: Kubernetes namespace to monitor (default: "default")
- `--enable-metrics`: Enable metrics collection from Prometheus
- `--prometheus-url`: URL of the Prometheus server (e.g., http://localhost:9090)
- `--enable-ml`: Enable ML capabilities for anomaly detection and remediation
- `--inject-anomaly`: Inject synthetic anomalies for demo purposes

## Monitoring Your Own Services

To monitor your own Kubernetes services:

1. Ensure your Kubernetes cluster is running and accessible via kubectl
2. Make sure Prometheus is set up and monitoring your services
3. Run Aegis Sentinel with the appropriate namespace and Prometheus URL

```bash
python3 run_k8s_demo.py --use-real-k8s --namespace your-namespace --enable-metrics --prometheus-url http://your-prometheus-url
```

## Troubleshooting

### Cannot connect to Kubernetes cluster

- Verify that kubectl is properly configured: `kubectl cluster-info`
- Check if the kubeconfig file is valid: `kubectl --kubeconfig=/path/to/kubeconfig get nodes`
- Ensure you have the necessary permissions to access the namespace

### Cannot connect to Prometheus

- Verify that Prometheus is running: `kubectl get pods -n monitoring`
- Check if Prometheus is accessible at the specified URL
- If using port forwarding, ensure the port forward is active

### Issues with Helm

If you're seeing errors like `ModuleNotFoundError: No module named 'glib'` when running Helm commands:

1. This indicates an issue with your Helm installation
2. You can run the demo without Prometheus metrics using:
   ```bash
   python3 run_k8s_demo.py --use-real-k8s --namespace aegis-demo
   ```
3. To fix Helm, try reinstalling it following the official instructions at: https://helm.sh/docs/intro/install/

### No services showing up in the graph

- Verify that services exist in the specified namespace: `kubectl get services -n your-namespace`
- Check if the services have the necessary labels and selectors
- Look at the logs for any parsing errors: `cat k8s_demo.log`

## Advanced Configuration

For advanced configuration options, refer to the documentation in the `docs/` directory.