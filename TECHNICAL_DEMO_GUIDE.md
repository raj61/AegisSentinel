# Aegis Technical Demo Guide

This guide provides detailed technical steps for demonstrating Aegis capabilities, with both UI and terminal-based approaches.

## Prerequisites

- Kubernetes cluster running (or minikube)
- Python 3.7+ installed
- Required Python packages (`pip install -r requirements.txt`)
- Prometheus running (optional, for metrics)

## Setup Commands

```bash
# Clone the repository if needed
git clone https://github.com/yourusername/CredHackathon.git
cd CredHackathon

# Install dependencies
pip install -r requirements.txt

# If using real Kubernetes, ensure you have access
kubectl get nodes

# If using minikube, start it
minikube start

# Deploy demo microservices (if not already deployed)
kubectl apply -f demo-microservices-with-metrics.yaml

# Setup Prometheus for metrics collection (optional)
python3 setup_prometheus.py
```

## Starting Aegis

### Basic Demo

```bash
# Start with demo YAML file
python3 run_k8s_demo.py
```

### Advanced Demo with Real Kubernetes

```bash
# Start with real Kubernetes and metrics
python3 run_k8s_demo.py --use-real-k8s --namespace aegis-demo --enable-metrics

# For ML-enhanced capabilities
python3 run_k8s_demo.py --use-real-k8s --namespace aegis-demo --enable-metrics --enable-ml
```

## API Endpoints Reference

> **Note about jq**: The commands below use `jq` for formatting JSON output. If you don't have jq installed, you can:
> - Install it: `apt-get install jq` (Ubuntu/Debian), `brew install jq` (macOS), or download from [stedolan.github.io/jq](https://stedolan.github.io/jq/)
> - Or simply remove the `| jq` part from the commands to see the raw JSON output

### Service Graph

```bash
# Get the entire service graph (with jq for formatting)
curl http://localhost:8080/api/graph | jq

# Without jq
curl http://localhost:8080/api/graph

# Get just the nodes
curl http://localhost:8080/api/graph/nodes | jq
# or without jq
curl http://localhost:8080/api/graph/nodes

# Get just the edges
curl http://localhost:8080/api/graph/edges | jq
# or without jq
curl http://localhost:8080/api/graph/edges
```

### Metrics

```bash
# Get metrics for a specific service
curl http://localhost:8080/api/metrics/service/aegis-demo/frontend | jq
# or without jq
curl http://localhost:8080/api/metrics/service/aegis-demo/frontend

# Get metrics for all services
curl http://localhost:8080/api/metrics/services | jq
# or without jq
curl http://localhost:8080/api/metrics/services

# Get historical metrics for a specific service and metric
curl http://localhost:8080/api/metrics/historical/aegis-demo/frontend/cpu_usage?step=5m | jq
curl http://localhost:8080/api/metrics/historical/aegis-demo/frontend/memory_usage?step=5m | jq
curl http://localhost:8080/api/metrics/historical/aegis-demo/frontend/latency_p95?step=5m | jq
curl http://localhost:8080/api/metrics/historical/aegis-demo/frontend/error_rate?step=5m | jq
# or without jq (example for cpu_usage)
curl http://localhost:8080/api/metrics/historical/aegis-demo/frontend/cpu_usage?step=5m
```

### Issues and Anomalies

```bash
# Get current issues
curl http://localhost:8080/api/issues | jq
# or without jq
curl http://localhost:8080/api/issues

# Inject an anomaly
curl -X POST http://localhost:8080/api/inject-anomaly \
  -H "Content-Type: application/json" \
  -d '{"type":"random"}' | jq
# or without jq
curl -X POST http://localhost:8080/api/inject-anomaly \
  -H "Content-Type: application/json" \
  -d '{"type":"random"}'

# Get health status
curl http://localhost:8080/api/health | jq
# or without jq
curl http://localhost:8080/api/health
```

## Demo Scenarios

### Scenario 1: Basic Service Visualization

1. Start Aegis with basic settings
   ```bash
   python3 run_k8s_demo.py
   ```

2. Open browser at http://localhost:8080

3. Explain the service graph:
   - Nodes represent services
   - Edges represent dependencies
   - Colors indicate service types and health status
   - Zoom and pan to navigate

4. Terminal alternative:
   ```bash
   curl http://localhost:8080/api/graph | jq
   echo "We have $(curl -s http://localhost:8080/api/graph/nodes | jq '.nodes | length') services with $(curl -s http://localhost:8080/api/graph/edges | jq '.edges | length') dependencies"
   ```

### Scenario 2: Metrics Monitoring

1. Start Aegis with metrics enabled
   ```bash
   python3 run_k8s_demo.py --enable-metrics
   ```

2. In the UI, click on a service node to view its metrics

3. Terminal alternative:
   ```bash
   # Show metrics for frontend service
   curl http://localhost:8080/api/metrics/service/aegis-demo/frontend | jq
   
   # Show historical CPU usage
   curl http://localhost:8080/api/metrics/historical/aegis-demo/frontend/cpu_usage?step=5m | jq
   ```

4. Explain the metrics:
   - CPU and memory usage
   - Response latency
   - Error rates
   - Historical trends

### Scenario 3: Anomaly Detection and Resolution

1. Start Aegis with metrics and ML enabled
   ```bash
   python3 run_k8s_demo.py --enable-metrics --enable-ml
   ```

2. In the UI, use the "Inject Anomaly" button or via terminal:
   ```bash
   curl -X POST http://localhost:8080/api/inject-anomaly \
     -H "Content-Type: application/json" \
     -d '{"type":"random"}' | jq
   ```

3. Observe the changes:
   - UI: Service turns red, metrics spike
   - Terminal: 
     ```bash
     curl http://localhost:8080/api/issues | jq
     ```

4. Wait for automatic resolution (60 seconds)
   - UI: Service returns to green
   - Terminal:
     ```bash
     echo "Waiting for resolution..."
     sleep 60
     curl http://localhost:8080/api/issues | jq
     ```

### Scenario 4: Infrastructure as Code Analysis

1. Run the Terraform analysis example with the required source argument
   ```bash
   # Analyze the AWS microservices Terraform example
   python3 examples/analyze_terraform.py --source examples/terraform/aws-microservices --detect --verbose
   
   # Or analyze with resolution suggestions
   python3 examples/analyze_terraform.py --source examples/terraform/aws-microservices --detect --resolve --verbose
   ```

2. Explain the output:
   - Detected resources
   - Potential issues
   - Security recommendations
   - Resolution suggestions (if using --resolve)

## Troubleshooting

### UI Issues

If the graph visualization isn't working properly:

1. Check browser console for errors
2. Refresh the page
3. Fall back to terminal commands
4. Restart the server with `--use-frontend` flag:
   ```bash
   python3 run_k8s_demo.py --use-frontend
   ```

### API Issues

If API endpoints aren't responding:

1. Check if the server is running
2. Verify the port is correct (default: 8080)
3. Check for errors in the terminal where the server is running
4. Restart the server

### Kubernetes Issues

If using real Kubernetes and encountering issues:

1. Verify cluster access:
   ```bash
   kubectl get nodes
   kubectl get pods -n aegis-demo
   ```

2. Check pod status:
   ```bash
   kubectl get pods -n aegis-demo
   kubectl describe pod [pod-name] -n aegis-demo
   ```

3. Fall back to demo mode:
   ```bash
   python3 run_k8s_demo.py
   ```

## Key Talking Points

When demonstrating Aegis, emphasize these key benefits:

1. **Real-time visibility**: "Aegis provides a complete view of your microservices architecture in real-time"

2. **Proactive monitoring**: "Instead of waiting for alerts, Aegis proactively detects anomalies before they impact users"

3. **Reduced MTTR**: "By automatically identifying root causes, Aegis reduces mean time to resolution"

4. **ML-enhanced analysis**: "Our machine learning models can detect subtle patterns that traditional monitoring misses"

5. **Infrastructure as Code integration**: "Aegis can analyze your IaC to prevent issues before deployment"

## Conclusion

End the demo by summarizing the key capabilities demonstrated and inviting questions. Be prepared to dive deeper into specific technical aspects based on audience interest.