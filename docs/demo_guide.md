# Aegis Sentinel: Demo Guide

This guide provides step-by-step instructions for running a demonstration of the Aegis Sentinel system. The demo showcases the key capabilities of the system, including service graph visualization, anomaly detection, root cause analysis, and automated remediation.

## Prerequisites

Before running the demo, ensure you have the following:

- Python 3.8+ installed
- Required Python packages (install using `pip install -r requirements.txt`)
- At least 4GB of free RAM (8GB recommended for the ML components)
- Docker (optional, for containerized demo)

## Demo Setup Options

You have three options for running the demo:

### Option 1: Quick Demo with Simulated Data (5 minutes)

This option uses pre-generated data to demonstrate the system's capabilities without requiring any external services or infrastructure.

### Option 2: Interactive Demo with Sample Microservices (15 minutes)

This option deploys a set of sample microservices locally and demonstrates the system's capabilities with real-time monitoring and anomaly injection.

### Option 3: Full Demo with Kubernetes Cluster (30+ minutes)

This option deploys the system on a Kubernetes cluster and demonstrates its capabilities in a production-like environment.

## Option 1: Quick Demo with Simulated Data

### Step 1: Prepare the Environment

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/aegis-sentinel/aegis-sentinel.git
cd aegis-sentinel

# Install dependencies
pip install -r requirements.txt

# Navigate to the demo directory
cd examples
```

### Step 2: Run the Demo Script

```bash
python aegis_sentinel_demo.py --inject-anomaly
```

This script will:
1. Load a pre-defined service graph
2. Initialize the ML components
3. Inject a simulated anomaly
4. Demonstrate detection, root cause analysis, and remediation

### Step 3: Key Features to Highlight

During the demo, the script will output information to the console. Key points to highlight:

1. **Service Graph Visualization**: The script will generate a service graph visualization showing the relationships between services.

2. **Anomaly Detection**: Watch for the "Anomaly Detected" message, which will show:
   - The affected service
   - The type of anomaly (metric, log, etc.)
   - The severity of the anomaly

3. **Root Cause Analysis**: After detecting anomalies, the system will perform root cause analysis:
   - Multiple services may show anomalies
   - The system will identify the root cause service
   - It will explain why this service was identified as the root cause

4. **Automated Remediation**: Finally, the system will demonstrate automated remediation:
   - Selection of an appropriate remediation action
   - Execution of the remediation
   - Verification of the remediation's effectiveness

## Option 2: Interactive Demo with Sample Microservices

### Step 1: Start the Sample Microservices

```bash
# Navigate to the demo directory
cd examples/microservices-demo

# Start the microservices
docker-compose up -d
```

This will start a set of interconnected microservices:
- Frontend (React)
- API Gateway (Express)
- User Service (Node.js)
- Product Service (Python)
- Database (MongoDB)
- Cache (Redis)

### Step 2: Start the Web Interface

```bash
# Navigate back to the project root
cd ../..

# Start the web interface
python run_web_interface.py --enable-ml --inject-anomaly
```

This will:
1. Start the web interface on http://localhost:8080
2. Enable ML-based anomaly detection and remediation
3. Inject a simulated anomaly after 30 seconds

### Step 3: Explore the Web Interface

Open your browser and navigate to http://localhost:8080. The web interface provides:

1. **Service Graph View**: Shows all services and their dependencies
   - Healthy services are shown in green
   - Services with warnings are shown in yellow
   - Services with critical issues are shown in red

2. **Anomaly Dashboard**: Shows detected anomalies
   - Click on an anomaly to see details
   - View the root cause analysis results
   - See the remediation actions taken

3. **Real-time Updates**: The interface updates in real-time as anomalies are detected and remediated

### Step 4: Inject Additional Anomalies (Optional)

You can inject additional anomalies to demonstrate the system's capabilities:

```bash
# Inject a CPU spike in the Product Service
curl http://localhost:8080/api/demo/inject-anomaly?service=product-service&type=cpu-spike

# Inject a memory leak in the User Service
curl http://localhost:8080/api/demo/inject-anomaly?service=user-service&type=memory-leak

# Inject a slow query in the Database
curl http://localhost:8080/api/demo/inject-anomaly?service=database&type=slow-query
```

### Step 5: Clean Up

When you're done with the demo:

```bash
# Stop the web interface (Ctrl+C in the terminal)

# Stop the microservices
cd examples/microservices-demo
docker-compose down
```

## Option 3: Full Demo with Kubernetes Cluster

### Step 1: Set Up a Kubernetes Cluster

You can use Minikube for a local cluster:

```bash
# Start Minikube
minikube start --memory=8192 --cpus=4

# Enable necessary addons
minikube addons enable metrics-server
minikube addons enable dashboard
```

Or use a cloud-based Kubernetes service like GKE, EKS, or AKS.

### Step 2: Deploy the Demo Application

```bash
# Navigate to the Kubernetes demo directory
cd examples/kubernetes-demo

# Deploy the demo application
kubectl apply -f manifests/
```

This will deploy:
- A microservices application
- Prometheus for metrics collection
- Elasticsearch for log collection
- Aegis Sentinel components

### Step 3: Access the Web Interface

```bash
# Get the URL for the web interface
kubectl get svc aegis-sentinel-web -o jsonpath="{.status.loadBalancer.ingress[0].ip}"
```

Open your browser and navigate to http://<IP_ADDRESS>:8080

### Step 4: Inject Anomalies

You can inject anomalies using the Kubernetes demo tools:

```bash
# Inject a CPU spike in the product service
kubectl apply -f anomalies/cpu-spike.yaml

# Inject a memory leak in the user service
kubectl apply -f anomalies/memory-leak.yaml

# Inject a network latency issue
kubectl apply -f anomalies/network-latency.yaml
```

### Step 5: Clean Up

When you're done with the demo:

```bash
# Delete the demo application
kubectl delete -f manifests/

# Stop Minikube (if using)
minikube stop
```

## Demo Script: Highlighting Key Features

When presenting the demo, follow this script to highlight the key features:

### 1. Introduction (2 minutes)

- Explain the challenges of monitoring and troubleshooting modern distributed systems
- Introduce Aegis Sentinel as a solution that combines service graph analysis, ML-based anomaly detection, and automated remediation

### 2. Service Graph Visualization (3 minutes)

- Show the service graph visualization
- Explain how the system builds this graph from infrastructure code and runtime observations
- Point out the different types of services and dependencies

### 3. Anomaly Detection (5 minutes)

- Wait for the injected anomaly to be detected
- Explain the different types of anomalies the system can detect:
  - Metric anomalies (CPU, memory, latency, etc.)
  - Log anomalies (error patterns, unusual messages)
  - Topology anomalies (unusual service interactions)
- Highlight the ML-based detection approach:
  - Multi-model ensemble for robust detection
  - Contextual awareness to reduce false positives
  - Predictive capabilities to detect issues before they impact users

### 4. Root Cause Analysis (5 minutes)

- Show how the system identifies the root cause of the issue
- Explain the multi-dimensional correlation:
  - Temporal correlation (anomalies close in time)
  - Spatial correlation (anomalies in connected services)
  - Causal inference (probabilistic cause-effect relationships)
- Highlight the example of tracing a frontend issue to a database problem

### 5. Automated Remediation (5 minutes)

- Show how the system selects and applies remediation actions
- Explain the reinforcement learning approach:
  - Learning from past remediation outcomes
  - Balancing exploration and exploitation
  - Adapting to changing system behavior
- Highlight the safety mechanisms:
  - Confidence thresholds for automated actions
  - Approval workflows for critical actions
  - Rollback capabilities

### 6. Q&A (10 minutes)

Be prepared to answer questions about:
- Technical implementation details
- Integration with existing monitoring systems
- Scalability and performance
- Handling edge cases and limitations

## Troubleshooting

If you encounter issues during the demo:

### Web Interface Not Loading

- Check if the web server is running: `ps aux | grep run_web_interface.py`
- Check for errors in the logs: `cat web_interface.log`
- Ensure port 8080 is not in use by another application

### Anomalies Not Being Detected

- Check if the ML components are enabled: `--enable-ml` flag should be used
- Check for errors in the logs: `cat api_server.log`
- Try injecting a more severe anomaly

### Microservices Not Starting

- Check Docker status: `docker ps`
- Check Docker logs: `docker-compose logs`
- Ensure ports are not in use by other applications

## Conclusion

This demo guide provides multiple options for demonstrating Aegis Sentinel's capabilities, from a quick demo with simulated data to a full deployment on a Kubernetes cluster. Choose the option that best fits your time constraints and available resources.

Remember to focus on the key value propositions during the demo:
1. Early detection of issues through ML-based anomaly detection
2. Faster troubleshooting through automated root cause analysis
3. Reduced downtime through automated remediation
4. Improved system understanding through service graph visualization