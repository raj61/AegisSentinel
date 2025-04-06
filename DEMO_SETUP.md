# Aegis Sentinel: Mind-Blowing Demo Setup Guide

This guide provides step-by-step instructions for setting up a compelling demonstration of Aegis Sentinel's ML-powered autonomous reliability capabilities.

## Demo Overview

The demo will showcase:
1. Real-time anomaly detection using ML
2. Predictive analytics for forecasting issues
3. Intelligent root cause analysis
4. Autonomous remediation using reinforcement learning
5. Learning from manual fixes

## Prerequisites

- Kubernetes cluster (minikube or kind for local demo)
- Docker
- Python 3.8+
- Required Python packages (see requirements.txt)
- Prometheus and Grafana (optional, for enhanced metrics visualization)

## Step 1: Set Up Demo Environment

### Option A: Using Kubernetes (Recommended)

#### Install Kubernetes Tools

Before setting up the demo environment, you need to install the necessary Kubernetes tools. Choose one of the following options:

#### Option 1: Minikube (Local Single-Node Cluster)

```bash
# For macOS (using Homebrew)
brew install minikube

# For Linux
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# For Windows (using Chocolatey)
choco install minikube

# Verify installation
minikube version
```

#### Option 2: Kind (Kubernetes IN Docker)

```bash
# For macOS (using Homebrew)
brew install kind

# For Linux
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.17.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind

# For Windows (using Chocolatey)
choco install kind

# Verify installation
kind version
```

#### Option 3: Docker Desktop (macOS and Windows)

1. Download and install Docker Desktop from https://www.docker.com/products/docker-desktop
2. Open Docker Desktop
3. Go to Settings/Preferences
4. Enable Kubernetes
5. Click Apply & Restart

### Create a Kubernetes Demo Cluster

Choose one of the following options based on your installation:

#### Using Minikube

Before starting minikube, you need to ensure you have a compatible driver installed. Minikube supports several drivers:

**Option 1: Docker Driver (recommended for most users)**

```bash
# Install Docker Desktop
# Download from: https://www.docker.com/products/docker-desktop/

# Verify Docker is running
docker version

# Start minikube with Docker driver
minikube start --driver=docker --cpus 4 --memory 8192
```

**Option 2: Hyperkit Driver (macOS)**

```bash
# Install hyperkit driver
brew install hyperkit
brew install minikube

# Install the hyperkit driver for minikube
minikube config set driver hyperkit

# Start minikube
minikube start --cpus 4 --memory 8192
```

**Option 3: VirtualBox Driver**

```bash
# Install VirtualBox
# Download from: https://www.virtualbox.org/wiki/Downloads

# Start minikube with VirtualBox driver
minikube start --driver=virtualbox --cpus 4 --memory 8192
```

**Option 4: SSH Driver (for remote machines)**

```bash
# Start minikube with SSH driver
minikube start --driver=ssh --ssh-ip-address=<IP> --ssh-user=<user> --ssh-key=<path/to/key> --cpus 4 --memory 8192
```

After starting minikube, enable the necessary addons:

```bash
# Enable necessary addons
minikube addons enable metrics-server
minikube addons enable dashboard
```

**Troubleshooting Minikube**

If you encounter issues with minikube, try the following:

```bash
# Check minikube status
minikube status

# Delete and recreate minikube if needed
minikube delete
minikube start --driver=<your-driver> --cpus 4 --memory 8192

# View minikube logs
minikube logs

# Check available drivers
minikube config view
```

#### Using Kind

```bash
# Create a kind configuration file
cat > kind-config.yaml << EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  kubeadmConfigPatches:
  - |
    kind: InitConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        node-labels: "ingress-ready=true"
  extraPortMappings:
  - containerPort: 80
    hostPort: 80
    protocol: TCP
  - containerPort: 443
    hostPort: 443
    protocol: TCP
- role: worker
- role: worker
EOF

# Create the cluster
kind create cluster --config kind-config.yaml --name aegis-demo

# Install metrics server
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

#### Using Docker Desktop

If you've enabled Kubernetes in Docker Desktop, you should already have a running cluster.

```bash
# Verify the cluster is running
kubectl get nodes
```

### Deploy Microservices Demo Application

There are two options for deploying the demo microservices application:

#### Option 1: Use the included example manifests

```bash
# Use the example Kubernetes manifests included in the Aegis Sentinel repository
cd examples/kubernetes

# Deploy the microservices demo application
kubectl apply -f microservices-demo.yaml

# Verify all services are running
kubectl get pods
```

#### Option 2: Deploy a standard microservices demo

```bash
# Clone the popular microservices demo application
git clone https://github.com/microservices-demo/microservices-demo.git
cd microservices-demo/deploy/kubernetes

# Deploy the application
kubectl apply -f complete-demo.yaml

# Verify all services are running
kubectl get pods
```

The demo application includes:
- Frontend service
- Multiple backend services
- Database service
- Message queue
- Cache service

You can also use any existing Kubernetes application by pointing Aegis Sentinel to its manifest files.

### Option B: Simulation Mode (No Kubernetes Required)

If you're unable to set up a Kubernetes environment, you can run the demo in simulation mode. This mode uses pre-recorded data and simulated services to demonstrate the capabilities of Aegis Sentinel without requiring an actual Kubernetes cluster.

#### Setup Simulation Environment

```bash
# Create a simulation directory
mkdir -p simulation/data

# Download sample data files
curl -L -o simulation/data/kubernetes-manifest.yaml https://raw.githubusercontent.com/yourusername/aegis-sentinel/main/examples/kubernetes/microservices-demo.yaml
curl -L -o simulation/data/sample-logs.jsonl https://raw.githubusercontent.com/yourusername/aegis-sentinel/main/examples/data/sample-logs.jsonl
curl -L -o simulation/data/sample-metrics.jsonl https://raw.githubusercontent.com/yourusername/aegis-sentinel/main/examples/data/sample-metrics.jsonl
```

#### Create Simulation Configuration

Create a file named `simulation-config.json` with the following content:

```json
{
  "manifest_path": "simulation/data/kubernetes-manifest.yaml",
  "logs_path": "simulation/data/sample-logs.jsonl",
  "metrics_path": "simulation/data/sample-metrics.jsonl",
  "anomalies": [
    {
      "type": "memory_leak",
      "service": "backend",
      "start_time": 60,
      "duration": 120
    },
    {
      "type": "network_latency",
      "service": "database",
      "start_time": 240,
      "duration": 120
    },
    {
      "type": "cpu_spike",
      "service": "frontend",
      "start_time": 420,
      "duration": 120
    }
  ],
  "remediation_actions": [
    {
      "action": "restart_pod",
      "service": "backend",
      "time": 180
    },
    {
      "action": "scale_up",
      "service": "frontend",
      "time": 540,
      "parameters": {
        "replicas": 3
      }
    }
  ]
}
```

#### Run in Simulation Mode

When running the demo, use the `--simulation` flag to run in simulation mode:

```bash
./run_demo.py --simulation --config simulation-config.json
```

This will:
1. Load the sample Kubernetes manifest to build the service graph
2. Replay the sample logs and metrics data
3. Simulate anomalies according to the configuration
4. Demonstrate ML-based detection and remediation without requiring a real Kubernetes cluster

## Step 2: Install Aegis Sentinel

```bash
# Clone the Aegis Sentinel repository
git clone https://github.com/yourusername/aegis-sentinel.git
cd aegis-sentinel

# Install dependencies
pip install -r requirements.txt

# Start the API server
./run_api_server.py --source ../aegis-sentinel-demo-app/kubernetes/microservices-demo.yaml &

# Start the web interface
./run_web_interface.py --source ../aegis-sentinel-demo-app/kubernetes/microservices-demo.yaml &
```

## Step 3: Set Up Anomaly Injection

Create a script to inject anomalies into the demo environment:

```bash
# Create the anomaly injection script
cat > inject_anomalies.py << 'EOF'
#!/usr/bin/env python3
"""
Anomaly Injection Script for Aegis Sentinel Demo
"""

import argparse
import time
import random
import subprocess
import kubernetes
from kubernetes import client, config

def parse_args():
    parser = argparse.ArgumentParser(description='Inject anomalies into the demo environment')
    parser.add_argument('--type', choices=['memory', 'cpu', 'network', 'disk', 'error', 'random'], 
                        default='random', help='Type of anomaly to inject')
    parser.add_argument('--service', type=str, help='Target service')
    parser.add_argument('--duration', type=int, default=300, help='Duration of anomaly in seconds')
    parser.add_argument('--delay', type=int, default=60, help='Delay before injection in seconds')
    return parser.parse_args()

def inject_memory_leak(service, namespace='default'):
    """Inject a memory leak into a service"""
    print(f"Injecting memory leak into {service}")
    cmd = f"kubectl exec deploy/{service} -n {namespace} -- bash -c 'dd if=/dev/zero of=/tmp/memory-leak bs=1M count=1000 &'"
    subprocess.run(cmd, shell=True)

def inject_cpu_spike(service, namespace='default'):
    """Inject a CPU spike into a service"""
    print(f"Injecting CPU spike into {service}")
    cmd = f"kubectl exec deploy/{service} -n {namespace} -- bash -c 'for i in $(seq 1 $(nproc)); do yes > /dev/null & done'"
    subprocess.run(cmd, shell=True)

def inject_network_latency(service, namespace='default'):
    """Inject network latency into a service"""
    print(f"Injecting network latency into {service}")
    cmd = f"kubectl exec deploy/{service} -n {namespace} -- bash -c 'tc qdisc add dev eth0 root netem delay 200ms 50ms'"
    subprocess.run(cmd, shell=True)

def inject_disk_pressure(service, namespace='default'):
    """Inject disk pressure into a service"""
    print(f"Injecting disk pressure into {service}")
    cmd = f"kubectl exec deploy/{service} -n {namespace} -- bash -c 'dd if=/dev/zero of=/tmp/disk-pressure bs=1M count=5000 &'"
    subprocess.run(cmd, shell=True)

def inject_error_logs(service, namespace='default'):
    """Inject error logs into a service"""
    print(f"Injecting error logs into {service}")
    error_messages = [
        "ERROR: Connection refused",
        "ERROR: Database connection timeout",
        "ERROR: Out of memory",
        "ERROR: Disk I/O error",
        "ERROR: Network timeout",
        "CRITICAL: Service unavailable",
        "WARNING: High latency detected",
        "ERROR: Authentication failed",
        "ERROR: Permission denied",
        "ERROR: Resource limit exceeded"
    ]
    
    for _ in range(20):
        error = random.choice(error_messages)
        cmd = f"kubectl exec deploy/{service} -n {namespace} -- bash -c 'echo \"{error}\" >> /var/log/app.log'"
        subprocess.run(cmd, shell=True)
        time.sleep(1)

def cleanup_anomalies(service, namespace='default'):
    """Clean up injected anomalies"""
    print(f"Cleaning up anomalies in {service}")
    cmds = [
        f"kubectl exec deploy/{service} -n {namespace} -- bash -c 'pkill -f \"dd if=/dev/zero\"'",
        f"kubectl exec deploy/{service} -n {namespace} -- bash -c 'pkill -f \"yes\"'",
        f"kubectl exec deploy/{service} -n {namespace} -- bash -c 'tc qdisc del dev eth0 root 2>/dev/null || true'",
        f"kubectl exec deploy/{service} -n {namespace} -- bash -c 'rm -f /tmp/memory-leak /tmp/disk-pressure'"
    ]
    
    for cmd in cmds:
        try:
            subprocess.run(cmd, shell=True)
        except Exception as e:
            print(f"Error during cleanup: {e}")

def main():
    args = parse_args()
    
    # Load Kubernetes configuration
    try:
        config.load_kube_config()
    except:
        config.load_incluster_config()
    
    # Get available services if none specified
    if not args.service:
        v1 = client.CoreV1Api()
        services = v1.list_service_for_all_namespaces().items
        deployments = client.AppsV1Api().list_deployment_for_all_namespaces().items
        deployment_names = [d.metadata.name for d in deployments]
        
        # Filter to get only services with deployments
        available_services = [s.metadata.name for s in services if s.metadata.name in deployment_names]
        
        if not available_services:
            print("No suitable services found for anomaly injection")
            return
        
        args.service = random.choice(available_services)
        print(f"Randomly selected service: {args.service}")
    
    # Wait before injection
    if args.delay > 0:
        print(f"Waiting {args.delay} seconds before injection...")
        time.sleep(args.delay)
    
    # Inject the anomaly
    anomaly_type = args.type
    if anomaly_type == 'random':
        anomaly_type = random.choice(['memory', 'cpu', 'network', 'disk', 'error'])
    
    if anomaly_type == 'memory':
        inject_memory_leak(args.service)
    elif anomaly_type == 'cpu':
        inject_cpu_spike(args.service)
    elif anomaly_type == 'network':
        inject_network_latency(args.service)
    elif anomaly_type == 'disk':
        inject_disk_pressure(args.service)
    elif anomaly_type == 'error':
        inject_error_logs(args.service)
    
    # Wait for the specified duration
    print(f"Anomaly injected. Waiting {args.duration} seconds...")
    time.sleep(args.duration)
    
    # Clean up
    cleanup_anomalies(args.service)
    print("Anomaly injection completed and cleaned up")

if __name__ == "__main__":
    main()
EOF

# Make the script executable
chmod +x inject_anomalies.py
```

## Step 4: Set Up the ML Demo Script

```bash
# Run the ML demo script
./examples/ml_demo.py --source ../aegis-sentinel-demo-app/kubernetes/microservices-demo.yaml --simulate
```

## Step 5: Prepare the Demo Script

Create a demo script that guides you through the demonstration:

### Demo Flow

1. **Introduction (2 minutes)**
   - Explain the challenges of modern reliability engineering
   - Introduce Aegis Sentinel as an AI-powered solution

2. **Service Graph Visualization (3 minutes)**
   - Show the service graph of the demo application
   - Explain the dependencies and potential failure points
   - Highlight how Aegis Sentinel automatically discovered the architecture

3. **Anomaly Detection Demo (5 minutes)**
   - Inject a memory leak into a backend service
   ```bash
   ./inject_anomalies.py --type memory --service backend-service
   ```
   - Watch as Aegis Sentinel detects the anomaly using ML
   - Show the anomaly scores and confidence levels
   - Compare detection time with traditional threshold-based monitoring

4. **Root Cause Analysis Demo (5 minutes)**
   - Inject a cascading failure scenario
   ```bash
   ./inject_anomalies.py --type network --service database-service
   ```
   - Watch as multiple services show symptoms
   - Demonstrate how Aegis Sentinel traces back to the root cause
   - Show the propagation path of the failure

5. **Autonomous Remediation Demo (5 minutes)**
   - Let Aegis Sentinel automatically remediate the issue
   - Show the reinforcement learning model selecting the optimal action
   - Demonstrate the system returning to normal state
   - Explain how the system learns from each remediation action

6. **Predictive Analytics Demo (5 minutes)**
   - Show how Aegis Sentinel predicts potential issues
   - Demonstrate forecasting of resource usage
   - Show how preventive actions can be taken before issues occur

7. **Learning from Manual Fixes (3 minutes)**
   - Demonstrate a manual fix for a complex issue
   - Show how Aegis Sentinel captures the fix signature
   - Demonstrate how it applies the learned fix to a similar issue

8. **Q&A (5-10 minutes)**

## Step 6: Prepare Visuals and Metrics

Create compelling visuals to show the impact:

1. **Before/After Metrics**
   - MTTR (Mean Time To Resolution) comparison
   - Issue detection time comparison
   - SLA/uptime improvement

2. **ROI Calculation**
   - Cost of downtime prevented
   - Engineering time saved
   - Customer satisfaction improvement

## Step 7: Demo Day Preparation

1. **Technical Setup**
   - Ensure all components are running smoothly
   - Have backup plans for any technical issues
   - Test the demo flow multiple times

2. **Presentation Setup**
   - Prepare slides for introduction and conclusion
   - Create handouts with key metrics and benefits
   - Set up screen recording for later sharing

3. **Rehearse**
   - Practice the entire demo flow
   - Time each section
   - Prepare answers for potential questions

## Demo Day Checklist

- [ ] All services running in Kubernetes cluster
- [ ] Aegis Sentinel API and web interface running
- [ ] Anomaly injection script tested
- [ ] ML demo script tested
- [ ] Presentation materials ready
- [ ] Backup plan for technical issues
- [ ] Demo environment isolated from external factors

## Troubleshooting

### Common Issues

1. **Kubernetes Connectivity Issues**
   ```bash
   # Check Kubernetes connectivity
   kubectl get nodes
   kubectl get pods
   ```

2. **Service Discovery Issues**
   ```bash
   # Check service endpoints
   kubectl get endpoints
   ```

3. **Resource Constraints**
   ```bash
   # Check resource usage
   kubectl top nodes
   kubectl top pods
   ```

4. **Anomaly Injection Failures**
   ```bash
   # Check pod access
   kubectl exec -it deploy/backend-service -- bash
   ```

## Conclusion

This demo setup will showcase the power of Aegis Sentinel's ML and AI capabilities for autonomous reliability engineering. The combination of real-time anomaly detection, predictive analytics, intelligent root cause analysis, and autonomous remediation creates a compelling demonstration of the future of SRE.

Remember, the key to a mind-blowing demo is showing how the system can:
1. Detect issues before humans would notice
2. Find the root cause faster than any human could
3. Fix problems automatically without human intervention
4. Learn and improve with each incident
5. Predict and prevent future issues

Good luck with your demo!