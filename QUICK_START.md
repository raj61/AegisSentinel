# Aegis Sentinel: Quick Start Guide

This guide provides the minimal steps needed to run the Aegis Sentinel demo in different environments.

## Prerequisites

- Python 3.8+
- pip (Python package manager)

## Step 1: Install Dependencies

```bash
# Install required Python packages
pip install -r requirements.txt
```

## Step 2: Choose Your Demo Mode

### Option A: Full Demo with Kubernetes

If you have a Kubernetes environment (Minikube, Kind, Docker Desktop, etc.):

1. **Ensure Kubernetes is running:**
   ```bash
   kubectl get nodes
   ```

2. **Run the demo:**
   ```bash
   ./run_demo.py
   ```

   The script will:
   - Detect your Kubernetes environment
   - Deploy the example microservices
   - Start the Aegis Sentinel services
   - Open the web interface
   - Run through the demo sequence

### Option B: Simulation Mode (No Kubernetes Required)

If you don't have Kubernetes or prefer not to use it:

1. **Run the demo in simulation mode:**
   ```bash
   ./run_demo.py --simulation
   ```

   The script will:
   - Use pre-recorded data to simulate a Kubernetes environment
   - Start the Aegis Sentinel services in simulation mode
   - Open the web interface
   - Simulate anomalies and remediation actions

## Demo Controls

- **Duration:** By default, the demo runs for 10 minutes. You can change this with:
  ```bash
  ./run_demo.py --duration 300  # Run for 5 minutes
  ```

- **Demo Type:** You can focus on specific aspects of the demo:
  ```bash
  ./run_demo.py --demo-type anomaly-detection
  ```
  Available types: `full`, `anomaly-detection`, `root-cause`, `remediation`, `prediction`, `learning`

- **Auto-inject:** Enable automatic anomaly injection:
  ```bash
  ./run_demo.py --auto-inject
  ```

- **Web Interface:** Access the web interface at:
  ```
  http://localhost:8080
  ```

- **API Server:** Access the API at:
  ```
  http://localhost:8000
  ```

## Stopping the Demo

Press `Ctrl+C` in the terminal where you started the demo to stop all processes.

## Troubleshooting

If you encounter issues:

1. **Check the logs:**
   ```bash
   ./run_demo.py --verbose
   ```

2. **Verify Kubernetes connectivity:**
   ```bash
   kubectl get nodes
   kubectl get pods
   ```

3. **Clean up and try again:**
   ```bash
   kubectl delete namespace aegis-demo
   ./run_demo.py
   ```

For more detailed setup instructions and options, see `DEMO_SETUP.md`.