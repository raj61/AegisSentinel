#!/usr/bin/env python3
"""
Aegis Sentinel Demo Runner

This script orchestrates a complete demo of Aegis Sentinel's ML and AI capabilities.
It sets up the environment, injects anomalies, and showcases the autonomous remediation.
"""

import os
import sys
import time
import argparse
import subprocess
import signal
import webbrowser
import random
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("demo-runner")

# Global variables
PROCESSES = []
DEMO_DIR = Path(__file__).parent.absolute()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Aegis Sentinel Demo')
    parser.add_argument(
        '--k8s-manifest',
        type=str,
        default='examples/kubernetes/microservices-demo.yaml',
        help='Path to Kubernetes manifest file (can be local or from microservices-demo)'
    )
    parser.add_argument(
        '--demo-type',
        type=str,
        choices=['full', 'anomaly-detection', 'root-cause', 'remediation', 'prediction', 'learning'],
        default='full',
        help='Type of demo to run'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=600,
        help='Duration of the demo in seconds'
    )
    parser.add_argument(
        '--auto-inject',
        action='store_true',
        help='Automatically inject anomalies during the demo'
    )
    parser.add_argument(
        '--web-port',
        type=int,
        default=8080,
        help='Port for the web interface'
    )
    parser.add_argument(
        '--api-port',
        type=int,
        default=8000,
        help='Port for the API server'
    )
    parser.add_argument(
        '--simulation',
        action='store_true',
        help='Run in simulation mode (no Kubernetes required)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='simulation-config.json',
        help='Path to simulation configuration file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    return parser.parse_args()

def run_command(command, background=False):
    """Run a shell command."""
    logger.info(f"Running command: {command}")
    
    if background:
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        PROCESSES.append(process)
        return process
    else:
        result = subprocess.run(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            logger.error(f"Command failed: {result.stderr}")
        return result

def detect_kubernetes_environment():
    """Detect which Kubernetes environment is available."""
    # Check minikube
    minikube = run_command("which minikube")
    if minikube.returncode == 0:
        # Check if minikube is running
        status = run_command("minikube status")
        if status.returncode == 0 and "Running" in status.stdout:
            return "minikube"
    
    # Check kind
    kind = run_command("which kind")
    if kind.returncode == 0:
        # Check if kind has any clusters
        clusters = run_command("kind get clusters")
        if clusters.returncode == 0 and clusters.stdout.strip():
            return "kind"
    
    # Check Docker Desktop (or any other Kubernetes)
    kubectl = run_command("which kubectl")
    if kubectl.returncode == 0:
        # Check if Kubernetes is running
        nodes = run_command("kubectl get nodes")
        if nodes.returncode == 0:
            return "generic"
    
    return None

def check_prerequisites():
    """Check if all prerequisites are installed."""
    logger.info("Checking prerequisites...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        logger.error("Python 3.8 or higher is required")
        return False
    
    # Check kubectl
    kubectl = run_command("which kubectl")
    if kubectl.returncode != 0:
        logger.error("kubectl is not installed. Please install kubectl first.")
        logger.info("Installation instructions: https://kubernetes.io/docs/tasks/tools/")
        return False
    
    # Detect Kubernetes environment
    k8s_env = detect_kubernetes_environment()
    if not k8s_env:
        logger.error("No running Kubernetes environment detected.")
        logger.info("Please set up a Kubernetes environment using one of the following:")
        logger.info("1. Minikube: https://minikube.sigs.k8s.io/docs/start/")
        logger.info("2. Kind: https://kind.sigs.k8s.io/docs/user/quick-start/")
        logger.info("3. Docker Desktop: Enable Kubernetes in settings")
        return False
    
    logger.info(f"Detected Kubernetes environment: {k8s_env}")
    
    # Store the environment type for later use
    global KUBERNETES_ENV
    KUBERNETES_ENV = k8s_env
    
    logger.info("All prerequisites are satisfied")
    return True

# Global variable to store the detected Kubernetes environment
KUBERNETES_ENV = None

def setup_environment(args):
    """Set up the demo environment."""
    logger.info("Setting up demo environment...")
    
    # Check if the manifest file exists
    manifest_path = Path(args.k8s_manifest)
    if not manifest_path.exists():
        logger.error(f"Manifest file not found: {args.k8s_manifest}")
        return False
    
    # Setup based on the detected Kubernetes environment
    if KUBERNETES_ENV == "minikube":
        logger.info("Setting up minikube environment...")
        
        # Ensure minikube is running with enough resources
        status = run_command("minikube status")
        if status.returncode != 0 or "Running" not in status.stdout:
            logger.info("Starting minikube...")
            run_command("minikube start --cpus 4 --memory 8192")
        
        # Enable necessary addons
        run_command("minikube addons enable metrics-server")
        run_command("minikube addons enable dashboard")
    
    elif KUBERNETES_ENV == "kind":
        logger.info("Setting up kind environment...")
        
        # Check if our demo cluster exists
        clusters = run_command("kind get clusters")
        if "aegis-demo" not in clusters.stdout:
            logger.info("Creating kind cluster 'aegis-demo'...")
            
            # Create a kind configuration file
            config_path = Path(DEMO_DIR) / "kind-config.yaml"
            if not config_path.exists():
                with open(config_path, 'w') as f:
                    f.write("""kind: Cluster
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
""")
            
            # Create the cluster
            run_command(f"kind create cluster --config {config_path} --name aegis-demo")
            
            # Set kubectl context to the new cluster
            run_command("kubectl config use-context kind-aegis-demo")
            
            # Install metrics server
            run_command("kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml")
    
    elif KUBERNETES_ENV == "generic":
        logger.info("Using existing Kubernetes environment...")
        # No special setup needed for Docker Desktop or other Kubernetes environments
    
    # Deploy the target application to be monitored
    logger.info(f"Deploying target application from {args.k8s_manifest}...")
    
    # Check if the manifest path is from the microservices-demo repository
    if "microservices-demo" in args.k8s_manifest and not Path(args.k8s_manifest).exists():
        logger.info("Manifest appears to be from microservices-demo repository. Checking if it's installed...")
        
        # Check if microservices-demo repository exists
        if not Path("microservices-demo").exists():
            logger.info("Cloning microservices-demo repository...")
            clone_result = run_command("git clone https://github.com/microservices-demo/microservices-demo.git")
            if clone_result.returncode != 0:
                logger.error("Failed to clone microservices-demo repository")
                return False
        
        # Use the complete-demo.yaml from the repository
        manifest_path = "microservices-demo/deploy/kubernetes/complete-demo.yaml"
        logger.info(f"Using manifest from cloned repository: {manifest_path}")
        
        deploy_result = run_command(f"kubectl apply -f {manifest_path}")
    else:
        # Use the specified manifest
        deploy_result = run_command(f"kubectl apply -f {args.k8s_manifest}")
    
    if deploy_result.returncode != 0:
        logger.error("Failed to deploy demo application")
        return False
    
    # Wait for all pods to be ready
    logger.info("Waiting for all pods to be ready...")
    time.sleep(10)  # Give some time for the deployment to start
    
    # Specify the namespace for the wait command
    wait_result = run_command("kubectl wait --for=condition=Ready pods --all -n aegis-demo --timeout=300s")
    if wait_result.returncode != 0:
        logger.warning("Not all pods are ready, but continuing anyway")
    
    logger.info("Demo environment is set up")
    return True

def start_aegis_sentinel(args):
    """Start the Aegis Sentinel API and web interface to monitor the deployed application."""
    logger.info("Starting Aegis Sentinel to monitor the target application...")
    
    # The manifest file is used as a source for Aegis Sentinel to understand the application structure
    # It's not deploying the application again, just analyzing it
    
    # Start the API server with the manifest as the source for analysis
    api_cmd = f"./run_api_server.py --source {args.k8s_manifest} --port {args.api_port}"
    api_process = run_command(api_cmd, background=True)
    
    # Start the web interface with the same manifest for visualization
    web_cmd = f"./run_web_interface.py --source {args.k8s_manifest} --port {args.web_port}"
    web_process = run_command(web_cmd, background=True)
    
    # Give some time for the servers to start
    logger.info("Waiting for servers to start...")
    time.sleep(5)
    
    # Check if the servers are running
    api_check = run_command(f"curl -s http://localhost:{args.api_port}/api/health")
    web_check = run_command(f"curl -s http://localhost:{args.web_port}")
    
    if api_check.returncode != 0 or web_check.returncode != 0:
        logger.warning("Servers may not be running properly, but continuing anyway")
    
    # Open the web interface in a browser
    webbrowser.open(f"http://localhost:{args.web_port}")
    
    logger.info("Aegis Sentinel is running")
    return True

def run_ml_demo(args):
    """Run the ML demo script."""
    logger.info("Running ML demo...")
    
    ml_cmd = f"./examples/ml_demo.py --source {args.k8s_manifest} --simulate"
    ml_process = run_command(ml_cmd, background=True)
    
    logger.info("ML demo is running")
    return True

def inject_anomaly(anomaly_type=None, service=None, duration=60):
    """Inject an anomaly into the demo environment."""
    if not anomaly_type:
        anomaly_type = random.choice(['memory', 'cpu', 'network', 'disk', 'error'])
    
    logger.info(f"Injecting {anomaly_type} anomaly...")
    
    cmd = f"./inject_anomalies.py --type {anomaly_type}"
    if service:
        cmd += f" --service {service}"
    cmd += f" --duration {duration}"
    
    anomaly_process = run_command(cmd, background=True)
    return anomaly_process

def run_demo_sequence(args):
    """Run the demo sequence based on the selected demo type."""
    logger.info(f"Running {args.demo_type} demo sequence...")
    
    if args.demo_type == 'full' or args.demo_type == 'anomaly-detection':
        logger.info("=== Anomaly Detection Demo ===")
        logger.info("Injecting memory leak anomaly...")
        inject_anomaly('memory', duration=120)
        time.sleep(30)
        logger.info("ML-based anomaly detection should now identify the memory leak")
        time.sleep(90)
    
    if args.demo_type == 'full' or args.demo_type == 'root-cause':
        logger.info("=== Root Cause Analysis Demo ===")
        logger.info("Injecting network latency anomaly to database service...")
        inject_anomaly('network', service='database', duration=120)
        time.sleep(30)
        logger.info("Root cause analysis should now trace the issue to the database service")
        time.sleep(90)
    
    if args.demo_type == 'full' or args.demo_type == 'remediation':
        logger.info("=== Autonomous Remediation Demo ===")
        logger.info("Injecting CPU spike anomaly...")
        inject_anomaly('cpu', duration=120)
        time.sleep(30)
        logger.info("Reinforcement learning should now select and execute a remediation action")
        time.sleep(90)
    
    if args.demo_type == 'full' or args.demo_type == 'prediction':
        logger.info("=== Predictive Analytics Demo ===")
        logger.info("Simulating gradual resource usage increase...")
        # This would be a custom anomaly that gradually increases resource usage
        time.sleep(30)
        logger.info("Predictive analytics should now forecast potential resource exhaustion")
        time.sleep(60)
    
    if args.demo_type == 'full' or args.demo_type == 'learning':
        logger.info("=== Learning from Manual Fixes Demo ===")
        logger.info("Simulating a manual fix...")
        # This would involve executing a manual remediation action
        time.sleep(30)
        logger.info("Remediation learner should now capture the fix signature")
        time.sleep(60)
    
    logger.info("Demo sequence completed")

def auto_inject_anomalies(args):
    """Automatically inject anomalies at random intervals."""
    logger.info("Starting automatic anomaly injection...")
    
    end_time = time.time() + args.duration
    while time.time() < end_time:
        # Wait a random amount of time between anomalies
        wait_time = random.randint(60, 180)
        time.sleep(wait_time)
        
        # Inject a random anomaly
        anomaly_type = random.choice(['memory', 'cpu', 'network', 'disk', 'error'])
        duration = random.randint(30, 90)
        inject_anomaly(anomaly_type, duration=duration)

def cleanup():
    """Clean up the demo environment."""
    logger.info("Cleaning up...")
    
    # Terminate all background processes
    for process in PROCESSES:
        try:
            process.terminate()
        except:
            pass
    
    # Give processes time to terminate
    time.sleep(2)
    
    # Kill any remaining processes
    for process in PROCESSES:
        try:
            if process.poll() is None:
                process.kill()
        except:
            pass
    
    logger.info("Cleanup completed")

def signal_handler(sig, frame):
    """Handle Ctrl+C."""
    logger.info("Received interrupt signal, shutting down...")
    cleanup()
    sys.exit(0)

def run_simulation_mode(args):
    """Run the demo in simulation mode (no Kubernetes required)."""
    logger.info("Running in simulation mode (no Kubernetes required)")
    
    # Check if the simulation config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Simulation config file not found: {args.config}")
        return False
    
    # Load the simulation configuration
    try:
        with open(config_path, 'r') as f:
            import json
            config = json.load(f)
        logger.info(f"Loaded simulation configuration from {args.config}")
    except Exception as e:
        logger.error(f"Error loading simulation configuration: {e}")
        return False
    
    # Check if the manifest file exists
    manifest_path = Path(config.get('manifest_path', ''))
    if not manifest_path.exists():
        logger.error(f"Manifest file not found: {manifest_path}")
        return False
    
    # Start the API server with the simulation flag
    api_cmd = f"./run_api_server.py --source {manifest_path} --port {args.api_port} --simulation"
    api_process = run_command(api_cmd, background=True)
    
    # Start the web interface with the simulation flag
    web_cmd = f"./run_web_interface.py --source {manifest_path} --port {args.web_port} --simulation"
    web_process = run_command(web_cmd, background=True)
    
    # Give some time for the servers to start
    logger.info("Waiting for servers to start...")
    time.sleep(5)
    
    # Open the web interface in a browser
    webbrowser.open(f"http://localhost:{args.web_port}")
    
    # Simulate anomalies according to the configuration
    logger.info("Simulating anomalies according to the configuration...")
    
    # Get the anomalies from the configuration
    anomalies = config.get('anomalies', [])
    
    # Create a thread for each anomaly
    for anomaly in anomalies:
        anomaly_type = anomaly.get('type', 'memory_leak')
        service = anomaly.get('service', 'backend')
        start_time = anomaly.get('start_time', 60)
        duration = anomaly.get('duration', 120)
        
        # Schedule the anomaly
        logger.info(f"Scheduling {anomaly_type} anomaly for {service} in {start_time} seconds")
        
        def inject_scheduled_anomaly(anomaly_type, service, duration):
            logger.info(f"Simulating {anomaly_type} anomaly for {service}")
            # In simulation mode, we don't actually inject anomalies,
            # but we log them to show what would happen
            time.sleep(duration)
            logger.info(f"Finished simulating {anomaly_type} anomaly for {service}")
        
        # Start a thread to inject the anomaly after the specified delay
        import threading
        thread = threading.Timer(
            start_time,
            inject_scheduled_anomaly,
            args=[anomaly_type, service, duration]
        )
        thread.daemon = True
        thread.start()
    
    # Simulate remediation actions according to the configuration
    remediation_actions = config.get('remediation_actions', [])
    
    # Create a thread for each remediation action
    for action in remediation_actions:
        action_type = action.get('action', 'restart_pod')
        service = action.get('service', 'backend')
        action_time = action.get('time', 180)
        parameters = action.get('parameters', {})
        
        # Schedule the remediation action
        logger.info(f"Scheduling {action_type} remediation for {service} in {action_time} seconds")
        
        def execute_scheduled_remediation(action_type, service, parameters):
            logger.info(f"Simulating {action_type} remediation for {service}")
            # In simulation mode, we don't actually execute remediation actions,
            # but we log them to show what would happen
            logger.info(f"Finished simulating {action_type} remediation for {service}")
        
        # Start a thread to execute the remediation action after the specified delay
        import threading
        thread = threading.Timer(
            action_time,
            execute_scheduled_remediation,
            args=[action_type, service, parameters]
        )
        thread.daemon = True
        thread.start()
    
    # Wait for the specified duration
    logger.info(f"Simulation is running. Press Ctrl+C to stop after {args.duration} seconds...")
    time.sleep(args.duration)
    
    logger.info("Simulation completed successfully")
    return True

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Check if we're running in simulation mode
        if args.simulation:
            if not run_simulation_mode(args):
                return 1
        else:
            # Check prerequisites
            if not check_prerequisites():
                return 1
            
            # Setup environment
            if not setup_environment(args):
                return 1
            
            # Start Aegis Sentinel
            if not start_aegis_sentinel(args):
                return 1
            
            # Run ML demo
            if not run_ml_demo(args):
                return 1
            
            # Run demo sequence
            run_demo_sequence(args)
            
            # Auto-inject anomalies if requested
            if args.auto_inject:
                auto_inject_anomalies(args)
            else:
                # Wait for the specified duration
                logger.info(f"Demo is running. Press Ctrl+C to stop after {args.duration} seconds...")
                time.sleep(args.duration)
        
        logger.info("Demo completed successfully")
        return 0
    
    except Exception as e:
        logger.exception(f"Error in demo: {e}")
        return 1
    
    finally:
        cleanup()

if __name__ == "__main__":
    sys.exit(main())