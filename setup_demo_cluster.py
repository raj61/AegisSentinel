#!/usr/bin/env python3
"""
Setup Demo Kubernetes Cluster
============================

This script helps set up a demo Kubernetes cluster using minikube and deploy
sample microservices for demonstration purposes.
"""

import os
import sys
import time
import logging
import subprocess
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('demo_setup.log')
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Set up a demo Kubernetes cluster')
    parser.add_argument('--start-minikube', action='store_true',
                        help='Start a minikube cluster')
    parser.add_argument('--deploy-services', action='store_true',
                        help='Deploy sample microservices')
    parser.add_argument('--setup-prometheus', action='store_true',
                        help='Set up Prometheus monitoring')
    parser.add_argument('--all', action='store_true',
                        help='Perform all setup steps')
    return parser.parse_args()

def run_command(command, shell=False):
    """Run a shell command and log the output."""
    logger.info(f"Running command: {command}")
    try:
        if shell:
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        else:
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        stdout, stderr = process.communicate()
        
        if stdout:
            logger.info(stdout)
        if stderr:
            logger.warning(stderr)
        
        if process.returncode != 0:
            logger.error(f"Command failed with return code {process.returncode}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    dependencies = ['kubectl', 'minikube', 'helm']
    missing = []
    
    for dep in dependencies:
        try:
            subprocess.run([dep, '--help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            missing.append(dep)
    
    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        logger.info("Please install the missing dependencies and try again.")
        logger.info("  - kubectl: https://kubernetes.io/docs/tasks/tools/")
        logger.info("  - minikube: https://minikube.sigs.k8s.io/docs/start/")
        logger.info("  - helm: https://helm.sh/docs/intro/install/")
        return False
    
    return True

def start_minikube():
    """Start a minikube cluster."""
    logger.info("Starting minikube cluster...")
    
    # Check if minikube is already running
    result = subprocess.run(['minikube', 'status'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0 and 'Running' in result.stdout.decode():
        logger.info("Minikube is already running")
        return True
    
    # Start minikube with enough resources for the demo
    return run_command('minikube start --cpus=2 --memory=4096 --addons=ingress')

def deploy_microservices():
    """Deploy sample microservices to the cluster."""
    logger.info("Deploying sample microservices...")
    
    # Create a namespace for our demo
    run_command('kubectl create namespace aegis-demo')
    
    # Create a demo microservices YAML file
    demo_yaml = """
apiVersion: v1
kind: Namespace
metadata:
  name: aegis-demo
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
  namespace: aegis-demo
  labels:
    app: frontend
    tier: frontend
spec:
  replicas: 1
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
        tier: frontend
      annotations:
        aegis.sentinel/dependencies: "backend-service"
    spec:
      containers:
      - name: frontend
        image: nginx:alpine
        ports:
        - containerPort: 80
        env:
        - name: BACKEND_URL
          value: "http://backend-service:8080"
---
apiVersion: v1
kind: Service
metadata:
  name: frontend-service
  namespace: aegis-demo
spec:
  selector:
    app: frontend
  ports:
  - port: 80
    targetPort: 80
  type: ClusterIP
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
  namespace: aegis-demo
  labels:
    app: backend
    tier: application
spec:
  replicas: 2
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
        tier: application
      annotations:
        aegis.sentinel/dependencies: "database-service,cache-service"
    spec:
      containers:
      - name: backend
        image: busybox
        command: ["/bin/sh", "-c", "while true; do echo 'Backend service running'; sleep 10; done"]
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          value: "mongodb://database-service:27017/demo"
        - name: CACHE_URL
          value: "redis://cache-service:6379"
---
apiVersion: v1
kind: Service
metadata:
  name: backend-service
  namespace: aegis-demo
spec:
  selector:
    app: backend
  ports:
  - port: 8080
    targetPort: 8080
  type: ClusterIP
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: database
  namespace: aegis-demo
  labels:
    app: database
spec:
  replicas: 1
  selector:
    matchLabels:
      app: database
  template:
    metadata:
      labels:
        app: database
    spec:
      containers:
      - name: database
        image: mongo:4.4
        ports:
        - containerPort: 27017
---
apiVersion: v1
kind: Service
metadata:
  name: database-service
  namespace: aegis-demo
spec:
  selector:
    app: database
  ports:
  - port: 27017
    targetPort: 27017
  type: ClusterIP
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cache
  namespace: aegis-demo
  labels:
    app: cache
spec:
  replicas: 1
  selector:
    matchLabels:
      app: cache
  template:
    metadata:
      labels:
        app: cache
    spec:
      containers:
      - name: cache
        image: redis:6-alpine
        ports:
        - containerPort: 6379
---
apiVersion: v1
kind: Service
metadata:
  name: cache-service
  namespace: aegis-demo
spec:
  selector:
    app: cache
  ports:
  - port: 6379
    targetPort: 6379
  type: ClusterIP
"""
    
    # Write the YAML to a file
    with open('demo-microservices.yaml', 'w') as f:
        f.write(demo_yaml)
    
    # Apply the YAML
    success = run_command('kubectl apply -f demo-microservices.yaml')
    
    if success:
        logger.info("Waiting for deployments to be ready...")
        time.sleep(10)  # Give some time for the deployments to start
        
        # Check deployment status
        run_command('kubectl get deployments -n aegis-demo')
        run_command('kubectl get services -n aegis-demo')
        
        return True
    
    return False

def setup_prometheus():
    """Set up Prometheus monitoring."""
    logger.info("Setting up Prometheus monitoring...")
    
    # Check if helm is working properly
    try:
        result = subprocess.run(['helm', 'version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logger.error(f"Helm is not working properly: {result.stderr}")
            logger.info("Skipping Prometheus setup. You can run the demo without metrics.")
            logger.info("To fix Helm, try reinstalling it following the instructions at: https://helm.sh/docs/intro/install/")
            logger.info("\nAlternatively, you can run the demo without metrics:")
            logger.info("python3 run_k8s_demo.py --use-real-k8s --namespace aegis-demo")
            return False
    except Exception as e:
        logger.error(f"Error checking Helm: {e}")
        logger.info("Skipping Prometheus setup. You can run the demo without metrics.")
        return False
    
    # Add Prometheus Helm repository
    run_command('helm repo add prometheus-community https://prometheus-community.github.io/helm-charts')
    run_command('helm repo update')
    
    # Install Prometheus stack
    success = run_command('helm install prometheus prometheus-community/kube-prometheus-stack --namespace monitoring --create-namespace')
    
    if success:
        logger.info("Waiting for Prometheus to be ready...")
        time.sleep(30)  # Give some time for Prometheus to start
        
        # Port forward Prometheus server
        logger.info("Starting port forward for Prometheus server...")
        subprocess.Popen(['kubectl', 'port-forward', 'service/prometheus-kube-prometheus-prometheus', '9090:9090', '-n', 'monitoring'],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        logger.info("Prometheus is now available at http://localhost:9090")
        
        # Port forward Grafana
        logger.info("Starting port forward for Grafana...")
        subprocess.Popen(['kubectl', 'port-forward', 'service/prometheus-grafana', '3000:80', '-n', 'monitoring'],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        logger.info("Grafana is now available at http://localhost:3000")
        logger.info("Default Grafana credentials: admin / prom-operator")
        
        return True
    
    logger.warning("Failed to set up Prometheus. You can run the demo without metrics.")
    logger.info("To run the demo without metrics:")
    logger.info("python3 run_k8s_demo.py --use-real-k8s --namespace aegis-demo")
    return False

def main():
    """Main function."""
    args = parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Determine which steps to perform
    do_minikube = args.start_minikube or args.all
    do_services = args.deploy_services or args.all
    do_prometheus = args.setup_prometheus or args.all
    
    # If no specific steps are requested, show help
    if not (do_minikube or do_services or do_prometheus):
        logger.info("No setup steps specified. Use --help to see available options.")
        sys.exit(0)
    
    # Perform the requested steps
    if do_minikube:
        if not start_minikube():
            logger.error("Failed to start minikube")
            sys.exit(1)
    
    if do_services:
        if not deploy_microservices():
            logger.error("Failed to deploy microservices")
            sys.exit(1)
    
    if do_prometheus:
        if not setup_prometheus():
            logger.error("Failed to set up Prometheus")
            sys.exit(1)
    
    # Print instructions for running the demo
    logger.info("\n" + "="*80)
    logger.info("Demo setup complete!")
    logger.info("="*80)
    logger.info("\nTo run the demo with real Kubernetes services:")
    logger.info("python3 run_k8s_demo.py --use-real-k8s --namespace aegis-demo --enable-metrics --prometheus-url http://localhost:9090")
    logger.info("\nThis will connect to your Kubernetes cluster and monitor the deployed services.")
    logger.info("="*80)

if __name__ == '__main__':
    main()