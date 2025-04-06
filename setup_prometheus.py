#!/usr/bin/env python3
"""
Setup Prometheus Monitoring
==========================

This script sets up Prometheus monitoring for Kubernetes services
without requiring Helm.
"""

import os
import sys
import time
import logging
import argparse
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('prometheus_setup.log')
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Set up Prometheus monitoring without Helm')
    parser.add_argument('--namespace', type=str, default='monitoring',
                        help='Kubernetes namespace for Prometheus')
    parser.add_argument('--port-forward', action='store_true',
                        help='Set up port forwarding for Prometheus')
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

def create_prometheus_yaml():
    """Create Prometheus YAML configuration."""
    prometheus_yaml = """
apiVersion: v1
kind: Namespace
metadata:
  name: monitoring
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    scrape_configs:
      - job_name: 'kubernetes-apiservers'
        kubernetes_sd_configs:
        - role: endpoints
        scheme: https
        tls_config:
          ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
        relabel_configs:
        - source_labels: [__meta_kubernetes_namespace, __meta_kubernetes_service_name, __meta_kubernetes_endpoint_port_name]
          action: keep
          regex: default;kubernetes;https
      
      - job_name: 'kubernetes-nodes'
        scheme: https
        tls_config:
          ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
        kubernetes_sd_configs:
        - role: node
        relabel_configs:
        - action: labelmap
          regex: __meta_kubernetes_node_label_(.+)
      
      - job_name: 'kubernetes-pods'
        kubernetes_sd_configs:
        - role: pod
        relabel_configs:
        - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
          action: keep
          regex: true
        - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
          action: replace
          target_label: __metrics_path__
          regex: (.+)
        - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
          action: replace
          regex: ([^:]+)(?::\\d+)?;(\\d+)
          replacement: $1:$2
          target_label: __address__
        - action: labelmap
          regex: __meta_kubernetes_pod_label_(.+)
        - source_labels: [__meta_kubernetes_namespace]
          action: replace
          target_label: kubernetes_namespace
        - source_labels: [__meta_kubernetes_pod_name]
          action: replace
          target_label: kubernetes_pod_name
      
      - job_name: 'kubernetes-services'
        kubernetes_sd_configs:
        - role: service
        metrics_path: /metrics
        relabel_configs:
        - source_labels: [__meta_kubernetes_service_annotation_prometheus_io_scrape]
          action: keep
          regex: true
        - source_labels: [__meta_kubernetes_service_annotation_prometheus_io_path]
          action: replace
          target_label: __metrics_path__
          regex: (.+)
        - source_labels: [__address__, __meta_kubernetes_service_annotation_prometheus_io_port]
          action: replace
          regex: ([^:]+)(?::\\d+)?;(\\d+)
          replacement: $1:$2
          target_label: __address__
        - action: labelmap
          regex: __meta_kubernetes_service_label_(.+)
        - source_labels: [__meta_kubernetes_namespace]
          action: replace
          target_label: kubernetes_namespace
        - source_labels: [__meta_kubernetes_service_name]
          action: replace
          target_label: kubernetes_name
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: prometheus
  namespace: monitoring
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: prometheus
rules:
- apiGroups: [""]
  resources:
  - nodes
  - nodes/proxy
  - services
  - endpoints
  - pods
  verbs: ["get", "list", "watch"]
- apiGroups:
  - extensions
  resources:
  - ingresses
  verbs: ["get", "list", "watch"]
- nonResourceURLs: ["/metrics"]
  verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: prometheus
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: prometheus
subjects:
- kind: ServiceAccount
  name: prometheus
  namespace: monitoring
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: monitoring
  labels:
    app: prometheus
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      serviceAccountName: prometheus
      containers:
      - name: prometheus
        image: prom/prometheus:v2.37.0
        args:
        - "--config.file=/etc/prometheus/prometheus.yml"
        - "--storage.tsdb.path=/prometheus"
        - "--web.console.libraries=/usr/share/prometheus/console_libraries"
        - "--web.console.templates=/usr/share/prometheus/consoles"
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: config-volume
          mountPath: /etc/prometheus
        - name: storage-volume
          mountPath: /prometheus
      volumes:
      - name: config-volume
        configMap:
          name: prometheus-config
      - name: storage-volume
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus
  namespace: monitoring
  annotations:
    prometheus.io/scrape: 'true'
    prometheus.io/port: '9090'
spec:
  selector:
    app: prometheus
  ports:
  - port: 9090
    targetPort: 9090
  type: ClusterIP
"""
    
    # Write the YAML to a file
    with open('prometheus.yaml', 'w') as f:
        f.write(prometheus_yaml)
    
    logger.info("Created prometheus.yaml")
    return True

def setup_prometheus(namespace):
    """Set up Prometheus in the specified namespace."""
    # Create the YAML file
    if not create_prometheus_yaml():
        return False
    
    # Apply the YAML
    success = run_command(f"kubectl apply -f prometheus.yaml")
    
    if success:
        logger.info(f"Prometheus deployed to namespace {namespace}")
        logger.info("Waiting for Prometheus to be ready...")
        time.sleep(10)  # Give some time for the deployment to start
        
        # Check deployment status
        run_command(f"kubectl get deployments -n {namespace}")
        run_command(f"kubectl get services -n {namespace}")
        
        return True
    
    return False

def setup_port_forwarding(namespace):
    """Set up port forwarding for Prometheus."""
    logger.info("Setting up port forwarding for Prometheus...")
    
    # Start port forwarding
    process = subprocess.Popen(
        ['kubectl', 'port-forward', 'service/prometheus', '9090:9090', '-n', namespace],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    logger.info("Prometheus is now available at http://localhost:9090")
    logger.info("Press Ctrl+C to stop port forwarding")
    
    return process

def main():
    """Main function."""
    args = parse_args()
    
    # Set up Prometheus
    if not setup_prometheus(args.namespace):
        logger.error("Failed to set up Prometheus")
        sys.exit(1)
    
    # Set up port forwarding if requested
    if args.port_forward:
        port_forward_process = setup_port_forwarding(args.namespace)
        
        try:
            # Keep the script running to maintain port forwarding
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Port forwarding interrupted by user")
            port_forward_process.terminate()
    
    # Print instructions
    logger.info("\n" + "="*80)
    logger.info("Prometheus setup complete!")
    logger.info("="*80)
    logger.info("\nTo access Prometheus:")
    logger.info("kubectl port-forward service/prometheus 9090:9090 -n monitoring")
    logger.info("\nTo run the demo with Prometheus metrics:")
    logger.info("python3 run_k8s_demo.py --use-real-k8s --namespace aegis-demo --enable-metrics --prometheus-url http://localhost:9090")
    logger.info("="*80)

if __name__ == '__main__':
    main()