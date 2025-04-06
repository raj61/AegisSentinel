#!/usr/bin/env python3
"""
Run Kubernetes Demo
==================

This script runs the web interface with a real Kubernetes service graph
parsed from the microservices-demo.yaml file.
"""

import os
import sys
import logging
import argparse
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('k8s_demo.log')
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the web interface with a real Kubernetes service graph')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080,
                        help='Port to bind to')
    parser.add_argument('--no-browser', action='store_true',
                        help='Do not open a browser window')
    parser.add_argument('--enable-ml', action='store_true',
                        help='Enable ML capabilities')
    parser.add_argument('--inject-anomaly', action='store_true',
                        help='Inject synthetic anomalies for demo purposes')
    parser.add_argument('--prometheus-url', type=str, default='',
                        help='URL of the Prometheus server (e.g., http://localhost:9090)')
    parser.add_argument('--enable-metrics', action='store_true',
                        help='Enable metrics collection from Prometheus')
    parser.add_argument('--use-real-k8s', action='store_true',
                        help='Connect to a real Kubernetes cluster instead of using the demo YAML')
    parser.add_argument('--kubeconfig', type=str, default='',
                        help='Path to kubeconfig file for connecting to a real Kubernetes cluster')
    parser.add_argument('--namespace', type=str, default='default',
                        help='Kubernetes namespace to monitor')
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Add the project root to the Python path
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    # Import modules
    from src.graph import ServiceGraph
    from src.parsers import get_parser
    from src.web.server import start_web_server
    from src.ml.integration import MLIntegrationEngine
    from src.metrics.metrics_integration import MetricsIntegration
    
    # Create a service graph
    service_graph = ServiceGraph()
    
    if args.use_real_k8s:
        # Connect to a real Kubernetes cluster
        try:
            from kubernetes import client, config
            
            # Load kubeconfig
            if args.kubeconfig:
                config.load_kube_config(config_file=args.kubeconfig)
            else:
                # Try to load from default location
                try:
                    config.load_kube_config()
                except:
                    # Try in-cluster config (if running inside a pod)
                    try:
                        config.load_incluster_config()
                    except:
                        logger.error("Could not load Kubernetes configuration. Please provide a valid kubeconfig file.")
                        sys.exit(1)
            
            # Create Kubernetes API client
            v1 = client.CoreV1Api()
            apps_v1 = client.AppsV1Api()
            
            # Get services
            logger.info(f"Fetching services from namespace {args.namespace}")
            services = v1.list_namespaced_service(args.namespace)
            
            # Get deployments
            deployments = apps_v1.list_namespaced_deployment(args.namespace)
            
            # Get pods
            pods = v1.list_namespaced_pod(args.namespace)
            
            # Create a temporary directory for the Kubernetes resources
            temp_dir = tempfile.mkdtemp()
            k8s_yaml_path = os.path.join(temp_dir, 'k8s-resources.yaml')
            
            # Write Kubernetes resources to a YAML file
            with open(k8s_yaml_path, 'w') as f:
                # Write services
                for service in services.items:
                    f.write("---\n")
                    f.write(f"apiVersion: v1\n")
                    f.write(f"kind: Service\n")
                    f.write(f"metadata:\n")
                    f.write(f"  name: {service.metadata.name}\n")
                    f.write(f"  namespace: {service.metadata.namespace}\n")
                    if service.metadata.labels:
                        f.write(f"  labels:\n")
                        for key, value in service.metadata.labels.items():
                            f.write(f"    {key}: {value}\n")
                    f.write(f"spec:\n")
                    if service.spec.selector:
                        f.write(f"  selector:\n")
                        for key, value in service.spec.selector.items():
                            f.write(f"    {key}: {value}\n")
                    if service.spec.ports:
                        f.write(f"  ports:\n")
                        for port in service.spec.ports:
                            f.write(f"  - port: {port.port}\n")
                            if port.target_port:
                                f.write(f"    targetPort: {port.target_port}\n")
                    if service.spec.type:
                        f.write(f"  type: {service.spec.type}\n")
                
                # Write deployments
                for deployment in deployments.items:
                    f.write("---\n")
                    f.write(f"apiVersion: apps/v1\n")
                    f.write(f"kind: Deployment\n")
                    f.write(f"metadata:\n")
                    f.write(f"  name: {deployment.metadata.name}\n")
                    f.write(f"  namespace: {deployment.metadata.namespace}\n")
                    if deployment.metadata.labels:
                        f.write(f"  labels:\n")
                        for key, value in deployment.metadata.labels.items():
                            f.write(f"    {key}: {value}\n")
                    f.write(f"spec:\n")
                    if deployment.spec.replicas:
                        f.write(f"  replicas: {deployment.spec.replicas}\n")
                    if deployment.spec.selector and deployment.spec.selector.match_labels:
                        f.write(f"  selector:\n")
                        f.write(f"    matchLabels:\n")
                        for key, value in deployment.spec.selector.match_labels.items():
                            f.write(f"      {key}: {value}\n")
                    if deployment.spec.template:
                        f.write(f"  template:\n")
                        f.write(f"    metadata:\n")
                        if deployment.spec.template.metadata and deployment.spec.template.metadata.labels:
                            f.write(f"      labels:\n")
                            for key, value in deployment.spec.template.metadata.labels.items():
                                f.write(f"        {key}: {value}\n")
            
            logger.info(f"Wrote Kubernetes resources to {k8s_yaml_path}")
            
            # Parse the YAML file
            parser = get_parser('kubernetes', temp_dir)
            parser.parse(k8s_yaml_path, service_graph)
            logger.info(f"Parsed real Kubernetes resources from cluster")
            
        except ImportError:
            logger.error("Kubernetes client library not installed. Please install it with 'pip install kubernetes'")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error connecting to Kubernetes cluster: {e}")
            sys.exit(1)
    else:
        # Use the demo YAML file
        temp_dir = tempfile.mkdtemp()
        k8s_yaml_path = os.path.join(temp_dir, 'microservices-demo.yaml')
        
        # Copy the microservices-demo.yaml file to the temporary directory
        with open('examples/kubernetes/microservices-demo.yaml', 'r') as src_file:
            with open(k8s_yaml_path, 'w') as dest_file:
                dest_file.write(src_file.read())
        
        logger.info(f"Copied microservices-demo.yaml to {k8s_yaml_path}")
        
        # Parse the YAML file
        try:
            from src.parsers.kubernetes_parser import KubernetesParser
            parser = KubernetesParser()
            parser.parse(k8s_yaml_path, service_graph)
            logger.info(f"Parsed demo Kubernetes resources from YAML file")
            
            # Verify that nodes were created
            node_count = service_graph.node_count()
            edge_count = service_graph.edge_count()
            logger.info(f"Created service graph with {node_count} nodes and {edge_count} edges")
            
            if node_count == 0:
                logger.warning("No nodes were created in the service graph. Check the YAML file and parser.")
                # Try to read the YAML file to verify its contents
                with open(k8s_yaml_path, 'r') as f:
                    yaml_content = f.read()
                    logger.info(f"YAML file content length: {len(yaml_content)} bytes")
        except Exception as e:
            logger.error(f"Error parsing Kubernetes YAML: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Infer relationships between services using traditional methods
    service_graph.infer_relationships()
    
    # If ML is enabled, enhance the graph with ML-based relationship detection
    if args.enable_ml:
        try:
            from src.ml.graph import MLServiceGraph
            
            logger.info("Enhancing service graph with ML-based relationship detection...")
            ml_service_graph = MLServiceGraph(service_graph)
            service_graph = ml_service_graph.enhance_graph()
            logger.info("ML-based graph enhancement complete")
        except Exception as e:
            logger.error(f"Error enhancing service graph with ML: {e}")
    
    logger.info(f"Built service graph with {service_graph.node_count()} nodes and {service_graph.edge_count()} edges")
    
    # Enable metrics collection if requested
    metrics_integration = None
    if args.enable_metrics:
        if args.prometheus_url:
            logger.info(f"Enabling metrics collection from Prometheus at {args.prometheus_url}")
            try:
                metrics_integration = MetricsIntegration(service_graph, args.prometheus_url)
                
                # Check if Prometheus is reachable
                if metrics_integration.prometheus_collector.check_health():
                    logger.info("Successfully connected to Prometheus")
                    
                    # Update metrics once immediately
                    metrics_integration.update_all_services()
                    
                    # Start metrics monitoring
                    metrics_integration.start_monitoring()
                    logger.info("Started metrics monitoring")
                else:
                    logger.error("Could not connect to Prometheus server")
            except Exception as e:
                logger.error(f"Error enabling metrics collection: {e}")
        else:
            logger.warning("Prometheus URL not provided, metrics collection disabled")
    
    # Enable ML capabilities if requested
    ml_engine = None
    if args.enable_ml:
        logger.info("Enabling ML capabilities")
        
        try:
            # Initialize ML components
            ml_engine = MLIntegrationEngine(service_graph)
            
            # Load ML models
            logger.info("Loading ML models from models")
            ml_engine.load_models("models")
            
            # Start ML monitoring
            ml_engine.start_monitoring()
            logger.info("Started ML monitoring")
            
            # Schedule anomaly injection if requested
            if args.inject_anomaly:
                logger.info("Scheduled anomaly injection")
                
                # Manually inject anomalies periodically
                import threading
                import random
                import time
                
                # Create a thread-safe flag to control the anomaly injection thread
                stop_anomaly_thread = threading.Event()
                
                def inject_anomaly():
                    try:
                        while not stop_anomaly_thread.is_set():
                            # Wait for 30 seconds before injecting an anomaly
                            # Use small intervals to check for stop flag
                            for _ in range(30):
                                if stop_anomaly_thread.is_set():
                                    return
                                time.sleep(1)
                            
                            try:
                                # Choose a random service to affect
                                services = service_graph.get_nodes()
                                if not services:
                                    logger.warning("No services available for anomaly injection")
                                    continue
                                    
                                affected_service = random.choice(services)
                                
                                # Create a synthetic issue ID
                                issue_id = f"synthetic-issue-{time.strftime('%Y%m%d%H%M%S')}"
                                
                                logger.info(f"Injected synthetic issue {issue_id} affecting service {affected_service}")
                                
                                # Update the health status of the affected service
                                if affected_service in service_graph.get_nodes():
                                    # Update the node in the graph using the proper method
                                    service_graph.update_node_attribute(affected_service, 'health_status', 'critical')
                                    logger.info(f"Updated health status of {affected_service} to critical")
                                    
                                    # Reset the health status after 60 seconds
                                    def reset_health():
                                        try:
                                            time.sleep(60)
                                            if affected_service in service_graph.get_nodes():
                                                service_graph.update_node_attribute(affected_service, 'health_status', 'healthy')
                                                logger.info(f"Reset health status of {affected_service} to healthy")
                                        except Exception as e:
                                            logger.error(f"Error resetting health status: {e}")
                                    
                                    # Start a thread to reset the health status
                                    reset_thread = threading.Thread(target=reset_health, daemon=True)
                                    reset_thread.start()
                            except Exception as e:
                                logger.error(f"Error in anomaly injection: {e}")
                    except Exception as e:
                        logger.error(f"Anomaly injection thread crashed: {e}")
                
                # Start the anomaly injection thread
                anomaly_thread = threading.Thread(target=inject_anomaly, daemon=True)
                anomaly_thread.start()
                
                # Store the stop flag for later cleanup
                ml_engine.anomaly_thread = anomaly_thread
                ml_engine.stop_anomaly_thread = stop_anomaly_thread
        except Exception as e:
            logger.error(f"Error enabling ML capabilities: {e}")
    
    # Start the web server
    start_web_server(
        service_graph=service_graph,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
        use_frontend=True,  # Always use the modern frontend
        metrics_integration=metrics_integration
    )
    
    try:
        # Keep the main thread alive
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
        
        # Stop ML monitoring if enabled
        if ml_engine:
            # Stop anomaly injection thread if it exists
            if hasattr(ml_engine, 'stop_anomaly_thread'):
                ml_engine.stop_anomaly_thread.set()
                logger.info("Stopping anomaly injection thread...")
                
                # Wait for the thread to finish (with timeout)
                if hasattr(ml_engine, 'anomaly_thread'):
                    ml_engine.anomaly_thread.join(timeout=2)
                    logger.info("Anomaly injection thread stopped")
            
            ml_engine.stop_monitoring()
            logger.info("Stopped ML monitoring")
            
        # Stop metrics monitoring if enabled
        if metrics_integration:
            metrics_integration.stop_monitoring_thread()
            logger.info("Stopped metrics monitoring")
        
        logger.info("Server stopped")

if __name__ == '__main__':
    main()