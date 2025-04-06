#!/usr/bin/env python3
"""
Aegis Sentinel Web Interface
===========================

This script starts the web interface with ML capabilities.
"""

import os
import sys
import argparse
import logging
import threading
import time
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.parsers.kubernetes_parser import KubernetesParser
from src.parsers.terraform_parser import TerraformParser
from src.parsers.cloudformation_parser import CloudFormationParser
from src.graph.service_graph import ServiceGraph
from src.web.server import start_web_server
from src.ml.integration import create_ml_integration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('web_interface.log')
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Aegis Sentinel Web Interface')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080,
                        help='Port to bind to')
    parser.add_argument('--k8s-file', type=str, default=None,
                        help='Path to Kubernetes manifest file')
    parser.add_argument('--tf-dir', type=str, default=None,
                        help='Path to Terraform directory')
    parser.add_argument('--cf-file', type=str, default=None,
                        help='Path to CloudFormation template file')
    parser.add_argument('--enable-ml', action='store_true',
                        help='Enable ML capabilities')
    parser.add_argument('--model-path', type=str, default='models',
                        help='Path for loading ML models')
    parser.add_argument('--inject-anomaly', action='store_true',
                        help='Inject a simulated anomaly after startup')
    parser.add_argument('--use-frontend', action='store_true',
                        help='Use the frontend folder instead of static files')
    return parser.parse_args()

def build_service_graph(args):
    """Build a service graph from configuration files."""
    graph = ServiceGraph()
    
    # Parse Kubernetes manifest if provided
    if args.k8s_file and os.path.exists(args.k8s_file):
        logger.info(f"Parsing Kubernetes manifest: {args.k8s_file}")
        parser = KubernetesParser()
        parser.parse_file(args.k8s_file)
        parser.build_graph(graph)
    
    # Parse Terraform configuration if provided
    if args.tf_dir and os.path.exists(args.tf_dir):
        logger.info(f"Parsing Terraform configuration: {args.tf_dir}")
        parser = TerraformParser()
        parser.parse_directory(args.tf_dir)
        parser.build_graph(graph)
    
    # Parse CloudFormation template if provided
    if args.cf_file and os.path.exists(args.cf_file):
        logger.info(f"Parsing CloudFormation template: {args.cf_file}")
        parser = CloudFormationParser()
        parser.parse_file(args.cf_file)
        parser.build_graph(graph)
    
    # If no configuration provided, use a demo graph
    if graph.node_count() == 0:
        logger.info("No configuration provided, using demo graph")
        _create_demo_graph(graph)
    
    # Infer relationships
    graph.infer_relationships()
    
    logger.info(f"Built service graph with {graph.node_count()} nodes and {graph.edge_count()} edges")
    return graph

def _create_demo_graph(graph):
    """Create a demo service graph."""
    # Add nodes
    graph.add_node('frontend',
                  name='Frontend',
                  kind='Deployment',
                  category='api',
                  health_status='healthy')
    
    graph.add_node('backend',
                  name='Backend API',
                  kind='Deployment',
                  category='api',
                  health_status='healthy')
    
    graph.add_node('auth',
                  name='Auth Service',
                  kind='Deployment',
                  category='api',
                  health_status='healthy')
    
    graph.add_node('database',
                  name='Database',
                  kind='StatefulSet',
                  category='database',
                  health_status='healthy')
    
    graph.add_node('cache',
                  name='Cache',
                  kind='Deployment',
                  category='cache',
                  health_status='healthy')
    
    graph.add_node('queue',
                  name='Message Queue',
                  kind='StatefulSet',
                  category='queue',
                  health_status='healthy')
    
    graph.add_node('worker',
                  name='Worker',
                  kind='Deployment',
                  category='compute',
                  health_status='healthy')
    
    graph.add_node('logger',
                  name='Logger',
                  kind='Deployment',
                  category='compute',
                  health_status='healthy')
    
    # Add edges
    graph.add_edge('frontend', 'backend', type='http')
    graph.add_edge('frontend', 'auth', type='http')
    graph.add_edge('backend', 'database', type='sql')
    graph.add_edge('backend', 'cache', type='redis')
    graph.add_edge('backend', 'queue', type='amqp')
    graph.add_edge('auth', 'database', type='sql')
    graph.add_edge('queue', 'worker', type='amqp')
    graph.add_edge('worker', 'database', type='sql')
    graph.add_edge('worker', 'logger', type='http')
    graph.add_edge('backend', 'logger', type='http')

def inject_anomaly(ml_engine):
    """Inject a simulated anomaly after a delay."""
    def _inject():
        # Wait a bit before injecting
        time.sleep(30)
        
        # Get a random service
        if ml_engine.service_graph:
            nodes = ml_engine.service_graph.get_nodes()
            if nodes:
                import random
                from datetime import datetime
                
                affected_service = random.choice(nodes)
                
                # Create a synthetic issue
                issue_id = f"synthetic-issue-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                issue = {
                    'id': issue_id,
                    'type': 'cpu_spike',
                    'severity': 4,
                    'description': 'Synthetic CPU spike for demonstration',
                    'affected_services': [affected_service],
                    'detected_at': datetime.now(),
                    'status': 'detected'
                }
                
                ml_engine.active_issues.append(issue)
                logger.info(f"Injected synthetic issue {issue_id} affecting service {affected_service}")
                
                # Trigger remediation
                ml_engine._trigger_remediation(issue)
                
                # Update service health status
                service_graph = ml_engine.service_graph
                node = service_graph.get_node(affected_service)
                if node:
                    node['health_status'] = 'critical'
                    logger.info(f"Updated health status of {affected_service} to critical")
    
    thread = threading.Thread(target=_inject)
    thread.daemon = True
    thread.start()
    logger.info("Scheduled anomaly injection")

def main():
    """Main function."""
    args = parse_args()
    
    # Build service graph
    graph = build_service_graph(args)
    
    # Create ML integration engine if enabled
    ml_engine = None
    if args.enable_ml:
        logger.info("Enabling ML capabilities")
        ml_engine = create_ml_integration(graph)
        
        # Load existing models if available
        if os.path.exists(args.model_path):
            logger.info(f"Loading ML models from {args.model_path}")
            ml_engine.load_models(args.model_path)
        
        # Start monitoring
        ml_engine.start_monitoring()
        logger.info("Started ML monitoring")
        
        # Inject anomaly if requested
        if args.inject_anomaly:
            inject_anomaly(ml_engine)
    
    # Start web server
    logger.info(f"Starting web server on {args.host}:{args.port}")
    server_thread = start_web_server(graph, host=args.host, port=args.port, open_browser=True, use_frontend=args.use_frontend)
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    finally:
        # Stop ML monitoring if enabled
        if ml_engine:
            ml_engine.stop_monitoring()
            logger.info("Stopped ML monitoring")
    
    logger.info("Server stopped")

if __name__ == '__main__':
    main()