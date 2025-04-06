#!/usr/bin/env python3
"""
Aegis Sentinel ML Demo
=====================

This script demonstrates the ML capabilities of Aegis Sentinel.
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.parsers.kubernetes_parser import KubernetesParser
from src.graph.service_graph import ServiceGraph
from src.ml.integration import create_ml_integration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('aegis_sentinel_demo.log')
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Aegis Sentinel ML Demo')
    parser.add_argument('--k8s-file', type=str, default='examples/kubernetes/microservices-demo.yaml',
                        help='Path to Kubernetes manifest file')
    parser.add_argument('--inject-anomaly', action='store_true',
                        help='Inject a simulated anomaly')
    parser.add_argument('--run-time', type=int, default=300,
                        help='How long to run the demo (seconds)')
    parser.add_argument('--save-models', action='store_true',
                        help='Save trained ML models')
    parser.add_argument('--load-models', action='store_true',
                        help='Load existing ML models')
    parser.add_argument('--model-path', type=str, default='models',
                        help='Path for saving/loading models')
    return parser.parse_args()

def build_service_graph(k8s_file):
    """Build a service graph from a Kubernetes manifest file."""
    logger.info(f"Building service graph from {k8s_file}")
    
    # Parse the Kubernetes manifest
    parser = KubernetesParser()
    parser.parse_file(k8s_file)
    
    # Build the service graph
    graph = ServiceGraph()
    parser.build_graph(graph)
    
    # Infer relationships
    graph.infer_relationships()
    
    logger.info(f"Built service graph with {graph.node_count()} nodes and {graph.edge_count()} edges")
    return graph

def inject_anomaly(ml_engine):
    """Inject a simulated anomaly."""
    logger.info("Injecting simulated anomaly")
    
    # Get a random service
    if ml_engine.service_graph:
        nodes = ml_engine.service_graph.get_nodes()
        if nodes:
            import random
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
            return issue
    
    logger.warning("Could not inject anomaly: no services available")
    return None

def print_status(ml_engine):
    """Print the current status of the ML engine."""
    print("\n" + "=" * 80)
    print(f"AEGIS SENTINEL STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Print active issues
    issues = ml_engine.get_active_issues()
    print(f"\nActive Issues: {len(issues)}")
    print("-" * 40)
    
    for issue in issues:
        status_color = {
            'detected': '\033[93m',  # Yellow
            'mitigating': '\033[94m',  # Blue
            'mitigated': '\033[92m',  # Green
            'failed': '\033[91m'  # Red
        }.get(issue['status'], '\033[0m')
        
        print(f"ID: {issue['id']}")
        print(f"Type: {issue['type']}")
        print(f"Severity: {issue['severity']}/5")
        print(f"Status: {status_color}{issue['status']}\033[0m")
        print(f"Affected Services: {', '.join(issue['affected_services'])}")
        print(f"Detected: {issue['detected_at']}")
        if issue.get('mitigated_at'):
            print(f"Mitigated: {issue['mitigated_at']}")
        print()
    
    # Print active remediations
    remediations = ml_engine.get_active_remediations()
    print(f"\nActive Remediations: {len(remediations)}")
    print("-" * 40)
    
    for remediation_id, remediation in remediations.items():
        status_color = {
            'in_progress': '\033[94m',  # Blue
            'completed': '\033[92m',  # Green
            'failed': '\033[91m'  # Red
        }.get(remediation['status'], '\033[0m')
        
        print(f"ID: {remediation_id}")
        print(f"For Issue: {remediation['issue_id']}")
        print(f"Action: {remediation['action'].name}")
        print(f"Status: {status_color}{remediation['status']}\033[0m")
        print(f"Progress: {remediation['progress']}%")
        print(f"Confidence: {remediation['confidence']:.2f}")
        print(f"Learner: {remediation['learner']}")
        print()
    
    print("=" * 80)

def main():
    """Main function."""
    args = parse_args()
    
    # Build service graph
    graph = build_service_graph(args.k8s_file)
    
    # Create ML integration engine
    ml_engine = create_ml_integration(graph)
    
    # Load existing models if requested
    if args.load_models:
        if os.path.exists(args.model_path):
            logger.info(f"Loading ML models from {args.model_path}")
            ml_engine.load_models(args.model_path)
        else:
            logger.warning(f"Model path {args.model_path} does not exist, skipping model loading")
    
    # Start monitoring
    ml_engine.start_monitoring()
    logger.info("Started ML monitoring")
    
    # Run the demo for the specified time
    start_time = time.time()
    end_time = start_time + args.run_time
    
    try:
        # Inject anomaly if requested
        if args.inject_anomaly:
            # Wait a bit before injecting
            time.sleep(10)
            inject_anomaly(ml_engine)
        
        # Main loop
        while time.time() < end_time:
            print_status(ml_engine)
            time.sleep(10)
        
        # Train models
        logger.info("Training ML models")
        ml_engine.train_models()
        
        # Save models if requested
        if args.save_models:
            os.makedirs(args.model_path, exist_ok=True)
            logger.info(f"Saving ML models to {args.model_path}")
            ml_engine.save_models(args.model_path)
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    finally:
        # Stop monitoring
        ml_engine.stop_monitoring()
        logger.info("Stopped ML monitoring")
    
    logger.info("Demo completed")

if __name__ == '__main__':
    main()