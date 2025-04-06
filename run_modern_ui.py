#!/usr/bin/env python3
"""
Run Modern UI Web Interface
==========================

This script runs the web interface with the modern UI.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

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
    parser = argparse.ArgumentParser(description='Run the web interface with modern UI')
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
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Add the project root to the Python path
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    # Import modules
    from src.graph import ServiceGraph
    from src.web.server import start_web_server
    
    # Create a demo service graph if no configuration is provided
    logger.info("No configuration provided, using demo graph")
    service_graph = ServiceGraph()
    
    # Add some demo nodes
    service_graph.add_node('frontend', name='Frontend', kind='service', category='api')
    service_graph.add_node('backend', name='Backend', kind='service', category='compute')
    service_graph.add_node('auth', name='Auth Service', kind='service', category='compute')
    service_graph.add_node('database', name='Database', kind='database', category='database')
    service_graph.add_node('cache', name='Cache', kind='cache', category='cache')
    service_graph.add_node('queue', name='Message Queue', kind='queue', category='queue')
    service_graph.add_node('storage', name='Object Storage', kind='storage', category='storage')
    service_graph.add_node('analytics', name='Analytics', kind='service', category='compute')
    
    # Add some demo edges
    service_graph.add_edge('frontend', 'backend')
    service_graph.add_edge('frontend', 'auth')
    service_graph.add_edge('backend', 'database')
    service_graph.add_edge('backend', 'cache')
    service_graph.add_edge('backend', 'queue')
    service_graph.add_edge('auth', 'database')
    service_graph.add_edge('queue', 'analytics')
    service_graph.add_edge('analytics', 'storage')
    service_graph.add_edge('backend', 'storage')
    service_graph.add_edge('analytics', 'database')
    
    # Infer relationships
    service_graph.infer_relationships()
    
    logger.info(f"Built service graph with {service_graph.node_count()} nodes and {service_graph.edge_count()} edges")
    
    # Enable ML capabilities if requested
    if args.enable_ml:
        logger.info("Enabling ML capabilities")
        
        # Import ML modules
        try:
            from src.ml.integration import MLIntegrationEngine
            from src.ml.anomaly_detection import AnomalyDetectionEngine
            from src.ml.learning.remediation_learner import RemediationLearningEngine
            
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
                
                def inject_anomaly():
                    while True:
                        # Wait for 30 seconds before injecting an anomaly
                        time.sleep(30)
                        
                        # Choose a random service to affect
                        services = ['frontend', 'backend', 'auth', 'database', 'cache', 'queue', 'storage', 'analytics']
                        affected_service = random.choice(services)
                        
                        # Create a synthetic issue ID
                        issue_id = f"synthetic-issue-{time.strftime('%Y%m%d%H%M%S')}"
                        
                        logger.info(f"Injected synthetic issue {issue_id} affecting service {affected_service}")
                        
                        # Update the health status of the affected service
                        if affected_service in service_graph.get_nodes():
                            # Get the current node data
                            node_data = service_graph.get_node(affected_service)
                            # Update the health status
                            node_data['health_status'] = 'critical'
                            # Update the node in the graph using the proper method
                            service_graph.update_node_attribute(affected_service, 'health_status', 'critical')
                            logger.info(f"Updated health status of {affected_service} to critical")
                
                # Start the anomaly injection thread
                anomaly_thread = threading.Thread(target=inject_anomaly, daemon=True)
                anomaly_thread.start()
        except Exception as e:
            logger.error(f"Error enabling ML capabilities: {e}")
    
    # Start the web server
    start_web_server(
        service_graph=service_graph,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
        use_frontend=True  # Always use the modern frontend
    )
    
    try:
        # Keep the main thread alive
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
        
        # Stop ML monitoring if enabled
        if args.enable_ml and 'ml_engine' in locals():
            ml_engine.stop_monitoring()
            logger.info("Stopped ML monitoring")
        
        logger.info("Server stopped")

if __name__ == '__main__':
    main()