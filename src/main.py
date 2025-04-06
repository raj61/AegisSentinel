#!/usr/bin/env python3
"""
Service Graph Builder for SRE
=============================

This application builds service graphs from Kubernetes or infrastructure code,
enabling auto-detection and auto-resolution of issues for SRE teams.
"""

import argparse
import logging
import sys
from pathlib import Path

# Import our modules
from src.parsers import get_parser
from src.graph import ServiceGraph
from src.detection import DetectionEngine
from src.resolution import ResolutionEngine
from src.api.server import start_api_server
from src.web.server import start_web_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Build service graphs from infrastructure code for SRE auto-detection and resolution'
    )
    parser.add_argument(
        '--source', '-s',
        type=str,
        required=True,
        help='Source directory or file containing infrastructure code'
    )
    parser.add_argument(
        '--type', '-t',
        type=str,
        choices=['kubernetes', 'terraform', 'cloudformation', 'auto'],
        default='auto',
        help='Type of infrastructure code (default: auto-detect)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='service_graph.png',
        help='Output file for the service graph visualization'
    )
    parser.add_argument(
        '--api',
        action='store_true',
        help='Start the API server'
    )
    parser.add_argument(
        '--web',
        action='store_true',
        help='Start the web interface'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port for the API/web server'
    )
    parser.add_argument(
        '--detect',
        action='store_true',
        help='Run detection algorithms on the service graph'
    )
    parser.add_argument(
        '--resolve',
        action='store_true',
        help='Attempt to auto-resolve detected issues'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    args = parse_args()
    
    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Starting service graph builder with source: {args.source}")
    
    # Get the appropriate parser for the infrastructure code
    source_path = Path(args.source)
    if not source_path.exists():
        logger.error(f"Source path does not exist: {args.source}")
        return 1
    
    # Get parser based on the specified type or auto-detect
    parser = get_parser(args.type, source_path)
    if not parser:
        logger.error(f"Could not determine parser for source: {args.source}")
        return 1
    
    # Parse the infrastructure code and build the service graph
    try:
        service_graph = ServiceGraph()
        parser.parse(source_path, service_graph)
        logger.info(f"Built initial service graph with {service_graph.node_count()} nodes and {service_graph.edge_count()} edges")
        
        # Infer additional relationships between services
        service_graph.infer_relationships()
        logger.info(f"Enhanced service graph with {service_graph.node_count()} nodes and {service_graph.edge_count()} edges")
        
        # Save the visualization
        service_graph.visualize(args.output)
        logger.info(f"Saved service graph visualization to {args.output}")
        
        # Run detection if requested
        if args.detect:
            detection_engine = DetectionEngine(service_graph)
            issues = detection_engine.detect_issues()
            logger.info(f"Detected {len(issues)} issues in the service graph")
            
            # Run resolution if requested
            if args.resolve and issues:
                resolution_engine = ResolutionEngine(service_graph)
                resolved = resolution_engine.resolve_issues(issues)
                logger.info(f"Resolved {len(resolved)} out of {len(issues)} issues")
        
        # Start API server if requested
        if args.api:
            start_api_server(service_graph, port=args.port)
        
        # Start web server if requested
        if args.web:
            start_web_server(service_graph, port=args.port)
        
        return 0
    
    except Exception as e:
        logger.exception(f"Error building service graph: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())