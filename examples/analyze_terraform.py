#!/usr/bin/env python3
"""
Analyze Terraform Code
====================

This script demonstrates how to use the service graph builder to analyze Terraform code.
It parses Terraform files and builds a service graph from the infrastructure code.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the parent directory to the path so we can import the src module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph import ServiceGraph
from src.parsers import get_parser
from src.detection import DetectionEngine
from src.resolution import ResolutionEngine

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze Terraform code and build a service graph'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Path to the Terraform code directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='terraform_service_graph.png',
        help='Output file for the service graph visualization'
    )
    parser.add_argument(
        '--detect',
        action='store_true',
        help='Detect issues in the service graph'
    )
    parser.add_argument(
        '--resolve',
        action='store_true',
        help='Suggest resolutions for detected issues'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Check if the source directory exists
    source_path = Path(args.source)
    if not source_path.exists():
        logger.error(f"Source path does not exist: {args.source}")
        return 1
    
    # Build the service graph
    logger.info(f"Building service graph from Terraform code in {args.source}")
    service_graph = ServiceGraph()
    parser = get_parser('terraform', source_path)
    parser.parse(source_path, service_graph)
    logger.info(f"Built service graph with {service_graph.node_count()} nodes and {service_graph.edge_count()} edges")
    
    # Save the visualization
    service_graph.visualize(args.output)
    logger.info(f"Saved service graph visualization to {args.output}")
    
    # Detect issues if requested
    if args.detect:
        logger.info("Detecting issues")
        detection_engine = DetectionEngine(service_graph)
        issues = detection_engine.detect_issues()
        logger.info(f"Detected {len(issues)} issues")
        
        # Print the issues
        for issue in issues:
            logger.info(f"Issue: {issue.type.value} - {issue.description}")
            
            # Print affected nodes
            if issue.affected_nodes:
                logger.info(f"  Affected nodes:")
                for node_id in issue.affected_nodes:
                    node = service_graph.get_node(node_id)
                    name = node.get('name', node_id)
                    type_or_kind = node.get('type', node.get('kind', 'unknown'))
                    logger.info(f"    - {name} ({type_or_kind})")
        
        # Resolve issues if requested
        if args.resolve and issues:
            logger.info("Suggesting resolutions")
            resolution_engine = ResolutionEngine(service_graph)
            resolutions = resolution_engine.resolve_issues(issues)
            logger.info(f"Generated {len(resolutions)} resolution suggestions")
            
            # Print the resolutions
            for resolution in resolutions:
                logger.info(f"Resolution for {resolution.issue.type.value}: {resolution.description}")
                if resolution.metadata.get('suggestion'):
                    logger.info(f"Suggestion: {resolution.metadata['suggestion']}")
    
    # Save the service graph as JSON
    json_output = Path(args.output).with_suffix('.json')
    service_graph.save_json(json_output)
    logger.info(f"Saved service graph data to {json_output}")
    
    logger.info("Done")
    return 0

if __name__ == "__main__":
    sys.exit(main())