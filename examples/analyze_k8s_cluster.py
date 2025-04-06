#!/usr/bin/env python3
"""
Analyze Kubernetes Cluster
========================

This script demonstrates how to use the service graph builder to analyze a Kubernetes cluster.
It connects to a Kubernetes cluster using the kubeconfig file and builds a service graph
from the live cluster resources.
"""

import os
import sys
import argparse
import logging
import tempfile
import subprocess
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
        description='Analyze a Kubernetes cluster and build a service graph'
    )
    parser.add_argument(
        '--kubeconfig',
        type=str,
        default=os.environ.get('KUBECONFIG', '~/.kube/config'),
        help='Path to the kubeconfig file'
    )
    parser.add_argument(
        '--namespace',
        type=str,
        default='',
        help='Kubernetes namespace to analyze (default: all namespaces)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='k8s_service_graph.png',
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

def export_k8s_resources(kubeconfig, namespace='', temp_dir=None):
    """
    Export Kubernetes resources to YAML files.
    
    Args:
        kubeconfig: Path to the kubeconfig file
        namespace: Kubernetes namespace to export (empty for all namespaces)
        temp_dir: Temporary directory to store the YAML files
        
    Returns:
        Path to the directory containing the YAML files
    """
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    
    # Expand the kubeconfig path
    kubeconfig = os.path.expanduser(kubeconfig)
    
    # Set the KUBECONFIG environment variable
    os.environ['KUBECONFIG'] = kubeconfig
    
    # Resource types to export
    resource_types = [
        'deployments',
        'services',
        'statefulsets',
        'daemonsets',
        'ingresses',
        'configmaps',
        'secrets',
        'pods',
        'replicasets',
        'horizontalpodautoscalers',
    ]
    
    # Export each resource type
    for resource_type in resource_types:
        # Build the kubectl command
        cmd = ['kubectl', 'get', resource_type, '-o', 'yaml']
        
        # Add namespace if specified
        if namespace:
            cmd.extend(['-n', namespace])
        else:
            cmd.append('--all-namespaces')
        
        # Run the command and save the output to a file
        output_file = os.path.join(temp_dir, f'{resource_type}.yaml')
        with open(output_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE)
    
    return temp_dir

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Export Kubernetes resources to YAML files
    logger.info(f"Exporting Kubernetes resources from {args.kubeconfig}")
    temp_dir = export_k8s_resources(args.kubeconfig, args.namespace)
    logger.info(f"Exported Kubernetes resources to {temp_dir}")
    
    # Build the service graph
    logger.info("Building service graph")
    service_graph = ServiceGraph()
    parser = get_parser('kubernetes', temp_dir)
    parser.parse(temp_dir, service_graph)
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
    
    logger.info("Done")

if __name__ == "__main__":
    main()