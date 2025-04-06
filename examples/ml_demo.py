#!/usr/bin/env python3
"""
ML Components Demo
===============

This script demonstrates the ML components of the Aegis Sentinel system.
"""

import os
import sys
import logging
import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add the parent directory to the path so we can import the src module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsers import get_parser
from src.graph import ServiceGraph
from src.detection import DetectionEngine
from src.resolution import ResolutionEngine
from src.anomaly import LogAnomalyDetector
from src.remediation.remediation_engine import RemediationEngine, RemediationAction

# Import ML components
from src.ml.anomaly import MLLogAnomalyDetector, MetricAnomalyDetector, MetricAnomalyScore
from src.ml.root_cause import RootCauseAnalyzer
from src.ml.remediation import RLRemediationEngine
from src.ml.learning import RemediationLearner, RemediationSignature

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Demonstrate ML components of Aegis Sentinel'
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
        '--log-file',
        type=str,
        help='Log file to analyze'
    )
    parser.add_argument(
        '--k8s-logs',
        type=str,
        help='Kubernetes logs to analyze (format: namespace/pod)'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train ML models'
    )
    parser.add_argument(
        '--simulate',
        action='store_true',
        help='Simulate anomalies and remediation'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the demo."""
    args = parse_args()
    
    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Starting ML components demo with source: {args.source}")
    
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
        logger.info(f"Built service graph with {service_graph.node_count()} nodes and {service_graph.edge_count()} edges")
        
        # Initialize components
        detection_engine = DetectionEngine(service_graph)
        resolution_engine = ResolutionEngine(service_graph)
        
        # Initialize ML components
        ml_log_anomaly_detector = MLLogAnomalyDetector.create_with_default_patterns()
        metric_anomaly_detector = MetricAnomalyDetector()
        root_cause_analyzer = RootCauseAnalyzer(service_graph)
        
        # Create remediation actions
        remediation_actions = [
            RemediationAction(
                name="restart_pod",
                description="Restart a pod",
                command="kubectl rollout restart deployment/${SERVICE} -n ${NAMESPACE}",
                service=None,
                timeout_seconds=60,
                max_retries=3,
                cooldown_seconds=60
            ),
            RemediationAction(
                name="scale_up",
                description="Scale up a deployment",
                command="kubectl scale deployment/${SERVICE} -n ${NAMESPACE} --replicas=${REPLICAS}",
                service=None,
                timeout_seconds=60,
                max_retries=3,
                cooldown_seconds=60
            ),
            RemediationAction(
                name="restart_database",
                description="Restart a database",
                command="kubectl rollout restart statefulset/${SERVICE} -n ${NAMESPACE}",
                service=None,
                timeout_seconds=120,
                max_retries=2,
                cooldown_seconds=120
            ),
        ]
        
        rl_remediation_engine = RLRemediationEngine(remediation_actions)
        remediation_learner = RemediationLearner()
        
        # Load existing signatures if available
        try:
            remediation_learner.load_signatures()
        except Exception as e:
            logger.warning(f"Could not load existing signatures: {e}")
        
        # Train ML models if requested
        if args.train:
            logger.info("Training ML models...")
            
            # Train RL remediation model
            rl_remediation_engine.train(total_timesteps=1000)
            
            # Train remediation learner model if there are enough signatures
            if len(remediation_learner.signatures) >= 10:
                remediation_learner.train_model()
            
            logger.info("Finished training ML models")
        else:
            # Try to load pre-trained models
            try:
                rl_remediation_engine.load_model()
            except Exception as e:
                logger.warning(f"Could not load pre-trained RL model: {e}")
            
            try:
                remediation_learner.load_model()
            except Exception as e:
                logger.warning(f"Could not load pre-trained remediation learner model: {e}")
        
        # Detect issues in the service graph
        issues = detection_engine.detect_issues()
        logger.info(f"Detected {len(issues)} issues in the service graph")
        
        # Process logs if provided
        if args.log_file:
            logger.info(f"Processing logs from {args.log_file}")
            
            # Create a file log collector
            from src.anomaly.log_collector import FileLogCollector
            log_collector = FileLogCollector(args.log_file, follow=True)
            
            # Add callback to process logs with ML anomaly detector
            log_collector.add_callback(lambda log_line, timestamp: ml_log_anomaly_detector.process_log(log_line, timestamp))
            
            # Start collecting logs
            log_collector.start()
            
            # Wait for some logs to be processed
            time.sleep(5)
            
            # Get anomalies
            anomalies = ml_log_anomaly_detector.get_anomalies(
                since=datetime.now() - timedelta(minutes=5),
                min_score=0.5
            )
            
            logger.info(f"Detected {len(anomalies)} log anomalies")
            
            # Add anomalies to root cause analyzer
            for anomaly in anomalies:
                # Extract service from the anomaly
                service = anomaly.service or "default"
                
                # Find the corresponding node in the service graph
                node_id = None
                for node in service_graph.get_nodes():
                    node_data = service_graph.get_node(node)
                    if node_data.get('name') == service or node == service:
                        node_id = node
                        break
                
                if node_id:
                    root_cause_analyzer.add_node_anomaly(node_id, anomaly)
        
        # Process Kubernetes logs if provided
        if args.k8s_logs:
            logger.info(f"Processing Kubernetes logs from {args.k8s_logs}")
            
            # Parse namespace/pod
            try:
                namespace, pod = args.k8s_logs.split('/')
            except ValueError:
                logger.error(f"Invalid Kubernetes logs format: {args.k8s_logs}. Expected format: namespace/pod")
                return 1
            
            # Create a Kubernetes log collector
            from src.anomaly.log_collector import KubernetesLogCollector
            k8s_log_collector = KubernetesLogCollector(namespace, pod)
            
            # Add callback to process logs with ML anomaly detector
            k8s_log_collector.add_callback(lambda log_line, timestamp: ml_log_anomaly_detector.process_log(log_line, timestamp))
            
            # Start collecting logs
            k8s_log_collector.start()
            
            # Wait for some logs to be processed
            time.sleep(5)
            
            # Get anomalies
            anomalies = ml_log_anomaly_detector.get_anomalies(
                since=datetime.now() - timedelta(minutes=5),
                min_score=0.5
            )
            
            logger.info(f"Detected {len(anomalies)} Kubernetes log anomalies")
            
            # Add anomalies to root cause analyzer
            for anomaly in anomalies:
                # Extract service from the anomaly
                service = anomaly.service or pod
                
                # Find the corresponding node in the service graph
                node_id = None
                for node in service_graph.get_nodes():
                    node_data = service_graph.get_node(node)
                    if node_data.get('name') == service or node == service:
                        node_id = node
                        break
                
                if node_id:
                    root_cause_analyzer.add_node_anomaly(node_id, anomaly)
        
        # Add issues to root cause analyzer
        for issue in issues:
            for node_id in issue.affected_nodes:
                root_cause_analyzer.add_node_issue(node_id, issue)
        
        # Analyze root causes
        root_causes = root_cause_analyzer.analyze_root_cause()
        logger.info(f"Identified {len(root_causes)} potential root causes")
        
        # Print root causes
        for i, root_cause in enumerate(root_causes):
            logger.info(f"Root cause {i+1}: {root_cause.node_id} (score: {root_cause.score:.2f})")
            
            # Get node details
            node = service_graph.get_node(root_cause.node_id)
            name = node.get('name', root_cause.node_id)
            kind = node.get('kind', node.get('type', 'unknown'))
            
            logger.info(f"  Node: {name} ({kind})")
            logger.info(f"  Issues: {len(root_cause.issues)}")
            logger.info(f"  Anomalies: {len(root_cause.anomalies)}")
            logger.info(f"  Metric anomalies: {len(root_cause.metric_anomalies)}")
            logger.info(f"  Propagation path: {' -> '.join(root_cause.propagation_path)}")
        
        # Simulate anomalies and remediation if requested
        if args.simulate:
            logger.info("Simulating anomalies and remediation...")
            
            # Create some simulated metric anomalies
            metric_anomalies = []
            
            for i, node_id in enumerate(service_graph.get_nodes()[:3]):
                # Get node details
                node = service_graph.get_node(node_id)
                name = node.get('name', node_id)
                
                # Create a metric anomaly
                from src.ml.anomaly.metric_anomaly import MetricDataPoint
                metric = MetricDataPoint(
                    timestamp=datetime.now(),
                    value=95.0,  # High CPU usage
                    metric_name="cpu_usage",
                    metric_type="cpu",
                    service=name,
                    dimensions={"instance": f"pod-{i}"}
                )
                
                # Create an anomaly score
                anomaly = MetricAnomalyScore(
                    metric=metric,
                    score=0.9,
                    timestamp=datetime.now(),
                    expected_value=50.0,
                    model_name="isolation_forest"
                )
                
                metric_anomalies.append(anomaly)
                
                # Add to root cause analyzer
                root_cause_analyzer.add_node_metric_anomaly(node_id, anomaly)
            
            logger.info(f"Created {len(metric_anomalies)} simulated metric anomalies")
            
            # Re-analyze root causes
            root_causes = root_cause_analyzer.analyze_root_cause()
            logger.info(f"Identified {len(root_causes)} potential root causes after adding metric anomalies")
            
            # Use RL remediation engine to select and execute actions
            for root_cause in root_causes:
                # Get the issues, anomalies, and metric anomalies for this node
                node_issues = root_cause.issues
                node_anomalies = root_cause.anomalies
                node_metric_anomalies = root_cause.metric_anomalies
                
                # Select an action using the RL model
                action = rl_remediation_engine.select_action(
                    node_issues,
                    node_anomalies,
                    node_metric_anomalies,
                    root_causes
                )
                
                if action:
                    logger.info(f"Selected action for {root_cause.node_id}: {action.name}")
                    logger.info(f"Command: {action.command}")
                    
                    # In a real system, we would execute the action here
                    # For the demo, we'll just simulate it
                    logger.info(f"Simulating execution of {action.name}...")
                    
                    # Capture as a manual fix for the remediation learner
                    remediation_learner.capture_manual_fix(
                        command=action.command,
                        issues=node_issues,
                        anomalies=node_anomalies,
                        metric_anomalies=node_metric_anomalies,
                        success=True
                    )
                    
                    logger.info(f"Captured remediation action in the learner")
                else:
                    logger.info(f"No action selected for {root_cause.node_id}")
            
            # Try to suggest an action using the remediation learner
            for root_cause in root_causes:
                # Get the issues, anomalies, and metric anomalies for this node
                node_issues = root_cause.issues
                node_anomalies = root_cause.anomalies
                node_metric_anomalies = root_cause.metric_anomalies
                
                # Suggest an action using the learner
                action = remediation_learner.suggest_action(
                    node_issues,
                    node_anomalies,
                    node_metric_anomalies
                )
                
                if action:
                    logger.info(f"Suggested action from learner for {root_cause.node_id}: {action.name}")
                    logger.info(f"Command: {action.command}")
                else:
                    logger.info(f"No action suggested by learner for {root_cause.node_id}")
        
        logger.info("ML components demo completed successfully")
        return 0
    
    except Exception as e:
        logger.exception(f"Error in ML components demo: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
