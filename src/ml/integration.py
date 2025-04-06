"""
ML Integration Module
===================

This module integrates the ML components with the core Aegis Sentinel system.
"""

import logging
import threading
import time
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

from src.graph import ServiceGraph
from src.ml.anomaly_detection import AnomalyDetectionEngine, create_default_engine as create_anomaly_engine
from src.ml.learning.remediation_learner import (
    RemediationLearningEngine, RemediationState, RemediationAction, RemediationExperience,
    create_default_engine as create_remediation_engine
)

logger = logging.getLogger(__name__)

class MLIntegrationEngine:
    """Engine for integrating ML components with the core system."""
    
    def __init__(self, 
                 service_graph: ServiceGraph = None,
                 anomaly_engine: AnomalyDetectionEngine = None,
                 remediation_engine: RemediationLearningEngine = None,
                 config: Dict[str, Any] = None):
        """Initialize the ML integration engine.
        
        Args:
            service_graph: ServiceGraph instance
            anomaly_engine: AnomalyDetectionEngine instance
            remediation_engine: RemediationLearningEngine instance
            config: Configuration parameters
        """
        self.service_graph = service_graph
        self.anomaly_engine = anomaly_engine or create_anomaly_engine()
        self.remediation_engine = remediation_engine or create_remediation_engine()
        self.config = config or {}
        
        self.metrics_history = {}
        self.log_history = {}
        self.active_issues = []
        self.active_remediations = {}
        
        self.monitoring_thread = None
        self.is_running = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def set_service_graph(self, service_graph: ServiceGraph) -> None:
        """Set the service graph.
        
        Args:
            service_graph: ServiceGraph instance
        """
        self.service_graph = service_graph
        self.logger.info("Updated service graph")
        
    def start_monitoring(self) -> None:
        """Start the monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.logger.warning("Monitoring thread is already running")
            return
        
        self.is_running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        self.logger.info("Started monitoring thread")
        
    def stop_monitoring(self) -> None:
        """Stop the monitoring thread."""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
            self.logger.info("Stopped monitoring thread")
        
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Collect metrics and logs
                metrics_data = self._collect_metrics()
                log_data = self._collect_logs()
                
                # Detect anomalies
                anomaly_results = self._detect_anomalies(metrics_data, log_data)
                
                # Process detected anomalies
                self._process_anomalies(anomaly_results)
                
                # Check active remediations
                self._check_remediations()
                
                # Sleep for the monitoring interval
                time.sleep(self.config.get('monitoring_interval', 60))
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(5)  # Sleep briefly before retrying
    
    def _collect_metrics(self) -> Dict[str, pd.DataFrame]:
        """Collect metrics from services.
        
        Returns:
            Dictionary mapping service IDs to metric DataFrames
        """
        # In a real implementation, we would:
        # 1. Query Prometheus/InfluxDB for metrics
        # 2. Process and normalize the metrics
        # 3. Store in the metrics history
        
        metrics_data = {}
        
        if self.service_graph:
            for node_id in self.service_graph.get_nodes():
                # Generate random metrics for demonstration
                now = datetime.now()
                timestamps = [now - timedelta(minutes=i) for i in range(10)]
                
                metrics_df = pd.DataFrame({
                    'timestamp': timestamps,
                    'cpu_usage': [50 + (i * 2) + ((i % 3) * 10) for i in range(10)],
                    'memory_usage': [30 + (i * 1.5) for i in range(10)],
                    'request_count': [100 + (i * 5) for i in range(10)],
                    'error_rate': [2 + (i * 0.5) for i in range(10)]
                })
                
                metrics_df.set_index('timestamp', inplace=True)
                metrics_data[node_id] = metrics_df
                
                # Store in history
                if node_id not in self.metrics_history:
                    self.metrics_history[node_id] = metrics_df
                else:
                    self.metrics_history[node_id] = pd.concat([
                        self.metrics_history[node_id],
                        metrics_df
                    ]).sort_index().drop_duplicates()
        
        return metrics_data
    
    def _collect_logs(self) -> Dict[str, pd.DataFrame]:
        """Collect logs from services.
        
        Returns:
            Dictionary mapping service IDs to log DataFrames
        """
        # In a real implementation, we would:
        # 1. Query log aggregation system (ELK, Loki)
        # 2. Process and normalize the logs
        # 3. Store in the log history
        
        log_data = {}
        
        if self.service_graph:
            for node_id in self.service_graph.get_nodes():
                # Generate random logs for demonstration
                now = datetime.now()
                timestamps = [now - timedelta(minutes=i) for i in range(5)]
                
                log_df = pd.DataFrame({
                    'timestamp': timestamps,
                    'level': ['INFO', 'INFO', 'WARN', 'INFO', 'ERROR'],
                    'message': [
                        f"Service {node_id} started",
                        f"Processed 100 requests",
                        f"High latency detected",
                        f"Cache hit ratio: 0.75",
                        f"Failed to connect to database"
                    ]
                })
                
                log_df.set_index('timestamp', inplace=True)
                log_data[node_id] = log_df
                
                # Store in history
                if node_id not in self.log_history:
                    self.log_history[node_id] = log_df
                else:
                    self.log_history[node_id] = pd.concat([
                        self.log_history[node_id],
                        log_df
                    ]).sort_index().drop_duplicates()
        
        return log_data
    
    def _detect_anomalies(self, metrics_data: Dict[str, pd.DataFrame], log_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Detect anomalies in metrics and logs.
        
        Args:
            metrics_data: Dictionary mapping service IDs to metric DataFrames
            log_data: Dictionary mapping service IDs to log DataFrames
            
        Returns:
            Dictionary mapping detector names to anomaly results
        """
        # Prepare data for anomaly detection
        detector_data = {
            'metrics': pd.concat(metrics_data.values()) if metrics_data else pd.DataFrame(),
            'logs': pd.concat(log_data.values()) if log_data else pd.DataFrame()
        }
        
        # Detect anomalies
        anomaly_results = self.anomaly_engine.detect_anomalies(detector_data)
        self.logger.info(f"Detected anomalies: {sum(df['is_anomaly'].sum() for df in anomaly_results.values())}")
        
        return anomaly_results
    
    def _process_anomalies(self, anomaly_results: Dict[str, pd.DataFrame]) -> None:
        """Process detected anomalies.
        
        Args:
            anomaly_results: Dictionary mapping detector names to anomaly results
        """
        # Extract anomalies
        anomalies = []
        
        for detector_name, results in anomaly_results.items():
            if 'is_anomaly' in results.columns:
                anomaly_indices = results[results['is_anomaly']].index
                for idx in anomaly_indices:
                    # In a real implementation, we would extract more information
                    anomalies.append({
                        'timestamp': idx,
                        'detector': detector_name,
                        'score': results.loc[idx, 'z_score'] if 'z_score' in results.columns else 0,
                        'affected_services': self._identify_affected_services(detector_name, idx)
                    })
        
        # Create issues from anomalies
        for anomaly in anomalies:
            issue_id = f"issue-{len(self.active_issues) + 1}"
            issue = {
                'id': issue_id,
                'type': f"{anomaly['detector']}_anomaly",
                'severity': min(5, int(anomaly['score'] * 2)) if anomaly['score'] else 3,
                'description': f"Anomaly detected by {anomaly['detector']} detector",
                'affected_services': anomaly['affected_services'],
                'detected_at': anomaly['timestamp'],
                'status': 'detected'
            }
            
            self.active_issues.append(issue)
            self.logger.info(f"Created issue {issue_id} from anomaly")
            
            # Trigger remediation
            self._trigger_remediation(issue)
    
    def _identify_affected_services(self, detector_name: str, timestamp) -> List[str]:
        """Identify services affected by an anomaly.
        
        Args:
            detector_name: Name of the detector that found the anomaly
            timestamp: Timestamp of the anomaly
            
        Returns:
            List of affected service IDs
        """
        # In a real implementation, we would analyze metrics and logs
        # to determine which services are affected
        
        # For demonstration, return random services
        if self.service_graph:
            nodes = self.service_graph.get_nodes()
            if nodes:
                # Return 1-3 random services
                import random
                return random.sample(nodes, min(3, len(nodes)))
        
        return []
    
    def _trigger_remediation(self, issue: Dict[str, Any]) -> None:
        """Trigger remediation for an issue.
        
        Args:
            issue: Issue to remediate
        """
        # Create remediation state
        state = RemediationState(
            issue_type=issue['type'],
            issue_severity=issue['severity'],
            affected_services=issue['affected_services'],
            service_states={service: 'degraded' for service in issue['affected_services']},
            previous_actions=[]
        )
        
        # Get remediation recommendation
        action, confidence, learner = self.remediation_engine.recommend_action(state)
        
        if action:
            # Start remediation
            remediation_id = f"remediation-{len(self.active_remediations) + 1}"
            
            remediation = {
                'id': remediation_id,
                'issue_id': issue['id'],
                'action': action,
                'confidence': confidence,
                'learner': learner,
                'initial_state': state,
                'start_time': datetime.now(),
                'status': 'in_progress',
                'progress': 0
            }
            
            self.active_remediations[remediation_id] = remediation
            
            # Update issue status
            for i, active_issue in enumerate(self.active_issues):
                if active_issue['id'] == issue['id']:
                    self.active_issues[i]['status'] = 'mitigating'
                    break
            
            self.logger.info(f"Started remediation {remediation_id} for issue {issue['id']} using action {action.name}")
            
            # In a real implementation, we would execute the action
            # For demonstration, we'll simulate progress in _check_remediations
    
    def _check_remediations(self) -> None:
        """Check and update active remediations."""
        completed_remediations = []
        
        for remediation_id, remediation in self.active_remediations.items():
            if remediation['status'] == 'in_progress':
                # Simulate progress
                elapsed = (datetime.now() - remediation['start_time']).total_seconds()
                estimated_duration = remediation['action'].estimated_duration or 60
                
                progress = min(100, int(elapsed / estimated_duration * 100))
                remediation['progress'] = progress
                
                # Check if complete
                if progress >= 100:
                    # Simulate success (80% chance)
                    import random
                    success = random.random() < 0.8
                    
                    # Update remediation status
                    remediation['status'] = 'completed' if success else 'failed'
                    remediation['end_time'] = datetime.now()
                    
                    # Update issue status
                    for i, issue in enumerate(self.active_issues):
                        if issue['id'] == remediation['issue_id']:
                            self.active_issues[i]['status'] = 'mitigated' if success else 'failed'
                            if success:
                                self.active_issues[i]['mitigated_at'] = datetime.now()
                            break
                    
                    # Record experience
                    next_state = RemediationState(
                        issue_type=remediation['initial_state'].issue_type,
                        issue_severity=remediation['initial_state'].issue_severity,
                        affected_services=remediation['initial_state'].affected_services,
                        service_states={
                            service: 'healthy' if success else 'degraded' 
                            for service in remediation['initial_state'].affected_services
                        },
                        previous_actions=remediation['initial_state'].previous_actions + [remediation['action'].action_id]
                    )
                    
                    experience = RemediationExperience(
                        initial_state=remediation['initial_state'],
                        action=remediation['action'],
                        next_state=next_state,
                        reward=1.0 if success else -0.5,
                        timestamp=remediation['start_time'],
                        success=success,
                        notes=f"Remediation {'succeeded' if success else 'failed'}"
                    )
                    
                    self.remediation_engine.record_experience(experience)
                    
                    self.logger.info(
                        f"Remediation {remediation_id} {remediation['status']} "
                        f"for issue {remediation['issue_id']}"
                    )
                    
                    completed_remediations.append(remediation_id)
        
        # Remove completed remediations
        for remediation_id in completed_remediations:
            del self.active_remediations[remediation_id]
    
    def get_active_issues(self) -> List[Dict[str, Any]]:
        """Get active issues.
        
        Returns:
            List of active issues
        """
        return self.active_issues
    
    def get_active_remediations(self) -> Dict[str, Dict[str, Any]]:
        """Get active remediations.
        
        Returns:
            Dictionary mapping remediation IDs to remediation info
        """
        return self.active_remediations
    
    def train_models(self) -> None:
        """Train all ML models."""
        # Train anomaly detection models
        if self.metrics_history:
            metrics_data = pd.concat(self.metrics_history.values())
            self.anomaly_engine.train_all({'metrics': metrics_data})
        
        if self.log_history:
            log_data = pd.concat(self.log_history.values())
            self.anomaly_engine.train_all({'logs': log_data})
        
        # Train remediation models
        self.remediation_engine.train_all()
        
        self.logger.info("Trained all ML models")
    
    def save_models(self, base_path: str) -> None:
        """Save all ML models to disk.
        
        Args:
            base_path: Base path for saving models
        """
        # Save anomaly detection models
        anomaly_path = f"{base_path}/anomaly"
        self.anomaly_engine.save_all(anomaly_path)
        
        # Save remediation models
        remediation_path = f"{base_path}/remediation"
        self.remediation_engine.save_all(remediation_path)
        
        self.logger.info(f"Saved all ML models to {base_path}")
    
    def load_models(self, base_path: str) -> None:
        """Load all ML models from disk.
        
        Args:
            base_path: Base path for loading models
        """
        # Load anomaly detection models
        anomaly_path = f"{base_path}/anomaly"
        self.anomaly_engine.load_all(anomaly_path)
        
        # Load remediation models
        remediation_path = f"{base_path}/remediation"
        self.remediation_engine.load_all(remediation_path)
        
        self.logger.info(f"Loaded all ML models from {base_path}")


def create_ml_integration(service_graph: ServiceGraph = None) -> MLIntegrationEngine:
    """Create an ML integration engine with default components.
    
    Args:
        service_graph: ServiceGraph instance
        
    Returns:
        Configured MLIntegrationEngine
    """
    # Create anomaly detection engine
    anomaly_engine = create_anomaly_engine()
    
    # Create remediation learning engine
    remediation_engine = create_remediation_engine()
    
    # Create and configure integration engine
    integration_engine = MLIntegrationEngine(
        service_graph=service_graph,
        anomaly_engine=anomaly_engine,
        remediation_engine=remediation_engine,
        config={
            'monitoring_interval': 60,  # seconds
            'training_interval': 3600,  # seconds
            'model_save_interval': 86400  # seconds
        }
    )
    
    return integration_engine