"""
ML-based Root Cause Analysis
========================

This module provides functionality for root cause analysis using machine learning.
"""

import logging
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Set, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import pickle
import os
from pathlib import Path
from dataclasses import dataclass

# ML libraries
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Import from existing modules
from src.graph import ServiceGraph
from src.detection import Issue, IssueType
from src.anomaly import AnomalyScore
from src.ml.anomaly.metric_anomaly import MetricAnomalyScore

logger = logging.getLogger(__name__)

@dataclass
class RootCauseScore:
    """
    Represents a root cause score for a node in the service graph.
    
    Attributes:
        node_id: ID of the node
        score: Root cause score (0-1, with 1 being the most likely root cause)
        timestamp: Timestamp of the score
        issues: List of issues associated with the node
        anomalies: List of anomalies associated with the node
        metric_anomalies: List of metric anomalies associated with the node
        propagation_path: Path of propagation from the root cause
    """
    node_id: str
    score: float
    timestamp: datetime
    issues: List[Issue] = None
    anomalies: List[AnomalyScore] = None
    metric_anomalies: List[MetricAnomalyScore] = None
    propagation_path: List[str] = None
    
    def __post_init__(self):
        """Initialize default values for optional fields."""
        if self.issues is None:
            self.issues = []
        if self.anomalies is None:
            self.anomalies = []
        if self.metric_anomalies is None:
            self.metric_anomalies = []
        if self.propagation_path is None:
            self.propagation_path = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the root cause score to a dictionary."""
        return {
            'node_id': self.node_id,
            'score': self.score,
            'timestamp': self.timestamp.isoformat(),
            'issues': [issue.type.value for issue in self.issues],
            'anomalies': [anomaly.pattern.name for anomaly in self.anomalies],
            'metric_anomalies': [anomaly.metric.metric_name for anomaly in self.metric_anomalies],
            'propagation_path': self.propagation_path
        }

class RootCauseAnalyzer:
    """
    Analyzer for root causes of issues and anomalies in the service graph.
    
    This class provides functionality for identifying the root causes of issues
    and anomalies in the service graph using graph algorithms and machine learning.
    """
    
    def __init__(self, service_graph: ServiceGraph):
        """
        Initialize the root cause analyzer.
        
        Args:
            service_graph: Service graph to analyze
        """
        self.service_graph = service_graph
        self.node_health_scores = {}
        self.node_anomalies = defaultdict(list)
        self.node_issues = defaultdict(list)
        self.node_metric_anomalies = defaultdict(list)
        self.root_cause_history = []
        self.max_history_size = 1000
        self.model_dir = Path("models/root_cause")
        
        # ML model for root cause prediction
        self.model = None
        self.feature_names = []
        self.training_data = []
        self.training_labels = []
    
    def update_node_health(self, node_id: str, health_score: float) -> None:
        """
        Update the health score for a node.
        
        Args:
            node_id: ID of the node
            health_score: Health score (0-1, with 1 being healthy)
        """
        self.node_health_scores[node_id] = health_score
    
    def add_node_anomaly(self, node_id: str, anomaly: AnomalyScore) -> None:
        """
        Add an anomaly to a node.
        
        Args:
            node_id: ID of the node
            anomaly: Anomaly score
        """
        self.node_anomalies[node_id].append(anomaly)
        
        # Trim the anomalies if needed
        if len(self.node_anomalies[node_id]) > 100:
            self.node_anomalies[node_id] = self.node_anomalies[node_id][-100:]
    
    def add_node_issue(self, node_id: str, issue: Issue) -> None:
        """
        Add an issue to a node.
        
        Args:
            node_id: ID of the node
            issue: Issue
        """
        self.node_issues[node_id].append(issue)
        
        # Trim the issues if needed
        if len(self.node_issues[node_id]) > 100:
            self.node_issues[node_id] = self.node_issues[node_id][-100:]
    
    def add_node_metric_anomaly(self, node_id: str, anomaly: MetricAnomalyScore) -> None:
        """
        Add a metric anomaly to a node.
        
        Args:
            node_id: ID of the node
            anomaly: Metric anomaly score
        """
        self.node_metric_anomalies[node_id].append(anomaly)
        
        # Trim the metric anomalies if needed
        if len(self.node_metric_anomalies[node_id]) > 100:
            self.node_metric_anomalies[node_id] = self.node_metric_anomalies[node_id][-100:]
    
    def analyze_root_cause(self) -> List[RootCauseScore]:
        """
        Analyze the root causes of issues and anomalies in the service graph.
        
        Returns:
            List of root cause scores
        """
        # First, use graph-based analysis
        graph_scores = self._analyze_graph_based()
        
        # Then, use ML-based analysis if a model is available
        ml_scores = self._analyze_ml_based() if self.model else []
        
        # Combine the scores
        combined_scores = self._combine_scores(graph_scores, ml_scores)
        
        # Add to history
        self.root_cause_history.extend(combined_scores)
        
        # Trim history if needed
        if len(self.root_cause_history) > self.max_history_size:
            self.root_cause_history = self.root_cause_history[-self.max_history_size:]
        
        return combined_scores
    
    def _analyze_graph_based(self) -> List[RootCauseScore]:
        """
        Analyze root causes using graph-based algorithms.
        
        Returns:
            List of root cause scores
        """
        scores = []
        
        # Get nodes with anomalies or issues
        problem_nodes = set()
        for node_id, anomalies in self.node_anomalies.items():
            if anomalies:
                problem_nodes.add(node_id)
        
        for node_id, issues in self.node_issues.items():
            if issues:
                problem_nodes.add(node_id)
        
        for node_id, anomalies in self.node_metric_anomalies.items():
            if anomalies:
                problem_nodes.add(node_id)
        
        if not problem_nodes:
            return scores
        
        # Create a subgraph of the problem nodes and their neighbors
        subgraph_nodes = set(problem_nodes)
        for node_id in problem_nodes:
            # Add predecessors (upstream services)
            subgraph_nodes.update(self.service_graph.graph.predecessors(node_id))
            
            # Add successors (downstream services)
            subgraph_nodes.update(self.service_graph.graph.successors(node_id))
        
        # Create a subgraph
        subgraph = self.service_graph.graph.subgraph(subgraph_nodes)
        
        # Calculate node scores based on various factors
        node_scores = {}
        
        for node_id in subgraph_nodes:
            # Start with a base score
            score = 0.0
            
            # Factor 1: Health score (invert it since lower health means higher likelihood of being root cause)
            health_score = self.node_health_scores.get(node_id, 1.0)
            score += (1.0 - health_score) * 0.3
            
            # Factor 2: Number and severity of anomalies
            anomalies = self.node_anomalies.get(node_id, [])
            if anomalies:
                anomaly_score = sum(a.score * (a.pattern.severity / 5.0) for a in anomalies) / len(anomalies)
                score += anomaly_score * 0.2
            
            # Factor 3: Number and severity of issues
            issues = self.node_issues.get(node_id, [])
            if issues:
                issue_score = sum(i.severity / 5.0 for i in issues) / len(issues)
                score += issue_score * 0.2
            
            # Factor 4: Number and severity of metric anomalies
            metric_anomalies = self.node_metric_anomalies.get(node_id, [])
            if metric_anomalies:
                metric_score = sum(a.score for a in metric_anomalies) / len(metric_anomalies)
                score += metric_score * 0.2
            
            # Factor 5: Topology - upstream services are more likely to be root causes
            # Count how many problem nodes this node affects
            affected_nodes = set()
            for problem_node in problem_nodes:
                if node_id != problem_node:
                    try:
                        # Check if there's a path from this node to the problem node
                        path = nx.shortest_path(subgraph, node_id, problem_node)
                        if path:
                            affected_nodes.add(problem_node)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass
            
            # Score based on how many problem nodes this node affects
            if problem_nodes:
                topology_score = len(affected_nodes) / len(problem_nodes)
                score += topology_score * 0.1
            
            # Store the score
            node_scores[node_id] = min(1.0, score)
        
        # Create RootCauseScore objects for nodes with non-zero scores
        timestamp = datetime.now()
        for node_id, score in node_scores.items():
            if score > 0:
                # Find propagation path
                propagation_path = []
                for problem_node in problem_nodes:
                    if node_id != problem_node:
                        try:
                            path = nx.shortest_path(subgraph, node_id, problem_node)
                            if path and len(path) > 1:
                                propagation_path.append(path)
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            pass
                
                # Flatten and deduplicate the propagation path
                flat_path = []
                for path in propagation_path:
                    flat_path.extend(path)
                flat_path = list(dict.fromkeys(flat_path))
                
                # Create the score object
                root_cause = RootCauseScore(
                    node_id=node_id,
                    score=score,
                    timestamp=timestamp,
                    issues=self.node_issues.get(node_id, []),
                    anomalies=self.node_anomalies.get(node_id, []),
                    metric_anomalies=self.node_metric_anomalies.get(node_id, []),
                    propagation_path=flat_path
                )
                
                scores.append(root_cause)
        
        # Sort by score in descending order
        scores.sort(key=lambda x: x.score, reverse=True)
        
        return scores
    
    def _analyze_ml_based(self) -> List[RootCauseScore]:
        """
        Analyze root causes using machine learning.
        
        Returns:
            List of root cause scores
        """
        if not self.model:
            return []
        
        scores = []
        timestamp = datetime.now()
        
        # Get all nodes
        all_nodes = set(self.service_graph.get_nodes())
        
        # Extract features for each node
        node_features = {}
        for node_id in all_nodes:
            features = self._extract_node_features(node_id)
            if features:
                node_features[node_id] = features
        
        if not node_features:
            return scores
        
        # Create a feature matrix
        X = []
        nodes = []
        for node_id, features in node_features.items():
            # Ensure the features match the expected order
            feature_vector = [features.get(name, 0.0) for name in self.feature_names]
            X.append(feature_vector)
            nodes.append(node_id)
        
        # Convert to numpy array
        X = np.array(X)
        
        # Predict root cause probabilities
        try:
            probabilities = self.model.predict_proba(X)[:, 1]  # Probability of being a root cause
            
            # Create RootCauseScore objects for nodes with non-zero probabilities
            for i, node_id in enumerate(nodes):
                prob = probabilities[i]
                if prob > 0.1:  # Threshold for considering a node as a potential root cause
                    # Create the score object
                    root_cause = RootCauseScore(
                        node_id=node_id,
                        score=prob,
                        timestamp=timestamp,
                        issues=self.node_issues.get(node_id, []),
                        anomalies=self.node_anomalies.get(node_id, []),
                        metric_anomalies=self.node_metric_anomalies.get(node_id, [])
                    )
                    
                    scores.append(root_cause)
            
            # Sort by score in descending order
            scores.sort(key=lambda x: x.score, reverse=True)
            
        except Exception as e:
            logger.error(f"Error predicting root causes with ML model: {e}")
        
        return scores
    
    def _combine_scores(self, graph_scores: List[RootCauseScore], 
                       ml_scores: List[RootCauseScore]) -> List[RootCauseScore]:
        """
        Combine scores from different analysis methods.
        
        Args:
            graph_scores: Scores from graph-based analysis
            ml_scores: Scores from ML-based analysis
            
        Returns:
            Combined scores
        """
        if not ml_scores:
            return graph_scores
        
        if not graph_scores:
            return ml_scores
        
        # Create a map of node IDs to scores
        combined_map = {}
        
        # Add graph scores
        for score in graph_scores:
            combined_map[score.node_id] = {
                'graph_score': score.score,
                'issues': score.issues,
                'anomalies': score.anomalies,
                'metric_anomalies': score.metric_anomalies,
                'propagation_path': score.propagation_path
            }
        
        # Add ML scores
        for score in ml_scores:
            if score.node_id in combined_map:
                combined_map[score.node_id]['ml_score'] = score.score
            else:
                combined_map[score.node_id] = {
                    'graph_score': 0.0,
                    'ml_score': score.score,
                    'issues': score.issues,
                    'anomalies': score.anomalies,
                    'metric_anomalies': score.metric_anomalies,
                    'propagation_path': score.propagation_path
                }
        
        # Combine scores (weighted average)
        combined_scores = []
        timestamp = datetime.now()
        
        for node_id, data in combined_map.items():
            graph_score = data.get('graph_score', 0.0)
            ml_score = data.get('ml_score', 0.0)
            
            # Weighted average (give more weight to ML if available)
            if ml_score > 0:
                combined_score = 0.4 * graph_score + 0.6 * ml_score
            else:
                combined_score = graph_score
            
            # Create a combined score object
            root_cause = RootCauseScore(
                node_id=node_id,
                score=combined_score,
                timestamp=timestamp,
                issues=data['issues'],
                anomalies=data['anomalies'],
                metric_anomalies=data['metric_anomalies'],
                propagation_path=data['propagation_path']
            )
            
            combined_scores.append(root_cause)
        
        # Sort by score in descending order
        combined_scores.sort(key=lambda x: x.score, reverse=True)
        
        return combined_scores
    
    def _extract_node_features(self, node_id: str) -> Dict[str, float]:
        """
        Extract features for a node for ML-based analysis.
        
        Args:
            node_id: ID of the node
            
        Returns:
            Dictionary of feature name to value
        """
        features = {}
        
        # Feature 1: Health score
        features['health_score'] = self.node_health_scores.get(node_id, 1.0)
        
        # Feature 2: Number of anomalies
        anomalies = self.node_anomalies.get(node_id, [])
        features['anomaly_count'] = len(anomalies)
        
        # Feature 3: Average anomaly score
        if anomalies:
            features['avg_anomaly_score'] = sum(a.score for a in anomalies) / len(anomalies)
        else:
            features['avg_anomaly_score'] = 0.0
        
        # Feature 4: Number of issues
        issues = self.node_issues.get(node_id, [])
        features['issue_count'] = len(issues)
        
        # Feature 5: Average issue severity
        if issues:
            features['avg_issue_severity'] = sum(i.severity for i in issues) / len(issues)
        else:
            features['avg_issue_severity'] = 0.0
        
        # Feature 6: Number of metric anomalies
        metric_anomalies = self.node_metric_anomalies.get(node_id, [])
        features['metric_anomaly_count'] = len(metric_anomalies)
        
        # Feature 7: Average metric anomaly score
        if metric_anomalies:
            features['avg_metric_anomaly_score'] = sum(a.score for a in metric_anomalies) / len(metric_anomalies)
        else:
            features['avg_metric_anomaly_score'] = 0.0
        
        # Feature 8: Node degree (in + out)
        in_degree = len(list(self.service_graph.graph.predecessors(node_id)))
        out_degree = len(list(self.service_graph.graph.successors(node_id)))
        features['node_degree'] = in_degree + out_degree
        
        # Feature 9: In-degree (number of incoming connections)
        features['in_degree'] = in_degree
        
        # Feature 10: Out-degree (number of outgoing connections)
        features['out_degree'] = out_degree
        
        # Feature 11: Ratio of in-degree to out-degree
        if out_degree > 0:
            features['in_out_ratio'] = in_degree / out_degree
        else:
            features['in_out_ratio'] = in_degree if in_degree > 0 else 0.0
        
        return features
    
    def train_model(self, labeled_data: List[Tuple[str, bool, datetime]]) -> None:
        """
        Train a machine learning model for root cause prediction.
        
        Args:
            labeled_data: List of (node_id, is_root_cause, timestamp) tuples
        """
        if not labeled_data:
            logger.warning("No labeled data provided for training")
            return
        
        # Extract features for each labeled node
        X = []
        y = []
        
        for node_id, is_root_cause, timestamp in labeled_data:
            # Get the node features at the time of the incident
            features = self._extract_node_features(node_id)
            if features:
                X.append(list(features.values()))
                y.append(1 if is_root_cause else 0)
                
                # Store feature names if not already stored
                if not self.feature_names:
                    self.feature_names = list(features.keys())
        
        if not X:
            logger.warning("No features extracted for labeled data")
            return
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train a random forest classifier
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        logger.info(f"Trained root cause prediction model: "
                   f"precision={precision:.2f}, recall={recall:.2f}, f1={f1:.2f}")
        
        # Store the model
        self.model = model
        
        # Save the model
        os.makedirs(self.model_dir, exist_ok=True)
        with open(f"{self.model_dir}/root_cause_model.pkl", 'wb') as f:
            pickle.dump({
                'model': model,
                'feature_names': self.feature_names
            }, f)
    
    def load_model(self, path: Optional[str] = None) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the model file (default: self.model_dir/root_cause_model.pkl)
            
        Returns:
            True if the model was loaded successfully, False otherwise
        """
        path = path or f"{self.model_dir}/root_cause_model.pkl"
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.feature_names = data['feature_names']
                
                logger.info(f"Loaded root cause prediction model from {path}")
                return True
                
        except Exception as e:
            logger.error(f"Error loading root cause prediction model: {e}")
            return False
