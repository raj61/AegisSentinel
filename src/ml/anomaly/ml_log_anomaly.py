"""
ML-based Log Anomaly Detection
==========================

This module provides functionality for detecting anomalies in logs using machine learning.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import re
import json
import pickle
import os
from pathlib import Path

# ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from pyod.models.autoencoder import AutoEncoder

# Import from existing modules
from src.anomaly.log_anomaly import LogPattern, AnomalyScore, LogAnomalyDetector

logger = logging.getLogger(__name__)

class LogFeatureExtractor:
    """
    Extract features from log lines for machine learning.
    
    This class provides functionality for extracting features from log lines
    that can be used for machine learning-based anomaly detection.
    """
    
    def __init__(self, max_features: int = 100, ngram_range: Tuple[int, int] = (1, 3)):
        """
        Initialize the log feature extractor.
        
        Args:
            max_features: Maximum number of features to extract
            ngram_range: Range of n-grams to extract
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            analyzer='word',
            stop_words='english'
        )
        self.scaler = StandardScaler(with_mean=False)  # Sparse matrices don't support centering
        self.pca = None
        self.n_components = None
        self.fitted = False
        
    def fit(self, log_lines: List[str], n_components: int = 10) -> None:
        """
        Fit the feature extractor to a set of log lines.
        
        Args:
            log_lines: List of log lines to fit to
            n_components: Number of PCA components to use
        """
        if not log_lines:
            logger.warning("No log lines provided for fitting")
            return
        
        # Fit the vectorizer
        X = self.vectorizer.fit_transform(log_lines)
        
        # Fit the scaler
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit PCA if there are enough samples
        if len(log_lines) > n_components:
            self.n_components = min(n_components, X_scaled.shape[1])
            self.pca = PCA(n_components=self.n_components)
            self.pca.fit(X_scaled.toarray())
        
        self.fitted = True
        logger.info(f"Fitted feature extractor on {len(log_lines)} log lines")
        
    def transform(self, log_lines: List[str]) -> np.ndarray:
        """
        Transform log lines into feature vectors.
        
        Args:
            log_lines: List of log lines to transform
            
        Returns:
            Feature vectors
        """
        if not self.fitted:
            logger.warning("Feature extractor not fitted")
            return np.array([])
        
        if not log_lines:
            return np.array([])
        
        # Transform with the vectorizer
        X = self.vectorizer.transform(log_lines)
        
        # Transform with the scaler
        X_scaled = self.scaler.transform(X)
        
        # Transform with PCA if available
        if self.pca is not None:
            return self.pca.transform(X_scaled.toarray())
        else:
            return X_scaled.toarray()
        
    def fit_transform(self, log_lines: List[str], n_components: int = 10) -> np.ndarray:
        """
        Fit the feature extractor and transform log lines.
        
        Args:
            log_lines: List of log lines to fit and transform
            n_components: Number of PCA components to use
            
        Returns:
            Feature vectors
        """
        self.fit(log_lines, n_components)
        return self.transform(log_lines)
    
    def save(self, path: str) -> None:
        """
        Save the feature extractor to a file.
        
        Args:
            path: Path to save to
        """
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'scaler': self.scaler,
                'pca': self.pca,
                'n_components': self.n_components,
                'fitted': self.fitted,
                'max_features': self.max_features,
                'ngram_range': self.ngram_range
            }, f)
        
    @classmethod
    def load(cls, path: str) -> 'LogFeatureExtractor':
        """
        Load a feature extractor from a file.
        
        Args:
            path: Path to load from
            
        Returns:
            Loaded feature extractor
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        extractor = cls(
            max_features=data['max_features'],
            ngram_range=data['ngram_range']
        )
        extractor.vectorizer = data['vectorizer']
        extractor.scaler = data['scaler']
        extractor.pca = data['pca']
        extractor.n_components = data['n_components']
        extractor.fitted = data['fitted']
        
        return extractor

class MLLogAnomalyDetector(LogAnomalyDetector):
    """
    Machine learning-based detector for anomalies in logs.
    
    This class extends the pattern-based LogAnomalyDetector with machine learning
    capabilities for detecting anomalies in logs.
    """
    
    def __init__(self):
        """Initialize the ML log anomaly detector."""
        super().__init__()
        self.feature_extractor = LogFeatureExtractor()
        self.models = {}
        self.training_data = defaultdict(list)
        self.max_training_samples = 10000
        self.min_training_samples = 100
        self.model_dir = Path("models/log_anomaly")
        
    def add_training_data(self, log_line: str, service: Optional[str] = None) -> None:
        """
        Add a log line to the training data.
        
        Args:
            log_line: Log line to add
            service: Service associated with the log line
        """
        service = service or "default"
        self.training_data[service].append(log_line)
        
        # Trim the training data if needed
        if len(self.training_data[service]) > self.max_training_samples:
            self.training_data[service] = self.training_data[service][-self.max_training_samples:]
    
    def train_models(self) -> None:
        """Train machine learning models on the collected training data."""
        for service, logs in self.training_data.items():
            if len(logs) < self.min_training_samples:
                logger.warning(f"Not enough training data for service {service}: "
                              f"{len(logs)}/{self.min_training_samples}")
                continue
            
            logger.info(f"Training models for service {service} with {len(logs)} samples")
            
            # Extract features
            X = self.feature_extractor.fit_transform(logs)
            
            # Train models
            models = {
                'isolation_forest': IsolationForest(contamination=0.05, random_state=42).fit(X),
                'local_outlier_factor': LocalOutlierFactor(contamination=0.05, novelty=True).fit(X),
                'dbscan': DBSCAN(eps=0.5, min_samples=5).fit(X)
            }
            
            # Train autoencoder if we have enough data
            if len(logs) >= 500:
                try:
                    autoencoder = AutoEncoder(
                        hidden_neurons=[X.shape[1], 16, 8, 16, X.shape[1]],
                        contamination=0.05,
                        epochs=100,
                        batch_size=32,
                        verbose=0
                    )
                    autoencoder.fit(X)
                    models['autoencoder'] = autoencoder
                except Exception as e:
                    logger.error(f"Error training autoencoder for service {service}: {e}")
            
            self.models[service] = models
            
            # Save the feature extractor
            os.makedirs(self.model_dir, exist_ok=True)
            self.feature_extractor.save(f"{self.model_dir}/{service}_feature_extractor.pkl")
            
            logger.info(f"Trained {len(models)} models for service {service}")
    
    def detect_anomalies_ml(self, log_line: str, service: Optional[str] = None) -> List[float]:
        """
        Detect anomalies in a log line using machine learning.
        
        Args:
            log_line: Log line to check
            service: Service associated with the log line
            
        Returns:
            List of anomaly scores from different models
        """
        service = service or "default"
        
        # Check if we have models for this service
        if service not in self.models:
            return []
        
        # Extract features
        X = self.feature_extractor.transform([log_line])
        
        if X.size == 0:
            return []
        
        # Get anomaly scores from each model
        scores = []
        
        # Isolation Forest
        if 'isolation_forest' in self.models[service]:
            # Convert to anomaly score (0-1)
            score = 1.0 - (self.models[service]['isolation_forest'].score_samples(X)[0] + 0.5)
            scores.append(min(1.0, max(0.0, score)))
        
        # Local Outlier Factor
        if 'local_outlier_factor' in self.models[service]:
            # Convert to anomaly score (0-1)
            score = -self.models[service]['local_outlier_factor'].score_samples(X)[0]
            score = 1.0 / (1.0 + np.exp(-score))  # Sigmoid to get 0-1
            scores.append(min(1.0, max(0.0, score)))
        
        # DBSCAN
        if 'dbscan' in self.models[service]:
            # Check if the point is an outlier (-1)
            labels = self.models[service]['dbscan'].fit_predict(X)
            score = 1.0 if labels[0] == -1 else 0.0
            scores.append(score)
        
        # Autoencoder
        if 'autoencoder' in self.models[service]:
            # Get anomaly score
            score = self.models[service]['autoencoder'].decision_function(X)[0]
            # Normalize to 0-1
            score = 1.0 / (1.0 + np.exp(-score))  # Sigmoid to get 0-1
            scores.append(min(1.0, max(0.0, score)))
        
        return scores
    
    def process_log(self, log_line: str, timestamp: Optional[datetime] = None) -> List[AnomalyScore]:
        """
        Process a log line and detect anomalies using both pattern-based and ML methods.
        
        Args:
            log_line: Log line to process
            timestamp: Timestamp of the log line (default: current time)
            
        Returns:
            List of anomaly scores for the log line
        """
        # First, use the pattern-based detection
        pattern_anomalies = super().process_log(log_line, timestamp)
        
        # Extract service from the log line if possible
        service = self._extract_service_from_log(log_line)
        
        # Add to training data
        self.add_training_data(log_line, service)
        
        # If we don't have enough training data yet, just return pattern-based anomalies
        if service not in self.models:
            return pattern_anomalies
        
        # Get ML anomaly scores
        ml_scores = self.detect_anomalies_ml(log_line, service)
        
        if not ml_scores:
            return pattern_anomalies
        
        # Combine the highest ML score with pattern-based anomalies
        max_ml_score = max(ml_scores)
        
        if max_ml_score > 0.7:  # Threshold for ML anomalies
            # Create a generic ML anomaly pattern
            ml_pattern = LogPattern(
                pattern=r".*",  # Match anything
                name="ml_anomaly",
                severity=3,  # Medium severity
                description="Machine learning detected anomaly",
                service=service,
                threshold=0.1,
                window_seconds=300
            )
            
            # Create an anomaly score
            ml_anomaly = AnomalyScore(
                pattern=ml_pattern,
                score=max_ml_score,
                timestamp=timestamp or datetime.now(),
                count=1,
                expected=0.1,
                log_samples=[log_line]
            )
            
            # Add to anomaly history
            self.anomaly_history.append(ml_anomaly)
            
            # Trim anomaly history if needed
            if len(self.anomaly_history) > self.max_anomaly_history:
                self.anomaly_history = self.anomaly_history[-self.max_anomaly_history:]
            
            # Return both pattern-based and ML anomalies
            return pattern_anomalies + [ml_anomaly]
        
        return pattern_anomalies
    
    def _extract_service_from_log(self, log_line: str) -> str:
        """
        Extract the service name from a log line.
        
        Args:
            log_line: Log line to extract from
            
        Returns:
            Service name or "default" if not found
        """
        # Try to extract service name from common log formats
        
        # Kubernetes log format: [service-name-hash]
        k8s_match = re.search(r'\[([a-zA-Z0-9-]+)-[a-z0-9]+-[a-z0-9]+\]', log_line)
        if k8s_match:
            return k8s_match.group(1)
        
        # JSON log format with service field
        try:
            log_json = json.loads(log_line)
            if 'service' in log_json:
                return log_json['service']
        except:
            pass
        
        # Common log format with service prefix
        service_match = re.search(r'^([a-zA-Z0-9-]+):', log_line)
        if service_match:
            return service_match.group(1)
        
        return "default"
    
    def save_models(self, directory: Optional[str] = None) -> None:
        """
        Save the trained models to disk.
        
        Args:
            directory: Directory to save to (default: self.model_dir)
        """
        directory = Path(directory) if directory else self.model_dir
        os.makedirs(directory, exist_ok=True)
        
        for service, models in self.models.items():
            for model_name, model in models.items():
                try:
                    with open(f"{directory}/{service}_{model_name}.pkl", 'wb') as f:
                        pickle.dump(model, f)
                except Exception as e:
                    logger.error(f"Error saving model {model_name} for service {service}: {e}")
        
        logger.info(f"Saved models to {directory}")
    
    def load_models(self, directory: Optional[str] = None) -> None:
        """
        Load trained models from disk.
        
        Args:
            directory: Directory to load from (default: self.model_dir)
        """
        directory = Path(directory) if directory else self.model_dir
        
        if not directory.exists():
            logger.warning(f"Model directory does not exist: {directory}")
            return
        
        # Load feature extractors first
        for path in directory.glob("*_feature_extractor.pkl"):
            try:
                service = path.name.replace("_feature_extractor.pkl", "")
                self.feature_extractor = LogFeatureExtractor.load(str(path))
                logger.info(f"Loaded feature extractor for service {service}")
            except Exception as e:
                logger.error(f"Error loading feature extractor from {path}: {e}")
        
        # Load models
        for path in directory.glob("*.pkl"):
            if "_feature_extractor.pkl" in path.name:
                continue
                
            try:
                parts = path.name.replace(".pkl", "").split("_")
                if len(parts) < 2:
                    continue
                    
                service = parts[0]
                model_name = "_".join(parts[1:])
                
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                
                if service not in self.models:
                    self.models[service] = {}
                
                self.models[service][model_name] = model
                logger.info(f"Loaded model {model_name} for service {service}")
            except Exception as e:
                logger.error(f"Error loading model from {path}: {e}")
        
        logger.info(f"Loaded models from {directory}")
    
    @classmethod
    def create_with_default_patterns(cls) -> 'MLLogAnomalyDetector':
        """
        Create an ML log anomaly detector with default patterns.
        
        Returns:
            MLLogAnomalyDetector instance with default patterns
        """
        detector = cls()
        
        # Add default patterns from the parent class
        default_patterns = LogAnomalyDetector.create_with_default_patterns().patterns
        detector.add_patterns(default_patterns)
        
        # Try to load pre-trained models
        try:
            detector.load_models()
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")
        
        return detector