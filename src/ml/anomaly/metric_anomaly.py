"""
Metric Anomaly Detection
====================

This module provides functionality for detecting anomalies in metrics using machine learning.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import pickle
import os
from pathlib import Path

# ML libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from pyod.models.autoencoder import AutoEncoder
from pyod.models.vae import VAE
from pyod.models.cblof import CBLOF

logger = logging.getLogger(__name__)

class MetricType:
    """Types of metrics that can be monitored."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CUSTOM = "custom"

@dataclass
class MetricDataPoint:
    """
    Represents a single metric data point.
    
    Attributes:
        timestamp: Timestamp of the data point
        value: Value of the metric
        metric_name: Name of the metric
        metric_type: Type of the metric
        service: Service associated with the metric
        dimensions: Additional dimensions (tags) for the metric
    """
    timestamp: datetime
    value: float
    metric_name: str
    metric_type: str
    service: Optional[str] = None
    dimensions: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the data point to a dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'value': self.value,
            'metric_name': self.metric_name,
            'metric_type': self.metric_type,
            'service': self.service,
            'dimensions': self.dimensions or {}
        }

@dataclass
class MetricAnomalyScore:
    """
    Represents an anomaly score for a metric.
    
    Attributes:
        metric: Metric data point that triggered the anomaly
        score: Anomaly score (0-1, with 1 being the most anomalous)
        timestamp: Timestamp of the anomaly
        expected_value: Expected value of the metric
        model_name: Name of the model that detected the anomaly
    """
    metric: MetricDataPoint
    score: float
    timestamp: datetime
    expected_value: Optional[float] = None
    model_name: Optional[str] = None
    
    @property
    def is_anomaly(self) -> bool:
        """Check if the score indicates an anomaly."""
        return self.score > 0.7
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the anomaly score to a dictionary."""
        return {
            'metric': self.metric.to_dict(),
            'score': self.score,
            'timestamp': self.timestamp.isoformat(),
            'expected_value': self.expected_value,
            'model_name': self.model_name,
            'is_anomaly': self.is_anomaly
        }

class MetricAnomalyDetector:
    """
    Detector for anomalies in metrics using machine learning.
    
    This class provides functionality for detecting anomalies in metrics
    using various machine learning algorithms.
    """
    
    def __init__(self):
        """Initialize the metric anomaly detector."""
        self.models = {}
        self.scalers = {}
        self.training_data = defaultdict(list)
        self.anomaly_history = []
        self.max_history_size = 10000
        self.max_anomaly_history = 1000
        self.model_dir = Path("models/metric_anomaly")
        
    def add_metric(self, metric: MetricDataPoint) -> None:
        """
        Add a metric data point for training and anomaly detection.
        
        Args:
            metric: Metric data point to add
        """
        # Create a key for this metric
        key = self._get_metric_key(metric)
        
        # Add to training data
        self.training_data[key].append((metric.timestamp, metric.value))
        
        # Trim the training data if needed
        if len(self.training_data[key]) > self.max_history_size:
            self.training_data[key] = self.training_data[key][-self.max_history_size:]
        
        # Detect anomalies
        anomaly = self.detect_anomaly(metric)
        if anomaly:
            self.anomaly_history.append(anomaly)
            
            # Trim anomaly history if needed
            if len(self.anomaly_history) > self.max_anomaly_history:
                self.anomaly_history = self.anomaly_history[-self.max_anomaly_history:]
            
            return anomaly
        
        return None
    
    def _get_metric_key(self, metric: MetricDataPoint) -> str:
        """
        Get a unique key for a metric.
        
        Args:
            metric: Metric data point
            
        Returns:
            Unique key for the metric
        """
        # Create a key based on metric name, type, and service
        key = f"{metric.metric_name}_{metric.metric_type}"
        if metric.service:
            key += f"_{metric.service}"
        
        # Add dimensions to the key if present
        if metric.dimensions:
            for k, v in sorted(metric.dimensions.items()):
                key += f"_{k}_{v}"
        
        return key
    
    def train_models(self, min_samples: int = 100) -> None:
        """
        Train machine learning models on the collected metrics.
        
        Args:
            min_samples: Minimum number of samples required for training
        """
        for key, data in self.training_data.items():
            if len(data) < min_samples:
                logger.warning(f"Not enough training data for metric {key}: "
                              f"{len(data)}/{min_samples}")
                continue
            
            logger.info(f"Training models for metric {key} with {len(data)} samples")
            
            # Extract timestamps and values
            timestamps, values = zip(*data)
            
            # Convert to numpy array
            X = np.array(values).reshape(-1, 1)
            
            # Create and fit a scaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[key] = scaler
            
            # Train models
            models = {
                'isolation_forest': IsolationForest(contamination=0.05, random_state=42).fit(X_scaled),
                'local_outlier_factor': LocalOutlierFactor(contamination=0.05, novelty=True).fit(X_scaled)
            }
            
            # Train more complex models if we have enough data
            if len(data) >= 500:
                try:
                    # CBLOF (Cluster-Based Local Outlier Factor)
                    cblof = CBLOF(
                        n_clusters=3,
                        contamination=0.05,
                        check_estimator=False,
                        random_state=42
                    )
                    cblof.fit(X_scaled)
                    models['cblof'] = cblof
                    
                    # Autoencoder
                    autoencoder = AutoEncoder(
                        hidden_neurons=[1, 4, 2, 4, 1],
                        contamination=0.05,
                        epochs=100,
                        batch_size=32,
                        verbose=0
                    )
                    autoencoder.fit(X_scaled)
                    models['autoencoder'] = autoencoder
                    
                    # VAE (Variational Autoencoder)
                    if len(data) >= 1000:
                        vae = VAE(
                            encoder_neurons=[1, 4, 2],
                            decoder_neurons=[2, 4, 1],
                            contamination=0.05,
                            epochs=100,
                            batch_size=32,
                            verbose=0
                        )
                        vae.fit(X_scaled)
                        models['vae'] = vae
                        
                except Exception as e:
                    logger.error(f"Error training complex models for metric {key}: {e}")
            
            self.models[key] = models
            
            logger.info(f"Trained {len(models)} models for metric {key}")
    
    def detect_anomaly(self, metric: MetricDataPoint) -> Optional[MetricAnomalyScore]:
        """
        Detect if a metric data point is anomalous.
        
        Args:
            metric: Metric data point to check
            
        Returns:
            Anomaly score if anomalous, None otherwise
        """
        key = self._get_metric_key(metric)
        
        # Check if we have models for this metric
        if key not in self.models:
            return None
        
        # Get the value and scale it
        X = np.array([[metric.value]])
        X_scaled = self.scalers[key].transform(X)
        
        # Get anomaly scores from each model
        scores = []
        model_names = []
        
        for model_name, model in self.models[key].items():
            try:
                if model_name == 'isolation_forest':
                    # Convert to anomaly score (0-1)
                    score = 1.0 - (model.score_samples(X_scaled)[0] + 0.5)
                    scores.append(min(1.0, max(0.0, score)))
                    model_names.append(model_name)
                    
                elif model_name == 'local_outlier_factor':
                    # Convert to anomaly score (0-1)
                    score = -model.score_samples(X_scaled)[0]
                    score = 1.0 / (1.0 + np.exp(-score))  # Sigmoid to get 0-1
                    scores.append(min(1.0, max(0.0, score)))
                    model_names.append(model_name)
                    
                else:
                    # For pyod models
                    score = model.decision_function(X_scaled)[0]
                    # Normalize to 0-1
                    score = 1.0 / (1.0 + np.exp(-score))  # Sigmoid to get 0-1
                    scores.append(min(1.0, max(0.0, score)))
                    model_names.append(model_name)
                    
            except Exception as e:
                logger.error(f"Error getting anomaly score from {model_name} for metric {key}: {e}")
        
        if not scores:
            return None
        
        # Get the highest score and corresponding model
        max_score = max(scores)
        max_model = model_names[scores.index(max_score)]
        
        # Only return an anomaly if the score is high enough
        if max_score > 0.7:
            # Calculate expected value (simple average of recent values)
            recent_values = [v for _, v in self.training_data[key][-10:]]
            expected_value = sum(recent_values) / len(recent_values) if recent_values else None
            
            return MetricAnomalyScore(
                metric=metric,
                score=max_score,
                timestamp=datetime.now(),
                expected_value=expected_value,
                model_name=max_model
            )
        
        return None
    
    def get_anomalies(self, since: Optional[datetime] = None,
                     service: Optional[str] = None,
                     metric_type: Optional[str] = None,
                     min_score: float = 0.7) -> List[MetricAnomalyScore]:
        """
        Get anomalies detected since a given timestamp.
        
        Args:
            since: Timestamp to get anomalies since (default: all anomalies)
            service: Filter by service (default: all services)
            metric_type: Filter by metric type (default: all types)
            min_score: Minimum anomaly score (default: 0.7)
            
        Returns:
            List of anomaly scores
        """
        filtered_anomalies = []
        
        for anomaly in self.anomaly_history:
            # Filter by timestamp
            if since is not None and anomaly.timestamp < since:
                continue
            
            # Filter by service
            if service is not None and anomaly.metric.service != service:
                continue
            
            # Filter by metric type
            if metric_type is not None and anomaly.metric.metric_type != metric_type:
                continue
            
            # Filter by score
            if anomaly.score < min_score:
                continue
            
            filtered_anomalies.append(anomaly)
        
        return filtered_anomalies
    
    def save_models(self, directory: Optional[str] = None) -> None:
        """
        Save the trained models to disk.
        
        Args:
            directory: Directory to save to (default: self.model_dir)
        """
        directory = Path(directory) if directory else self.model_dir
        os.makedirs(directory, exist_ok=True)
        
        # Save models
        for key, models in self.models.items():
            for model_name, model in models.items():
                try:
                    with open(f"{directory}/{key}_{model_name}.pkl", 'wb') as f:
                        pickle.dump(model, f)
                except Exception as e:
                    logger.error(f"Error saving model {model_name} for metric {key}: {e}")
        
        # Save scalers
        for key, scaler in self.scalers.items():
            try:
                with open(f"{directory}/{key}_scaler.pkl", 'wb') as f:
                    pickle.dump(scaler, f)
            except Exception as e:
                logger.error(f"Error saving scaler for metric {key}: {e}")
        
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
        
        # Load scalers
        for path in directory.glob("*_scaler.pkl"):
            try:
                key = path.name.replace("_scaler.pkl", "")
                with open(path, 'rb') as f:
                    self.scalers[key] = pickle.load(f)
                logger.info(f"Loaded scaler for metric {key}")
            except Exception as e:
                logger.error(f"Error loading scaler from {path}: {e}")
        
        # Load models
        for path in directory.glob("*.pkl"):
            if "_scaler.pkl" in path.name:
                continue
                
            try:
                parts = path.name.replace(".pkl", "").split("_")
                if len(parts) < 2:
                    continue
                    
                model_name = parts[-1]
                key = "_".join(parts[:-1])
                
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                
                if key not in self.models:
                    self.models[key] = {}
                
                self.models[key][model_name] = model
                logger.info(f"Loaded model {model_name} for metric {key}")
            except Exception as e:
                logger.error(f"Error loading model from {path}: {e}")
        
        logger.info(f"Loaded models from {directory}")

class MetricCollector:
    """
    Collector for metrics from various sources.
    
    This class provides functionality for collecting metrics from various sources
    and passing them to the anomaly detector.
    """
    
    def __init__(self, anomaly_detector: MetricAnomalyDetector):
        """
        Initialize the metric collector.
        
        Args:
            anomaly_detector: Anomaly detector to use
        """
        self.anomaly_detector = anomaly_detector
        self.callbacks = []
    
    def add_callback(self, callback):
        """
        Add a callback function to be called for each anomaly.
        
        Args:
            callback: Function to call with the anomaly
        """
        self.callbacks.append(callback)
    
    def process_metric(self, metric: MetricDataPoint) -> Optional[MetricAnomalyScore]:
        """
        Process a metric data point and check for anomalies.
        
        Args:
            metric: Metric data point to process
            
        Returns:
            Anomaly score if anomalous, None otherwise
        """
        anomaly = self.anomaly_detector.add_metric(metric)
        
        if anomaly:
            # Call callbacks
            for callback in self.callbacks:
                try:
                    callback(anomaly)
                except Exception as e:
                    logger.error(f"Error in metric anomaly callback: {e}")
        
        return anomaly

class PrometheusMetricCollector(MetricCollector):
    """
    Collector for metrics from Prometheus.
    
    This class provides functionality for collecting metrics from Prometheus
    and passing them to the anomaly detector.
    """
    
    def __init__(self, anomaly_detector: MetricAnomalyDetector, 
                prometheus_url: str = "http://localhost:9090"):
        """
        Initialize the Prometheus metric collector.
        
        Args:
            anomaly_detector: Anomaly detector to use
            prometheus_url: URL of the Prometheus server
        """
        super().__init__(anomaly_detector)
        self.prometheus_url = prometheus_url
        self.running = False
        self.thread = None
        self.query_interval = 60  # seconds
        
        # Import prometheus client here to avoid dependency issues
        try:
            from prometheus_api_client import PrometheusConnect
            self.prometheus = PrometheusConnect(url=prometheus_url)
        except ImportError:
            logger.error("prometheus_api_client not installed. "
                        "Please install it with: pip install prometheus-api-client")
            self.prometheus = None
    
    def start(self, metrics: List[Dict[str, Any]] = None) -> None:
        """
        Start collecting metrics from Prometheus.
        
        Args:
            metrics: List of metrics to collect, each with:
                - query: Prometheus query
                - metric_name: Name of the metric
                - metric_type: Type of the metric
                - service: Service associated with the metric (optional)
        """
        if self.running:
            return
        
        if not self.prometheus:
            logger.error("Prometheus client not initialized")
            return
        
        self.metrics = metrics or []
        self.running = True
        self.thread = threading.Thread(target=self._collect_metrics)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self) -> None:
        """Stop collecting metrics from Prometheus."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
    
    def _collect_metrics(self) -> None:
        """Collect metrics from Prometheus."""
        while self.running:
            try:
                for metric_config in self.metrics:
                    query = metric_config['query']
                    metric_name = metric_config['metric_name']
                    metric_type = metric_config['metric_type']
                    service = metric_config.get('service')
                    
                    # Execute the query
                    result = self.prometheus.custom_query(query)
                    
                    # Process the results
                    for item in result:
                        try:
                            # Extract value and timestamp
                            value = float(item['value'][1])
                            timestamp = datetime.fromtimestamp(item['value'][0])
                            
                            # Extract dimensions (labels)
                            dimensions = {k: v for k, v in item['metric'].items() 
                                         if k not in ['__name__', 'job', 'instance']}
                            
                            # Extract service from labels if not provided
                            if not service and 'service' in item['metric']:
                                service = item['metric']['service']
                            
                            # Create metric data point
                            metric = MetricDataPoint(
                                timestamp=timestamp,
                                value=value,
                                metric_name=metric_name,
                                metric_type=metric_type,
                                service=service,
                                dimensions=dimensions
                            )
                            
                            # Process the metric
                            self.process_metric(metric)
                            
                        except Exception as e:
                            logger.error(f"Error processing Prometheus metric: {e}")
                
            except Exception as e:
                logger.error(f"Error collecting metrics from Prometheus: {e}")
            
            # Sleep until next collection
            time.sleep(self.query_interval)