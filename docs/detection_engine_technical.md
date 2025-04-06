# Aegis Sentinel: Detection Engine Technical Deep Dive

## Introduction

The Detection Engine is the core intelligence component of Aegis Sentinel, responsible for identifying anomalies, predicting potential failures, and pinpointing the root causes of issues across complex distributed systems. This document provides a technical deep dive into the architecture, algorithms, and implementation details of the Detection Engine, suitable for principal architects and senior developers.

## Architecture Overview

![Detection Engine Architecture](https://via.placeholder.com/900x500?text=Detection+Engine+Architecture)

The Detection Engine is built on a layered architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                     Detection Engine                         │
├─────────────┬─────────────┬────────────────┬────────────────┤
│ Time Series │    Log      │   Topology     │  Correlation   │
│  Analyzer   │  Analyzer   │   Analyzer     │    Engine      │
├─────────────┴─────────────┴────────────────┴────────────────┤
│                    Feature Extraction Layer                  │
├─────────────────────────────────────────────────────────────┤
│                      Data Access Layer                       │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Data Access Layer**: Interfaces with various data sources (metrics, logs, traces, events)
2. **Feature Extraction Layer**: Transforms raw data into ML-ready features
3. **Analysis Modules**:
   - Time Series Analyzer: Detects anomalies in metric data
   - Log Analyzer: Identifies unusual patterns in log data
   - Topology Analyzer: Detects anomalies in service relationships
4. **Correlation Engine**: Combines signals from multiple analyzers to identify complex issues

## Advanced Detection Techniques

### 1. Multi-Model Ensemble for Time Series Anomaly Detection

The Detection Engine employs a sophisticated ensemble approach that combines multiple models:

```python
class TimeSeriesAnomalyDetector:
    def __init__(self):
        # Statistical models
        self.statistical_models = {
            'z_score': ZScoreDetector(window_size=100, threshold=3.0),
            'mad': MedianAbsoluteDeviationDetector(window_size=100, threshold=3.0),
            'iqr': InterquartileRangeDetector(window_size=100, threshold=1.5)
        }
        
        # Machine learning models
        self.ml_models = {
            'isolation_forest': IsolationForestDetector(n_estimators=100, contamination=0.01),
            'one_class_svm': OneClassSVMDetector(nu=0.01, kernel='rbf'),
            'lof': LocalOutlierFactorDetector(n_neighbors=20, contamination=0.01)
        }
        
        # Deep learning models
        self.dl_models = {
            'autoencoder': LSTMAutoencoderDetector(
                sequence_length=30,
                hidden_size=64,
                num_layers=2,
                learning_rate=0.001
            ),
            'prophet': ProphetDetector(
                changepoint_prior_scale=0.05,
                seasonality_mode='multiplicative'
            )
        }
        
        # Model weights learned from historical performance
        self.model_weights = self._initialize_weights()
        
        # Adaptive thresholding
        self.threshold_adapter = AdaptiveThresholdManager()
    
    def detect_anomalies(self, time_series, metadata=None):
        """
        Detect anomalies in a time series using ensemble approach
        """
        results = {}
        explanations = {}
        
        # Run all models and collect results
        for name, model in {**self.statistical_models, **self.ml_models, **self.dl_models}.items():
            results[name] = model.detect(time_series)
            explanations[name] = model.explain()
        
        # Combine results using weighted voting
        ensemble_scores = self._combine_scores(results)
        
        # Apply adaptive thresholding
        threshold = self.threshold_adapter.get_threshold(time_series, ensemble_scores, metadata)
        
        # Generate final anomalies with explanations
        anomalies = []
        for i, score in enumerate(ensemble_scores):
            if score > threshold:
                anomaly = {
                    'index': i,
                    'timestamp': metadata.get('timestamps', [None])[i] if metadata else None,
                    'value': time_series[i],
                    'score': score,
                    'threshold': threshold,
                    'contributing_models': self._get_contributing_models(results, i),
                    'explanation': self._generate_explanation(explanations, i)
                }
                anomalies.append(anomaly)
        
        # Update model weights based on performance
        if metadata and 'labeled_anomalies' in metadata:
            self._update_weights(results, metadata['labeled_anomalies'])
        
        return anomalies
```

Key advantages of this ensemble approach:
- **Robustness**: Different models capture different types of anomalies
- **Adaptability**: Weights adjust based on model performance
- **Explainability**: Provides detailed explanations of why an anomaly was detected
- **Reduced false positives**: Consensus approach reduces spurious detections

### 2. Contextual Anomaly Detection

The Detection Engine considers context when identifying anomalies:

```python
class ContextualAnomalyDetector:
    def __init__(self, base_detector):
        self.base_detector = base_detector
        self.context_models = {}
    
    def detect_with_context(self, time_series, context):
        """
        Detect anomalies considering contextual information
        
        Args:
            time_series: The time series data
            context: Dict with contextual information like:
                     - time_of_day
                     - day_of_week
                     - is_holiday
                     - deployment_status
                     - traffic_level
        """
        # Get context key
        context_key = self._get_context_key(context)
        
        # Get or create context-specific model
        if context_key not in self.context_models:
            self.context_models[context_key] = self._create_context_model(context)
        
        context_model = self.context_models[context_key]
        
        # Get base anomalies
        base_anomalies = self.base_detector.detect_anomalies(time_series)
        
        # Adjust anomaly scores based on context
        adjusted_anomalies = []
        for anomaly in base_anomalies:
            # Get context-specific threshold
            context_threshold = context_model.get_threshold(time_series, anomaly['index'])
            
            # Adjust score based on context
            context_factor = self._calculate_context_factor(anomaly, context)
            adjusted_score = anomaly['score'] * context_factor
            
            # Only keep if it exceeds the context-specific threshold
            if adjusted_score > context_threshold:
                adjusted_anomaly = anomaly.copy()
                adjusted_anomaly['original_score'] = anomaly['score']
                adjusted_anomaly['score'] = adjusted_score
                adjusted_anomaly['context_factor'] = context_factor
                adjusted_anomaly['context_threshold'] = context_threshold
                adjusted_anomaly['context'] = context
                adjusted_anomalies.append(adjusted_anomaly)
        
        return adjusted_anomalies
```

This approach enables:
- **Deployment-Aware Detection**: Different thresholds during deployments
- **Time-Aware Detection**: Different baselines for different times of day/week
- **Load-Aware Detection**: Adjusts sensitivity based on traffic levels

### 3. Log-Based Anomaly Detection

The Detection Engine uses advanced NLP techniques for log analysis:

```python
class LogAnomalyDetector:
    def __init__(self, embedding_model, clustering_model):
        self.embedding_model = embedding_model  # BERT/Word2Vec model
        self.clustering_model = clustering_model  # DBSCAN/HDBSCAN
        self.normal_patterns = {}
        self.sequence_model = self._build_sequence_model()
    
    def _build_sequence_model(self):
        """Build a sequence model for detecting unusual log sequences"""
        return LSTM(
            input_size=300,  # Embedding dimension
            hidden_size=128,
            num_layers=2,
            bidirectional=True
        )
    
    def detect_anomalies(self, logs, window_size=10):
        """Detect anomalies in logs using multiple techniques"""
        # Extract log templates
        templates = self._extract_templates(logs)
        
        # Get embeddings
        embeddings = self.embedding_model.encode(templates)
        
        # Detect clustering anomalies
        clustering_anomalies = self._detect_clustering_anomalies(embeddings)
        
        # Detect sequence anomalies
        sequence_anomalies = self._detect_sequence_anomalies(embeddings, window_size)
        
        # Detect frequency anomalies
        frequency_anomalies = self._detect_frequency_anomalies(templates)
        
        # Combine results
        all_anomalies = self._combine_anomalies(
            clustering_anomalies, 
            sequence_anomalies, 
            frequency_anomalies
        )
        
        return all_anomalies
    
    def _detect_clustering_anomalies(self, embeddings):
        """Detect logs that don't fit into normal clusters"""
        # Cluster the embeddings
        clusters = self.clustering_model.fit_predict(embeddings)
        
        # Identify outliers (typically labeled as -1 by DBSCAN/HDBSCAN)
        anomalies = [i for i, c in enumerate(clusters) if c == -1]
        
        return anomalies
    
    def _detect_sequence_anomalies(self, embeddings, window_size):
        """Detect unusual sequences of logs"""
        anomalies = []
        
        # Create sequences
        sequences = []
        for i in range(len(embeddings) - window_size + 1):
            sequences.append(embeddings[i:i+window_size])
        
        if not sequences:
            return anomalies
        
        # Convert to tensor
        sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
        
        # Get sequence predictions
        predictions = self.sequence_model(sequences_tensor)
        
        # Calculate prediction error
        errors = torch.mean(torch.abs(predictions - sequences_tensor[:, -1]), dim=1)
        
        # Identify anomalies
        threshold = torch.mean(errors) + 2 * torch.std(errors)
        for i, error in enumerate(errors):
            if error > threshold:
                anomalies.append(i + window_size - 1)
        
        return anomalies
    
    def _detect_frequency_anomalies(self, templates):
        """Detect unusual changes in log message frequency"""
        # Count template occurrences
        template_counts = Counter(templates)
        
        anomalies = []
        for i, template in enumerate(templates):
            # Skip if we haven't seen this template before
            if template not in self.normal_patterns:
                self.normal_patterns[template] = {
                    'count': 0,
                    'history': []
                }
            
            # Update counts
            self.normal_patterns[template]['count'] += 1
            
            # Check if frequency is anomalous
            if len(self.normal_patterns[template]['history']) >= 30:
                expected = np.mean(self.normal_patterns[template]['history'])
                std_dev = np.std(self.normal_patterns[template]['history'])
                
                current = template_counts[template]
                z_score = (current - expected) / (std_dev + 1e-10)
                
                if abs(z_score) > 3:
                    anomalies.append(i)
            
            # Update history
            self.normal_patterns[template]['history'].append(template_counts[template])
            if len(self.normal_patterns[template]['history']) > 100:
                self.normal_patterns[template]['history'].pop(0)
        
        return anomalies
```

This approach enables:
- **Content-Based Detection**: Identifies semantically unusual log messages
- **Sequence-Based Detection**: Detects unusual sequences of log events
- **Frequency-Based Detection**: Identifies unusual changes in log patterns

### 4. Topology-Based Anomaly Detection

The Detection Engine analyzes service relationships to detect structural anomalies:

```python
class TopologyAnomalyDetector:
    def __init__(self, graph_embedding_model):
        self.graph_embedding_model = graph_embedding_model
        self.normal_embeddings = []
        self.edge_models = {}
    
    def detect_anomalies(self, service_graph):
        """Detect anomalies in service topology"""
        # Extract graph features
        graph_features = self._extract_graph_features(service_graph)
        
        # Get graph embedding
        graph_embedding = self.graph_embedding_model.encode(service_graph)
        
        # Detect structural anomalies
        structural_anomalies = self._detect_structural_anomalies(graph_embedding)
        
        # Detect edge anomalies
        edge_anomalies = self._detect_edge_anomalies(service_graph)
        
        # Detect subgraph anomalies
        subgraph_anomalies = self._detect_subgraph_anomalies(service_graph, graph_features)
        
        # Combine results
        all_anomalies = {
            'structural': structural_anomalies,
            'edge': edge_anomalies,
            'subgraph': subgraph_anomalies
        }
        
        return all_anomalies
    
    def _extract_graph_features(self, graph):
        """Extract features from the graph"""
        features = {}
        
        # Node-level features
        for node in graph.nodes():
            features[node] = {
                'degree': graph.degree(node),
                'in_degree': graph.in_degree(node),
                'out_degree': graph.out_degree(node),
                'clustering': nx.clustering(graph, node),
                'betweenness': nx.betweenness_centrality(graph)[node],
                'pagerank': nx.pagerank(graph)[node]
            }
        
        # Graph-level features
        features['graph'] = {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'avg_clustering': nx.average_clustering(graph),
            'diameter': nx.diameter(graph) if nx.is_connected(graph) else -1
        }
        
        return features
```

This approach enables:
- **Structural Anomaly Detection**: Identifies unusual changes in graph structure
- **Edge Anomaly Detection**: Detects unusual traffic patterns between services
- **Subgraph Anomaly Detection**: Identifies problematic service clusters

### 5. Multi-Dimensional Correlation Engine

The Detection Engine correlates signals across multiple dimensions:

```python
class CorrelationEngine:
    def __init__(self):
        self.causal_model = CausalModel()
        self.temporal_correlator = TemporalCorrelator(max_lag=30)  # 30 second max lag
        self.spatial_correlator = SpatialCorrelator()
    
    def correlate_anomalies(self, anomalies, service_graph, time_window=300):
        """
        Correlate anomalies across multiple dimensions
        
        Args:
            anomalies: Dict of anomalies from different detectors
            service_graph: Service dependency graph
            time_window: Time window for correlation in seconds
        """
        # Group anomalies by time (within the specified window)
        time_groups = self._group_by_time(anomalies, time_window)
        
        correlated_incidents = []
        
        for time_group in time_groups:
            # Perform spatial correlation (using service graph)
            spatial_groups = self.spatial_correlator.correlate(
                time_group, service_graph
            )
            
            # For each spatially correlated group
            for spatial_group in spatial_groups:
                # Perform causal analysis
                root_causes = self.causal_model.infer_causes(
                    spatial_group, service_graph
                )
                
                # Create incident
                incident = {
                    'anomalies': spatial_group,
                    'root_causes': root_causes,
                    'start_time': min(a['timestamp'] for a in spatial_group if 'timestamp' in a),
                    'services_affected': self._get_affected_services(spatial_group),
                    'severity': self._calculate_severity(spatial_group, root_causes)
                }
                
                correlated_incidents.append(incident)
        
        return correlated_incidents
```

This approach enables:
- **Temporal Correlation**: Relates events occurring close in time
- **Spatial Correlation**: Relates events affecting connected services
- **Causal Correlation**: Identifies root causes using causal inference

## Predictive Capabilities

The Detection Engine includes predictive capabilities to identify issues before they become critical:

```python
class PredictiveAnalyzer:
    def __init__(self):
        self.forecasting_models = {
            'cpu': ProphetModel(changepoint_prior_scale=0.05),
            'memory': ProphetModel(changepoint_prior_scale=0.05),
            'latency': LSTMModel(sequence_length=60, hidden_size=128),
            'traffic': LSTMModel(sequence_length=60, hidden_size=128)
        }
        self.failure_predictor = FailurePredictor()
    
    def predict_issues(self, metrics, service_graph, prediction_horizon=3600):
        """
        Predict potential issues within the prediction horizon
        
        Args:
            metrics: Dict of time series metrics
            service_graph: Service dependency graph
            prediction_horizon: Prediction horizon in seconds
        """
        # Generate forecasts for each metric
        forecasts = {}
        for metric_name, time_series in metrics.items():
            model_type = self._get_model_type(metric_name)
            if model_type in self.forecasting_models:
                forecasts[metric_name] = self.forecasting_models[model_type].forecast(
                    time_series, prediction_horizon
                )
        
        # Detect anomalies in forecasts
        forecast_anomalies = {}
        for metric_name, forecast in forecasts.items():
            forecast_anomalies[metric_name] = self._detect_forecast_anomalies(
                forecast, metrics[metric_name]
            )
        
        # Predict potential failures
        potential_failures = self.failure_predictor.predict(
            forecasts, forecast_anomalies, service_graph
        )
        
        return {
            'forecasts': forecasts,
            'forecast_anomalies': forecast_anomalies,
            'potential_failures': potential_failures
        }
```

This approach enables:
- **Metric Forecasting**: Predicts future metric values
- **Anomaly Forecasting**: Identifies future anomalies before they occur
- **Failure Prediction**: Predicts potential service failures

## Conclusion

The Detection Engine of Aegis Sentinel represents a significant advancement in anomaly detection and root cause analysis for complex distributed systems. By combining multiple detection techniques, contextual awareness, and correlation capabilities, it provides accurate, explainable, and actionable insights into system behavior.

The key technical innovations include:
1. **Multi-model ensemble approach** for robust anomaly detection
2. **Contextual awareness** to reduce false positives
3. **Advanced NLP techniques** for log analysis
4. **Graph-based algorithms** for topology analysis
5. **Multi-dimensional correlation** for incident identification
6. **Predictive capabilities** for proactive issue resolution

These capabilities enable Aegis Sentinel to detect issues earlier, identify root causes faster, and provide more actionable insights than traditional monitoring systems.
