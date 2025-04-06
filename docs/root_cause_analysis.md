# Root Cause Analysis in Aegis Sentinel

## Introduction

One of the most challenging aspects of troubleshooting in modern distributed systems is determining the actual root cause of an issue. When a problem occurs in a microservices architecture, the symptoms often manifest in multiple services, making it difficult to identify the true source of the problem. For example, a database slowdown might cause API timeouts, which then lead to frontend errors visible to users.

Aegis Sentinel addresses this challenge through a sophisticated root cause analysis system that correlates anomalies across services, analyzes causal relationships, and identifies the most likely source of issues. This document explains the technical implementation of this system and how it traces issues across service boundaries.

## Architecture Overview

![Root Cause Analysis Architecture](https://via.placeholder.com/900x500?text=Root+Cause+Analysis+Architecture)

The root cause analysis system consists of several interconnected components:

```
┌─────────────────────────────────────────────────────────────┐
│                  Root Cause Analysis Engine                  │
├─────────────┬─────────────┬────────────────┬────────────────┤
│  Temporal   │   Spatial   │    Causal      │   Bayesian     │
│ Correlation │ Correlation │   Inference    │   Networks     │
├─────────────┴─────────────┴────────────────┴────────────────┤
│                    Service Graph Analysis                    │
├─────────────────────────────────────────────────────────────┤
│                      Anomaly Correlation                     │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Service Dependency Graph

The foundation of root cause analysis is a comprehensive service dependency graph that models relationships between services:

```python
class ServiceDependencyGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.dependency_types = {}
        self.dependency_strengths = {}
        
    def add_service(self, service_id, metadata=None):
        """Add a service to the graph"""
        self.graph.add_node(service_id, **(metadata or {}))
        
    def add_dependency(self, source, target, dependency_type, strength=1.0):
        """Add a dependency between services"""
        self.graph.add_edge(source, target)
        edge_key = (source, target)
        self.dependency_types[edge_key] = dependency_type
        self.dependency_strengths[edge_key] = strength
```

The service dependency graph is built from multiple sources:

1. **Static Analysis**: Parsing infrastructure-as-code (Kubernetes, Terraform, etc.)
2. **Dynamic Discovery**: Analyzing network traffic and API calls
3. **Manual Configuration**: Expert-provided dependency information
4. **Inference**: Automatically inferring relationships from correlated behavior

### 2. Temporal Correlation

The first step in root cause analysis is identifying anomalies that occur close together in time:

```python
class TemporalCorrelator:
    def __init__(self, max_time_window=300):  # 5 minutes default
        self.max_time_window = max_time_window
        
    def correlate(self, anomalies):
        """Group anomalies that occur close together in time"""
        if not anomalies:
            return []
            
        # Sort anomalies by timestamp
        sorted_anomalies = sorted(anomalies, key=lambda a: a['timestamp'])
        
        # Group anomalies within time windows
        groups = []
        current_group = [sorted_anomalies[0]]
        
        for anomaly in sorted_anomalies[1:]:
            # Check if this anomaly is within the time window of the current group
            if anomaly['timestamp'] - current_group[0]['timestamp'] <= self.max_time_window:
                current_group.append(anomaly)
            else:
                # Start a new group
                groups.append(current_group)
                current_group = [anomaly]
                
        # Add the last group
        if current_group:
            groups.append(current_group)
            
        return groups
```

Key aspects of temporal correlation:

- **Adaptive Time Windows**: The time window adjusts based on system characteristics
- **Propagation Delay Modeling**: Accounts for expected delays between services
- **Periodic Pattern Recognition**: Distinguishes between related and coincidental anomalies

### 3. Spatial Correlation

Next, the system analyzes the service topology to identify related anomalies:

```python
class SpatialCorrelator:
    def __init__(self, service_graph, max_distance=3):
        self.service_graph = service_graph
        self.max_distance = max_distance
        
    def correlate(self, temporal_groups):
        """Further group anomalies based on service topology"""
        spatial_groups = []
        
        for temporal_group in temporal_groups:
            # Get affected services
            affected_services = set(anomaly['service_id'] for anomaly in temporal_group)
            
            # Create a subgraph of affected services and nearby services
            subgraph = self._create_relevant_subgraph(affected_services)
            
            # Find connected components in the subgraph
            connected_components = list(nx.connected_components(subgraph.to_undirected()))
            
            # Group anomalies by connected component
            for component in connected_components:
                component_anomalies = [
                    anomaly for anomaly in temporal_group
                    if anomaly['service_id'] in component
                ]
                
                if component_anomalies:
                    spatial_groups.append(component_anomalies)
                    
        return spatial_groups
```

Key aspects of spatial correlation:

- **Topological Distance**: Considers services that are closely connected in the dependency graph
- **Bidirectional Analysis**: Examines both upstream and downstream dependencies
- **Connection Strength**: Weighs relationships based on dependency strength

### 4. Causal Inference

The core of root cause analysis is causal inference, which determines the most likely cause-effect relationships:

```python
class CausalInferenceEngine:
    def __init__(self, service_graph):
        self.service_graph = service_graph
        self.bayesian_network = self._build_bayesian_network()
        
    def _build_bayesian_network(self):
        """Build a Bayesian network from the service graph"""
        G = nx.DiGraph()
        
        # Add nodes
        for node in self.service_graph.graph.nodes():
            G.add_node(node)
            
        # Add edges with conditional probability tables
        for source, target in self.service_graph.graph.edges():
            G.add_edge(source, target)
            
            # Initialize CPTs based on dependency strength
            strength = self.service_graph.get_dependency_strength(source, target)
            
            # Higher strength means higher probability of target failing if source fails
            cpt = {
                'source_healthy': {
                    'target_healthy': 0.99,
                    'target_failing': 0.01
                },
                'source_failing': {
                    'target_healthy': 1.0 - strength,
                    'target_failing': strength
                }
            }
            
            G.edges[source, target]['cpt'] = cpt
            
        return G
        
    def infer_root_causes(self, anomalies):
        """Infer the most likely root causes for a set of anomalies"""
        # Extract affected services and their states
        evidence = {}
        for anomaly in anomalies:
            service_id = anomaly['service_id']
            evidence[service_id] = 'failing'
            
        # Perform inference
        root_causes = self._infer_causes(evidence)
        
        return root_causes
```

Key aspects of causal inference:

- **Bayesian Networks**: Model probabilistic relationships between services
- **Conditional Probability Tables**: Capture the likelihood of failure propagation
- **Multi-path Analysis**: Considers all possible paths through which failures can propagate
- **Strength-weighted Scoring**: Prioritizes stronger causal relationships

### 5. Temporal Causality Analysis

To further refine root cause analysis, the system analyzes the temporal order of anomalies:

```python
class TemporalCausalityAnalyzer:
    def __init__(self, service_graph):
        self.service_graph = service_graph
        
    def analyze(self, anomalies):
        """Analyze temporal ordering of anomalies to refine causality"""
        if not anomalies or len(anomalies) < 2:
            return anomalies
            
        # Sort anomalies by timestamp
        sorted_anomalies = sorted(anomalies, key=lambda a: a['timestamp'])
        
        # Build a directed graph of temporal relationships
        temporal_graph = nx.DiGraph()
        
        # Add nodes for each anomaly
        for i, anomaly in enumerate(sorted_anomalies):
            temporal_graph.add_node(i, **anomaly)
            
        # Add edges for temporal relationships
        for i in range(len(sorted_anomalies) - 1):
            for j in range(i + 1, len(sorted_anomalies)):
                # Check if there's a dependency path from i to j
                source_service = sorted_anomalies[i]['service_id']
                target_service = sorted_anomalies[j]['service_id']
                
                # Check if there's a path in the service graph
                paths = self.service_graph.get_all_paths(source_service, target_service)
                
                if paths:
                    # Calculate expected propagation delay
                    expected_delay = self._calculate_expected_delay(paths)
                    
                    # Calculate actual delay
                    actual_delay = sorted_anomalies[j]['timestamp'] - sorted_anomalies[i]['timestamp']
                    
                    # If actual delay is consistent with expected delay,
                    # add an edge indicating potential causality
                    if 0 <= actual_delay <= expected_delay * 1.5:  # Allow some buffer
                        temporal_graph.add_edge(i, j, 
                                              expected_delay=expected_delay,
                                              actual_delay=actual_delay)
        
        # Find root nodes (potential root causes)
        root_nodes = [n for n in temporal_graph.nodes() if temporal_graph.in_degree(n) == 0]
        
        # Extract the corresponding anomalies
        root_anomalies = [sorted_anomalies[i] for i in root_nodes]
        
        return root_anomalies
```

Key aspects of temporal causality analysis:

- **Propagation Delay Modeling**: Accounts for the time it takes for failures to propagate
- **Consistency Checking**: Verifies that temporal ordering is consistent with causal relationships
- **Multi-path Consideration**: Analyzes all possible propagation paths

### 6. Granger Causality Testing

For metric-based anomalies, the system employs Granger causality testing to identify causal relationships:

```python
class GrangerCausalityTester:
    def __init__(self, max_lag=5):
        self.max_lag = max_lag
        
    def test_causality(self, time_series_dict):
        """
        Test for Granger causality between multiple time series
        
        Args:
            time_series_dict: Dict mapping service IDs to their metric time series
            
        Returns:
            Dict mapping (source, target) pairs to causality scores
        """
        causality_scores = {}
        
        # Get all pairs of services
        service_ids = list(time_series_dict.keys())
        
        for i, source in enumerate(service_ids):
            for j, target in enumerate(service_ids):
                if i != j:  # Don't test self-causality
                    # Get time series data
                    source_data = time_series_dict[source]
                    target_data = time_series_dict[target]
                    
                    # Test Granger causality
                    score = self._granger_causality_test(source_data, target_data)
                    
                    causality_scores[(source, target)] = score
        
        return causality_scores
```

Key aspects of Granger causality testing:

- **Time Series Analysis**: Tests if past values of one metric help predict another
- **Multiple Lag Testing**: Considers various time delays for causal relationships
- **Statistical Significance**: Quantifies the strength of causal evidence

## Practical Example: Frontend Error Traced to Database Issue

Let's walk through a concrete example of how Aegis Sentinel would trace a frontend error to its root cause in a database service:

### Initial Anomaly Detection

1. **Frontend Service**: Users report slow page loads and errors
   - Log anomaly: Increased rate of timeout errors
   - Metric anomaly: Increased response time and error rate

2. **API Gateway Service**: No visible errors to users but internal monitoring detects issues
   - Metric anomaly: Increased latency
   - Log anomaly: Increased rate of timeout errors from downstream services

3. **User Service**: Internal service with no direct user impact
   - Metric anomaly: Increased latency in database queries
   - Log anomaly: Database connection pool exhaustion

4. **Database Service**: Core infrastructure service
   - Metric anomaly: High CPU utilization
   - Metric anomaly: Increased disk I/O wait time
   - Log anomaly: Slow query warnings

### Root Cause Analysis Process

1. **Temporal Correlation**:
   - All anomalies occurred within a 2-minute window
   - Database anomalies appeared first, followed by User Service, then API Gateway, and finally Frontend

2. **Spatial Correlation**:
   - Service dependency graph shows:
     ```
     Frontend → API Gateway → User Service → Database
     ```
   - All affected services are connected in the dependency graph

3. **Causal Inference**:
   - Bayesian network analysis shows high probability that Database issues caused User Service issues
   - User Service issues likely caused API Gateway issues
   - API Gateway issues likely caused Frontend issues

4. **Temporal Causality Analysis**:
   - Timeline confirms Database issues preceded other issues
   - Propagation delays between services match expected patterns

5. **Granger Causality Testing**:
   - Database CPU metric Granger-causes User Service latency
   - User Service latency Granger-causes API Gateway latency
   - API Gateway latency Granger-causes Frontend response time

### Final Root Cause Determination

The system identifies the Database service as the root cause with high confidence (0.92 score) and provides the following explanation:

> "Database service is likely causing issues in 3 dependent services: User Service, API Gateway, and Frontend. Database showed anomalies before other affected services. Strong statistical evidence that Database metrics predict subsequent issues. High CPU utilization and disk I/O wait time in Database service preceded connection pool exhaustion in User Service, which led to timeouts in API Gateway and ultimately errors visible to users in Frontend. High confidence (score: 0.92)"

## Key Advantages of This Approach

1. **Cross-Service Tracing**: Identifies root causes even when they're several services removed from the visible symptoms

2. **Multiple Evidence Types**: Combines structural, temporal, and statistical evidence for more accurate results

3. **Probabilistic Reasoning**: Uses Bayesian networks to handle uncertainty and incomplete information

4. **Adaptive Learning**: Continuously improves based on feedback and historical data

5. **Explainable Results**: Provides detailed explanations of why a particular service was identified as the root cause

## Conclusion

Aegis Sentinel's root cause analysis system addresses one of the most challenging aspects of operating distributed systems: tracing issues across service boundaries to identify their true source. By combining multiple analytical approaches—temporal correlation, spatial correlation, causal inference, and statistical testing—the system can accurately identify root causes even in complex microservice architectures.

This capability dramatically reduces mean time to resolution (MTTR) by eliminating the time-consuming process of manually tracing issues through service dependencies. Instead of starting with the symptoms and working backward, engineers can immediately focus on the actual source of the problem, leading to faster resolution and reduced service impact.
