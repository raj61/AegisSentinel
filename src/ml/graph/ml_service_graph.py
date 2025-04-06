"""
ML-Enhanced Service Graph
========================

This module provides an ML-enhanced service graph implementation that uses machine learning
to better detect relationships between services.
"""

import logging
import json
import numpy as np
from typing import Dict, List, Set, Any, Optional, Union, Tuple
from pathlib import Path

from src.graph import ServiceGraph

logger = logging.getLogger(__name__)

class MLServiceGraph:
    """
    ML-enhanced service graph implementation.
    
    This class extends the base ServiceGraph with machine learning capabilities
    to better detect relationships between services.
    """
    
    def __init__(self, service_graph: ServiceGraph = None):
        """
        Initialize the ML-enhanced service graph.
        
        Args:
            service_graph: Existing ServiceGraph instance to enhance
        """
        self.service_graph = service_graph or ServiceGraph()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Embedding cache for service nodes
        self.embeddings = {}
        
        # Similarity threshold for relationship detection
        self.similarity_threshold = 0.7
        
        # Confidence scores for different relationship types
        self.confidence_scores = {
            'direct': 0.9,
            'inferred': 0.7,
            'ml-inferred': 0.8
        }
    
    def enhance_graph(self) -> ServiceGraph:
        """
        Enhance the service graph with ML-detected relationships.
        
        Returns:
            Enhanced ServiceGraph instance
        """
        self.logger.info("Enhancing service graph with ML...")
        
        # Step 1: Generate embeddings for all services
        self._generate_service_embeddings()
        
        # Step 2: Detect relationships based on embeddings
        self._detect_relationships_from_embeddings()
        
        # Step 3: Analyze traffic patterns
        self._analyze_traffic_patterns()
        
        # Step 4: Analyze log correlations
        self._analyze_log_correlations()
        
        # Step 5: Analyze deployment patterns
        self._analyze_deployment_patterns()
        
        # Step 6: Prune unlikely relationships
        self._prune_unlikely_relationships()
        
        self.logger.info("Service graph enhancement complete")
        return self.service_graph
    
    def _generate_service_embeddings(self) -> None:
        """
        Generate embeddings for all services in the graph.
        
        This method creates vector representations of services based on their
        attributes, names, and relationships.
        """
        self.logger.info("Generating service embeddings...")
        
        # Get all nodes in the graph
        nodes = self.service_graph.get_nodes()
        
        for node_id in nodes:
            # Get node attributes
            node = self.service_graph.get_node(node_id)
            
            # Extract features for embedding
            features = self._extract_node_features(node_id, node)
            
            # Create a simple embedding (in a real implementation, this would use a proper embedding model)
            embedding = self._create_embedding_from_features(features)
            
            # Store the embedding
            self.embeddings[node_id] = embedding
    
    def _extract_node_features(self, node_id: str, node: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from a node for embedding generation.
        
        Args:
            node_id: Node ID
            node: Node attributes
            
        Returns:
            Dictionary of features
        """
        features = {
            'id': node_id,
            'name': node.get('name', node_id),
            'kind': node.get('kind', 'unknown'),
            'category': node.get('category', 'unknown'),
            'labels': node.get('labels', {}),
            'annotations': node.get('annotations', {}),
            'ports': node.get('ports', []),
            'environment': node.get('environment', {}),
            'health_status': node.get('health_status', 'unknown')
        }
        
        # Extract service name parts
        name_parts = features['name'].split('-')
        features['name_parts'] = name_parts
        
        # Extract namespace if available
        if '/' in node_id:
            namespace, name = node_id.split('/', 1)
            features['namespace'] = namespace
        else:
            features['namespace'] = 'default'
        
        return features
    
    def _create_embedding_from_features(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Create an embedding vector from node features.
        
        Args:
            features: Node features
            
        Returns:
            Embedding vector as numpy array
        """
        # In a real implementation, this would use a proper embedding model
        # For now, we'll create a simple embedding based on feature hashing
        
        # Initialize a fixed-size embedding vector
        embedding_size = 64
        embedding = np.zeros(embedding_size)
        
        # Add components based on different features
        
        # 1. Name and ID contribution
        name = features['name'].lower()
        for i, char in enumerate(name[:10]):  # Use first 10 chars
            pos = i % embedding_size
            embedding[pos] += ord(char) / 255.0
        
        # 2. Kind and category contribution
        kind = features['kind'].lower()
        category = features['category'].lower()
        
        for i, char in enumerate(kind[:5]):  # Use first 5 chars
            pos = (i * 2) % embedding_size
            embedding[pos] += ord(char) / 255.0
        
        for i, char in enumerate(category[:5]):  # Use first 5 chars
            pos = (i * 2 + 1) % embedding_size
            embedding[pos] += ord(char) / 255.0
        
        # 3. Labels contribution
        for key, value in features['labels'].items():
            key_hash = hash(key) % embedding_size
            value_hash = hash(str(value)) % embedding_size
            embedding[key_hash] += 0.5
            embedding[value_hash] += 0.5
        
        # 4. Ports contribution
        for port in features['ports']:
            port_num = port.get('port', 0)
            port_pos = port_num % embedding_size
            embedding[port_pos] += 0.3
        
        # 5. Environment variables contribution
        for key, value in features['environment'].items():
            key_hash = hash(key) % embedding_size
            embedding[key_hash] += 0.2
        
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _detect_relationships_from_embeddings(self) -> None:
        """
        Detect relationships between services based on their embeddings.
        """
        self.logger.info("Detecting relationships from embeddings...")
        
        # Get all nodes in the graph
        nodes = self.service_graph.get_nodes()
        
        # Compare each pair of nodes
        for i, node1_id in enumerate(nodes):
            for node2_id in nodes[i+1:]:
                # Skip if nodes are the same
                if node1_id == node2_id:
                    continue
                
                # Get embeddings
                embedding1 = self.embeddings.get(node1_id)
                embedding2 = self.embeddings.get(node2_id)
                
                if embedding1 is None or embedding2 is None:
                    continue
                
                # Calculate similarity
                similarity = self._calculate_similarity(embedding1, embedding2)
                
                # If similarity is above threshold, add a relationship
                if similarity > self.similarity_threshold:
                    # Determine direction based on node types
                    node1 = self.service_graph.get_node(node1_id)
                    node2 = self.service_graph.get_node(node2_id)
                    
                    direction = self._determine_relationship_direction(node1_id, node1, node2_id, node2)
                    
                    if direction == 1:
                        # node1 -> node2
                        self.service_graph.add_edge(
                            node1_id, 
                            node2_id, 
                            type="ml-inferred", 
                            confidence=similarity,
                            similarity=similarity
                        )
                        self.logger.debug(f"ML-inferred relationship: {node1_id} -> {node2_id} (similarity: {similarity:.2f})")
                    elif direction == -1:
                        # node2 -> node1
                        self.service_graph.add_edge(
                            node2_id, 
                            node1_id, 
                            type="ml-inferred", 
                            confidence=similarity,
                            similarity=similarity
                        )
                        self.logger.debug(f"ML-inferred relationship: {node2_id} -> {node1_id} (similarity: {similarity:.2f})")
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score between 0 and 1
        """
        # Use cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _determine_relationship_direction(self, node1_id: str, node1: Dict[str, Any], 
                                         node2_id: str, node2: Dict[str, Any]) -> int:
        """
        Determine the direction of a relationship between two nodes.
        
        Args:
            node1_id: First node ID
            node1: First node attributes
            node2_id: Second node ID
            node2: Second node attributes
            
        Returns:
            1 if node1 -> node2, -1 if node2 -> node1, 0 if undetermined
        """
        # Get node kinds
        kind1 = node1.get('kind', '').lower()
        kind2 = node2.get('kind', '').lower()
        
        # Get node categories
        category1 = node1.get('category', '').lower()
        category2 = node2.get('category', '').lower()
        
        # Get node names
        name1 = node1.get('name', node1_id).lower()
        name2 = node2.get('name', node2_id).lower()
        
        # Rule 1: Frontend -> Backend
        if 'frontend' in category1 and 'backend' in category2:
            return 1
        if 'frontend' in category2 and 'backend' in category1:
            return -1
        
        # Rule 2: Backend -> Database
        if 'backend' in category1 and 'database' in category2:
            return 1
        if 'backend' in category2 and 'database' in category1:
            return -1
        
        # Rule 3: Backend -> Cache
        if 'backend' in category1 and 'cache' in category2:
            return 1
        if 'backend' in category2 and 'cache' in category1:
            return -1
        
        # Rule 4: API Gateway -> Service
        if 'api' in category1 and 'api' not in category2:
            return 1
        if 'api' in category2 and 'api' not in category1:
            return -1
        
        # Rule 5: Service -> Queue
        if 'queue' not in category1 and 'queue' in category2:
            return 1
        if 'queue' not in category2 and 'queue' in category1:
            return -1
        
        # Rule 6: Based on name patterns
        if name1 in name2 and 'client' in name1:
            return 1
        if name2 in name1 and 'client' in name2:
            return -1
        
        # Rule 7: Based on port exposure
        ports1 = node1.get('ports', [])
        ports2 = node2.get('ports', [])
        
        if ports1 and not ports2:
            return -1  # node2 -> node1 (node1 exposes ports)
        if ports2 and not ports1:
            return 1   # node1 -> node2 (node2 exposes ports)
        
        # If direction is still undetermined, use alphabetical order as a fallback
        return 1 if node1_id < node2_id else -1
    
    def _analyze_traffic_patterns(self) -> None:
        """
        Analyze traffic patterns to detect relationships.
        """
        self.logger.info("Analyzing traffic patterns...")
        
        # In a real implementation, this would analyze traffic data from monitoring systems
        # For now, we'll use a simple heuristic based on node attributes
        
        # Get all nodes in the graph
        nodes = self.service_graph.get_nodes()
        
        for node_id in nodes:
            node = self.service_graph.get_node(node_id)
            
            # Check if the node has traffic metrics
            if 'request_rate' in node or 'latency_p95' in node:
                # Find potential dependencies based on traffic patterns
                for other_id in nodes:
                    if node_id == other_id:
                        continue
                    
                    other = self.service_graph.get_node(other_id)
                    
                    # Check if the other node has traffic metrics
                    if 'request_rate' in other or 'latency_p95' in other:
                        # Check for correlated traffic patterns
                        # In a real implementation, this would use time series correlation
                        # For now, we'll use a simple heuristic
                        
                        # If both nodes have similar request rates, they might be related
                        if 'request_rate' in node and 'request_rate' in other:
                            node_rate = node['request_rate']
                            other_rate = other['request_rate']
                            
                            # Calculate ratio
                            ratio = min(node_rate, other_rate) / max(node_rate, other_rate) if max(node_rate, other_rate) > 0 else 0
                            
                            # If ratio is high, add a relationship
                            if ratio > 0.8:
                                # Determine direction based on latency
                                if 'latency_p95' in node and 'latency_p95' in other:
                                    if node['latency_p95'] > other['latency_p95']:
                                        # node_id likely calls other_id
                                        self.service_graph.add_edge(
                                            node_id, 
                                            other_id, 
                                            type="traffic-inferred", 
                                            confidence=ratio,
                                            ratio=ratio
                                        )
                                        self.logger.debug(f"Traffic-inferred relationship: {node_id} -> {other_id} (ratio: {ratio:.2f})")
    
    def _analyze_log_correlations(self) -> None:
        """
        Analyze log correlations to detect relationships.
        """
        self.logger.info("Analyzing log correlations...")
        
        # In a real implementation, this would analyze log data to find correlations
        # For now, we'll use a simple heuristic based on node attributes
        
        # Get all nodes in the graph
        nodes = self.service_graph.get_nodes()
        
        for node_id in nodes:
            node = self.service_graph.get_node(node_id)
            
            # Check if the node has error metrics
            if 'error_rate' in node:
                # Find potential dependencies based on error correlations
                for other_id in nodes:
                    if node_id == other_id:
                        continue
                    
                    other = self.service_graph.get_node(other_id)
                    
                    # Check if the other node has error metrics
                    if 'error_rate' in other:
                        # If both nodes have similar error rates, they might be related
                        node_error = node['error_rate']
                        other_error = other['error_rate']
                        
                        # If both have high error rates, they might be related
                        if node_error > 0.05 and other_error > 0.05:
                            # Determine direction based on error rate
                            if node_error > other_error:
                                # node_id likely calls other_id
                                self.service_graph.add_edge(
                                    node_id, 
                                    other_id, 
                                    type="error-inferred", 
                                    confidence=0.7,
                                    error_correlation=True
                                )
                                self.logger.debug(f"Error-inferred relationship: {node_id} -> {other_id}")
    
    def _analyze_deployment_patterns(self) -> None:
        """
        Analyze deployment patterns to detect relationships.
        """
        self.logger.info("Analyzing deployment patterns...")
        
        # In a real implementation, this would analyze deployment data
        # For now, we'll use a simple heuristic based on node attributes
        
        # Get all nodes in the graph
        nodes = self.service_graph.get_nodes()
        
        # Group nodes by namespace
        namespaces = {}
        for node_id in nodes:
            namespace = node_id.split('/')[0] if '/' in node_id else 'default'
            if namespace not in namespaces:
                namespaces[namespace] = []
            namespaces[namespace].append(node_id)
        
        # For each namespace, analyze deployment patterns
        for namespace, ns_nodes in namespaces.items():
            # Group nodes by deployment pattern
            deployments = {}
            for node_id in ns_nodes:
                node = self.service_graph.get_node(node_id)
                kind = node.get('kind', '').lower()
                
                if kind in ['deployment', 'statefulset', 'daemonset']:
                    name = node.get('name', node_id)
                    base_name = name.split('-')[0]
                    
                    if base_name not in deployments:
                        deployments[base_name] = []
                    deployments[base_name].append(node_id)
            
            # For each deployment group, infer relationships
            for base_name, dep_nodes in deployments.items():
                if len(dep_nodes) > 1:
                    # If multiple nodes share the same deployment pattern, they might be related
                    for i, node1_id in enumerate(dep_nodes):
                        for node2_id in dep_nodes[i+1:]:
                            # Add bidirectional relationship
                            self.service_graph.add_edge(
                                node1_id, 
                                node2_id, 
                                type="deployment-inferred", 
                                confidence=0.6,
                                deployment_group=base_name
                            )
                            self.logger.debug(f"Deployment-inferred relationship: {node1_id} <-> {node2_id}")
                            
                            self.service_graph.add_edge(
                                node2_id, 
                                node1_id, 
                                type="deployment-inferred", 
                                confidence=0.6,
                                deployment_group=base_name
                            )
    
    def _prune_unlikely_relationships(self) -> None:
        """
        Prune unlikely relationships from the graph.
        """
        self.logger.info("Pruning unlikely relationships...")
        
        # Get all edges in the graph
        edges = self.service_graph.get_edges()
        
        # Identify edges to remove
        edges_to_remove = []
        for source, target, attrs in edges:
            # Skip edges with high confidence
            if attrs.get('confidence', 0) > 0.8:
                continue
            
            # Skip edges with direct evidence
            if attrs.get('type') == 'direct':
                continue
            
            # Check for contradictory edges
            for s2, t2, a2 in edges:
                if s2 == target and t2 == source:
                    # Contradictory edge found
                    # Keep the one with higher confidence
                    if attrs.get('confidence', 0) < a2.get('confidence', 0):
                        edges_to_remove.append((source, target))
                        break
        
        # Remove identified edges
        for source, target in edges_to_remove:
            # In a real implementation, we would remove the edge
            # For now, we'll just log it
            self.logger.debug(f"Pruning unlikely relationship: {source} -> {target}")