"""
Service Graph
===========

This module provides the ServiceGraph class for representing service dependencies.
"""

import logging
import json
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)

class ServiceGraph:
    """
    A graph representation of service dependencies.
    
    This class wraps a NetworkX DiGraph with additional functionality for
    service dependency analysis and visualization.
    """
    
    def __init__(self):
        """Initialize an empty service graph."""
        self.graph = nx.DiGraph()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def add_node(self, node_id: str, **attrs) -> None:
        """
        Add a node to the graph with the given attributes.
        
        Args:
            node_id: Unique identifier for the node
            **attrs: Node attributes
        """
        self.graph.add_node(node_id, **attrs)
    
    def add_edge(self, source: str, target: str, **attrs) -> None:
        """
        Add an edge from source to target with the given attributes.
        
        Args:
            source: Source node ID
            target: Target node ID
            **attrs: Edge attributes
        """
        # Only add the edge if both nodes exist
        if self.has_node(source) and self.has_node(target):
            # Check if edge already exists
            if self.graph.has_edge(source, target):
                # Update attributes of existing edge
                for key, value in attrs.items():
                    self.graph[source][target][key] = value
            else:
                # Add new edge
                self.graph.add_edge(source, target, **attrs)
        else:
            missing = []
            if not self.has_node(source):
                missing.append(source)
            if not self.has_node(target):
                missing.append(target)
            self.logger.warning(f"Cannot add edge {source} -> {target}: nodes {', '.join(missing)} do not exist")
    
    def has_node(self, node_id: str) -> bool:
        """
        Check if a node exists in the graph.
        
        Args:
            node_id: Node ID to check
            
        Returns:
            True if the node exists, False otherwise
        """
        return node_id in self.graph.nodes
    
    def get_node(self, node_id: str) -> Dict[str, Any]:
        """
        Get a node's attributes.
        
        Args:
            node_id: Node ID to get
            
        Returns:
            Dictionary of node attributes
            
        Raises:
            KeyError: If the node does not exist
        """
        if not self.has_node(node_id):
            raise KeyError(f"Node {node_id} does not exist")
        
        return dict(self.graph.nodes[node_id])
    
    def get_nodes(self) -> List[str]:
        """
        Get all node IDs in the graph.
        
        Returns:
            List of node IDs
        """
        return list(self.graph.nodes)
    
    def get_edges(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Get all edges in the graph.
        
        Returns:
            List of (source, target, attributes) tuples
        """
        return [(u, v, dict(d)) for u, v, d in self.graph.edges(data=True)]
    
    def get_node_attribute(self, node_id: str, attr: str, default: Any = None) -> Any:
        """
        Get a specific attribute of a node.
        
        Args:
            node_id: Node ID
            attr: Attribute name
            default: Default value to return if the attribute does not exist
            
        Returns:
            Attribute value, or default if the attribute does not exist
        """
        if not self.has_node(node_id):
            return default
        
        return self.graph.nodes[node_id].get(attr, default)
    
    def has_node_attribute(self, node_id: str, attr: str) -> bool:
        """
        Check if a node has a specific attribute.
        
        Args:
            node_id: Node ID
            attr: Attribute name
            
        Returns:
            True if the node has the attribute, False otherwise
        """
        if not self.has_node(node_id):
            return False
        
        return attr in self.graph.nodes[node_id]
    
    def add_node_attribute(self, node_id: str, **attrs) -> None:
        """
        Add attributes to a node.
        
        Args:
            node_id: Node ID
            **attrs: Attributes to add
            
        Raises:
            KeyError: If the node does not exist
        """
        if not self.has_node(node_id):
            raise KeyError(f"Node {node_id} does not exist")
        
        for attr, value in attrs.items():
            self.graph.nodes[node_id][attr] = value
            
    def update_node_attribute(self, node_id: str, attr: str, value: Any) -> None:
        """
        Update a specific attribute of a node.
        
        Args:
            node_id: Node ID
            attr: Attribute name
            value: New attribute value
            
        Raises:
            KeyError: If the node does not exist
        """
        if not self.has_node(node_id):
            raise KeyError(f"Node {node_id} does not exist")
        
        self.graph.nodes[node_id][attr] = value
        self.logger.info(f"Updated node {node_id} attribute {attr} to {value}")
    
    def get_successors(self, node_id: str) -> List[str]:
        """
        Get the successors of a node (outgoing edges).
        
        Args:
            node_id: Node ID
            
        Returns:
            List of successor node IDs
            
        Raises:
            KeyError: If the node does not exist
        """
        if not self.has_node(node_id):
            raise KeyError(f"Node {node_id} does not exist")
        
        return list(self.graph.successors(node_id))
    
    def get_predecessors(self, node_id: str) -> List[str]:
        """
        Get the predecessors of a node (incoming edges).
        
        Args:
            node_id: Node ID
            
        Returns:
            List of predecessor node IDs
            
        Raises:
            KeyError: If the node does not exist
        """
        if not self.has_node(node_id):
            raise KeyError(f"Node {node_id} does not exist")
        
        return list(self.graph.predecessors(node_id))
    
    def node_count(self) -> int:
        """
        Get the number of nodes in the graph.
        
        Returns:
            Number of nodes
        """
        return self.graph.number_of_nodes()
    
    def edge_count(self) -> int:
        """
        Get the number of edges in the graph.
        
        Returns:
            Number of edges
        """
        return self.graph.number_of_edges()
    
    def visualize(self, output_path: Union[str, Path], format: str = None) -> None:
        """
        Visualize the service graph and save it to a file.
        
        Args:
            output_path: Path to save the visualization
            format: Output format (png, pdf, svg, etc.). If None, inferred from the file extension.
        """
        if isinstance(output_path, str):
            output_path = Path(output_path)
        
        # Determine the format from the file extension if not specified
        if format is None:
            format = output_path.suffix.lstrip('.')
            if not format:
                format = 'png'
                output_path = output_path.with_suffix(f'.{format}')
        
        # Create the directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set up the figure
        plt.figure(figsize=(12, 8))
        
        # Create a layout for the graph
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Get node categories for coloring
        node_categories = {}
        for node_id in self.graph.nodes:
            category = self.get_node_attribute(node_id, 'category')
            if category:
                if category not in node_categories:
                    node_categories[category] = []
                node_categories[category].append(node_id)
        
        # Define colors for different categories
        category_colors = {
            'compute': 'skyblue',
            'serverless': 'lightgreen',
            'container': 'lightcoral',
            'kubernetes': 'orange',
            'api': 'yellow',
            'loadbalancer': 'pink',
            'database': 'lightblue',
            'cache': 'lightgrey',
            'queue': 'tan',
            'topic': 'wheat',
            'subscription': 'beige',
            'storage': 'lightcyan',
            'network': 'lavender',
            'security': 'mistyrose',
            'stack': 'lightsteelblue',
            'data': 'honeydew',
            'other': 'white'
        }
        
        # Draw nodes by category
        for category, nodes in node_categories.items():
            nx.draw_networkx_nodes(
                self.graph,
                pos,
                nodelist=nodes,
                node_color=category_colors.get(category, 'white'),
                node_size=1000,
                alpha=0.8,
                label=category
            )
        
        # Draw nodes without a category
        uncategorized = [n for n in self.graph.nodes if not self.get_node_attribute(n, 'category')]
        if uncategorized:
            nx.draw_networkx_nodes(
                self.graph,
                pos,
                nodelist=uncategorized,
                node_color='white',
                node_size=1000,
                alpha=0.8,
                label='uncategorized'
            )
        
        # Draw edges
        nx.draw_networkx_edges(
            self.graph,
            pos,
            width=1.0,
            alpha=0.5,
            arrowsize=20,
            arrowstyle='->'
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            self.graph,
            pos,
            labels={n: self.get_node_attribute(n, 'name', n) for n in self.graph.nodes},
            font_size=8,
            font_weight='bold'
        )
        
        # Add a legend
        plt.legend(scatterpoints=1, loc='lower right')
        
        # Add a title
        plt.title(f"Service Dependency Graph ({self.node_count()} services, {self.edge_count()} dependencies)")
        
        # Remove axis
        plt.axis('off')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, format=format, dpi=300)
        plt.close()
        
        self.logger.info(f"Saved service graph visualization to {output_path}")
    
    def to_dict(self, simplified: bool = False) -> Dict[str, Any]:
        """
        Convert the service graph to a dictionary representation.
        
        Args:
            simplified: If True, simplify the graph by merging related nodes
            
        Returns:
            Dictionary representation of the graph
        """
        if not simplified:
            # Return the full graph
            return {
                'nodes': [
                    {
                        'id': node_id,
                        **dict(attrs)
                    }
                    for node_id, attrs in self.graph.nodes(data=True)
                ],
                'edges': [
                    {
                        'source': source,
                        'target': target,
                        **dict(attrs)
                    }
                    for source, target, attrs in self.graph.edges(data=True)
                ]
            }
        else:
            # Return a simplified graph with merged nodes
            service_groups = self._group_related_services()
            simplified_nodes = []
            simplified_edges = []
            node_mapping = {}  # Maps original node IDs to simplified node IDs
            
            # Create simplified nodes
            for base_name, nodes in service_groups.items():
                primary_node = self._get_primary_node(nodes)
                if not primary_node:
                    continue
                
                # Use the primary node's attributes
                attrs = dict(self.graph.nodes[primary_node])
                
                # Create a simplified node
                simplified_node = {
                    'id': base_name,
                    'name': base_name,
                    'kind': 'Service',  # Simplified kind
                    **attrs
                }
                
                # Add the simplified node
                simplified_nodes.append(simplified_node)
                
                # Map all nodes in this group to the simplified node
                for node in nodes:
                    node_mapping[node] = base_name
            
            # Create simplified edges
            for source, target, attrs in self.graph.edges(data=True):
                if source in node_mapping and target in node_mapping:
                    simplified_source = node_mapping[source]
                    simplified_target = node_mapping[target]
                    
                    # Skip self-loops
                    if simplified_source == simplified_target:
                        continue
                    
                    # Check if this edge already exists
                    edge_exists = False
                    for edge in simplified_edges:
                        if edge['source'] == simplified_source and edge['target'] == simplified_target:
                            edge_exists = True
                            break
                    
                    if not edge_exists:
                        simplified_edges.append({
                            'source': simplified_source,
                            'target': simplified_target,
                            **dict(attrs)
                        })
            
            return {
                'nodes': simplified_nodes,
                'edges': simplified_edges
            }
    
    def to_json(self, indent: int = 2, simplified: bool = False) -> str:
        """
        Convert the service graph to a JSON string.
        
        Args:
            indent: Number of spaces for indentation
            simplified: If True, simplify the graph by merging related nodes
            
        Returns:
            JSON string representation of the graph
        """
        return json.dumps(self.to_dict(simplified=simplified), indent=indent)
    
    def save_json(self, output_path: Union[str, Path]) -> None:
        """
        Save the service graph to a JSON file.
        
        Args:
            output_path: Path to save the JSON file
        """
        if isinstance(output_path, str):
            output_path = Path(output_path)
        
        # Create the directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(self.to_json())
        
        self.logger.info(f"Saved service graph to {output_path}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ServiceGraph':
        """
        Create a service graph from a dictionary representation.
        
        Args:
            data: Dictionary representation of the graph
            
        Returns:
            ServiceGraph instance
        """
        graph = cls()
        
        # Add nodes
        for node_data in data.get('nodes', []):
            node_id = node_data.pop('id')
            graph.add_node(node_id, **node_data)
        
        # Add edges
        for edge_data in data.get('edges', []):
            source = edge_data.pop('source')
            target = edge_data.pop('target')
            graph.add_edge(source, target, **edge_data)
        
        return graph
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ServiceGraph':
        """
        Create a service graph from a JSON string.
        
        Args:
            json_str: JSON string representation of the graph
            
        Returns:
            ServiceGraph instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @classmethod
    def load_json(cls, input_path: Union[str, Path]) -> 'ServiceGraph':
        """
        Load a service graph from a JSON file.
        
        Args:
            input_path: Path to the JSON file
            
        Returns:
            ServiceGraph instance
        """
        if isinstance(input_path, str):
            input_path = Path(input_path)
        
        with open(input_path, 'r') as f:
            json_str = f.read()
        
        return cls.from_json(json_str)
    
    def infer_relationships(self) -> None:
        """
        Infer relationships between services based on common patterns and naming conventions.
        This method adds edges to the graph based on service types, names, and other heuristics.
        It also handles duplicate nodes and avoids circular dependencies.
        """
        self.logger.info("Inferring relationships between services...")
        
        # First, identify and group related nodes (e.g., backend and backend-service)
        service_groups = self._group_related_services()
        
        # Collect primary nodes by category
        frontend_groups = []
        backend_groups = []
        database_groups = []
        cache_groups = []
        queue_groups = []
        
        for group_name, nodes in service_groups.items():
            group_name_lower = group_name.lower()
            
            # Check for frontend services
            if any(pattern in group_name_lower for pattern in ['frontend', 'ui', 'web', 'client']):
                frontend_groups.append((group_name, nodes))
            
            # Check for backend/API services
            elif any(pattern in group_name_lower for pattern in ['backend', 'api', 'server']):
                backend_groups.append((group_name, nodes))
            
            # Check for database services
            elif any(pattern in group_name_lower for pattern in ['db', 'database', 'sql', 'mongo', 'postgres', 'mysql']):
                database_groups.append((group_name, nodes))
            
            # Check for cache services
            elif any(pattern in group_name_lower for pattern in ['cache', 'redis', 'memcached']):
                cache_groups.append((group_name, nodes))
            
            # Check for queue/messaging services
            elif any(pattern in group_name_lower for pattern in ['queue', 'mq', 'kafka', 'rabbit', 'nats']):
                queue_groups.append((group_name, nodes))
        
        # Apply common microservice patterns using the primary nodes from each group
        
        # Pattern 1: Frontend -> Backend
        for frontend_name, frontend_nodes in frontend_groups:
            for backend_name, backend_nodes in backend_groups:
                # Get the primary nodes from each group
                frontend_primary = self._get_primary_node(frontend_nodes)
                backend_primary = self._get_primary_node(backend_nodes)
                
                if frontend_primary and backend_primary and frontend_primary != backend_primary:
                    self.add_edge(frontend_primary, backend_primary, type="inferred-dependency", confidence=0.8)
                    self.logger.debug(f"Inferred frontend->backend: {frontend_primary} -> {backend_primary}")
        
        # Pattern 2: Backend -> Database
        for backend_name, backend_nodes in backend_groups:
            for db_name, db_nodes in database_groups:
                # Get the primary nodes from each group
                backend_primary = self._get_primary_node(backend_nodes)
                db_primary = self._get_primary_node(db_nodes)
                
                if backend_primary and db_primary and backend_primary != db_primary:
                    self.add_edge(backend_primary, db_primary, type="inferred-dependency", confidence=0.9)
                    self.logger.debug(f"Inferred backend->database: {backend_primary} -> {db_primary}")
        
        # Pattern 3: Backend -> Cache
        for backend_name, backend_nodes in backend_groups:
            for cache_name, cache_nodes in cache_groups:
                # Get the primary nodes from each group
                backend_primary = self._get_primary_node(backend_nodes)
                cache_primary = self._get_primary_node(cache_nodes)
                
                if backend_primary and cache_primary and backend_primary != cache_primary:
                    self.add_edge(backend_primary, cache_primary, type="inferred-dependency", confidence=0.7)
                    self.logger.debug(f"Inferred backend->cache: {backend_primary} -> {cache_primary}")
        
        # Pattern 4: Backend -> Queue
        for backend_name, backend_nodes in backend_groups:
            for queue_name, queue_nodes in queue_groups:
                # Get the primary nodes from each group
                backend_primary = self._get_primary_node(backend_nodes)
                queue_primary = self._get_primary_node(queue_nodes)
                
                if backend_primary and queue_primary and backend_primary != queue_primary:
                    self.add_edge(backend_primary, queue_primary, type="inferred-dependency", confidence=0.7)
                    self.logger.debug(f"Inferred backend->queue: {backend_primary} -> {queue_primary}")
        
        self.logger.info(f"Inference complete. Added relationships based on common patterns.")
    
    def _group_related_services(self) -> Dict[str, List[str]]:
        """
        Group related services together (e.g., backend and backend-service).
        
        Returns:
            Dictionary mapping base service names to lists of node IDs
        """
        service_groups = {}
        
        for node_id in self.get_nodes():
            # Extract the service name without namespace
            full_name = node_id.split('/')[-1]
            
            # Remove common suffixes
            base_name = full_name
            for suffix in ['-service', '-deployment', '-pod', '-container', '-db', '-cache', '-queue']:
                if base_name.endswith(suffix):
                    base_name = base_name[:-len(suffix)]
                    break
            
            # Add to the appropriate group
            if base_name not in service_groups:
                service_groups[base_name] = []
            service_groups[base_name].append(node_id)
        
        return service_groups
    
    def _get_primary_node(self, nodes: List[str]) -> Optional[str]:
        """
        Get the primary node from a group of related nodes.
        
        Args:
            nodes: List of node IDs
            
        Returns:
            The primary node ID, or None if no suitable node is found
        """
        if not nodes:
            return None
        
        # Prefer Kubernetes Deployment over Service
        for node_id in nodes:
            kind = self.get_node_attribute(node_id, 'kind')
            if kind in ['Deployment', 'StatefulSet', 'DaemonSet']:
                return node_id
        
        # Otherwise, prefer Service over other types
        for node_id in nodes:
            kind = self.get_node_attribute(node_id, 'kind')
            if kind == 'Service':
                return node_id
        
        # If no Deployment or Service, just return the first node
        return nodes[0]