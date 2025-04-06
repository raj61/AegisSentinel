"""
Detection Engine
==============

This module provides the DetectionEngine class for detecting issues in service graphs.
"""

import logging
import networkx as nx
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Set, Any, Optional, Union, Tuple

from src.graph import ServiceGraph

logger = logging.getLogger(__name__)

class IssueType(Enum):
    """Types of issues that can be detected in a service graph."""
    SINGLE_POINT_OF_FAILURE = "single_point_of_failure"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    ORPHANED_SERVICE = "orphaned_service"
    HIGH_FANOUT = "high_fanout"
    HIGH_FANIN = "high_fanin"
    MISSING_DEPENDENCY = "missing_dependency"
    NETWORK_PARTITION = "network_partition"
    UNREACHABLE_SERVICE = "unreachable_service"
    SECURITY_GROUP_MISCONFIGURATION = "security_group_misconfiguration"
    RESOURCE_LIMIT = "resource_limit"
    LOAD_BALANCER_MISCONFIGURATION = "load_balancer_misconfiguration"
    DATABASE_SINGLE_INSTANCE = "database_single_instance"
    MISSING_HEALTH_CHECK = "missing_health_check"
    MISSING_AUTOSCALING = "missing_autoscaling"
    MISSING_MONITORING = "missing_monitoring"
    MISSING_LOGGING = "missing_logging"
    MISSING_BACKUP = "missing_backup"
    MISSING_DISASTER_RECOVERY = "missing_disaster_recovery"
    MISSING_SECURITY_POLICY = "missing_security_policy"
    MISSING_ENCRYPTION = "missing_encryption"
    MISSING_AUTHENTICATION = "missing_authentication"
    MISSING_AUTHORIZATION = "missing_authorization"
    MISSING_RATE_LIMITING = "missing_rate_limiting"
    MISSING_CIRCUIT_BREAKER = "missing_circuit_breaker"
    MISSING_RETRY_POLICY = "missing_retry_policy"
    MISSING_TIMEOUT_POLICY = "missing_timeout_policy"
    MISSING_FALLBACK_POLICY = "missing_fallback_policy"
    CUSTOM = "custom"

@dataclass
class Issue:
    """
    Represents an issue detected in a service graph.
    
    Attributes:
        type: Type of the issue
        severity: Severity level (1-5, with 5 being the most severe)
        description: Human-readable description of the issue
        affected_nodes: List of node IDs affected by the issue
        affected_edges: List of (source, target) tuples representing affected edges
        metadata: Additional metadata about the issue
        detected_at: Timestamp when the issue was detected
        mitigated_at: Timestamp when the issue was mitigated (None if not mitigated)
        mitigation_action: Description of the action taken to mitigate the issue (None if not mitigated)
        status: Current status of the issue (e.g., "detected", "mitigating", "mitigated")
    """
    type: IssueType
    severity: int
    description: str
    affected_nodes: List[str]
    affected_edges: List[Tuple[str, str]]
    metadata: Dict[str, Any]
    detected_at: Optional[str] = None
    mitigated_at: Optional[str] = None
    mitigation_action: Optional[str] = None
    status: str = "detected"

class DetectionEngine:
    """
    Engine for detecting issues in service graphs.
    
    This class provides methods for detecting various types of issues in a service graph,
    such as single points of failure, circular dependencies, etc.
    """
    
    def __init__(self, service_graph: ServiceGraph):
        """
        Initialize the detection engine.
        
        Args:
            service_graph: ServiceGraph instance to analyze
        """
        self.service_graph = service_graph
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Import datetime here to avoid circular imports
        from datetime import datetime
        self.datetime = datetime
    
    def detect_issues(self) -> List[Issue]:
        """
        Detect all issues in the service graph.
        
        Returns:
            List of detected issues
        """
        issues = []
        
        # Run all detection methods
        issues.extend(self.detect_single_points_of_failure())
        issues.extend(self.detect_circular_dependencies())
        issues.extend(self.detect_orphaned_services())
        issues.extend(self.detect_high_fanout())
        issues.extend(self.detect_high_fanin())
        issues.extend(self.detect_network_partitions())
        issues.extend(self.detect_database_single_instances())
        issues.extend(self.detect_missing_health_checks())
        issues.extend(self.detect_missing_autoscaling())
        
        return issues
    
    def detect_single_points_of_failure(self) -> List[Issue]:
        """
        Detect single points of failure in the service graph.
        
        A single point of failure is a node that, if removed, would disconnect the graph.
        
        Returns:
            List of detected issues
        """
        issues = []
        
        # Convert the directed graph to an undirected graph for articulation point analysis
        undirected_graph = self.service_graph.graph.to_undirected()
        
        # Find articulation points (nodes that would disconnect the graph if removed)
        articulation_points = list(nx.articulation_points(undirected_graph))
        
        for node_id in articulation_points:
            # Get node details
            node = self.service_graph.get_node(node_id)
            name = node.get('name', node_id)
            kind = node.get('kind', node.get('type', 'unknown'))
            
            # Create an issue with timestamp
            current_time = self.datetime.now().isoformat()
            issue = Issue(
                type=IssueType.SINGLE_POINT_OF_FAILURE,
                severity=4,  # High severity
                description=f"Single point of failure detected: {name} ({kind})",
                affected_nodes=[node_id],
                affected_edges=[],
                metadata={
                    'node_details': node,
                    'incoming_connections': len(list(self.service_graph.graph.predecessors(node_id))),
                    'outgoing_connections': len(list(self.service_graph.graph.successors(node_id)))
                },
                detected_at=current_time,
                status="detected"
            )
            
            issues.append(issue)
        
        return issues
    
    def detect_circular_dependencies(self) -> List[Issue]:
        """
        Detect circular dependencies in the service graph.
        
        A circular dependency is a cycle in the directed graph.
        
        Returns:
            List of detected issues
        """
        issues = []
        
        # Find all simple cycles in the graph
        try:
            cycles = list(nx.simple_cycles(self.service_graph.graph))
            
            for cycle in cycles:
                if len(cycle) > 1:  # Ignore self-loops
                    # Create edges for the cycle
                    cycle_edges = []
                    for i in range(len(cycle)):
                        source = cycle[i]
                        target = cycle[(i + 1) % len(cycle)]
                        cycle_edges.append((source, target))
                    
                    # Create an issue
                    issue = Issue(
                        type=IssueType.CIRCULAR_DEPENDENCY,
                        severity=3,  # Medium severity
                        description=f"Circular dependency detected: {' -> '.join(cycle)} -> {cycle[0]}",
                        affected_nodes=cycle,
                        affected_edges=cycle_edges,
                        metadata={
                            'cycle_length': len(cycle)
                        }
                    )
                    
                    issues.append(issue)
        
        except nx.NetworkXNoCycle:
            # No cycles found
            pass
        
        return issues
    
    def detect_orphaned_services(self) -> List[Issue]:
        """
        Detect orphaned services in the service graph.
        
        An orphaned service is a node with no incoming or outgoing edges.
        
        Returns:
            List of detected issues
        """
        issues = []
        
        for node_id in self.service_graph.get_nodes():
            predecessors = list(self.service_graph.graph.predecessors(node_id))
            successors = list(self.service_graph.graph.successors(node_id))
            
            if not predecessors and not successors:
                # Get node details
                node = self.service_graph.get_node(node_id)
                name = node.get('name', node_id)
                kind = node.get('kind', node.get('type', 'unknown'))
                
                # Create an issue
                issue = Issue(
                    type=IssueType.ORPHANED_SERVICE,
                    severity=2,  # Low severity
                    description=f"Orphaned service detected: {name} ({kind})",
                    affected_nodes=[node_id],
                    affected_edges=[],
                    metadata={
                        'node_details': node
                    }
                )
                
                issues.append(issue)
        
        return issues
    
    def detect_high_fanout(self, threshold: int = 10) -> List[Issue]:
        """
        Detect services with high fanout (many outgoing connections).
        
        Args:
            threshold: Threshold for considering a fanout as high
            
        Returns:
            List of detected issues
        """
        issues = []
        
        for node_id in self.service_graph.get_nodes():
            successors = list(self.service_graph.graph.successors(node_id))
            
            if len(successors) >= threshold:
                # Get node details
                node = self.service_graph.get_node(node_id)
                name = node.get('name', node_id)
                kind = node.get('kind', node.get('type', 'unknown'))
                
                # Create edges for the fanout
                fanout_edges = [(node_id, succ) for succ in successors]
                
                # Create an issue
                issue = Issue(
                    type=IssueType.HIGH_FANOUT,
                    severity=2,  # Low severity
                    description=f"High fanout detected: {name} ({kind}) has {len(successors)} outgoing connections",
                    affected_nodes=[node_id] + successors,
                    affected_edges=fanout_edges,
                    metadata={
                        'node_details': node,
                        'fanout_count': len(successors)
                    }
                )
                
                issues.append(issue)
        
        return issues
    
    def detect_high_fanin(self, threshold: int = 10) -> List[Issue]:
        """
        Detect services with high fanin (many incoming connections).
        
        Args:
            threshold: Threshold for considering a fanin as high
            
        Returns:
            List of detected issues
        """
        issues = []
        
        for node_id in self.service_graph.get_nodes():
            predecessors = list(self.service_graph.graph.predecessors(node_id))
            
            if len(predecessors) >= threshold:
                # Get node details
                node = self.service_graph.get_node(node_id)
                name = node.get('name', node_id)
                kind = node.get('kind', node.get('type', 'unknown'))
                
                # Create edges for the fanin
                fanin_edges = [(pred, node_id) for pred in predecessors]
                
                # Create an issue
                issue = Issue(
                    type=IssueType.HIGH_FANIN,
                    severity=2,  # Low severity
                    description=f"High fanin detected: {name} ({kind}) has {len(predecessors)} incoming connections",
                    affected_nodes=[node_id] + predecessors,
                    affected_edges=fanin_edges,
                    metadata={
                        'node_details': node,
                        'fanin_count': len(predecessors)
                    }
                )
                
                issues.append(issue)
        
        return issues
    
    def detect_network_partitions(self) -> List[Issue]:
        """
        Detect network partitions in the service graph.
        
        A network partition is a disconnected component in the undirected graph.
        
        Returns:
            List of detected issues
        """
        issues = []
        
        # Convert the directed graph to an undirected graph for connected component analysis
        undirected_graph = self.service_graph.graph.to_undirected()
        
        # Find connected components
        connected_components = list(nx.connected_components(undirected_graph))
        
        if len(connected_components) > 1:
            # Multiple connected components found
            for i, component in enumerate(connected_components):
                component_nodes = list(component)
                
                # Create an issue
                issue = Issue(
                    type=IssueType.NETWORK_PARTITION,
                    severity=3,  # Medium severity
                    description=f"Network partition detected: Component {i+1} with {len(component_nodes)} services",
                    affected_nodes=component_nodes,
                    affected_edges=[],
                    metadata={
                        'component_index': i,
                        'component_size': len(component_nodes)
                    }
                )
                
                issues.append(issue)
        
        return issues
    
    def detect_database_single_instances(self) -> List[Issue]:
        """
        Detect database services that are single instances (not replicated).
        
        Returns:
            List of detected issues
        """
        issues = []
        
        for node_id in self.service_graph.get_nodes():
            node = self.service_graph.get_node(node_id)
            
            # Check if it's a database service
            is_database = False
            
            # Check by category
            category = node.get('category')
            if category in ['database', 'cache']:
                is_database = True
            
            # Check by type or kind
            type_or_kind = node.get('type', node.get('kind', ''))
            if any(db_type in type_or_kind.lower() for db_type in ['db', 'database', 'sql', 'nosql', 'redis', 'memcached', 'cache']):
                is_database = True
            
            if is_database:
                # Check if it's replicated
                is_replicated = False
                
                # Check for replication indicators in attributes
                for attr, value in node.items():
                    if isinstance(value, str) and any(rep in attr.lower() for rep in ['replica', 'replication', 'cluster']):
                        is_replicated = True
                        break
                
                if not is_replicated:
                    name = node.get('name', node_id)
                    
                    # Create an issue
                    issue = Issue(
                        type=IssueType.DATABASE_SINGLE_INSTANCE,
                        severity=4,  # High severity
                        description=f"Database single instance detected: {name} is not replicated",
                        affected_nodes=[node_id],
                        affected_edges=[],
                        metadata={
                            'node_details': node
                        }
                    )
                    
                    issues.append(issue)
        
        return issues
    
    def detect_missing_health_checks(self) -> List[Issue]:
        """
        Detect services that might be missing health checks.
        
        Returns:
            List of detected issues
        """
        issues = []
        
        for node_id in self.service_graph.get_nodes():
            node = self.service_graph.get_node(node_id)
            
            # Skip certain types of resources that don't typically have health checks
            category = node.get('category', '')
            if category in ['network', 'security', 'storage']:
                continue
            
            # Check if there's any indication of health checks
            has_health_check = False
            
            # Check node attributes for health check indicators
            for attr, value in node.items():
                if isinstance(attr, str) and 'health' in attr.lower():
                    has_health_check = True
                    break
            
            if not has_health_check:
                # Check if it's a service that should have health checks
                type_or_kind = node.get('type', node.get('kind', ''))
                if any(svc_type in type_or_kind.lower() for svc_type in ['service', 'deployment', 'instance', 'function', 'container']):
                    name = node.get('name', node_id)
                    
                    # Create an issue
                    issue = Issue(
                        type=IssueType.MISSING_HEALTH_CHECK,
                        severity=3,  # Medium severity
                        description=f"Missing health check detected: {name} does not have a health check configured",
                        affected_nodes=[node_id],
                        affected_edges=[],
                        metadata={
                            'node_details': node
                        }
                    )
                    
                    issues.append(issue)
        
        return issues
    
    def detect_missing_autoscaling(self) -> List[Issue]:
        """
        Detect services that might be missing autoscaling configuration.
        
        Returns:
            List of detected issues
        """
        issues = []
        
        for node_id in self.service_graph.get_nodes():
            node = self.service_graph.get_node(node_id)
            
            # Skip certain types of resources that don't typically have autoscaling
            category = node.get('category', '')
            if category in ['network', 'security', 'storage', 'data']:
                continue
            
            # Check if there's any indication of autoscaling
            has_autoscaling = False
            
            # Check node attributes for autoscaling indicators
            for attr, value in node.items():
                if isinstance(attr, str) and any(scale in attr.lower() for scale in ['scale', 'auto', 'replica']):
                    has_autoscaling = True
                    break
            
            if not has_autoscaling:
                # Check if it's a service that should have autoscaling
                type_or_kind = node.get('type', node.get('kind', ''))
                if any(svc_type in type_or_kind.lower() for svc_type in ['service', 'deployment', 'instance', 'function', 'container']):
                    name = node.get('name', node_id)
                    
                    # Create an issue
                    issue = Issue(
                        type=IssueType.MISSING_AUTOSCALING,
                        severity=2,  # Low severity
                        description=f"Missing autoscaling detected: {name} does not have autoscaling configured",
                        affected_nodes=[node_id],
                        affected_edges=[],
                        metadata={
                            'node_details': node
                        }
                    )
                    
                    issues.append(issue)
        
        return issues