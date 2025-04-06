"""
Resolution Engine
===============

This module provides the ResolutionEngine class for resolving issues in service graphs.
"""

import logging
import networkx as nx
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Set, Any, Optional, Union, Tuple

from src.graph import ServiceGraph
from src.detection import DetectionEngine, Issue, IssueType

logger = logging.getLogger(__name__)

class ResolutionStatus(Enum):
    """Status of a resolution attempt."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    MANUAL_INTERVENTION_REQUIRED = "manual_intervention_required"

@dataclass
class Resolution:
    """
    Represents a resolution for an issue in a service graph.
    
    Attributes:
        issue: The issue being resolved
        status: Status of the resolution attempt
        description: Human-readable description of the resolution
        changes: List of changes made to resolve the issue
        metadata: Additional metadata about the resolution
        timestamp: When the resolution was applied
        duration_ms: How long the resolution took to apply (in milliseconds)
    """
    issue: Issue
    status: ResolutionStatus
    description: str
    changes: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: Optional[str] = None
    duration_ms: Optional[int] = None

class ResolutionEngine:
    """
    Engine for resolving issues in service graphs.
    
    This class provides methods for resolving various types of issues in a service graph,
    such as single points of failure, circular dependencies, etc.
    """
    
    def __init__(self, service_graph: ServiceGraph):
        """
        Initialize the resolution engine.
        
        Args:
            service_graph: ServiceGraph instance to modify
        """
        self.service_graph = service_graph
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Import datetime here to avoid circular imports
        from datetime import datetime
        self.datetime = datetime
    
    def resolve_issues(self, issues: List[Issue]) -> List[Resolution]:
        """
        Resolve a list of issues in the service graph.
        
        Args:
            issues: List of issues to resolve
            
        Returns:
            List of resolutions
        """
        resolutions = []
        
        for issue in issues:
            resolution = self.resolve_issue(issue)
            if resolution:
                resolutions.append(resolution)
        
        return resolutions
    
    def resolve_issue(self, issue: Issue) -> Optional[Resolution]:
        """
        Resolve a single issue in the service graph.
        
        Args:
            issue: Issue to resolve
            
        Returns:
            Resolution if successful, None otherwise
        """
        # Dispatch to the appropriate resolution method based on issue type
        if issue.type == IssueType.SINGLE_POINT_OF_FAILURE:
            return self.resolve_single_point_of_failure(issue)
        elif issue.type == IssueType.CIRCULAR_DEPENDENCY:
            return self.resolve_circular_dependency(issue)
        elif issue.type == IssueType.ORPHANED_SERVICE:
            return self.resolve_orphaned_service(issue)
        elif issue.type == IssueType.HIGH_FANOUT:
            return self.resolve_high_fanout(issue)
        elif issue.type == IssueType.HIGH_FANIN:
            return self.resolve_high_fanin(issue)
        elif issue.type == IssueType.NETWORK_PARTITION:
            return self.resolve_network_partition(issue)
        elif issue.type == IssueType.DATABASE_SINGLE_INSTANCE:
            return self.resolve_database_single_instance(issue)
        elif issue.type == IssueType.MISSING_HEALTH_CHECK:
            return self.resolve_missing_health_check(issue)
        elif issue.type == IssueType.MISSING_AUTOSCALING:
            return self.resolve_missing_autoscaling(issue)
        else:
            self.logger.warning(f"No resolution method available for issue type: {issue.type}")
            return None
    
    def resolve_single_point_of_failure(self, issue: Issue) -> Resolution:
        """
        Resolve a single point of failure issue.
        
        Args:
            issue: Issue to resolve
            
        Returns:
            Resolution
        """
        # For a single point of failure, we suggest adding redundancy
        if not issue.affected_nodes:
            return Resolution(
                issue=issue,
                status=ResolutionStatus.FAILED,
                description="Cannot resolve single point of failure: no affected nodes",
                changes=[],
                metadata={}
            )
        
        node_id = issue.affected_nodes[0]
        node = self.service_graph.get_node(node_id)
        name = node.get('name', node_id)
        kind = node.get('kind', node.get('type', 'unknown'))
        
        # Generate a suggestion for adding redundancy
        suggestion = f"Add redundancy for {name} ({kind}) by:"
        
        if kind.lower() in ['deployment', 'statefulset', 'daemonset']:
            suggestion += "\n- Increasing the number of replicas"
            suggestion += "\n- Adding pod anti-affinity rules to ensure pods are scheduled on different nodes"
            suggestion += "\n- Using a PodDisruptionBudget to ensure minimum availability during disruptions"
        elif kind.lower() in ['service']:
            suggestion += "\n- Ensuring the service has multiple backend pods"
            suggestion += "\n- Using a multi-zone or multi-region deployment"
        elif 'database' in kind.lower() or kind.lower() in ['rds', 'dynamodb', 'cosmosdb']:
            suggestion += "\n- Setting up database replication or clustering"
            suggestion += "\n- Implementing read replicas"
            suggestion += "\n- Setting up automatic failover"
        else:
            suggestion += "\n- Deploying multiple instances across different availability zones"
            suggestion += "\n- Implementing a load balancer to distribute traffic"
            suggestion += "\n- Setting up automatic failover mechanisms"
        
        # Get current timestamp
        current_time = self.datetime.now().isoformat()
        
        # Update the issue status and add mitigation timestamp
        issue.status = "mitigating"
        issue.mitigation_action = "Adding redundancy"
        
        # Return a resolution that requires manual intervention
        return Resolution(
            issue=issue,
            status=ResolutionStatus.MANUAL_INTERVENTION_REQUIRED,
            description=f"Single point of failure resolution for {name} requires manual intervention",
            changes=[],
            metadata={
                'suggestion': suggestion
            },
            timestamp=current_time
        )
    
    def resolve_circular_dependency(self, issue: Issue) -> Resolution:
        """
        Resolve a circular dependency issue.
        
        Args:
            issue: Issue to resolve
            
        Returns:
            Resolution
        """
        # For a circular dependency, we suggest breaking the cycle
        if not issue.affected_edges:
            return Resolution(
                issue=issue,
                status=ResolutionStatus.FAILED,
                description="Cannot resolve circular dependency: no affected edges",
                changes=[],
                metadata={}
            )
        
        # Find the edge to break (we'll choose the one with the least impact)
        edge_to_break = issue.affected_edges[0]  # Default to the first edge
        min_impact = float('inf')
        
        for source, target in issue.affected_edges:
            # Calculate the impact of breaking this edge
            # Impact is measured by the number of paths that would be affected
            impact = 0
            for s in self.service_graph.get_nodes():
                for t in self.service_graph.get_nodes():
                    if s != source or t != target:  # Skip the edge we're considering breaking
                        try:
                            # Count paths that would be affected
                            paths = list(nx.all_simple_paths(self.service_graph.graph, s, t))
                            for path in paths:
                                if source in path and target in path and path.index(source) + 1 == path.index(target):
                                    impact += 1
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            pass
            
            if impact < min_impact:
                min_impact = impact
                edge_to_break = (source, target)
        
        source, target = edge_to_break
        source_name = self.service_graph.get_node_attribute(source, 'name', source)
        target_name = self.service_graph.get_node_attribute(target, 'name', target)
        
        # Generate a suggestion for breaking the cycle
        suggestion = f"Break the circular dependency by modifying the relationship between {source_name} and {target_name}:"
        suggestion += "\n- Consider introducing an asynchronous communication pattern"
        suggestion += "\n- Use an event-driven architecture with a message queue"
        suggestion += "\n- Extract shared functionality into a separate service"
        suggestion += "\n- Implement the Circuit Breaker pattern"
        suggestion += "\n- Use the Bulkhead pattern to isolate failures"
        
        # Return a resolution that requires manual intervention
        return Resolution(
            issue=issue,
            status=ResolutionStatus.MANUAL_INTERVENTION_REQUIRED,
            description=f"Circular dependency resolution requires breaking the dependency between {source_name} and {target_name}",
            changes=[],
            metadata={
                'suggestion': suggestion,
                'edge_to_break': edge_to_break
            }
        )
    
    def resolve_orphaned_service(self, issue: Issue) -> Resolution:
        """
        Resolve an orphaned service issue.
        
        Args:
            issue: Issue to resolve
            
        Returns:
            Resolution
        """
        # For an orphaned service, we suggest connecting it to the graph
        if not issue.affected_nodes:
            return Resolution(
                issue=issue,
                status=ResolutionStatus.FAILED,
                description="Cannot resolve orphaned service: no affected nodes",
                changes=[],
                metadata={}
            )
        
        node_id = issue.affected_nodes[0]
        node = self.service_graph.get_node(node_id)
        name = node.get('name', node_id)
        kind = node.get('kind', node.get('type', 'unknown'))
        
        # Find potential services to connect to
        potential_connections = []
        for other_id in self.service_graph.get_nodes():
            if other_id != node_id:
                other = self.service_graph.get_node(other_id)
                other_name = other.get('name', other_id)
                other_kind = other.get('kind', other.get('type', 'unknown'))
                
                # Check if the service types are compatible
                if self._are_services_compatible(kind, other_kind):
                    potential_connections.append((other_id, other_name, other_kind))
        
        # Generate a suggestion for connecting the orphaned service
        suggestion = f"Connect the orphaned service {name} ({kind}) to the service graph:"
        
        if potential_connections:
            suggestion += "\n\nPotential connections:"
            for other_id, other_name, other_kind in potential_connections[:5]:  # Limit to 5 suggestions
                suggestion += f"\n- Connect to {other_name} ({other_kind})"
        else:
            suggestion += "\n- No compatible services found in the graph"
            suggestion += "\n- Consider adding a new service that can interact with this one"
            suggestion += "\n- Or review if this service is actually needed"
        
        # Return a resolution that requires manual intervention
        return Resolution(
            issue=issue,
            status=ResolutionStatus.MANUAL_INTERVENTION_REQUIRED,
            description=f"Orphaned service resolution for {name} requires manual intervention",
            changes=[],
            metadata={
                'suggestion': suggestion,
                'potential_connections': potential_connections
            }
        )
    
    def resolve_high_fanout(self, issue: Issue) -> Resolution:
        """
        Resolve a high fanout issue.
        
        Args:
            issue: Issue to resolve
            
        Returns:
            Resolution
        """
        # For high fanout, we suggest introducing intermediary services or a facade
        if not issue.affected_nodes:
            return Resolution(
                issue=issue,
                status=ResolutionStatus.FAILED,
                description="Cannot resolve high fanout: no affected nodes",
                changes=[],
                metadata={}
            )
        
        node_id = issue.affected_nodes[0]  # The service with high fanout
        node = self.service_graph.get_node(node_id)
        name = node.get('name', node_id)
        kind = node.get('kind', node.get('type', 'unknown'))
        
        # Get the outgoing connections
        outgoing = []
        for target in self.service_graph.get_successors(node_id):
            target_name = self.service_graph.get_node_attribute(target, 'name', target)
            target_kind = self.service_graph.get_node_attribute(target, 'kind', self.service_graph.get_node_attribute(target, 'type', 'unknown'))
            outgoing.append((target, target_name, target_kind))
        
        # Group outgoing connections by type
        grouped_connections = {}
        for target, target_name, target_kind in outgoing:
            if target_kind not in grouped_connections:
                grouped_connections[target_kind] = []
            grouped_connections[target_kind].append((target, target_name))
        
        # Generate a suggestion for reducing fanout
        suggestion = f"Reduce the fanout of {name} ({kind}) by:"
        suggestion += "\n- Introducing intermediary services or a facade pattern"
        suggestion += "\n- Using a message broker or event bus for communication"
        suggestion += "\n- Implementing service aggregation"
        
        if grouped_connections:
            suggestion += "\n\nPotential groupings by service type:"
            for group_kind, targets in grouped_connections.items():
                if len(targets) > 1:
                    target_names = [name for _, name in targets]
                    suggestion += f"\n- Group {len(targets)} {group_kind} services: {', '.join(target_names[:3])}"
                    if len(target_names) > 3:
                        suggestion += f" and {len(target_names) - 3} more"
        
        # Return a resolution that requires manual intervention
        return Resolution(
            issue=issue,
            status=ResolutionStatus.MANUAL_INTERVENTION_REQUIRED,
            description=f"High fanout resolution for {name} requires manual intervention",
            changes=[],
            metadata={
                'suggestion': suggestion,
                'grouped_connections': grouped_connections
            }
        )
    
    def resolve_high_fanin(self, issue: Issue) -> Resolution:
        """
        Resolve a high fanin issue.
        
        Args:
            issue: Issue to resolve
            
        Returns:
            Resolution
        """
        # For high fanin, we suggest introducing a load balancer or a proxy
        if not issue.affected_nodes:
            return Resolution(
                issue=issue,
                status=ResolutionStatus.FAILED,
                description="Cannot resolve high fanin: no affected nodes",
                changes=[],
                metadata={}
            )
        
        node_id = issue.affected_nodes[0]  # The service with high fanin
        node = self.service_graph.get_node(node_id)
        name = node.get('name', node_id)
        kind = node.get('kind', node.get('type', 'unknown'))
        
        # Get the incoming connections
        incoming = []
        for source in self.service_graph.get_predecessors(node_id):
            source_name = self.service_graph.get_node_attribute(source, 'name', source)
            source_kind = self.service_graph.get_node_attribute(source, 'kind', self.service_graph.get_node_attribute(source, 'type', 'unknown'))
            incoming.append((source, source_name, source_kind))
        
        # Generate a suggestion for reducing fanin
        suggestion = f"Reduce the fanin of {name} ({kind}) by:"
        suggestion += "\n- Introducing a load balancer or API gateway"
        suggestion += "\n- Implementing a service mesh"
        suggestion += "\n- Using a caching layer"
        suggestion += "\n- Scaling the service horizontally"
        suggestion += "\n- Implementing rate limiting and circuit breakers"
        
        # Return a resolution that requires manual intervention
        return Resolution(
            issue=issue,
            status=ResolutionStatus.MANUAL_INTERVENTION_REQUIRED,
            description=f"High fanin resolution for {name} requires manual intervention",
            changes=[],
            metadata={
                'suggestion': suggestion,
                'incoming_connections': len(incoming)
            }
        )
    
    def resolve_network_partition(self, issue: Issue) -> Resolution:
        """
        Resolve a network partition issue.
        
        Args:
            issue: Issue to resolve
            
        Returns:
            Resolution
        """
        # For a network partition, we suggest connecting the partitions
        if not issue.affected_nodes or len(issue.affected_nodes) < 2:
            return Resolution(
                issue=issue,
                status=ResolutionStatus.FAILED,
                description="Cannot resolve network partition: insufficient affected nodes",
                changes=[],
                metadata={}
            )
        
        # Find another partition to connect to
        other_partitions = []
        for other_issue in issue.metadata.get('related_issues', []):
            if other_issue.type == IssueType.NETWORK_PARTITION and other_issue != issue:
                other_partitions.append(other_issue)
        
        # Generate a suggestion for connecting the partitions
        suggestion = f"Connect the network partition (component {issue.metadata.get('component_index', 'unknown')}) to the rest of the service graph:"
        
        if other_partitions:
            suggestion += "\n\nPotential connections between partitions:"
            for i, other_issue in enumerate(other_partitions[:3]):  # Limit to 3 suggestions
                other_component = other_issue.metadata.get('component_index', 'unknown')
                suggestion += f"\n- Connect component {issue.metadata.get('component_index', 'unknown')} to component {other_component}"
                
                # Suggest specific services to connect
                if issue.affected_nodes and other_issue.affected_nodes:
                    node_id = issue.affected_nodes[0]
                    other_node_id = other_issue.affected_nodes[0]
                    
                    node_name = self.service_graph.get_node_attribute(node_id, 'name', node_id)
                    other_node_name = self.service_graph.get_node_attribute(other_node_id, 'name', other_node_id)
                    
                    suggestion += f"\n  - For example, connect {node_name} to {other_node_name}"
        else:
            suggestion += "\n- No other partitions found to connect to"
            suggestion += "\n- Consider if this partition should be a separate system"
            suggestion += "\n- Or add a new service that can bridge the partitions"
        
        # Return a resolution that requires manual intervention
        return Resolution(
            issue=issue,
            status=ResolutionStatus.MANUAL_INTERVENTION_REQUIRED,
            description=f"Network partition resolution requires manual intervention",
            changes=[],
            metadata={
                'suggestion': suggestion
            }
        )
    
    def resolve_database_single_instance(self, issue: Issue) -> Resolution:
        """
        Resolve a database single instance issue.
        
        Args:
            issue: Issue to resolve
            
        Returns:
            Resolution
        """
        # For a database single instance, we suggest adding replication
        if not issue.affected_nodes:
            return Resolution(
                issue=issue,
                status=ResolutionStatus.FAILED,
                description="Cannot resolve database single instance: no affected nodes",
                changes=[],
                metadata={}
            )
        
        node_id = issue.affected_nodes[0]
        node = self.service_graph.get_node(node_id)
        name = node.get('name', node_id)
        kind = node.get('kind', node.get('type', 'unknown'))
        
        # Generate a suggestion for adding replication
        suggestion = f"Add replication for the database {name} ({kind}) by:"
        
        if 'aws' in kind.lower() or node.get('provider', '').lower() == 'aws':
            suggestion += "\n- For RDS: Enable Multi-AZ deployment"
            suggestion += "\n- For DynamoDB: Enable global tables"
            suggestion += "\n- For ElastiCache: Set up a replication group"
        elif 'azure' in kind.lower() or node.get('provider', '').lower() == 'azure':
            suggestion += "\n- For Azure SQL: Enable geo-replication"
            suggestion += "\n- For Cosmos DB: Configure multi-region writes"
            suggestion += "\n- For Azure Cache for Redis: Use Premium tier with clustering"
        elif 'google' in kind.lower() or node.get('provider', '').lower() == 'google':
            suggestion += "\n- For Cloud SQL: Set up read replicas and high availability"
            suggestion += "\n- For Firestore: Use multi-region configuration"
            suggestion += "\n- For Memorystore: Enable high availability"
        else:
            suggestion += "\n- Set up primary-replica replication"
            suggestion += "\n- Configure automatic failover"
            suggestion += "\n- Implement regular backups"
            suggestion += "\n- Consider using a managed database service with built-in replication"
        
        # Return a resolution that requires manual intervention
        return Resolution(
            issue=issue,
            status=ResolutionStatus.MANUAL_INTERVENTION_REQUIRED,
            description=f"Database single instance resolution for {name} requires manual intervention",
            changes=[],
            metadata={
                'suggestion': suggestion
            }
        )
    
    def resolve_missing_health_check(self, issue: Issue) -> Resolution:
        """
        Resolve a missing health check issue.
        
        Args:
            issue: Issue to resolve
            
        Returns:
            Resolution
        """
        # For a missing health check, we suggest adding health check endpoints
        if not issue.affected_nodes:
            return Resolution(
                issue=issue,
                status=ResolutionStatus.FAILED,
                description="Cannot resolve missing health check: no affected nodes",
                changes=[],
                metadata={}
            )
        
        node_id = issue.affected_nodes[0]
        node = self.service_graph.get_node(node_id)
        name = node.get('name', node_id)
        kind = node.get('kind', node.get('type', 'unknown'))
        
        # Generate a suggestion for adding health checks
        suggestion = f"Add health checks for {name} ({kind}) by:"
        
        if kind.lower() in ['deployment', 'statefulset', 'daemonset', 'pod']:
            suggestion += "\n- Adding readiness and liveness probes to the Kubernetes manifest"
            suggestion += "\n- Implementing a /health or /status endpoint in your application"
            suggestion += "\n- Configuring appropriate timeouts and thresholds"
        elif 'function' in kind.lower():
            suggestion += "\n- Implementing a health check endpoint"
            suggestion += "\n- Setting up CloudWatch alarms (AWS) or equivalent monitoring"
            suggestion += "\n- Configuring appropriate retry policies"
        else:
            suggestion += "\n- Implementing a health check endpoint in your application"
            suggestion += "\n- Setting up external monitoring to periodically check the endpoint"
            suggestion += "\n- Configuring alerts for when the health check fails"
        
        # Return a resolution that requires manual intervention
        return Resolution(
            issue=issue,
            status=ResolutionStatus.MANUAL_INTERVENTION_REQUIRED,
            description=f"Missing health check resolution for {name} requires manual intervention",
            changes=[],
            metadata={
                'suggestion': suggestion
            }
        )
    
    def resolve_missing_autoscaling(self, issue: Issue) -> Resolution:
        """
        Resolve a missing autoscaling issue.
        
        Args:
            issue: Issue to resolve
            
        Returns:
            Resolution
        """
        # For missing autoscaling, we suggest adding autoscaling configuration
        if not issue.affected_nodes:
            return Resolution(
                issue=issue,
                status=ResolutionStatus.FAILED,
                description="Cannot resolve missing autoscaling: no affected nodes",
                changes=[],
                metadata={}
            )
        
        node_id = issue.affected_nodes[0]
        node = self.service_graph.get_node(node_id)
        name = node.get('name', node_id)
        kind = node.get('kind', node.get('type', 'unknown'))
        
        # Generate a suggestion for adding autoscaling
        suggestion = f"Add autoscaling for {name} ({kind}) by:"
        
        if kind.lower() in ['deployment', 'statefulset', 'replicaset']:
            suggestion += "\n- Adding a Horizontal Pod Autoscaler (HPA) to the Kubernetes manifest"
            suggestion += "\n- Configuring resource requests and limits"
            suggestion += "\n- Setting appropriate min and max replicas"
            suggestion += "\n- Choosing appropriate metrics (CPU, memory, custom metrics)"
        elif 'function' in kind.lower():
            suggestion += "\n- Enabling provisioned concurrency (AWS Lambda)"
            suggestion += "\n- Configuring appropriate memory and timeout settings"
            suggestion += "\n- Setting up auto-scaling based on invocation metrics"
        elif 'instance' in kind.lower() or 'vm' in kind.lower():
            suggestion += "\n- Setting up an auto-scaling group (AWS) or VM scale set (Azure)"
            suggestion += "\n- Configuring appropriate scaling policies based on CPU, memory, or custom metrics"
            suggestion += "\n- Setting appropriate min and max instance counts"
            suggestion += "\n- Configuring scale-in protection if needed"
        else:
            suggestion += "\n- Implementing a scaling mechanism appropriate for the service type"
            suggestion += "\n- Setting up monitoring to track resource usage"
            suggestion += "\n- Configuring alerts for high resource usage"
            suggestion += "\n- Testing the scaling behavior under load"
        
        # Return a resolution that requires manual intervention
        return Resolution(
            issue=issue,
            status=ResolutionStatus.MANUAL_INTERVENTION_REQUIRED,
            description=f"Missing autoscaling resolution for {name} requires manual intervention",
            changes=[],
            metadata={
                'suggestion': suggestion
            }
        )
    
    def _are_services_compatible(self, kind1: str, kind2: str) -> bool:
        """
        Check if two service types are compatible for connection.
        
        Args:
            kind1: First service type
            kind2: Second service type
            
        Returns:
            True if the services are compatible, False otherwise
        """
        # This is a simplified compatibility check
        # In a real implementation, this would be more sophisticated
        
        # Convert to lowercase for case-insensitive comparison
        kind1 = kind1.lower()
        kind2 = kind2.lower()
        
        # Some basic compatibility rules
        if 'frontend' in kind1 and 'api' in kind2:
            return True
        if 'api' in kind1 and 'database' in kind2:
            return True
        if 'service' in kind1 and 'service' in kind2:
            return True
        if 'function' in kind1 and ('api' in kind2 or 'database' in kind2):
            return True
        
        # Default to true for now - in a real implementation, we would have more rules
        return True