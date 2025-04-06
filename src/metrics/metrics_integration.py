"""
Metrics Integration
=================

This module provides functionality to integrate metrics with the service graph.
"""

import logging
import time
import threading
from typing import Dict, Any, Optional

from src.graph import ServiceGraph
from src.metrics.prometheus_collector import PrometheusCollector

logger = logging.getLogger(__name__)

class MetricsIntegration:
    """
    Integration of metrics with the service graph.
    
    This class provides methods to update the service graph with metrics
    from Prometheus and other sources.
    """
    
    def __init__(self, service_graph: ServiceGraph, prometheus_url: str = "http://localhost:9090"):
        """
        Initialize the metrics integration.
        
        Args:
            service_graph: ServiceGraph instance to update with metrics
            prometheus_url: URL of the Prometheus server
        """
        self.service_graph = service_graph
        self.prometheus_collector = PrometheusCollector(prometheus_url)
        self.monitoring_thread = None
        self.stop_monitoring = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def update_service_metrics(self, service_id: str, metrics: Dict[str, Any]) -> None:
        """
        Update a service node with metrics.
        
        Args:
            service_id: ID of the service node in the graph
            metrics: Dictionary of metrics to add to the node
        """
        if not self.service_graph.has_node(service_id):
            self.logger.warning(f"Service {service_id} not found in graph")
            return
        
        # Add metrics as node attributes
        for metric_name, metric_value in metrics.items():
            self.service_graph.add_node_attribute(service_id, **{metric_name: metric_value})
        
        # Update health status based on metrics
        health_status = self._determine_health_status(metrics)
        if health_status:
            self.service_graph.update_node_attribute(service_id, 'health_status', health_status)
    
    def _determine_health_status(self, metrics: Dict[str, Any]) -> Optional[str]:
        """
        Determine the health status of a service based on its metrics.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Health status string ('healthy', 'warning', 'critical') or None if unknown
        """
        # Check error rate
        if 'error_rate' in metrics:
            error_rate = metrics['error_rate']
            if error_rate > 0.1:  # More than 10% errors
                return 'critical'
            elif error_rate > 0.01:  # More than 1% errors
                return 'warning'
        
        # Check latency
        if 'latency_p95' in metrics:
            latency = metrics['latency_p95']
            if latency > 1.0:  # More than 1 second
                return 'critical'
            elif latency > 0.5:  # More than 500ms
                return 'warning'
        
        # Check CPU usage
        if 'cpu_usage' in metrics:
            cpu_usage = metrics['cpu_usage']
            if cpu_usage > 0.9:  # More than 90% CPU
                return 'critical'
            elif cpu_usage > 0.7:  # More than 70% CPU
                return 'warning'
        
        # Check memory usage
        if 'memory_usage' in metrics:
            memory_usage = metrics['memory_usage']
            if memory_usage > 1024:  # More than 1GB
                return 'warning'
        
        # If we have metrics but no issues, the service is healthy
        if metrics:
            return 'healthy'
        
        # If we don't have any metrics, we don't know the health status
        return None
    
    def update_all_services(self, namespace: str = "default") -> None:
        """
        Update all services in the graph with metrics.
        
        Args:
            namespace: Kubernetes namespace
        """
        # Get all service nodes in the graph
        service_nodes = []
        for node_id in self.service_graph.get_nodes():
            node = self.service_graph.get_node(node_id)
            if node.get('kind') in ['Service', 'Deployment', 'Pod']:
                service_nodes.append(node_id)
        
        # Get metrics for each service
        for service_id in service_nodes:
            # Extract service name from node ID (namespace/name)
            parts = service_id.split('/')
            if len(parts) == 2:
                namespace, service_name = parts
            else:
                service_name = service_id
            
            try:
                metrics = self.prometheus_collector.get_service_metrics(service_name, namespace)
                self.update_service_metrics(service_id, metrics)
            except Exception as e:
                self.logger.error(f"Error updating metrics for service {service_id}: {e}")
    
    def start_monitoring(self, interval: int = 60) -> None:
        """
        Start a background thread to periodically update metrics.
        
        Args:
            interval: Update interval in seconds
        """
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.logger.warning("Monitoring thread already running")
            return
        
        self.stop_monitoring = False
        
        def monitor_metrics():
            while not self.stop_monitoring:
                try:
                    self.update_all_services()
                    self.logger.info("Updated service metrics")
                except Exception as e:
                    self.logger.error(f"Error in metrics monitoring thread: {e}")
                
                # Sleep for the specified interval
                for _ in range(interval):
                    if self.stop_monitoring:
                        break
                    time.sleep(1)
        
        self.monitoring_thread = threading.Thread(target=monitor_metrics, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Started metrics monitoring thread")
    
    def stop_monitoring_thread(self) -> None:
        """Stop the metrics monitoring thread."""
        self.stop_monitoring = True
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
            self.logger.info("Stopped metrics monitoring thread")
    
    def get_historical_metrics(self, service_id: str, metric: str, 
                              start: Optional[str] = None, end: Optional[str] = None, 
                              step: str = "1m") -> Dict[str, Any]:
        """
        Get historical metrics for a service.
        
        Args:
            service_id: ID of the service node in the graph
            metric: Metric to retrieve (cpu_usage, memory_usage, request_rate, error_rate, latency_p95)
            start: Start time (RFC3339 or Unix timestamp), defaults to 1 hour ago
            end: End time (RFC3339 or Unix timestamp), defaults to now
            step: Query resolution step width (e.g., "15s", "1m", "1h")
            
        Returns:
            Dictionary with service ID, metric name, and data points
        """
        if not self.service_graph.has_node(service_id):
            self.logger.warning(f"Service {service_id} not found in graph")
            return {"service_id": service_id, "metric": metric, "data_points": []}
        
        # Extract service name from node ID (namespace/name)
        parts = service_id.split('/')
        if len(parts) == 2:
            namespace, service_name = parts
        else:
            service_name = service_id
            namespace = "default"
        
        try:
            data_points = self.prometheus_collector.get_historical_metrics(
                service_name, metric, namespace, start, end, step
            )
            
            return {
                "service_id": service_id,
                "metric": metric,
                "data_points": data_points
            }
        except Exception as e:
            self.logger.error(f"Error getting historical metrics for service {service_id}: {e}")
            return {"service_id": service_id, "metric": metric, "data_points": []}