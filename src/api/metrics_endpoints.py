"""
Metrics API Endpoints
===================

This module provides API endpoints for metrics data.
"""

import logging
import json
from typing import Dict, Any, Optional
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

from src.metrics.metrics_integration import MetricsIntegration

logger = logging.getLogger(__name__)

class MetricsEndpoints:
    """
    API endpoints for metrics data.
    """
    
    def __init__(self, metrics_integration: Optional[MetricsIntegration] = None):
        """
        Initialize the metrics endpoints.
        
        Args:
            metrics_integration: MetricsIntegration instance
        """
        self.metrics_integration = metrics_integration
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def handle_request(self, handler: BaseHTTPRequestHandler, path: str) -> bool:
        """
        Handle an API request for metrics.
        
        Args:
            handler: BaseHTTPRequestHandler instance
            path: Request path
            
        Returns:
            True if the request was handled, False otherwise
        """
        if path.startswith('/api/metrics'):
            if not self.metrics_integration:
                self._send_error(handler, 503, "Metrics collection not enabled")
                return True
            
            # Parse query parameters
            parsed_url = urlparse(path)
            query_params = parse_qs(parsed_url.query)
            
            # Handle different metrics endpoints
            if path.startswith('/api/metrics/service/'):
                # Extract service ID from path
                # The service ID might contain slashes, so we need to handle it carefully
                service_path = path[len('/api/metrics/service/'):]
                # Remove query parameters if any
                if '?' in service_path:
                    service_path = service_path.split('?')[0]
                # The service ID is the entire path
                service_id = service_path
                return self._handle_service_metrics(handler, service_id, query_params)
            elif path == '/api/metrics/services':
                return self._handle_all_services_metrics(handler)
            elif path.startswith('/api/metrics/historical/'):
                # Extract service ID and metric from path
                path_part = path[len('/api/metrics/historical/'):]
                # Find the last slash which separates service_id from metric
                last_slash_index = path_part.rfind('/')
                if last_slash_index != -1:
                    service_id = path_part[:last_slash_index]
                    metric = path_part[last_slash_index+1:]
                    # Remove query parameters from metric if any
                    if '?' in metric:
                        metric = metric.split('?')[0]
                    return self._handle_historical_metrics(handler, service_id, metric, query_params)
            
            # If we get here, the metrics endpoint was not recognized
            self._send_error(handler, 404, "Metrics endpoint not found")
            return True
        
        return False
    
    def _handle_service_metrics(self, handler: BaseHTTPRequestHandler, service_id: str, 
                               query_params: Dict[str, Any]) -> bool:
        """
        Handle a request for service metrics.
        
        Args:
            handler: BaseHTTPRequestHandler instance
            service_id: ID of the service
            query_params: Query parameters
            
        Returns:
            True if the request was handled
        """
        try:
            # Get service metrics
            if service_id in self.metrics_integration.service_graph.get_nodes():
                node = self.metrics_integration.service_graph.get_node(service_id)
                
                # Extract metrics from node attributes
                metrics = {}
                for key, value in node.items():
                    if key in ['cpu_usage', 'memory_usage', 'request_rate', 'error_rate', 'latency_p95']:
                        metrics[key] = value
                
                self._send_json_response(handler, {
                    'service_id': service_id,
                    'metrics': metrics
                })
            else:
                self._send_error(handler, 404, f"Service {service_id} not found")
        except Exception as e:
            self.logger.error(f"Error handling service metrics request: {e}")
            self._send_error(handler, 500, f"Internal server error: {str(e)}")
        
        return True
    
    def _handle_all_services_metrics(self, handler: BaseHTTPRequestHandler) -> bool:
        """
        Handle a request for all services metrics.
        
        Args:
            handler: BaseHTTPRequestHandler instance
            
        Returns:
            True if the request was handled
        """
        try:
            # Get all service nodes
            service_metrics = []
            for node_id in self.metrics_integration.service_graph.get_nodes():
                node = self.metrics_integration.service_graph.get_node(node_id)
                
                # Extract metrics from node attributes
                metrics = {}
                for key, value in node.items():
                    if key in ['cpu_usage', 'memory_usage', 'request_rate', 'error_rate', 'latency_p95']:
                        metrics[key] = value
                
                if metrics:
                    service_metrics.append({
                        'service_id': node_id,
                        'name': node.get('name', node_id),
                        'kind': node.get('kind', 'unknown'),
                        'health_status': node.get('health_status', 'unknown'),
                        'metrics': metrics
                    })
            
            self._send_json_response(handler, {
                'services': service_metrics
            })
        except Exception as e:
            self.logger.error(f"Error handling all services metrics request: {e}")
            self._send_error(handler, 500, f"Internal server error: {str(e)}")
        
        return True
    
    def _handle_historical_metrics(self, handler: BaseHTTPRequestHandler, service_id: str, 
                                  metric: str, query_params: Dict[str, Any]) -> bool:
        """
        Handle a request for historical metrics.
        
        Args:
            handler: BaseHTTPRequestHandler instance
            service_id: ID of the service
            metric: Metric name
            query_params: Query parameters
            
        Returns:
            True if the request was handled
        """
        try:
            # Get query parameters
            start = query_params.get('start', [None])[0]
            end = query_params.get('end', [None])[0]
            step = query_params.get('step', ['1m'])[0]
            
            # Get historical metrics
            result = self.metrics_integration.get_historical_metrics(
                service_id, metric, start, end, step
            )
            
            self._send_json_response(handler, result)
        except Exception as e:
            self.logger.error(f"Error handling historical metrics request: {e}")
            self._send_error(handler, 500, f"Internal server error: {str(e)}")
        
        return True
    
    def _send_json_response(self, handler: BaseHTTPRequestHandler, data: Dict[str, Any]) -> None:
        """
        Send a JSON response.
        
        Args:
            handler: BaseHTTPRequestHandler instance
            data: Data to send as JSON
        """
        response = json.dumps(data).encode('utf-8')
        handler.send_response(200)
        handler.send_header('Content-Type', 'application/json')
        handler.send_header('Content-Length', len(response))
        handler.send_header('Access-Control-Allow-Origin', '*')
        handler.end_headers()
        handler.wfile.write(response)
    
    def _send_error(self, handler: BaseHTTPRequestHandler, code: int, message: str) -> None:
        """
        Send an error response.
        
        Args:
            handler: BaseHTTPRequestHandler instance
            code: HTTP status code
            message: Error message
        """
        response = json.dumps({
            'error': message
        }).encode('utf-8')
        handler.send_response(code)
        handler.send_header('Content-Type', 'application/json')
        handler.send_header('Content-Length', len(response))
        handler.send_header('Access-Control-Allow-Origin', '*')
        handler.end_headers()
        handler.wfile.write(response)