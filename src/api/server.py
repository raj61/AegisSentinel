"""
API Server
=========

This module provides a REST API server for interacting with the service graph builder.
"""

import logging
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import time

from src.graph import ServiceGraph
from src.detection import DetectionEngine, Issue
from src.resolution import ResolutionEngine, Resolution
from src.parsers import get_parser

logger = logging.getLogger(__name__)

class ServiceGraphAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the service graph API."""
    
    def __init__(self, *args, service_graph: ServiceGraph = None, **kwargs):
        self.service_graph = service_graph
        super().__init__(*args, **kwargs)
    
    def _set_headers(self, status_code=200, content_type='application/json'):
        """Set the response headers."""
        self.send_response(status_code)
        self.send_header('Content-type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS."""
        self._set_headers()
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query = parse_qs(parsed_url.query)
        
        try:
            if path == '/api/graph':
                self._handle_get_graph()
            elif path == '/api/graph/nodes':
                self._handle_get_nodes()
            elif path == '/api/graph/edges':
                self._handle_get_edges()
            elif path.startswith('/api/graph/node/'):
                node_id = path.split('/')[-1]
                self._handle_get_node(node_id)
            elif path == '/api/issues':
                self._handle_get_issues()
            elif path == '/api/resolutions':
                self._handle_get_resolutions()
            elif path == '/api/health':
                self._handle_health_check()
            else:
                self._set_headers(404)
                self.wfile.write(json.dumps({'error': 'Not found'}).encode())
        except Exception as e:
            logger.exception(f"Error handling GET request: {e}")
            self._set_headers(500)
            self.wfile.write(json.dumps({'error': str(e)}).encode())
    
    def do_POST(self):
        """Handle POST requests."""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode())
            
            if self.path == '/api/graph/parse':
                self._handle_parse_graph(data)
            elif self.path == '/api/issues/detect':
                self._handle_detect_issues()
            elif self.path == '/api/issues/resolve':
                self._handle_resolve_issues(data)
            else:
                self._set_headers(404)
                self.wfile.write(json.dumps({'error': 'Not found'}).encode())
        except json.JSONDecodeError:
            self._set_headers(400)
            self.wfile.write(json.dumps({'error': 'Invalid JSON'}).encode())
        except Exception as e:
            logger.exception(f"Error handling POST request: {e}")
            self._set_headers(500)
            self.wfile.write(json.dumps({'error': str(e)}).encode())
    
    def _handle_get_graph(self):
        """Handle GET /api/graph request."""
        if not self.service_graph:
            self._set_headers(404)
            self.wfile.write(json.dumps({'error': 'No service graph available'}).encode())
            return
        
        self._set_headers()
        self.wfile.write(self.service_graph.to_json().encode())
    
    def _handle_get_nodes(self):
        """Handle GET /api/graph/nodes request."""
        if not self.service_graph:
            self._set_headers(404)
            self.wfile.write(json.dumps({'error': 'No service graph available'}).encode())
            return
        
        nodes = []
        for node_id in self.service_graph.get_nodes():
            node = self.service_graph.get_node(node_id)
            nodes.append({'id': node_id, **node})
        
        self._set_headers()
        self.wfile.write(json.dumps({'nodes': nodes}).encode())
    
    def _handle_get_edges(self):
        """Handle GET /api/graph/edges request."""
        if not self.service_graph:
            self._set_headers(404)
            self.wfile.write(json.dumps({'error': 'No service graph available'}).encode())
            return
        
        edges = []
        for source, target, attrs in self.service_graph.get_edges():
            edges.append({'source': source, 'target': target, **attrs})
        
        self._set_headers()
        self.wfile.write(json.dumps({'edges': edges}).encode())
    
    def _handle_get_node(self, node_id):
        """Handle GET /api/graph/node/{node_id} request."""
        if not self.service_graph:
            self._set_headers(404)
            self.wfile.write(json.dumps({'error': 'No service graph available'}).encode())
            return
        
        try:
            node = self.service_graph.get_node(node_id)
            self._set_headers()
            self.wfile.write(json.dumps({'id': node_id, **node}).encode())
        except KeyError:
            self._set_headers(404)
            self.wfile.write(json.dumps({'error': f'Node {node_id} not found'}).encode())
    
    def _handle_parse_graph(self, data):
        """Handle POST /api/graph/parse request."""
        if 'source' not in data:
            self._set_headers(400)
            self.wfile.write(json.dumps({'error': 'Missing source parameter'}).encode())
            return
        
        source_path = data['source']
        parser_type = data.get('type', 'auto')
        
        try:
            # Get the appropriate parser
            parser = get_parser(parser_type, source_path)
            if not parser:
                self._set_headers(400)
                self.wfile.write(json.dumps({'error': f'Could not determine parser for source: {source_path}'}).encode())
                return
            
            # Parse the infrastructure code
            self.server.service_graph = ServiceGraph()
            parser.parse(source_path, self.server.service_graph)
            
            # Update the handler's reference to the service graph
            self.service_graph = self.server.service_graph
            
            self._set_headers()
            self.wfile.write(json.dumps({
                'message': 'Graph parsed successfully',
                'nodes': self.service_graph.node_count(),
                'edges': self.service_graph.edge_count()
            }).encode())
        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({'error': f'Error parsing graph: {str(e)}'}).encode())
    
    def _handle_get_issues(self):
        """Handle GET /api/issues request."""
        if not self.service_graph:
            self._set_headers(404)
            self.wfile.write(json.dumps({'error': 'No service graph available'}).encode())
            return
        
        try:
            # Detect issues
            detection_engine = DetectionEngine(self.service_graph)
            issues = detection_engine.detect_issues()
            
            # Convert issues to JSON-serializable format
            issues_json = []
            for issue in issues:
                issues_json.append({
                    'type': issue.type.value,
                    'severity': issue.severity,
                    'description': issue.description,
                    'affected_nodes': issue.affected_nodes,
                    'affected_edges': issue.affected_edges,
                    'metadata': issue.metadata
                })
            
            self._set_headers()
            self.wfile.write(json.dumps({'issues': issues_json}).encode())
        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({'error': f'Error detecting issues: {str(e)}'}).encode())
    
    def _handle_detect_issues(self):
        """Handle POST /api/issues/detect request."""
        if not self.service_graph:
            self._set_headers(404)
            self.wfile.write(json.dumps({'error': 'No service graph available'}).encode())
            return
        
        try:
            # Detect issues
            detection_engine = DetectionEngine(self.service_graph)
            issues = detection_engine.detect_issues()
            
            # Store issues in the server
            self.server.issues = issues
            
            # Convert issues to JSON-serializable format
            issues_json = []
            for issue in issues:
                issues_json.append({
                    'type': issue.type.value,
                    'severity': issue.severity,
                    'description': issue.description,
                    'affected_nodes': issue.affected_nodes,
                    'affected_edges': issue.affected_edges,
                    'metadata': issue.metadata
                })
            
            self._set_headers()
            self.wfile.write(json.dumps({'issues': issues_json}).encode())
        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({'error': f'Error detecting issues: {str(e)}'}).encode())
    
    def _handle_resolve_issues(self, data):
        """Handle POST /api/issues/resolve request."""
        if not self.service_graph:
            self._set_headers(404)
            self.wfile.write(json.dumps({'error': 'No service graph available'}).encode())
            return
        
        if not hasattr(self.server, 'issues') or not self.server.issues:
            self._set_headers(400)
            self.wfile.write(json.dumps({'error': 'No issues detected yet'}).encode())
            return
        
        try:
            # Resolve issues
            resolution_engine = ResolutionEngine(self.service_graph)
            resolutions = resolution_engine.resolve_issues(self.server.issues)
            
            # Store resolutions in the server
            self.server.resolutions = resolutions
            
            # Convert resolutions to JSON-serializable format
            resolutions_json = []
            for resolution in resolutions:
                resolutions_json.append({
                    'issue_type': resolution.issue.type.value,
                    'status': resolution.status.value,
                    'description': resolution.description,
                    'changes': resolution.changes,
                    'metadata': resolution.metadata
                })
            
            self._set_headers()
            self.wfile.write(json.dumps({'resolutions': resolutions_json}).encode())
        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({'error': f'Error resolving issues: {str(e)}'}).encode())
    
    def _handle_get_resolutions(self):
        """Handle GET /api/resolutions request."""
        if not hasattr(self.server, 'resolutions') or not self.server.resolutions:
            self._set_headers(404)
            self.wfile.write(json.dumps({'error': 'No resolutions available'}).encode())
            return
        
        # Convert resolutions to JSON-serializable format
        resolutions_json = []
        for resolution in self.server.resolutions:
            resolutions_json.append({
                'issue_type': resolution.issue.type.value,
                'status': resolution.status.value,
                'description': resolution.description,
                'changes': resolution.changes,
                'metadata': resolution.metadata
            })
        
        self._set_headers()
        self.wfile.write(json.dumps({'resolutions': resolutions_json}).encode())
    
    def _handle_health_check(self):
        """Handle GET /api/health request."""
        self._set_headers()
        self.wfile.write(json.dumps({
            'status': 'ok',
            'timestamp': time.time(),
            'graph': {
                'available': self.service_graph is not None,
                'nodes': self.service_graph.node_count() if self.service_graph else 0,
                'edges': self.service_graph.edge_count() if self.service_graph else 0
            }
        }).encode())

class ServiceGraphAPIServer(HTTPServer):
    """HTTP server for the service graph API."""
    
    def __init__(self, server_address, service_graph=None):
        """Initialize the server with a service graph."""
        self.service_graph = service_graph
        super().__init__(server_address, self.handler_class)
    
    @property
    def handler_class(self):
        """Get the request handler class with the service graph injected."""
        server = self
        
        class Handler(ServiceGraphAPIHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, service_graph=server.service_graph, **kwargs)
        
        return Handler

def start_api_server(service_graph: ServiceGraph, host: str = '0.0.0.0', port: int = 8000) -> threading.Thread:
    """
    Start the API server in a separate thread.
    
    Args:
        service_graph: ServiceGraph instance to serve
        host: Host to bind to
        port: Port to bind to
        
    Returns:
        Thread running the server
    """
    server = ServiceGraphAPIServer((host, port), service_graph)
    
    def run_server():
        logger.info(f"Starting API server on {host}:{port}")
        server.serve_forever()
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    
    return thread