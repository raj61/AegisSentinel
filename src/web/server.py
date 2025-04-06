"""
Web Server
=========

This module provides a web server for visualizing and interacting with the service graph.
"""

import logging
import json
import os
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver
import webbrowser

from src.graph import ServiceGraph

logger = logging.getLogger(__name__)

# Directory containing web assets
WEB_DIR = Path(__file__).parent / 'static'

class ServiceGraphWebHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for the service graph web interface."""
    
    def __init__(self, *args, service_graph: ServiceGraph = None, use_frontend: bool = False, metrics_integration=None, **kwargs):
        self.service_graph = service_graph
        self.use_frontend = use_frontend
        self.frontend_dir = Path(__file__).parent.parent.parent / 'frontend'
        self.metrics_integration = metrics_integration
        super().__init__(*args, **kwargs)
    
    def translate_path(self, path):
        """Translate URL path to file system path."""
        # Map root to index.html
        if path == '/':
            if self.use_frontend and (self.frontend_dir / 'public' / 'index.html').exists():
                return str(self.frontend_dir / 'public' / 'index.html')
            else:
                return str(WEB_DIR / 'index.html')
        
        # Remove leading slash
        path = path.lstrip('/')
        
        # Check if it's an API request
        if path.startswith('api/'):
            return path
        
        # Check if it's a frontend resource
        if self.use_frontend:
            # Check for frontend/src path pattern
            if path.startswith('frontend/src/'):
                component_path = path.replace('frontend/src/', '')
                src_path = self.frontend_dir / 'src' / component_path
                if src_path.exists():
                    return str(src_path)
            
            # First check in public directory
            public_path = self.frontend_dir / 'public' / path
            if public_path.exists():
                return str(public_path)
            
            # Then check in src directory
            src_path = self.frontend_dir / 'src' / path
            if src_path.exists():
                return str(src_path)
            
            # Check for direct component references
            if path.startswith('components/'):
                component_path = self.frontend_dir / 'src' / path
                if component_path.exists():
                    return str(component_path)
        
        # Otherwise, serve static files
        return str(WEB_DIR / path)
    
    def do_GET(self):
        """Handle GET requests."""
        # Check if it's an API request
        if self.path.startswith('/api/'):
            self.handle_api_request()
            return
        
        # Otherwise, serve static files
        return SimpleHTTPRequestHandler.do_GET(self)
        
    def do_POST(self):
        """Handle POST requests."""
        # Check if it's an API request
        if self.path.startswith('/api/'):
            self.handle_api_request()
            return
        
        # Otherwise, return 404
        self.send_response(404)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'error': 'Not found'}).encode())
    
    def handle_api_request(self):
        """Handle API requests."""
        if self.path == '/api/graph':
            self.handle_api_graph()
        elif self.path == '/api/graph/nodes':
            self.handle_api_nodes()
        elif self.path == '/api/graph/edges':
            self.handle_api_edges()
        elif self.path == '/api/issues':
            self.handle_api_issues()
        elif self.path == '/api/health':
            self.handle_api_health()
        elif self.path == '/api/inject-anomaly' and self.command == 'POST':
            self.handle_api_inject_anomaly()
        elif self.path.startswith('/api/metrics') and self.metrics_integration:
            self.handle_api_metrics()
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'Not found'}).encode())
    
    def handle_api_graph(self):
        """Handle GET /api/graph request."""
        if not self.service_graph:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'No service graph available'}).encode())
            return
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        # Use simplified graph to avoid duplicate nodes and circular dependencies
        self.wfile.write(self.service_graph.to_json(simplified=True).encode())
    
    def handle_api_nodes(self):
        """Handle GET /api/graph/nodes request."""
        if not self.service_graph:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'No service graph available'}).encode())
            return
        
        nodes = []
        for node_id in self.service_graph.get_nodes():
            node = self.service_graph.get_node(node_id)
            nodes.append({'id': node_id, **node})
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'nodes': nodes}).encode())
    
    def handle_api_edges(self):
        """Handle GET /api/graph/edges request."""
        if not self.service_graph:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'No service graph available'}).encode())
            return
        
        edges = []
        for source, target, attrs in self.service_graph.get_edges():
            edges.append({'source': source, 'target': target, **attrs})
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'edges': edges}).encode())
    
    def handle_api_issues(self):
        """Handle GET /api/issues request."""
        if not self.service_graph:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'No service graph available'}).encode())
            return
        
        try:
            from src.detection import DetectionEngine
            
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
                    'metadata': issue.metadata,
                    'detected_at': issue.detected_at,
                    'mitigated_at': issue.mitigated_at,
                    'mitigation_action': issue.mitigation_action,
                    'status': issue.status
                })
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'issues': issues_json}).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': f'Error detecting issues: {str(e)}'}).encode())
    
    def handle_api_health(self):
        """Handle GET /api/health request."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({
            'status': 'ok',
            'graph': {
                'available': self.service_graph is not None,
                'nodes': self.service_graph.node_count() if self.service_graph else 0,
                'edges': self.service_graph.edge_count() if self.service_graph else 0
            }
        }).encode())
    
    def handle_api_inject_anomaly(self):
        """Handle POST /api/inject-anomaly request."""
        if not self.service_graph:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'No service graph available'}).encode())
            return
        
        try:
            # Get request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            # Choose a random service to affect
            import random
            services = self.service_graph.get_nodes()
            affected_service = random.choice(services)
            
            # Create a synthetic issue ID
            import time
            issue_id = f"synthetic-issue-{time.strftime('%Y%m%d%H%M%S')}"
            
            # Update the health status of the affected service
            if affected_service in self.service_graph.get_nodes():
                # Update the node in the graph
                self.service_graph.update_node_attribute(affected_service, 'health_status', 'critical')
                logger.info(f"Injected synthetic issue {issue_id} affecting service {affected_service}")
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'success': True,
                    'issue_id': issue_id,
                    'affected_service': affected_service
                }).encode())
            else:
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': f'Service {affected_service} not found'}).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': f'Error injecting anomaly: {str(e)}'}).encode())
    
    def handle_api_metrics(self):
        """Handle GET /api/metrics request."""
        if not self.metrics_integration:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'Metrics integration not available'}).encode())
            return
        
        try:
            from src.api.metrics_endpoints import MetricsEndpoints
            
            # Create metrics endpoints handler
            metrics_endpoints = MetricsEndpoints(self.metrics_integration)
            
            # Handle the request
            if metrics_endpoints.handle_request(self, self.path):
                return
            
            # If we get here, the metrics endpoint was not recognized
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'Metrics endpoint not found'}).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': f'Error handling metrics request: {str(e)}'}).encode())

class ServiceGraphWebServer(socketserver.ThreadingMixIn, HTTPServer):
    """HTTP server for the service graph web interface."""
    
    def __init__(self, server_address, service_graph=None, use_frontend=False, metrics_integration=None):
        """Initialize the server with a service graph."""
        self.service_graph = service_graph
        self.use_frontend = use_frontend
        self.metrics_integration = metrics_integration
        super().__init__(server_address, self.handler_class)
    
    @property
    def handler_class(self):
        """Get the request handler class with the service graph injected."""
        server = self
        
        class Handler(ServiceGraphWebHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, service_graph=server.service_graph, 
                                use_frontend=server.use_frontend,
                                metrics_integration=server.metrics_integration, **kwargs)
        
        return Handler

def create_basic_html():
    """Create a basic HTML file for the web interface."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Service Graph Viewer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        header {
            background-color: #333;
            color: white;
            padding: 1rem;
        }
        h1 {
            margin: 0;
        }
        .container {
            display: flex;
            flex: 1;
            overflow: hidden;
        }
        .sidebar {
            width: 300px;
            background-color: #f5f5f5;
            padding: 1rem;
            overflow-y: auto;
            border-right: 1px solid #ddd;
        }
        .graph-container {
            flex: 1;
            overflow: hidden;
            position: relative;
        }
        #graph {
            width: 100%;
            height: 100%;
        }
        .controls {
            position: absolute;
            top: 10px;
            right: 10px;
            background-color: rgba(255, 255, 255, 0.8);
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
    </style>
</head>
<body>
    <header>
        <h1>Service Graph Viewer</h1>
    </header>
    <div class="container">
        <div class="sidebar">
            <h2>Issues</h2>
            <div id="issues-container">
                <div class="loading">Loading issues...</div>
            </div>
        </div>
        <div class="graph-container">
            <div id="graph">
                <div class="loading">Loading service graph...</div>
            </div>
            <div class="controls">
                <button id="zoom-in">+</button>
                <button id="zoom-out">-</button>
                <button id="reset">Reset</button>
            </div>
        </div>
    </div>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script>
        // Fetch the service graph data
        fetch('/api/graph')
            .then(response => response.json())
            .then(data => {
                console.log('Graph data:', data);
                // Render the graph here
            })
            .catch(error => {
                console.error('Error fetching graph:', error);
            });
        
        // Fetch issues
        fetch('/api/issues')
            .then(response => response.json())
            .then(data => {
                console.log('Issues data:', data);
                // Render issues here
            })
            .catch(error => {
                console.error('Error fetching issues:', error);
            });
    </script>
</body>
</html>
"""

def start_web_server(service_graph: ServiceGraph, host: str = '0.0.0.0', port: int = 8080, 
                    open_browser: bool = True, use_frontend: bool = True,
                    metrics_integration=None) -> threading.Thread:
    """
    Start the web server in a separate thread.
    
    Args:
        service_graph: ServiceGraph instance to serve
        host: Host to bind to
        port: Port to bind to
        open_browser: Whether to open a browser window
        use_frontend: Whether to use the frontend folder instead of static files
        metrics_integration: Optional metrics integration instance
        
    Returns:
        Thread running the server
    """
    # Create the static directory if it doesn't exist
    os.makedirs(WEB_DIR, exist_ok=True)
    
    # Determine which frontend to use
    if use_frontend:
        # Use the frontend folder
        frontend_dir = Path(__file__).parent.parent.parent / 'frontend'
        if not frontend_dir.exists():
            logger.warning(f"Frontend directory {frontend_dir} does not exist, falling back to static files")
            use_frontend = False
    
    if not use_frontend:
        # Always recreate the index.html file to ensure latest changes are applied
        index_html = WEB_DIR / 'index.html'
        with open(index_html, 'w') as f:
            f.write(create_basic_html())
        logger.info(f"Created/updated web interface HTML at {index_html}")
    
    server = ServiceGraphWebServer((host, port), service_graph, use_frontend, metrics_integration)
    
    def run_server():
        logger.info(f"Starting web server on {host}:{port}")
        server.serve_forever()
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    
    # Open browser if requested
    if open_browser:
        webbrowser.open(f"http://localhost:{port}")
    
    return thread
