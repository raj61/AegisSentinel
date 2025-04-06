"""
Test Kubernetes Parser
====================

This module contains tests for the Kubernetes parser.
"""

import os
import sys
import unittest
from pathlib import Path

# Add the parent directory to the path so we can import the src module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph import ServiceGraph
from src.parsers import get_parser
from src.detection import DetectionEngine

class TestKubernetesParser(unittest.TestCase):
    """Test case for the Kubernetes parser."""
    
    def setUp(self):
        """Set up the test case."""
        # Create a sample Kubernetes YAML
        self.sample_yaml = """
apiVersion: v1
kind: Service
metadata:
  name: user-service
spec:
  selector:
    app: user
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
spec:
  template:
    spec:
      containers:
      - name: frontend-container
        env:
        - name: USER_SERVICE_URL
          value: http://user-service:8080
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
spec:
  template:
    spec:
      containers:
      - name: backend-container
        env:
        - name: USER_SERVICE_URL
          value: http://user-service:8080
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ingress
spec:
  rules:
  - http:
      paths:
      - path: /frontend
        pathType: Prefix
        backend:
          service:
            name: frontend
            port:
              number: 80
"""
        
        # Write the sample YAML to a temporary file
        self.temp_file = Path("temp_k8s.yaml")
        with open(self.temp_file, "w") as f:
            f.write(self.sample_yaml)
    
    def tearDown(self):
        """Clean up after the test case."""
        # Remove the temporary file
        if self.temp_file.exists():
            self.temp_file.unlink()
    
    def test_kubernetes_parser(self):
        """Test the Kubernetes parser."""
        # Get the parser
        parser = get_parser("kubernetes", self.temp_file)
        self.assertIsNotNone(parser)
        
        # Parse the YAML
        service_graph = ServiceGraph()
        parser.parse(self.temp_file, service_graph)
        
        # Check that the graph was built correctly
        self.assertGreater(service_graph.node_count(), 0)
        self.assertGreater(service_graph.edge_count(), 0)
        
        # Check that the nodes were added
        nodes = service_graph.get_nodes()
        node_names = [service_graph.get_node_attribute(node, "name") for node in nodes]
        self.assertIn("user-service", node_names)
        self.assertIn("frontend", node_names)
        self.assertIn("backend", node_names)
        self.assertIn("ingress", node_names)
        
        # Check that the edges were added
        edges = service_graph.get_edges()
        self.assertTrue(any(source == "default/frontend" and target == "default/user-service" for source, target, _ in edges))
        self.assertTrue(any(source == "default/backend" and target == "default/user-service" for source, target, _ in edges))
        self.assertTrue(any(source == "default/ingress" and target == "default/frontend" for source, target, _ in edges))
    
    def test_issue_detection(self):
        """Test issue detection."""
        # Get the parser
        parser = get_parser("kubernetes", self.temp_file)
        self.assertIsNotNone(parser)
        
        # Parse the YAML
        service_graph = ServiceGraph()
        parser.parse(self.temp_file, service_graph)
        
        # Detect issues
        detection_engine = DetectionEngine(service_graph)
        issues = detection_engine.detect_issues()
        
        # Check that issues were detected
        self.assertGreater(len(issues), 0)
        
        # Print the issues
        print(f"\nDetected {len(issues)} issues:")
        for issue in issues:
            print(f"- {issue.type.value}: {issue.description}")

if __name__ == "__main__":
    unittest.main()