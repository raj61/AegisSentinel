import yaml
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# Sample Kubernetes YAML mockup (as string for demo)
k8s_yaml = """
---
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

# Function to parse the above YAML and build the service graph
def build_service_graph_from_k8s(yaml_str):
    docs = list(yaml.safe_load_all(yaml_str))
    graph = nx.DiGraph()
    for doc in docs:
        kind = doc.get("kind", "")
        metadata = doc.get("metadata", {})
        name = metadata.get("name", "")

        if kind in ("Deployment", "StatefulSet"):
            graph.add_node(name)
            containers = doc["spec"]["template"]["spec"]["containers"]
            for container in containers:
                env_vars = container.get("env", [])
                for env in env_vars:
                    val = env.get("value", "")
                    if "svc.cluster.local" in val or ".svc" in val or "http://" in val:
                        target = val.split("//")[-1].split(":")[0]
                        graph.add_edge(name, target)

        elif kind == "Service":
            graph.add_node(name)

        elif kind == "Ingress":
            rules = doc["spec"].get("rules", [])
            for rule in rules:
                paths = rule.get("http", {}).get("paths", [])
                for path in paths:
                    backend = path.get("backend", {})
                    svc = backend.get("service", {}).get("name")
                    if svc:
                        graph.add_edge("ingress", svc)
    return graph

# Build and visualize the graph
service_graph = build_service_graph_from_k8s(k8s_yaml)
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(service_graph, seed=42)
nx.draw(service_graph, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='gray', arrows=True)
plt.title("Kubernetes Service Dependency Graph")
plt.tight_layout()
graph_img_path = "/mnt/data/k8s_service_graph.png"
plt.savefig(graph_img_path)
graph_img_path
