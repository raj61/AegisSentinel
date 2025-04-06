"""
Kubernetes Parser
===============

This module provides a parser for Kubernetes YAML manifests.
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, List, Union, Any

from src.graph import ServiceGraph
from src.parsers.base_parser import BaseParser, ParseError

logger = logging.getLogger(__name__)

class KubernetesParser(BaseParser):
    """
    Parser for Kubernetes YAML manifests.
    
    This parser can handle both individual YAML files and directories containing multiple YAML files.
    It extracts service dependencies from various Kubernetes resources like Deployments, Services,
    StatefulSets, Ingresses, etc.
    """
    
    def __init__(self):
        """Initialize the Kubernetes parser."""
        super().__init__()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def get_name(self) -> str:
        """Get the name of the parser."""
        return "Kubernetes"
    
    def parse(self, source_path: Union[str, Path], service_graph: ServiceGraph) -> None:
        """
        Parse Kubernetes YAML manifests and populate the service graph.
        
        Args:
            source_path: Path to the source file or directory
            service_graph: ServiceGraph instance to populate
            
        Raises:
            ValueError: If the source path does not exist or is not readable
            ParseError: If there is an error parsing the Kubernetes manifests
        """
        source_path = self._validate_source_path(source_path)
        
        try:
            if source_path.is_file():
                self._parse_file(source_path, service_graph)
            else:  # It's a directory
                self._parse_directory(source_path, service_graph)
        except Exception as e:
            raise ParseError(f"Error parsing Kubernetes manifests: {e}")
    
    def _parse_directory(self, directory_path: Path, service_graph: ServiceGraph) -> None:
        """
        Parse all Kubernetes YAML files in a directory.
        
        Args:
            directory_path: Path to the directory
            service_graph: ServiceGraph instance to populate
        """
        yaml_files = list(directory_path.glob("**/*.yaml")) + list(directory_path.glob("**/*.yml"))
        
        if not yaml_files:
            self.logger.warning(f"No YAML files found in {directory_path}")
            return
        
        for yaml_file in yaml_files:
            self._parse_file(yaml_file, service_graph)
    
    def _parse_file(self, file_path: Path, service_graph: ServiceGraph) -> None:
        """
        Parse a single Kubernetes YAML file.
        
        Args:
            file_path: Path to the YAML file
            service_graph: ServiceGraph instance to populate
        """
        self.logger.debug(f"Parsing Kubernetes YAML file: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Parse all YAML documents in the file
            docs = list(yaml.safe_load_all(content))
            
            for doc in docs:
                if not doc:
                    continue
                
                self._process_k8s_resource(doc, service_graph)
        
        except yaml.YAMLError as e:
            self.logger.warning(f"Error parsing YAML file {file_path}: {e}")
        except Exception as e:
            self.logger.warning(f"Error processing file {file_path}: {e}")
    
    def _process_k8s_resource(self, resource: Dict[str, Any], service_graph: ServiceGraph) -> None:
        """
        Process a Kubernetes resource and update the service graph.
        
        Args:
            resource: Kubernetes resource as a dictionary
            service_graph: ServiceGraph instance to populate
        """
        kind = resource.get("kind", "")
        if not kind:
            return
        
        metadata = resource.get("metadata", {})
        name = metadata.get("name", "")
        namespace = metadata.get("namespace", "default")
        
        if not name:
            return
        
        # Create a unique node ID that includes the namespace
        node_id = f"{namespace}/{name}"
        
        # Add node with appropriate attributes
        service_graph.add_node(
            node_id,
            name=name,
            namespace=namespace,
            kind=kind,
            labels=metadata.get("labels", {}),
        )
        
        # Process different resource types
        if kind in ("Deployment", "StatefulSet", "DaemonSet", "Job", "CronJob"):
            self._process_workload(resource, node_id, service_graph)
        elif kind == "Service":
            self._process_service(resource, node_id, service_graph)
        elif kind == "Ingress":
            self._process_ingress(resource, node_id, service_graph)
        elif kind == "ConfigMap" or kind == "Secret":
            self._process_config(resource, node_id, service_graph)
    
    def _process_workload(self, resource: Dict[str, Any], node_id: str, service_graph: ServiceGraph) -> None:
        """
        Process a Kubernetes workload resource (Deployment, StatefulSet, etc.).
        
        Args:
            resource: Kubernetes resource as a dictionary
            node_id: Node ID in the service graph
            service_graph: ServiceGraph instance to populate
        """
        spec = resource.get("spec", {})
        template = spec.get("template", {})
        pod_spec = template.get("spec", {})
        
        # Check for annotations that specify dependencies
        template_metadata = template.get("metadata", {})
        annotations = template_metadata.get("annotations", {})
        
        # Process aegis.sentinel/dependencies annotation
        if "aegis.sentinel/dependencies" in annotations:
            dependencies = annotations["aegis.sentinel/dependencies"].split(",")
            namespace = resource.get("metadata", {}).get("namespace", "default")
            
            for dependency in dependencies:
                dependency = dependency.strip()
                if dependency:
                    # Check if dependency includes namespace
                    if "." in dependency:
                        parts = dependency.split(".")
                        target = parts[0]
                        target_namespace = parts[1]
                    else:
                        target = dependency
                        target_namespace = namespace
                    
                    # Create target node if it doesn't exist
                    target_id = f"{target_namespace}/{target}"
                    if not service_graph.has_node(target_id):
                        service_graph.add_node(
                            target_id,
                            name=target,
                            namespace=target_namespace,
                            kind="Service",  # Assume it's a service
                        )
                        self.logger.debug(f"Created missing service node: {target_id}")
                    
                    service_graph.add_edge(node_id, target_id, type="explicit-dependency")
                    self.logger.debug(f"Added explicit dependency: {node_id} -> {target_id}")
        
        # Process containers
        containers = pod_spec.get("containers", [])
        for container in containers:
            # Set node category based on container image
            image = container.get("image", "")
            if image:
                category = self._infer_category_from_image(image)
                if category:
                    service_graph.add_node_attribute(node_id, category=category)
            
            # Check environment variables for service references
            env_vars = container.get("env", [])
            for env in env_vars:
                # Check for direct value references
                val = env.get("value", "")
                if val:
                    self._extract_service_refs_from_string(val, node_id, service_graph)
                
                # Check for environment variable names that suggest dependencies
                name = env.get("name", "")
                if name:
                    if "_HOST" in name or "_URL" in name or "_ADDR" in name or "_SERVICE" in name or "_ENDPOINT" in name:
                        # Extract service name from env var name
                        service_hint = name.split("_")[0].lower()
                        if service_hint:
                            # Try to find matching services in the graph
                            for other_node_id in service_graph.get_nodes():
                                other_node = service_graph.get_node(other_node_id)
                                if other_node.get("kind") == "Service" and service_hint in other_node_id.lower():
                                    service_graph.add_edge(node_id, other_node_id, type="implied-dependency")
                                    self.logger.debug(f"Added implied dependency from env var name: {node_id} -> {other_node_id}")
            
            # Check command and args for service references
            command = container.get("command", [])
            args = container.get("args", [])
            
            for cmd_part in command + args:
                if isinstance(cmd_part, str):
                    self._extract_service_refs_from_string(cmd_part, node_id, service_graph)
            
            # Check volume mounts for ConfigMap and Secret references
            volume_mounts = container.get("volumeMounts", [])
            for mount in volume_mounts:
                volume_name = mount.get("name", "")
                
                # Find the corresponding volume in the pod spec
                volumes = pod_spec.get("volumes", [])
                for volume in volumes:
                    if volume.get("name") == volume_name:
                        if "configMap" in volume:
                            config_name = volume["configMap"].get("name", "")
                            if config_name:
                                config_id = f"{resource.get('metadata', {}).get('namespace', 'default')}/{config_name}"
                                service_graph.add_edge(node_id, config_id, type="config-dependency")
                        
                        elif "secret" in volume:
                            secret_name = volume["secret"].get("secretName", "")
                            if secret_name:
                                secret_id = f"{resource.get('metadata', {}).get('namespace', 'default')}/{secret_name}"
                                service_graph.add_edge(node_id, secret_id, type="secret-dependency")
    
    def _process_service(self, resource: Dict[str, Any], node_id: str, service_graph: ServiceGraph) -> None:
        """
        Process a Kubernetes Service resource.
        
        Args:
            resource: Kubernetes resource as a dictionary
            node_id: Node ID in the service graph
            service_graph: ServiceGraph instance to populate
        """
        spec = resource.get("spec", {})
        selector = spec.get("selector", {})
        
        if selector:
            # Add service type and port information
            service_graph.add_node_attribute(
                node_id,
                service_type=spec.get("type", "ClusterIP"),
                ports=spec.get("ports", []),
                selector=selector
            )
            
            # Find workloads that match this service's selector
            for other_node_id in service_graph.get_nodes():
                other_node = service_graph.get_node(other_node_id)
                if other_node.get("kind") in ("Deployment", "StatefulSet", "DaemonSet"):
                    labels = other_node.get("labels", {})
                    
                    # Check if all selector keys match the workload's labels
                    if all(labels.get(k) == v for k, v in selector.items()):
                        service_graph.add_edge(node_id, other_node_id, type="service-selector")
    
    def _process_ingress(self, resource: Dict[str, Any], node_id: str, service_graph: ServiceGraph) -> None:
        """
        Process a Kubernetes Ingress resource.
        
        Args:
            resource: Kubernetes resource as a dictionary
            node_id: Node ID in the service graph
            service_graph: ServiceGraph instance to populate
        """
        spec = resource.get("spec", {})
        rules = spec.get("rules", [])
        
        for rule in rules:
            host = rule.get("host", "*")
            http = rule.get("http", {})
            paths = http.get("paths", [])
            
            for path in paths:
                path_type = path.get("pathType", "Prefix")
                path_value = path.get("path", "/")
                
                # Handle different Ingress API versions
                backend = path.get("backend", {})
                service_name = None
                service_port = None
                
                if "service" in backend:
                    # Ingress API v1
                    service = backend.get("service", {})
                    service_name = service.get("name")
                    port = service.get("port", {})
                    service_port = port.get("number") or port.get("name")
                else:
                    # Ingress API v1beta1
                    service_name = backend.get("serviceName")
                    service_port = backend.get("servicePort")
                
                if service_name:
                    namespace = resource.get("metadata", {}).get("namespace", "default")
                    service_id = f"{namespace}/{service_name}"
                    
                    # Add edge from ingress to service
                    service_graph.add_edge(
                        node_id,
                        service_id,
                        type="ingress-route",
                        host=host,
                        path=path_value,
                        path_type=path_type,
                        port=service_port
                    )
    
    def _process_config(self, resource: Dict[str, Any], node_id: str, service_graph: ServiceGraph) -> None:
        """
        Process a Kubernetes ConfigMap or Secret resource.
        
        Args:
            resource: Kubernetes resource as a dictionary
            node_id: Node ID in the service graph
            service_graph: ServiceGraph instance to populate
        """
        # Add data keys as attributes (but not the values for security reasons)
        data = resource.get("data", {})
        if data:
            service_graph.add_node_attribute(
                node_id,
                data_keys=list(data.keys())
            )
            
            # Set appropriate category
            kind = resource.get("kind", "")
            if kind == "ConfigMap":
                service_graph.add_node_attribute(node_id, category="config")
            elif kind == "Secret":
                service_graph.add_node_attribute(node_id, category="security")
    
    def _extract_service_refs_from_string(self, text: str, node_id: str, service_graph: ServiceGraph) -> None:
        """
        Extract service references from a string and add edges to the service graph.
        
        Args:
            text: String to extract service references from
            node_id: Source node ID
            service_graph: ServiceGraph instance to update
        """
        # Check for Kubernetes service DNS patterns
        if "svc.cluster.local" in text or ".svc" in text:
            # Extract service name from DNS name
            parts = text.split(".")
            target = parts[0]
            target_namespace = "default"
            
            if len(parts) > 1:
                target_namespace = parts[1]
            
            target_id = f"{target_namespace}/{target}"
            service_graph.add_edge(node_id, target_id, type="service-dependency")
            self.logger.debug(f"Added service dependency from string: {node_id} -> {target_id}")
        
        # Check for HTTP/HTTPS URLs
        elif "http://" in text or "https://" in text:
            # Extract hostname from URL
            try:
                from urllib.parse import urlparse
                url = urlparse(text)
                hostname = url.hostname
                
                if hostname:
                    # Extract service name from hostname
                    service_name = hostname.split(".")[0]
                    
                    # Try to find matching services in the graph
                    for other_node_id in service_graph.get_nodes():
                        other_node = service_graph.get_node(other_node_id)
                        if other_node.get("kind") == "Service" and service_name in other_node_id:
                            service_graph.add_edge(node_id, other_node_id, type="url-dependency")
                            self.logger.debug(f"Added URL dependency: {node_id} -> {other_node_id}")
            except Exception as e:
                self.logger.debug(f"Error parsing URL {text}: {e}")
    
    def _infer_category_from_image(self, image: str) -> str:
        """
        Infer the service category from the container image.
        
        Args:
            image: Container image name
            
        Returns:
            Category string or None if no category could be inferred
        """
        image_lower = image.lower()
        
        # Database images
        if any(db in image_lower for db in ["postgres", "mysql", "mariadb", "mongodb", "redis", "cassandra", "elasticsearch", "couchdb"]):
            return "database"
        
        # Cache images
        elif any(cache in image_lower for cache in ["redis", "memcached", "hazelcast"]):
            return "cache"
        
        # Queue/messaging images
        elif any(queue in image_lower for queue in ["rabbitmq", "kafka", "activemq", "nats"]):
            return "queue"
        
        # Web/API images
        elif any(web in image_lower for web in ["nginx", "httpd", "apache", "traefik", "envoy", "haproxy"]):
            return "api"
        
        # Application server images
        elif any(app in image_lower for app in ["tomcat", "jetty", "wildfly", "websphere", "weblogic"]):
            return "compute"
        
        # Container/Kubernetes related images
        elif any(k8s in image_lower for k8s in ["k8s", "kube", "kubernetes"]):
            return "kubernetes"
        
        # Default for common base images
        elif any(base in image_lower for base in ["alpine", "debian", "ubuntu", "centos", "fedora", "busybox"]):
            return "container"
        
        return None