"""
Terraform Parser
==============

This module provides a parser for Terraform configuration files.
"""

import logging
import json
import re
from pathlib import Path
from typing import Dict, List, Union, Any, Set

from src.graph import ServiceGraph
from src.parsers.base_parser import BaseParser, ParseError

logger = logging.getLogger(__name__)

class TerraformParser(BaseParser):
    """
    Parser for Terraform configuration files.
    
    This parser can handle Terraform configuration files (.tf) and extract
    service dependencies from various resource types.
    """
    
    def __init__(self):
        """Initialize the Terraform parser."""
        super().__init__()
        self.resource_types = {
            # AWS resources
            "aws_instance": "compute",
            "aws_lambda_function": "serverless",
            "aws_ecs_service": "container",
            "aws_ecs_task_definition": "container",
            "aws_eks_cluster": "kubernetes",
            "aws_api_gateway_rest_api": "api",
            "aws_api_gateway_resource": "api",
            "aws_api_gateway_method": "api",
            "aws_api_gateway_integration": "api",
            "aws_lb": "loadbalancer",
            "aws_lb_listener": "loadbalancer",
            "aws_lb_target_group": "loadbalancer",
            "aws_lb_target_group_attachment": "loadbalancer",
            "aws_rds_cluster": "database",
            "aws_rds_instance": "database",
            "aws_dynamodb_table": "database",
            "aws_elasticache_cluster": "cache",
            "aws_sqs_queue": "queue",
            "aws_sns_topic": "topic",
            "aws_sns_topic_subscription": "subscription",
            
            # Azure resources
            "azurerm_virtual_machine": "compute",
            "azurerm_function_app": "serverless",
            "azurerm_container_group": "container",
            "azurerm_kubernetes_cluster": "kubernetes",
            "azurerm_api_management": "api",
            "azurerm_api_management_api": "api",
            "azurerm_lb": "loadbalancer",
            "azurerm_sql_server": "database",
            "azurerm_sql_database": "database",
            "azurerm_cosmosdb_account": "database",
            "azurerm_redis_cache": "cache",
            "azurerm_servicebus_queue": "queue",
            "azurerm_servicebus_topic": "topic",
            "azurerm_servicebus_subscription": "subscription",
            
            # GCP resources
            "google_compute_instance": "compute",
            "google_cloudfunctions_function": "serverless",
            "google_container_cluster": "kubernetes",
            "google_cloud_run_service": "container",
            "google_sql_database_instance": "database",
            "google_sql_database": "database",
            "google_pubsub_topic": "topic",
            "google_pubsub_subscription": "subscription",
            
            # Generic Kubernetes resources via Terraform Kubernetes provider
            "kubernetes_deployment": "kubernetes",
            "kubernetes_service": "kubernetes",
            "kubernetes_ingress": "kubernetes",
            "kubernetes_stateful_set": "kubernetes",
            "kubernetes_config_map": "kubernetes",
            "kubernetes_secret": "kubernetes",
        }
    
    def get_name(self) -> str:
        """Get the name of the parser."""
        return "Terraform"
    
    def parse(self, source_path: Union[str, Path], service_graph: ServiceGraph) -> None:
        """
        Parse Terraform configuration files and populate the service graph.
        
        Args:
            source_path: Path to the source file or directory
            service_graph: ServiceGraph instance to populate
            
        Raises:
            ValueError: If the source path does not exist or is not readable
            ParseError: If there is an error parsing the Terraform files
        """
        source_path = self._validate_source_path(source_path)
        
        try:
            # First, run terraform init and terraform plan to get the JSON plan
            # This requires terraform to be installed
            # For simplicity, we'll parse the .tf files directly in this implementation
            
            if source_path.is_file():
                self._parse_file(source_path, service_graph)
            else:  # It's a directory
                self._parse_directory(source_path, service_graph)
                
            # After parsing all files, process the references to connect nodes
            self._process_references(service_graph)
            
        except Exception as e:
            raise ParseError(f"Error parsing Terraform files: {e}")
    
    def _parse_directory(self, directory_path: Path, service_graph: ServiceGraph) -> None:
        """
        Parse all Terraform files in a directory.
        
        Args:
            directory_path: Path to the directory
            service_graph: ServiceGraph instance to populate
        """
        tf_files = list(directory_path.glob("**/*.tf"))
        
        if not tf_files:
            self.logger.warning(f"No Terraform files found in {directory_path}")
            return
        
        for tf_file in tf_files:
            self._parse_file(tf_file, service_graph)
    
    def _parse_file(self, file_path: Path, service_graph: ServiceGraph) -> None:
        """
        Parse a single Terraform file.
        
        Args:
            file_path: Path to the Terraform file
            service_graph: ServiceGraph instance to populate
        """
        self.logger.debug(f"Parsing Terraform file: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Extract resource blocks
            resource_pattern = r'resource\s+"([^"]+)"\s+"([^"]+)"\s+{([^}]*)}'
            resource_matches = re.finditer(resource_pattern, content, re.DOTALL)
            
            for match in resource_matches:
                resource_type = match.group(1)
                resource_name = match.group(2)
                resource_body = match.group(3)
                
                self._process_resource(resource_type, resource_name, resource_body, service_graph)
            
            # Extract data blocks (for references)
            data_pattern = r'data\s+"([^"]+)"\s+"([^"]+)"\s+{([^}]*)}'
            data_matches = re.finditer(data_pattern, content, re.DOTALL)
            
            for match in data_matches:
                data_type = match.group(1)
                data_name = match.group(2)
                data_body = match.group(3)
                
                self._process_data(data_type, data_name, data_body, service_graph)
                
        except Exception as e:
            self.logger.warning(f"Error processing Terraform file {file_path}: {e}")
    
    def _process_resource(self, resource_type: str, resource_name: str, resource_body: str, service_graph: ServiceGraph) -> None:
        """
        Process a Terraform resource block and update the service graph.
        
        Args:
            resource_type: Type of the Terraform resource
            resource_name: Name of the resource
            resource_body: Body of the resource block
            service_graph: ServiceGraph instance to populate
        """
        # Create a unique node ID
        node_id = f"{resource_type}.{resource_name}"
        
        # Determine the category of the resource
        category = self.resource_types.get(resource_type, "other")
        
        # Add node with appropriate attributes
        service_graph.add_node(
            node_id,
            name=resource_name,
            type=resource_type,
            category=category,
            provider=resource_type.split("_")[0] if "_" in resource_type else "unknown"
        )
        
        # Extract references to other resources
        self._extract_references(node_id, resource_body, service_graph)
    
    def _process_data(self, data_type: str, data_name: str, data_body: str, service_graph: ServiceGraph) -> None:
        """
        Process a Terraform data block and update the service graph.
        
        Args:
            data_type: Type of the Terraform data source
            data_name: Name of the data source
            data_body: Body of the data block
            service_graph: ServiceGraph instance to populate
        """
        # Create a unique node ID
        node_id = f"data.{data_type}.{data_name}"
        
        # Add node with appropriate attributes
        service_graph.add_node(
            node_id,
            name=data_name,
            type=f"data.{data_type}",
            category="data",
            provider=data_type.split("_")[0] if "_" in data_type else "unknown"
        )
    
    def _extract_references(self, node_id: str, body: str, service_graph: ServiceGraph) -> None:
        """
        Extract references to other resources from a resource body.
        
        Args:
            node_id: ID of the node in the service graph
            body: Body of the resource block
            service_graph: ServiceGraph instance to populate
        """
        # Look for references like "${aws_instance.web.id}" or "${var.instance_type}"
        reference_pattern = r'\${([^}]*)}'
        references = re.findall(reference_pattern, body)
        
        for ref in references:
            # Skip variable references and functions
            if ref.startswith("var.") or "(" in ref:
                continue
            
            # Add reference to the node
            if not service_graph.has_node_attribute(node_id, "references"):
                service_graph.add_node_attribute(node_id, references=[])
            
            refs = service_graph.get_node_attribute(node_id, "references")
            if ref not in refs:
                refs.append(ref)
                service_graph.add_node_attribute(node_id, references=refs)
    
    def _process_references(self, service_graph: ServiceGraph) -> None:
        """
        Process references between resources and create edges in the service graph.
        
        Args:
            service_graph: ServiceGraph instance to update
        """
        for node_id in service_graph.get_nodes():
            references = service_graph.get_node_attribute(node_id, "references", [])
            
            for ref in references:
                # Handle different reference formats
                parts = ref.split(".")
                
                if len(parts) >= 2:
                    # Simple reference like "aws_instance.web"
                    if parts[0] in self.resource_types:
                        target_id = f"{parts[0]}.{parts[1]}"
                        if service_graph.has_node(target_id):
                            service_graph.add_edge(node_id, target_id, type="reference")
                    
                    # Data source reference like "data.aws_ami.ubuntu"
                    elif parts[0] == "data" and len(parts) >= 3:
                        target_id = f"data.{parts[1]}.{parts[2]}"
                        if service_graph.has_node(target_id):
                            service_graph.add_edge(node_id, target_id, type="data-reference")