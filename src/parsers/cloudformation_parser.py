"""
CloudFormation Parser
===================

This module provides a parser for AWS CloudFormation templates.
"""

import logging
import json
import re
from pathlib import Path
from typing import Dict, List, Union, Any, Set

from src.graph import ServiceGraph
from src.parsers.base_parser import BaseParser, ParseError

logger = logging.getLogger(__name__)

class CloudFormationParser(BaseParser):
    """
    Parser for AWS CloudFormation templates.
    
    This parser can handle CloudFormation templates in JSON or YAML format
    and extract service dependencies from various resource types.
    """
    
    def __init__(self):
        """Initialize the CloudFormation parser."""
        super().__init__()
        self.resource_categories = {
            # Compute
            "AWS::EC2::Instance": "compute",
            "AWS::Lambda::Function": "serverless",
            "AWS::Batch::JobDefinition": "compute",
            "AWS::Batch::ComputeEnvironment": "compute",
            
            # Containers
            "AWS::ECS::Service": "container",
            "AWS::ECS::TaskDefinition": "container",
            "AWS::ECS::Cluster": "container",
            "AWS::EKS::Cluster": "kubernetes",
            
            # API and Web
            "AWS::ApiGateway::RestApi": "api",
            "AWS::ApiGateway::Resource": "api",
            "AWS::ApiGateway::Method": "api",
            "AWS::ApiGateway::Stage": "api",
            "AWS::ApiGatewayV2::Api": "api",
            "AWS::ApiGatewayV2::Route": "api",
            "AWS::ApiGatewayV2::Integration": "api",
            
            # Load Balancing
            "AWS::ElasticLoadBalancing::LoadBalancer": "loadbalancer",
            "AWS::ElasticLoadBalancingV2::LoadBalancer": "loadbalancer",
            "AWS::ElasticLoadBalancingV2::Listener": "loadbalancer",
            "AWS::ElasticLoadBalancingV2::TargetGroup": "loadbalancer",
            
            # Database
            "AWS::RDS::DBInstance": "database",
            "AWS::RDS::DBCluster": "database",
            "AWS::DynamoDB::Table": "database",
            "AWS::ElastiCache::CacheCluster": "cache",
            "AWS::ElastiCache::ReplicationGroup": "cache",
            "AWS::Neptune::DBCluster": "database",
            "AWS::Redshift::Cluster": "database",
            
            # Messaging
            "AWS::SQS::Queue": "queue",
            "AWS::SNS::Topic": "topic",
            "AWS::SNS::Subscription": "subscription",
            "AWS::EventBridge::Rule": "event",
            "AWS::Events::Rule": "event",
            "AWS::Kinesis::Stream": "stream",
            "AWS::KinesisFirehose::DeliveryStream": "stream",
            
            # Storage
            "AWS::S3::Bucket": "storage",
            "AWS::EFS::FileSystem": "storage",
            
            # Networking
            "AWS::EC2::VPC": "network",
            "AWS::EC2::Subnet": "network",
            "AWS::EC2::SecurityGroup": "network",
            "AWS::EC2::RouteTable": "network",
            "AWS::EC2::InternetGateway": "network",
            
            # IAM
            "AWS::IAM::Role": "security",
            "AWS::IAM::Policy": "security",
            "AWS::IAM::ManagedPolicy": "security",
        }
    
    def get_name(self) -> str:
        """Get the name of the parser."""
        return "CloudFormation"
    
    def parse(self, source_path: Union[str, Path], service_graph: ServiceGraph) -> None:
        """
        Parse CloudFormation templates and populate the service graph.
        
        Args:
            source_path: Path to the source file or directory
            service_graph: ServiceGraph instance to populate
            
        Raises:
            ValueError: If the source path does not exist or is not readable
            ParseError: If there is an error parsing the CloudFormation templates
        """
        source_path = self._validate_source_path(source_path)
        
        try:
            if source_path.is_file():
                self._parse_file(source_path, service_graph)
            else:  # It's a directory
                self._parse_directory(source_path, service_graph)
                
            # After parsing all files, process the references to connect nodes
            self._process_references(service_graph)
            
        except Exception as e:
            raise ParseError(f"Error parsing CloudFormation templates: {e}")
    
    def _parse_directory(self, directory_path: Path, service_graph: ServiceGraph) -> None:
        """
        Parse all CloudFormation templates in a directory.
        
        Args:
            directory_path: Path to the directory
            service_graph: ServiceGraph instance to populate
        """
        # Look for JSON and YAML files that might be CloudFormation templates
        template_files = list(directory_path.glob("**/*.json")) + \
                         list(directory_path.glob("**/*.yaml")) + \
                         list(directory_path.glob("**/*.yml")) + \
                         list(directory_path.glob("**/*.template"))
        
        if not template_files:
            self.logger.warning(f"No potential CloudFormation templates found in {directory_path}")
            return
        
        for template_file in template_files:
            try:
                # Check if it's a CloudFormation template
                with open(template_file, 'r') as f:
                    content = f.read()
                    
                    # Simple heuristic to identify CloudFormation templates
                    if '"AWSTemplateFormatVersion"' in content or '"Resources"' in content or \
                       'AWSTemplateFormatVersion:' in content or 'Resources:' in content:
                        self._parse_file(template_file, service_graph)
            except Exception as e:
                self.logger.warning(f"Error checking if {template_file} is a CloudFormation template: {e}")
    
    def _parse_file(self, file_path: Path, service_graph: ServiceGraph) -> None:
        """
        Parse a single CloudFormation template.
        
        Args:
            file_path: Path to the CloudFormation template
            service_graph: ServiceGraph instance to populate
        """
        self.logger.debug(f"Parsing CloudFormation template: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Determine if it's JSON or YAML
            try:
                template = json.loads(content)
            except json.JSONDecodeError:
                # Try to parse as YAML
                import yaml
                template = yaml.safe_load(content)
            
            # Extract stack name from the file name
            stack_name = file_path.stem
            
            # Process the template
            self._process_template(template, stack_name, service_graph)
            
        except Exception as e:
            self.logger.warning(f"Error processing CloudFormation template {file_path}: {e}")
    
    def _process_template(self, template: Dict[str, Any], stack_name: str, service_graph: ServiceGraph) -> None:
        """
        Process a CloudFormation template and update the service graph.
        
        Args:
            template: CloudFormation template as a dictionary
            stack_name: Name of the CloudFormation stack
            service_graph: ServiceGraph instance to populate
        """
        # Add a node for the stack itself
        stack_id = f"stack.{stack_name}"
        service_graph.add_node(
            stack_id,
            name=stack_name,
            type="AWS::CloudFormation::Stack",
            category="stack",
            provider="aws"
        )
        
        # Process resources
        resources = template.get("Resources", {})
        for resource_id, resource in resources.items():
            resource_type = resource.get("Type")
            if not resource_type:
                continue
            
            # Create a unique node ID
            node_id = f"{stack_name}.{resource_id}"
            
            # Determine the category of the resource
            category = self.resource_categories.get(resource_type, "other")
            
            # Add node with appropriate attributes
            service_graph.add_node(
                node_id,
                name=resource_id,
                type=resource_type,
                category=category,
                stack=stack_name,
                provider="aws"
            )
            
            # Add edge from stack to resource
            service_graph.add_edge(stack_id, node_id, type="contains")
            
            # Extract properties for references
            properties = resource.get("Properties", {})
            if properties:
                service_graph.add_node_attribute(node_id, properties=properties)
                
                # Extract references from properties
                self._extract_references(node_id, properties, service_graph)
    
    def _extract_references(self, node_id: str, obj: Any, service_graph: ServiceGraph) -> None:
        """
        Recursively extract references from a CloudFormation object.
        
        Args:
            node_id: ID of the node in the service graph
            obj: CloudFormation object to extract references from
            service_graph: ServiceGraph instance to populate
        """
        if isinstance(obj, dict):
            # Check for Ref
            if "Ref" in obj and len(obj) == 1:
                ref = obj["Ref"]
                self._add_reference(node_id, ref, "Ref", service_graph)
            
            # Check for GetAtt
            elif "Fn::GetAtt" in obj and len(obj) == 1:
                get_att = obj["Fn::GetAtt"]
                if isinstance(get_att, list) and len(get_att) == 2:
                    ref = get_att[0]
                    attr = get_att[1]
                    self._add_reference(node_id, ref, f"GetAtt.{attr}", service_graph)
            
            # Recursively check all values
            for key, value in obj.items():
                self._extract_references(node_id, value, service_graph)
        
        elif isinstance(obj, list):
            # Recursively check all items
            for item in obj:
                self._extract_references(node_id, item, service_graph)
    
    def _add_reference(self, node_id: str, ref: str, ref_type: str, service_graph: ServiceGraph) -> None:
        """
        Add a reference to a node.
        
        Args:
            node_id: ID of the node in the service graph
            ref: Reference value
            ref_type: Type of reference (Ref, GetAtt, etc.)
            service_graph: ServiceGraph instance to populate
        """
        # Skip AWS pseudo-parameters
        if ref in ["AWS::Region", "AWS::AccountId", "AWS::NotificationARNs", 
                  "AWS::NoValue", "AWS::Partition", "AWS::StackId", 
                  "AWS::StackName", "AWS::URLSuffix"]:
            return
        
        # Add reference to the node
        if not service_graph.has_node_attribute(node_id, "references"):
            service_graph.add_node_attribute(node_id, references=[])
        
        refs = service_graph.get_node_attribute(node_id, "references")
        ref_info = {"ref": ref, "type": ref_type}
        if ref_info not in refs:
            refs.append(ref_info)
            service_graph.add_node_attribute(node_id, references=refs)
    
    def _process_references(self, service_graph: ServiceGraph) -> None:
        """
        Process references between resources and create edges in the service graph.
        
        Args:
            service_graph: ServiceGraph instance to update
        """
        # Get all nodes
        nodes = service_graph.get_nodes()
        
        # Create a mapping of resource logical IDs to node IDs
        resource_map = {}
        for node_id in nodes:
            node = service_graph.get_node(node_id)
            if "stack" in node and "name" in node:
                stack = node["stack"]
                name = node["name"]
                resource_map[name] = node_id
        
        # Process references
        for node_id in nodes:
            references = service_graph.get_node_attribute(node_id, "references", [])
            
            for ref_info in references:
                ref = ref_info["ref"]
                ref_type = ref_info["type"]
                
                # Check if the reference points to a known resource
                if ref in resource_map:
                    target_id = resource_map[ref]
                    service_graph.add_edge(node_id, target_id, type=ref_type.lower())