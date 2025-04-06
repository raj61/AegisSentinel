"""
Infrastructure Code Parsers
==========================

This module provides parsers for different types of infrastructure code.
"""

import logging
from pathlib import Path
from typing import Optional, Union

from .base_parser import BaseParser
from .kubernetes_parser import KubernetesParser
from .terraform_parser import TerraformParser
from .cloudformation_parser import CloudFormationParser

logger = logging.getLogger(__name__)

def get_parser(parser_type: str, source_path: Union[str, Path]) -> Optional[BaseParser]:
    """
    Get the appropriate parser for the given infrastructure code type.
    
    Args:
        parser_type: Type of infrastructure code ('kubernetes', 'terraform', 'cloudformation', or 'auto')
        source_path: Path to the source file or directory
        
    Returns:
        An instance of the appropriate parser, or None if no parser could be determined
    """
    if isinstance(source_path, str):
        source_path = Path(source_path)
    
    if parser_type == 'auto':
        # Auto-detect the parser type based on file extensions and content
        if source_path.is_file():
            if source_path.suffix in ['.yaml', '.yml']:
                # Check if it's a Kubernetes YAML file
                with open(source_path, 'r') as f:
                    content = f.read()
                    if 'apiVersion:' in content and ('kind:' in content):
                        return KubernetesParser()
            elif source_path.suffix == '.tf':
                return TerraformParser()
            elif source_path.suffix in ['.json', '.template']:
                # Check if it's a CloudFormation template
                with open(source_path, 'r') as f:
                    content = f.read()
                    if '"AWSTemplateFormatVersion"' in content or '"Resources"' in content:
                        return CloudFormationParser()
        else:  # It's a directory
            # Check for kubernetes manifests
            k8s_files = list(source_path.glob('**/*.yaml')) + list(source_path.glob('**/*.yml'))
            if k8s_files:
                return KubernetesParser()
            
            # Check for terraform files
            tf_files = list(source_path.glob('**/*.tf'))
            if tf_files:
                return TerraformParser()
            
            # Check for CloudFormation templates
            cf_files = list(source_path.glob('**/*.json')) + list(source_path.glob('**/*.template'))
            if cf_files:
                return CloudFormationParser()
        
        logger.warning(f"Could not auto-detect parser type for {source_path}")
        return None
    
    # Explicit parser type
    if parser_type == 'kubernetes':
        return KubernetesParser()
    elif parser_type == 'terraform':
        return TerraformParser()
    elif parser_type == 'cloudformation':
        return CloudFormationParser()
    
    logger.warning(f"Unknown parser type: {parser_type}")
    return None