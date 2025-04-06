"""
Base Parser
==========

This module defines the base parser class that all infrastructure code parsers must implement.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from src.graph import ServiceGraph

logger = logging.getLogger(__name__)

class BaseParser(ABC):
    """
    Base class for all infrastructure code parsers.
    
    This abstract class defines the interface that all parsers must implement.
    """
    
    def __init__(self):
        """Initialize the parser."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def parse(self, source_path: Union[str, Path], service_graph: ServiceGraph) -> None:
        """
        Parse the infrastructure code and populate the service graph.
        
        Args:
            source_path: Path to the source file or directory
            service_graph: ServiceGraph instance to populate
            
        Raises:
            ValueError: If the source path does not exist or is not readable
            ParseError: If there is an error parsing the infrastructure code
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the parser.
        
        Returns:
            The name of the parser (e.g., 'Kubernetes', 'Terraform')
        """
        pass
    
    def _validate_source_path(self, source_path: Union[str, Path]) -> Path:
        """
        Validate that the source path exists and is readable.
        
        Args:
            source_path: Path to the source file or directory
            
        Returns:
            Path object representing the validated source path
            
        Raises:
            ValueError: If the source path does not exist or is not readable
        """
        if isinstance(source_path, str):
            source_path = Path(source_path)
        
        if not source_path.exists():
            raise ValueError(f"Source path does not exist: {source_path}")
        
        if source_path.is_file() and not source_path.is_file():
            raise ValueError(f"Source path is not a file: {source_path}")
        
        if source_path.is_dir() and not source_path.is_dir():
            raise ValueError(f"Source path is not a directory: {source_path}")
        
        return source_path

class ParseError(Exception):
    """Exception raised when there is an error parsing infrastructure code."""
    pass