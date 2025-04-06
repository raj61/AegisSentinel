"""
API Module
=========

This module provides a REST API for interacting with the service graph builder.
"""

from .server import start_api_server

__all__ = ['start_api_server']