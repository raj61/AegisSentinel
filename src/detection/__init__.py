"""
Detection Module
==============

This module provides functionality for detecting issues in service graphs.
"""

from .detection_engine import DetectionEngine, Issue, IssueType

__all__ = ['DetectionEngine', 'Issue', 'IssueType']