"""
Log Anomaly Detection
===================

This module provides functionality for detecting anomalies in logs.
"""

import re
import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Set, Any, Optional, Union, Pattern, Tuple
from collections import defaultdict
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class LogPattern:
    """
    Represents a log pattern for anomaly detection.
    
    Attributes:
        pattern: Regular expression pattern to match logs
        name: Name of the pattern
        severity: Severity level (1-5, with 5 being the most severe)
        description: Description of the pattern
        service: Service associated with the pattern
        threshold: Threshold for anomaly detection (e.g., occurrences per minute)
        window_seconds: Time window for anomaly detection in seconds
    """
    pattern: str
    name: str
    severity: int
    description: str
    service: Optional[str] = None
    threshold: float = 1.0
    window_seconds: int = 60
    
    def __post_init__(self):
        """Compile the regex pattern."""
        self.compiled_pattern = re.compile(self.pattern)
    
    def matches(self, log_line: str) -> bool:
        """
        Check if the log line matches the pattern.
        
        Args:
            log_line: Log line to check
            
        Returns:
            True if the log line matches the pattern, False otherwise
        """
        return bool(self.compiled_pattern.search(log_line))

@dataclass
class AnomalyScore:
    """
    Represents an anomaly score for a log pattern.
    
    Attributes:
        pattern: Log pattern that triggered the anomaly
        score: Anomaly score (0-1, with 1 being the most anomalous)
        timestamp: Timestamp of the anomaly
        count: Number of occurrences in the time window
        expected: Expected number of occurrences in the time window
        log_samples: Sample log lines that triggered the anomaly
    """
    pattern: LogPattern
    score: float
    timestamp: datetime
    count: int
    expected: float
    log_samples: List[str]
    
    @property
    def is_anomaly(self) -> bool:
        """Check if the score indicates an anomaly."""
        return self.score > 0.5
    
    @property
    def severity(self) -> int:
        """Get the severity level of the anomaly."""
        return self.pattern.severity
    
    @property
    def service(self) -> Optional[str]:
        """Get the service associated with the anomaly."""
        return self.pattern.service
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the anomaly score to a dictionary."""
        return {
            'pattern_name': self.pattern.name,
            'score': self.score,
            'timestamp': self.timestamp.isoformat(),
            'count': self.count,
            'expected': self.expected,
            'log_samples': self.log_samples[:5],  # Limit to 5 samples
            'severity': self.severity,
            'service': self.service,
            'description': self.pattern.description,
            'is_anomaly': self.is_anomaly
        }

class LogAnomalyDetector:
    """
    Detector for anomalies in logs based on patterns and statistical analysis.
    
    This class provides functionality for detecting anomalies in logs using
    predefined patterns and statistical analysis of log frequencies.
    """
    
    def __init__(self):
        """Initialize the log anomaly detector."""
        self.patterns: List[LogPattern] = []
        self.log_history: Dict[str, List[Tuple[datetime, str]]] = defaultdict(list)
        self.baseline: Dict[str, float] = {}
        self.anomaly_history: List[AnomalyScore] = []
        self.max_history_size = 10000  # Maximum number of log lines to keep in history
        self.max_anomaly_history = 1000  # Maximum number of anomalies to keep in history
    
    def add_pattern(self, pattern: LogPattern) -> None:
        """
        Add a log pattern for anomaly detection.
        
        Args:
            pattern: Log pattern to add
        """
        self.patterns.append(pattern)
        self.baseline[pattern.name] = pattern.threshold
    
    def add_patterns(self, patterns: List[LogPattern]) -> None:
        """
        Add multiple log patterns for anomaly detection.
        
        Args:
            patterns: List of log patterns to add
        """
        for pattern in patterns:
            self.add_pattern(pattern)
    
    def process_log(self, log_line: str, timestamp: Optional[datetime] = None) -> List[AnomalyScore]:
        """
        Process a log line and detect anomalies.
        
        Args:
            log_line: Log line to process
            timestamp: Timestamp of the log line (default: current time)
            
        Returns:
            List of anomaly scores for the log line
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        anomalies = []
        
        # Check each pattern
        for pattern in self.patterns:
            if pattern.matches(log_line):
                # Add to history
                self.log_history[pattern.name].append((timestamp, log_line))
                
                # Check for anomalies
                anomaly = self._check_anomaly(pattern, timestamp)
                if anomaly:
                    anomalies.append(anomaly)
                    self.anomaly_history.append(anomaly)
                    
                    # Trim anomaly history if needed
                    if len(self.anomaly_history) > self.max_anomaly_history:
                        self.anomaly_history = self.anomaly_history[-self.max_anomaly_history:]
        
        # Trim log history if needed
        for pattern_name, history in self.log_history.items():
            if len(history) > self.max_history_size:
                self.log_history[pattern_name] = history[-self.max_history_size:]
        
        return anomalies
    
    def process_logs(self, logs: List[str], timestamps: Optional[List[datetime]] = None) -> List[AnomalyScore]:
        """
        Process multiple log lines and detect anomalies.
        
        Args:
            logs: List of log lines to process
            timestamps: List of timestamps for the log lines (default: current time for all)
            
        Returns:
            List of anomaly scores for the log lines
        """
        if timestamps is None:
            timestamps = [datetime.now() for _ in logs]
        
        anomalies = []
        for log_line, timestamp in zip(logs, timestamps):
            anomalies.extend(self.process_log(log_line, timestamp))
        
        return anomalies
    
    def _check_anomaly(self, pattern: LogPattern, timestamp: datetime) -> Optional[AnomalyScore]:
        """
        Check if there's an anomaly for a pattern at a given timestamp.
        
        Args:
            pattern: Log pattern to check
            timestamp: Timestamp to check
            
        Returns:
            Anomaly score if there's an anomaly, None otherwise
        """
        # Get logs in the time window
        window_start = timestamp - timedelta(seconds=pattern.window_seconds)
        logs_in_window = [
            log for ts, log in self.log_history[pattern.name]
            if ts >= window_start and ts <= timestamp
        ]
        
        # Count occurrences
        count = len(logs_in_window)
        
        # Get expected count
        expected = self.baseline[pattern.name]
        
        # Calculate score
        if expected == 0:
            score = 1.0 if count > 0 else 0.0
        else:
            ratio = count / expected
            if ratio <= 1.0:
                score = 0.0  # No anomaly if count is less than or equal to expected
            else:
                # Score increases with ratio, capped at 1.0
                score = min(1.0, (ratio - 1.0) / 9.0)  # Reaches 1.0 when ratio is 10
        
        # Return anomaly if score is high enough
        if score > 0.0:
            return AnomalyScore(
                pattern=pattern,
                score=score,
                timestamp=timestamp,
                count=count,
                expected=expected,
                log_samples=logs_in_window[:10]  # Limit to 10 samples
            )
        
        return None
    
    def update_baseline(self, pattern_name: str, new_baseline: float) -> None:
        """
        Update the baseline for a pattern.
        
        Args:
            pattern_name: Name of the pattern
            new_baseline: New baseline value
        """
        if pattern_name in self.baseline:
            self.baseline[pattern_name] = new_baseline
    
    def get_anomalies(self, since: Optional[datetime] = None, 
                     service: Optional[str] = None,
                     min_severity: int = 1,
                     min_score: float = 0.5) -> List[AnomalyScore]:
        """
        Get anomalies detected since a given timestamp.
        
        Args:
            since: Timestamp to get anomalies since (default: all anomalies)
            service: Filter by service (default: all services)
            min_severity: Minimum severity level (default: 1)
            min_score: Minimum anomaly score (default: 0.5)
            
        Returns:
            List of anomaly scores
        """
        filtered_anomalies = []
        
        for anomaly in self.anomaly_history:
            # Filter by timestamp
            if since is not None and anomaly.timestamp < since:
                continue
            
            # Filter by service
            if service is not None and anomaly.service != service:
                continue
            
            # Filter by severity
            if anomaly.severity < min_severity:
                continue
            
            # Filter by score
            if anomaly.score < min_score:
                continue
            
            filtered_anomalies.append(anomaly)
        
        return filtered_anomalies
    
    def get_anomaly_count(self, since: Optional[datetime] = None,
                         service: Optional[str] = None,
                         min_severity: int = 1,
                         min_score: float = 0.5) -> int:
        """
        Get the count of anomalies detected since a given timestamp.
        
        Args:
            since: Timestamp to get anomalies since (default: all anomalies)
            service: Filter by service (default: all services)
            min_severity: Minimum severity level (default: 1)
            min_score: Minimum anomaly score (default: 0.5)
            
        Returns:
            Count of anomalies
        """
        return len(self.get_anomalies(since, service, min_severity, min_score))
    
    def clear_history(self) -> None:
        """Clear the log and anomaly history."""
        self.log_history.clear()
        self.anomaly_history.clear()
    
    @classmethod
    def create_with_default_patterns(cls) -> 'LogAnomalyDetector':
        """
        Create a log anomaly detector with default patterns.
        
        Returns:
            LogAnomalyDetector instance with default patterns
        """
        detector = cls()
        
        # Add some default patterns
        default_patterns = [
            LogPattern(
                pattern=r"Error|Exception|Failed|Failure|Timeout",
                name="general_error",
                severity=3,
                description="General error pattern",
                threshold=5.0,
                window_seconds=300
            ),
            LogPattern(
                pattern=r"OutOfMemory|OOM|memory\s+usage|memory\s+limit",
                name="memory_error",
                severity=4,
                description="Memory-related error",
                threshold=1.0,
                window_seconds=300
            ),
            LogPattern(
                pattern=r"CPU\s+usage|CPU\s+limit|high\s+CPU",
                name="cpu_error",
                severity=3,
                description="CPU-related error",
                threshold=1.0,
                window_seconds=300
            ),
            LogPattern(
                pattern=r"Disk\s+full|no\s+space\s+left|storage\s+limit",
                name="disk_error",
                severity=4,
                description="Disk-related error",
                threshold=1.0,
                window_seconds=300
            ),
            LogPattern(
                pattern=r"Connection\s+refused|Connection\s+reset|Connection\s+timeout",
                name="connection_error",
                severity=3,
                description="Connection-related error",
                threshold=3.0,
                window_seconds=300
            ),
            LogPattern(
                pattern=r"Authentication\s+failed|Unauthorized|Permission\s+denied|Access\s+denied",
                name="auth_error",
                severity=3,
                description="Authentication-related error",
                threshold=3.0,
                window_seconds=300
            ),
            LogPattern(
                pattern=r"Database\s+error|SQL\s+error|query\s+failed",
                name="database_error",
                severity=4,
                description="Database-related error",
                threshold=2.0,
                window_seconds=300
            ),
            LogPattern(
                pattern=r"5[0-9]{2}|HTTP\s+5[0-9]{2}",
                name="http_5xx",
                severity=3,
                description="HTTP 5xx error",
                threshold=5.0,
                window_seconds=300
            ),
            LogPattern(
                pattern=r"4[0-9]{2}|HTTP\s+4[0-9]{2}",
                name="http_4xx",
                severity=2,
                description="HTTP 4xx error",
                threshold=10.0,
                window_seconds=300
            ),
            LogPattern(
                pattern=r"Crash|Panic|Fatal|Segmentation\s+fault|SIGSEGV",
                name="crash",
                severity=5,
                description="Application crash",
                threshold=1.0,
                window_seconds=300
            ),
        ]
        
        detector.add_patterns(default_patterns)
        return detector