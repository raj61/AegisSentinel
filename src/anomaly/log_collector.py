"""
Log Collector
===========

This module provides functionality for collecting logs from various sources.
"""

import os
import re
import json
import logging
import subprocess
import time
from datetime import datetime, timedelta
from typing import Dict, List, Set, Any, Optional, Union, Callable, Iterator, Tuple
from pathlib import Path
import threading
import queue

logger = logging.getLogger(__name__)

class LogCollector:
    """
    Base class for log collectors.
    
    This class provides the interface for collecting logs from various sources.
    """
    
    def __init__(self):
        """Initialize the log collector."""
        self.callbacks: List[Callable[[str, datetime], None]] = []
    
    def add_callback(self, callback: Callable[[str, datetime], None]) -> None:
        """
        Add a callback function to be called for each log line.
        
        Args:
            callback: Function to call with (log_line, timestamp) for each log line
        """
        self.callbacks.append(callback)
    
    def process_log(self, log_line: str, timestamp: Optional[datetime] = None) -> None:
        """
        Process a log line and call all callbacks.
        
        Args:
            log_line: Log line to process
            timestamp: Timestamp of the log line (default: current time)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        for callback in self.callbacks:
            try:
                callback(log_line, timestamp)
            except Exception as e:
                logger.error(f"Error in log callback: {e}")
    
    def start(self) -> None:
        """Start collecting logs."""
        raise NotImplementedError("Subclasses must implement start()")
    
    def stop(self) -> None:
        """Stop collecting logs."""
        raise NotImplementedError("Subclasses must implement stop()")

class FileLogCollector(LogCollector):
    """
    Collector for logs from files.
    
    This class provides functionality for collecting logs from files.
    """
    
    def __init__(self, file_path: Union[str, Path], follow: bool = True):
        """
        Initialize the file log collector.
        
        Args:
            file_path: Path to the log file
            follow: Whether to follow the file for new logs (like tail -f)
        """
        super().__init__()
        self.file_path = Path(file_path)
        self.follow = follow
        self.running = False
        self.thread = None
    
    def start(self) -> None:
        """Start collecting logs from the file."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._collect_logs)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self) -> None:
        """Stop collecting logs from the file."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
    
    def _collect_logs(self) -> None:
        """Collect logs from the file."""
        if not self.file_path.exists():
            logger.error(f"Log file does not exist: {self.file_path}")
            return
        
        with open(self.file_path, 'r') as f:
            # Go to the end of the file if following
            if self.follow:
                f.seek(0, os.SEEK_END)
            
            # Read existing content if not following
            if not self.follow:
                for line in f:
                    line = line.rstrip('\n')
                    self.process_log(line)
            
            # Follow the file for new content
            while self.running and self.follow:
                line = f.readline()
                if line:
                    line = line.rstrip('\n')
                    self.process_log(line)
                else:
                    time.sleep(0.1)

class KubernetesLogCollector(LogCollector):
    """
    Collector for logs from Kubernetes pods.
    
    This class provides functionality for collecting logs from Kubernetes pods.
    """
    
    def __init__(self, namespace: str, pod_name: str, container: Optional[str] = None,
                previous: bool = False, since: Optional[str] = None):
        """
        Initialize the Kubernetes log collector.
        
        Args:
            namespace: Kubernetes namespace
            pod_name: Name of the pod
            container: Name of the container (default: first container in the pod)
            previous: Whether to get logs from the previous instance of the container
            since: Get logs since this time (e.g., '1h', '10m', '5s')
        """
        super().__init__()
        self.namespace = namespace
        self.pod_name = pod_name
        self.container = container
        self.previous = previous
        self.since = since
        self.running = False
        self.process = None
        self.thread = None
    
    def start(self) -> None:
        """Start collecting logs from the Kubernetes pod."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._collect_logs)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self) -> None:
        """Stop collecting logs from the Kubernetes pod."""
        self.running = False
        if self.process:
            try:
                self.process.terminate()
            except:
                pass
        
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
    
    def _collect_logs(self) -> None:
        """Collect logs from the Kubernetes pod."""
        cmd = ['kubectl', 'logs', '-n', self.namespace, self.pod_name]
        
        if self.container:
            cmd.extend(['-c', self.container])
        
        if self.previous:
            cmd.append('-p')
        
        if self.since:
            cmd.extend(['--since', self.since])
        
        # Follow the logs
        cmd.append('-f')
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Read stdout
            for line in self.process.stdout:
                if not self.running:
                    break
                
                line = line.rstrip('\n')
                self.process_log(line)
            
            # Wait for the process to finish
            self.process.wait()
            
        except Exception as e:
            logger.error(f"Error collecting Kubernetes logs: {e}")
        
        finally:
            self.running = False

class MultiLogCollector(LogCollector):
    """
    Collector for logs from multiple sources.
    
    This class provides functionality for collecting logs from multiple sources.
    """
    
    def __init__(self):
        """Initialize the multi log collector."""
        super().__init__()
        self.collectors: List[LogCollector] = []
    
    def add_collector(self, collector: LogCollector) -> None:
        """
        Add a log collector.
        
        Args:
            collector: Log collector to add
        """
        # Add our callbacks to the collector
        for callback in self.callbacks:
            collector.add_callback(callback)
        
        self.collectors.append(collector)
    
    def add_callback(self, callback: Callable[[str, datetime], None]) -> None:
        """
        Add a callback function to be called for each log line.
        
        Args:
            callback: Function to call with (log_line, timestamp) for each log line
        """
        super().add_callback(callback)
        
        # Add the callback to all collectors
        for collector in self.collectors:
            collector.add_callback(callback)
    
    def start(self) -> None:
        """Start collecting logs from all sources."""
        for collector in self.collectors:
            collector.start()
    
    def stop(self) -> None:
        """Stop collecting logs from all sources."""
        for collector in self.collectors:
            collector.stop()

class LogBuffer:
    """
    Buffer for storing logs.
    
    This class provides functionality for storing logs in a buffer.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize the log buffer.
        
        Args:
            max_size: Maximum number of log lines to store
        """
        self.logs: List[Tuple[str, datetime]] = []
        self.max_size = max_size
        self.lock = threading.Lock()
    
    def add_log(self, log_line: str, timestamp: Optional[datetime] = None) -> None:
        """
        Add a log line to the buffer.
        
        Args:
            log_line: Log line to add
            timestamp: Timestamp of the log line (default: current time)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        with self.lock:
            self.logs.append((log_line, timestamp))
            
            # Trim the buffer if needed
            if len(self.logs) > self.max_size:
                self.logs = self.logs[-self.max_size:]
    
    def get_logs(self, count: Optional[int] = None, 
                since: Optional[datetime] = None) -> List[Tuple[str, datetime]]:
        """
        Get logs from the buffer.
        
        Args:
            count: Maximum number of logs to return (default: all)
            since: Get logs since this timestamp (default: all)
            
        Returns:
            List of (log_line, timestamp) tuples
        """
        with self.lock:
            # Filter by timestamp
            if since is not None:
                logs = [(log, ts) for log, ts in self.logs if ts >= since]
            else:
                logs = self.logs.copy()
            
            # Limit by count
            if count is not None:
                logs = logs[-count:]
            
            return logs
    
    def clear(self) -> None:
        """Clear the log buffer."""
        with self.lock:
            self.logs.clear()