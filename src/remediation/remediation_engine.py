"""
Remediation Engine
===============

This module provides functionality for remediating issues in the system.
"""

import logging
import subprocess
import json
import time
import re
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Set, Any, Optional, Union, Callable, Tuple
from datetime import datetime
import threading
import queue

from src.anomaly import AnomalyScore
from src.detection import Issue

logger = logging.getLogger(__name__)

class RemediationStatus(Enum):
    """Status of a remediation action."""
    SUCCESS = "success"
    FAILURE = "failure"
    IN_PROGRESS = "in_progress"
    PENDING = "pending"
    SKIPPED = "skipped"

@dataclass
class RemediationAction:
    """
    Represents a remediation action.
    
    Attributes:
        name: Name of the action
        description: Description of the action
        command: Command to execute
        service: Service associated with the action
        timeout_seconds: Timeout for the action in seconds
        max_retries: Maximum number of retries
        cooldown_seconds: Cooldown period between retries in seconds
    """
    name: str
    description: str
    command: str
    service: Optional[str] = None
    timeout_seconds: int = 60
    max_retries: int = 3
    cooldown_seconds: int = 60
    
    def execute(self) -> Tuple[bool, str]:
        """
        Execute the remediation action.
        
        Returns:
            Tuple of (success, output)
        """
        try:
            logger.info(f"Executing remediation action: {self.name}")
            logger.debug(f"Command: {self.command}")
            
            process = subprocess.Popen(
                self.command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.timeout_seconds)
                return_code = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                return False, f"Timeout after {self.timeout_seconds} seconds"
            
            if return_code == 0:
                return True, stdout
            else:
                return False, f"Command failed with return code {return_code}: {stderr}"
            
        except Exception as e:
            logger.error(f"Error executing remediation action: {e}")
            return False, str(e)

@dataclass
class RemediationResult:
    """
    Represents the result of a remediation action.
    
    Attributes:
        action: Remediation action that was executed
        status: Status of the remediation
        output: Output of the remediation action
        timestamp: Timestamp of the remediation
        issue: Issue that triggered the remediation
        anomaly: Anomaly that triggered the remediation
        retry_count: Number of retries
    """
    action: RemediationAction
    status: RemediationStatus
    output: str
    timestamp: datetime
    issue: Optional[Issue] = None
    anomaly: Optional[AnomalyScore] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the remediation result to a dictionary."""
        return {
            'action_name': self.action.name,
            'status': self.status.value,
            'output': self.output,
            'timestamp': self.timestamp.isoformat(),
            'issue_type': self.issue.type.value if self.issue else None,
            'anomaly_pattern': self.anomaly.pattern.name if self.anomaly else None,
            'service': self.action.service,
            'retry_count': self.retry_count
        }

@dataclass
class RemediationPolicy:
    """
    Represents a policy for when to apply a remediation action.
    
    Attributes:
        name: Name of the policy
        description: Description of the policy
        action: Remediation action to execute
        issue_types: List of issue types that trigger this policy
        anomaly_patterns: List of anomaly patterns that trigger this policy
        services: List of services that this policy applies to
        min_severity: Minimum severity level for issues/anomalies
        min_score: Minimum anomaly score
        enabled: Whether the policy is enabled
    """
    name: str
    description: str
    action: RemediationAction
    issue_types: Optional[List[str]] = None
    anomaly_patterns: Optional[List[str]] = None
    services: Optional[List[str]] = None
    min_severity: int = 1
    min_score: float = 0.5
    enabled: bool = True
    
    def matches_issue(self, issue: Issue) -> bool:
        """
        Check if the policy matches an issue.
        
        Args:
            issue: Issue to check
            
        Returns:
            True if the policy matches the issue, False otherwise
        """
        if not self.enabled:
            return False
        
        # Check issue type
        if self.issue_types is not None:
            if issue.type.value not in self.issue_types:
                return False
        
        # Check severity
        if issue.severity < self.min_severity:
            return False
        
        # Check service
        if self.services is not None and self.action.service is not None:
            if self.action.service not in self.services:
                return False
        
        return True
    
    def matches_anomaly(self, anomaly: AnomalyScore) -> bool:
        """
        Check if the policy matches an anomaly.
        
        Args:
            anomaly: Anomaly to check
            
        Returns:
            True if the policy matches the anomaly, False otherwise
        """
        if not self.enabled:
            return False
        
        # Check anomaly pattern
        if self.anomaly_patterns is not None:
            if anomaly.pattern.name not in self.anomaly_patterns:
                return False
        
        # Check severity
        if anomaly.pattern.severity < self.min_severity:
            return False
        
        # Check score
        if anomaly.score < self.min_score:
            return False
        
        # Check service
        if self.services is not None and anomaly.service is not None:
            if anomaly.service not in self.services:
                return False
        
        return True

class RemediationEngine:
    """
    Engine for remediating issues in the system.
    
    This class provides functionality for executing remediation actions
    based on policies.
    """
    
    def __init__(self):
        """Initialize the remediation engine."""
        self.policies: List[RemediationPolicy] = []
        self.results: List[RemediationResult] = []
        self.max_results = 1000
        self.callbacks: List[Callable[[RemediationResult], None]] = []
        self.lock = threading.Lock()
        
        # Queue for pending remediations
        self.queue = queue.Queue()
        self.worker_thread = None
        self.running = False
    
    def add_policy(self, policy: RemediationPolicy) -> None:
        """
        Add a remediation policy.
        
        Args:
            policy: Remediation policy to add
        """
        with self.lock:
            self.policies.append(policy)
    
    def add_policies(self, policies: List[RemediationPolicy]) -> None:
        """
        Add multiple remediation policies.
        
        Args:
            policies: List of remediation policies to add
        """
        with self.lock:
            self.policies.extend(policies)
    
    def add_callback(self, callback: Callable[[RemediationResult], None]) -> None:
        """
        Add a callback function to be called for each remediation result.
        
        Args:
            callback: Function to call with the remediation result
        """
        self.callbacks.append(callback)
    
    def start(self) -> None:
        """Start the remediation engine."""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()
    
    def stop(self) -> None:
        """Stop the remediation engine."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)
            self.worker_thread = None
    
    def _worker(self) -> None:
        """Worker thread for processing the remediation queue."""
        while self.running:
            try:
                # Get the next item from the queue
                item = self.queue.get(timeout=1.0)
                
                # Process the item
                if isinstance(item, tuple) and len(item) == 2:
                    if isinstance(item[0], Issue):
                        self._process_issue(item[0], item[1])
                    elif isinstance(item[0], AnomalyScore):
                        self._process_anomaly(item[0], item[1])
                
                # Mark the item as done
                self.queue.task_done()
                
            except queue.Empty:
                # No items in the queue
                pass
            except Exception as e:
                logger.error(f"Error in remediation worker: {e}")
    
    def _process_issue(self, issue: Issue, retry_map: Dict[str, int]) -> None:
        """
        Process an issue and execute matching policies.
        
        Args:
            issue: Issue to process
            retry_map: Map of policy names to retry counts
        """
        # Find matching policies
        matching_policies = []
        with self.lock:
            for policy in self.policies:
                if policy.matches_issue(issue):
                    matching_policies.append(policy)
        
        # Execute matching policies
        for policy in matching_policies:
            # Check if we've exceeded the retry limit
            retry_count = retry_map.get(policy.name, 0)
            if retry_count >= policy.action.max_retries:
                logger.warning(f"Skipping policy {policy.name} for issue {issue.type.value}: "
                              f"exceeded retry limit ({retry_count}/{policy.action.max_retries})")
                
                # Add a skipped result
                result = RemediationResult(
                    action=policy.action,
                    status=RemediationStatus.SKIPPED,
                    output=f"Exceeded retry limit ({retry_count}/{policy.action.max_retries})",
                    timestamp=datetime.now(),
                    issue=issue,
                    anomaly=None,
                    retry_count=retry_count
                )
                self._add_result(result)
                continue
            
            # Execute the action
            success, output = policy.action.execute()
            
            # Update the result
            status = RemediationStatus.SUCCESS if success else RemediationStatus.FAILURE
            result = RemediationResult(
                action=policy.action,
                status=status,
                output=output,
                timestamp=datetime.now(),
                issue=issue,
                anomaly=None,
                retry_count=retry_count
            )
            self._add_result(result)
            
            # Update the retry count
            if not success:
                retry_map[policy.name] = retry_count + 1
                
                # Schedule a retry if needed
                if retry_count + 1 < policy.action.max_retries:
                    logger.info(f"Scheduling retry for policy {policy.name} "
                               f"({retry_count + 1}/{policy.action.max_retries})")
                    
                    # Add to the queue after the cooldown period
                    threading.Timer(
                        policy.action.cooldown_seconds,
                        lambda: self.queue.put((issue, retry_map))
                    ).start()
    
    def _process_anomaly(self, anomaly: AnomalyScore, retry_map: Dict[str, int]) -> None:
        """
        Process an anomaly and execute matching policies.
        
        Args:
            anomaly: Anomaly to process
            retry_map: Map of policy names to retry counts
        """
        # Find matching policies
        matching_policies = []
        with self.lock:
            for policy in self.policies:
                if policy.matches_anomaly(anomaly):
                    matching_policies.append(policy)
        
        # Execute matching policies
        for policy in matching_policies:
            # Check if we've exceeded the retry limit
            retry_count = retry_map.get(policy.name, 0)
            if retry_count >= policy.action.max_retries:
                logger.warning(f"Skipping policy {policy.name} for anomaly {anomaly.pattern.name}: "
                              f"exceeded retry limit ({retry_count}/{policy.action.max_retries})")
                
                # Add a skipped result
                result = RemediationResult(
                    action=policy.action,
                    status=RemediationStatus.SKIPPED,
                    output=f"Exceeded retry limit ({retry_count}/{policy.action.max_retries})",
                    timestamp=datetime.now(),
                    issue=None,
                    anomaly=anomaly,
                    retry_count=retry_count
                )
                self._add_result(result)
                continue
            
            # Execute the action
            success, output = policy.action.execute()
            
            # Update the result
            status = RemediationStatus.SUCCESS if success else RemediationStatus.FAILURE
            result = RemediationResult(
                action=policy.action,
                status=status,
                output=output,
                timestamp=datetime.now(),
                issue=None,
                anomaly=anomaly,
                retry_count=retry_count
            )
            self._add_result(result)
            
            # Update the retry count
            if not success:
                retry_map[policy.name] = retry_count + 1
                
                # Schedule a retry if needed
                if retry_count + 1 < policy.action.max_retries:
                    logger.info(f"Scheduling retry for policy {policy.name} "
                               f"({retry_count + 1}/{policy.action.max_retries})")
                    
                    # Add to the queue after the cooldown period
                    threading.Timer(
                        policy.action.cooldown_seconds,
                        lambda: self.queue.put((anomaly, retry_map))
                    ).start()
    
    def _add_result(self, result: RemediationResult) -> None:
        """
        Add a remediation result.
        
        Args:
            result: Remediation result to add
        """
        with self.lock:
            self.results.append(result)
            
            # Trim the results if needed
            if len(self.results) > self.max_results:
                self.results = self.results[-self.max_results:]
        
        # Call the callbacks
        for callback in self.callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Error in remediation callback: {e}")
    
    def remediate_issue(self, issue: Issue) -> None:
        """
        Remediate an issue.
        
        Args:
            issue: Issue to remediate
        """
        # Add to the queue
        self.queue.put((issue, {}))
    
    def remediate_anomaly(self, anomaly: AnomalyScore) -> None:
        """
        Remediate an anomaly.
        
        Args:
            anomaly: Anomaly to remediate
        """
        # Add to the queue
        self.queue.put((anomaly, {}))
    
    def get_results(self, count: Optional[int] = None, 
                   since: Optional[datetime] = None,
                   status: Optional[RemediationStatus] = None) -> List[RemediationResult]:
        """
        Get remediation results.
        
        Args:
            count: Maximum number of results to return (default: all)
            since: Get results since this timestamp (default: all)
            status: Filter by status (default: all)
            
        Returns:
            List of remediation results
        """
        with self.lock:
            # Filter by timestamp
            if since is not None:
                results = [r for r in self.results if r.timestamp >= since]
            else:
                results = self.results.copy()
            
            # Filter by status
            if status is not None:
                results = [r for r in results if r.status == status]
            
            # Limit by count
            if count is not None:
                results = results[-count:]
            
            return results
    
    def get_success_rate(self, since: Optional[datetime] = None) -> float:
        """
        Get the success rate of remediation actions.
        
        Args:
            since: Get success rate since this timestamp (default: all)
            
        Returns:
            Success rate (0-1)
        """
        results = self.get_results(since=since)
        
        if not results:
            return 0.0
        
        success_count = sum(1 for r in results if r.status == RemediationStatus.SUCCESS)
        return success_count / len(results)
    
    @classmethod
    def create_with_default_policies(cls) -> 'RemediationEngine':
        """
        Create a remediation engine with default policies.
        
        Returns:
            RemediationEngine instance with default policies
        """
        engine = cls()
        
        # Add some default policies for Kubernetes
        default_policies = [
            # Restart a pod
            RemediationPolicy(
                name="restart_pod",
                description="Restart a pod",
                action=RemediationAction(
                    name="restart_pod",
                    description="Restart a pod",
                    command="kubectl rollout restart deployment/${SERVICE} -n ${NAMESPACE}",
                    service=None,
                    timeout_seconds=60,
                    max_retries=3,
                    cooldown_seconds=60
                ),
                issue_types=["single_point_of_failure", "unreachable_service"],
                anomaly_patterns=["crash", "memory_error", "cpu_error"],
                services=None,
                min_severity=3,
                min_score=0.7,
                enabled=True
            ),
            
            # Scale up a deployment
            RemediationPolicy(
                name="scale_up",
                description="Scale up a deployment",
                action=RemediationAction(
                    name="scale_up",
                    description="Scale up a deployment",
                    command="kubectl scale deployment/${SERVICE} -n ${NAMESPACE} --replicas=${REPLICAS}",
                    service=None,
                    timeout_seconds=60,
                    max_retries=3,
                    cooldown_seconds=60
                ),
                issue_types=["high_fanin", "missing_autoscaling"],
                anomaly_patterns=["cpu_error"],
                services=None,
                min_severity=3,
                min_score=0.7,
                enabled=True
            ),
            
            # Restart a database
            RemediationPolicy(
                name="restart_database",
                description="Restart a database",
                action=RemediationAction(
                    name="restart_database",
                    description="Restart a database",
                    command="kubectl rollout restart statefulset/${SERVICE} -n ${NAMESPACE}",
                    service=None,
                    timeout_seconds=120,
                    max_retries=2,
                    cooldown_seconds=120
                ),
                issue_types=["database_single_instance"],
                anomaly_patterns=["database_error"],
                services=None,
                min_severity=4,
                min_score=0.8,
                enabled=True
            ),
        ]
        
        engine.add_policies(default_policies)
        return engine