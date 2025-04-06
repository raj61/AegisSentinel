"""
Machine Learning Module
=====================

This module provides machine learning capabilities for Aegis Sentinel.
"""

from src.ml.anomaly_detection import (
    AnomalyDetector,
    TimeSeriesAnomalyDetector,
    LogAnomalyDetector,
    MLBasedAnomalyDetector,
    AnomalyDetectionEngine,
    create_default_engine as create_anomaly_engine
)

from src.ml.learning.remediation_learner import (
    RemediationAction,
    RemediationState,
    RemediationExperience,
    RemediationLearner,
    RuleLearner,
    ReinforcementLearner,
    RemediationLearningEngine,
    create_default_engine as create_remediation_engine
)

from src.ml.integration import (
    MLIntegrationEngine,
    create_ml_integration
)

__all__ = [
    # Anomaly detection
    'AnomalyDetector',
    'TimeSeriesAnomalyDetector',
    'LogAnomalyDetector',
    'MLBasedAnomalyDetector',
    'AnomalyDetectionEngine',
    'create_anomaly_engine',
    
    # Remediation learning
    'RemediationAction',
    'RemediationState',
    'RemediationExperience',
    'RemediationLearner',
    'RuleLearner',
    'ReinforcementLearner',
    'RemediationLearningEngine',
    'create_remediation_engine',
    
    # Integration
    'MLIntegrationEngine',
    'create_ml_integration'
]