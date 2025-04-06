#!/usr/bin/env python3
"""
Initialize ML models for Aegis Sentinel demo
"""

import os
import numpy as np
import json
import pickle
from pathlib import Path

def create_directory(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

def create_metrics_model():
    """Create a simple metrics anomaly detection model"""
    print("Creating metrics model...")
    model_path = "models/anomaly/metrics_model.npy"
    
    # Create a simple threshold-based model
    model = {
        'cpu_threshold': 80.0,
        'memory_threshold': 90.0,
        'latency_threshold': 500.0,
        'error_rate_threshold': 0.05,
        'z_score_threshold': 3.0
    }
    
    # Save as numpy array
    np.save(model_path, np.array([model]))
    print(f"Metrics model saved to {model_path}")

def create_logs_model():
    """Create a simple log anomaly detection model"""
    print("Creating logs model...")
    model_path = "models/anomaly/logs_model.npy"
    
    # Create a simple keyword-based model
    model = {
        'error_keywords': ['error', 'exception', 'fail', 'critical', 'crash'],
        'warning_keywords': ['warning', 'warn', 'timeout', 'slow'],
        'threshold': 0.7
    }
    
    # Save as numpy array
    np.save(model_path, np.array([model]))
    print(f"Logs model saved to {model_path}")

def create_ml_model():
    """Create a simple ML-based anomaly detection model"""
    print("Creating ML model...")
    model_path = "models/anomaly/ml_model.npy"
    
    # Create a simple isolation forest-like model
    model = {
        'contamination': 0.01,
        'n_estimators': 100,
        'max_samples': 'auto',
        'threshold': 0.5
    }
    
    # Save as numpy array
    np.save(model_path, np.array([model]))
    print(f"ML model saved to {model_path}")

def create_rules_model():
    """Create a simple rules-based remediation model"""
    print("Creating rules model...")
    model_path = "models/remediation/rules_model.json"
    
    # Create a simple rules model
    model = {
        'rules': [
            {
                'condition': {'type': 'cpu_spike', 'severity': {'min': 3}},
                'action': 'Scale Service',
                'parameters': {'replicas': '+1'}
            },
            {
                'condition': {'type': 'memory_leak', 'severity': {'min': 4}},
                'action': 'Restart Service',
                'parameters': {}
            },
            {
                'condition': {'type': 'error_spike', 'severity': {'min': 3}},
                'action': 'Rollback Deployment',
                'parameters': {'to_version': 'previous'}
            },
            {
                'condition': {'type': 'disk_full', 'severity': {'min': 4}},
                'action': 'Clear Cache',
                'parameters': {}
            }
        ]
    }
    
    # Save as JSON
    with open(model_path, 'w') as f:
        json.dump(model, f, indent=2)
    print(f"Rules model saved to {model_path}")

def create_rl_model():
    """Create a simple RL-based remediation model"""
    print("Creating RL model...")
    model_path = "models/remediation/rl_model.json"
    
    # Create a simple Q-learning model
    model = {
        'q_table': {
            'cpu_spike_critical': {
                'Scale Service': 0.8,
                'Restart Service': 0.4,
                'Rollback Deployment': 0.2,
                'Drain Node': 0.1,
                'Clear Cache': 0.0
            },
            'memory_leak_critical': {
                'Scale Service': 0.3,
                'Restart Service': 0.9,
                'Rollback Deployment': 0.4,
                'Drain Node': 0.2,
                'Clear Cache': 0.5
            },
            'error_spike_critical': {
                'Scale Service': 0.2,
                'Restart Service': 0.5,
                'Rollback Deployment': 0.9,
                'Drain Node': 0.1,
                'Clear Cache': 0.0
            }
        },
        'learning_rate': 0.1,
        'discount_factor': 0.9,
        'exploration_rate': 0.1
    }
    
    # Save as JSON
    with open(model_path, 'w') as f:
        json.dump(model, f, indent=2)
    print(f"RL model saved to {model_path}")

def create_action_library():
    """Create an action library"""
    print("Creating action library...")
    model_path = "models/remediation/action_library.json"
    
    # Create a simple action library
    model = {
        'restart_service': {
            'action_id': 'restart_service',
            'name': 'Restart Service',
            'description': 'Restart a service to clear its state',
            'target_type': 'service',
            'parameters': {},
            'preconditions': ['service_exists'],
            'estimated_duration': 30,
            'risk_level': 2
        },
        'scale_service': {
            'action_id': 'scale_service',
            'name': 'Scale Service',
            'description': 'Scale a service to handle more load',
            'target_type': 'service',
            'parameters': {'replicas': 'int'},
            'preconditions': ['service_exists', 'can_scale'],
            'estimated_duration': 60,
            'risk_level': 1
        },
        'rollback_deployment': {
            'action_id': 'rollback_deployment',
            'name': 'Rollback Deployment',
            'description': 'Rollback to a previous version',
            'target_type': 'deployment',
            'parameters': {'to_version': 'string'},
            'preconditions': ['service_exists', 'has_previous_version'],
            'estimated_duration': 120,
            'risk_level': 3
        },
        'drain_node': {
            'action_id': 'drain_node',
            'name': 'Drain Node',
            'description': 'Drain a node to move workloads',
            'target_type': 'node',
            'parameters': {'node': 'string'},
            'preconditions': ['node_exists', 'can_drain'],
            'estimated_duration': 300,
            'risk_level': 3
        },
        'clear_cache': {
            'action_id': 'clear_cache',
            'name': 'Clear Cache',
            'description': 'Clear a service cache',
            'target_type': 'service',
            'parameters': {},
            'preconditions': ['service_exists', 'has_cache'],
            'estimated_duration': 15,
            'risk_level': 1
        }
    }
    
    # Save as JSON
    with open(model_path, 'w') as f:
        json.dump(model, f, indent=2)
    print(f"Action library saved to {model_path}")

def create_experience_history():
    """Create an experience history"""
    print("Creating experience history...")
    model_path = "models/remediation/experience_history.json"
    
    # Create a simple experience history with the correct format
    experiences = [
        {
            'initial_state': {
                'metrics': {'cpu': 95.0, 'memory': 80.0, 'latency': 500},
                'issue_type': 'cpu_spike',
                'issue_severity': 4,
                'affected_services': ['frontend', 'backend'],
                'service_states': {'frontend': 'degraded', 'backend': 'critical'},
                'previous_actions': []
            },
            'action': {
                'action_id': 'scale_service',
                'name': 'Scale Service',
                'description': 'Scale a service to handle more load',
                'target_type': 'service',
                'parameters': {'replicas': '+1'},
                'preconditions': ['service_exists', 'can_scale'],
                'estimated_duration': 60,
                'risk_level': 1
            },
            'next_state': {
                'metrics': {'cpu': 70.0, 'memory': 75.0, 'latency': 200},
                'issue_type': 'cpu_spike',
                'issue_severity': 2,
                'affected_services': ['frontend', 'backend'],
                'service_states': {'frontend': 'healthy', 'backend': 'healthy'},
                'previous_actions': ['scale_service']
            },
            'reward': 1.0,
            'timestamp': '2025-04-01T12:00:00+00:00',
            'success': True,
            'notes': 'Successfully scaled service to handle CPU spike'
        },
        {
            'initial_state': {
                'metrics': {'cpu': 60.0, 'memory': 95.0, 'latency': 300},
                'issue_type': 'memory_leak',
                'issue_severity': 4,
                'affected_services': ['auth'],
                'service_states': {'auth': 'critical'},
                'previous_actions': []
            },
            'action': {
                'action_id': 'restart_service',
                'name': 'Restart Service',
                'description': 'Restart a service to clear its state',
                'target_type': 'service',
                'parameters': {},
                'preconditions': ['service_exists'],
                'estimated_duration': 30,
                'risk_level': 2
            },
            'next_state': {
                'metrics': {'cpu': 55.0, 'memory': 40.0, 'latency': 150},
                'issue_type': 'memory_leak',
                'issue_severity': 0,
                'affected_services': [],
                'service_states': {'auth': 'healthy'},
                'previous_actions': ['restart_service']
            },
            'reward': 1.0,
            'timestamp': '2025-04-02T14:30:00+00:00',
            'success': True,
            'notes': 'Successfully restarted service to clear memory leak'
        },
        {
            'initial_state': {
                'metrics': {'cpu': 70.0, 'memory': 60.0, 'latency': 800, 'error_rate': 0.15},
                'issue_type': 'error_spike',
                'issue_severity': 5,
                'affected_services': ['api', 'database'],
                'service_states': {'api': 'critical', 'database': 'degraded'},
                'previous_actions': []
            },
            'action': {
                'action_id': 'rollback_deployment',
                'name': 'Rollback Deployment',
                'description': 'Rollback to a previous version',
                'target_type': 'deployment',
                'parameters': {'to_version': 'v1.2.3'},
                'preconditions': ['service_exists', 'has_previous_version'],
                'estimated_duration': 120,
                'risk_level': 3
            },
            'next_state': {
                'metrics': {'cpu': 50.0, 'memory': 55.0, 'latency': 200, 'error_rate': 0.01},
                'issue_type': 'error_spike',
                'issue_severity': 1,
                'affected_services': [],
                'service_states': {'api': 'healthy', 'database': 'healthy'},
                'previous_actions': ['rollback_deployment']
            },
            'reward': 1.0,
            'timestamp': '2025-04-03T09:15:00+00:00',
            'success': True,
            'notes': 'Successfully rolled back deployment to resolve error spike'
        }
    ]
    
    # Save as JSON
    with open(model_path, 'w') as f:
        json.dump(experiences, f, indent=2)
    print(f"Experience history saved to {model_path}")

def main():
    """Main function"""
    # Create directories
    create_directory("models/anomaly")
    create_directory("models/remediation")
    
    # Create anomaly detection models
    create_metrics_model()
    create_logs_model()
    create_ml_model()
    
    # Create remediation models
    create_rules_model()
    create_rl_model()
    create_action_library()
    create_experience_history()
    
    print("All models initialized successfully!")

if __name__ == "__main__":
    main()