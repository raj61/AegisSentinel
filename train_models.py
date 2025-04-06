#!/usr/bin/env python3
"""
Train ML Models for Aegis Sentinel
=================================

This script trains the ML models used by Aegis Sentinel.
"""

import os
import sys
import argparse
import logging
import numpy as np
import json
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ML Models for Aegis Sentinel')
    parser.add_argument('--model-path', type=str, default='models',
                        help='Path for saving ML models')
    parser.add_argument('--data-path', type=str, default='simulation/data',
                        help='Path for loading training data')
    return parser.parse_args()

def train_metrics_model(model_path, data_path):
    """Train the metrics anomaly detection model."""
    logger.info("Training metrics anomaly detection model...")
    
    # Always use synthetic data for consistency
    logger.info("Generating synthetic metrics data")
    metrics_data = []
    for i in range(1000):
        metrics_data.append({
            'timestamp': f'2025-04-{i % 30 + 1:02d}T{i % 24:02d}:00:00Z',
            'service': f'service-{i % 5}',
            'metrics': {
                'cpu': np.random.normal(50, 15),
                'memory': np.random.normal(60, 10),
                'latency': np.random.normal(200, 50),
                'error_rate': np.random.exponential(0.01)
            }
        })
    
    # Extract features
    cpu_values = [m['metrics'].get('cpu', 0) for m in metrics_data if 'metrics' in m]
    memory_values = [m['metrics'].get('memory', 0) for m in metrics_data if 'metrics' in m]
    latency_values = [m['metrics'].get('latency', 0) for m in metrics_data if 'metrics' in m]
    error_rate_values = [m['metrics'].get('error_rate', 0) for m in metrics_data if 'metrics' in m]
    
    # Calculate baseline statistics
    cpu_mean = np.mean(cpu_values)
    cpu_std = np.std(cpu_values)
    memory_mean = np.mean(memory_values)
    memory_std = np.std(memory_values)
    latency_mean = np.mean(latency_values)
    latency_std = np.std(latency_values)
    error_rate_mean = np.mean(error_rate_values)
    error_rate_std = np.std(error_rate_values)
    
    # Create model in the format expected by the detector
    model = {
        'cpu_threshold': float(np.percentile(cpu_values, 95)),
        'memory_threshold': float(np.percentile(memory_values, 95)),
        'latency_threshold': float(np.percentile(latency_values, 95)),
        'error_rate_threshold': float(np.percentile(error_rate_values, 95)),
        'z_score_threshold': 3.0,
        'cpu_mean': float(cpu_mean),
        'cpu_std': float(cpu_std),
        'memory_mean': float(memory_mean),
        'memory_std': float(memory_std),
        'latency_mean': float(latency_mean),
        'latency_std': float(latency_std),
        'error_rate_mean': float(error_rate_mean),
        'error_rate_std': float(error_rate_std)
    }
    
    # Save model
    model_file = Path(model_path) / 'anomaly' / 'metrics_model.npy'
    model_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(model_file, np.array([model]))
    
    logger.info(f"Metrics model saved to {model_file}")
    return model

def train_logs_model(model_path, data_path):
    """Train the logs anomaly detection model."""
    logger.info("Training logs anomaly detection model...")
    
    # Always use synthetic data for consistency
    logger.info("Generating synthetic logs data")
    logs_data = []
    log_templates = [
        "Application started",
        "Request received: GET /api/v1/users",
        "Request completed in {time}ms",
        "Database query executed in {time}ms",
        "Cache hit ratio: {ratio}",
        "Warning: Slow query detected",
        "Error: Connection refused",
        "Error: Timeout waiting for response",
        "Critical: Out of memory",
        "Fatal: System crash detected"
    ]
    
    for i in range(1000):
        severity = np.random.choice(['INFO', 'WARNING', 'ERROR', 'CRITICAL'], p=[0.7, 0.2, 0.08, 0.02])
        template = np.random.choice(log_templates)
        
        if '{time}' in template:
            template = template.replace('{time}', str(int(np.random.exponential(100))))
        if '{ratio}' in template:
            template = template.replace('{ratio}', f"{np.random.uniform(0.5, 1.0):.2f}")
            
        logs_data.append({
            'timestamp': f'2025-04-{i % 30 + 1:02d}T{i % 24:02d}:00:00Z',
            'service': f'service-{i % 5}',
            'severity': severity,
            'message': template
        })
    
    # Extract features - word frequency analysis
    all_words = []
    for log in logs_data:
        if 'message' in log:
            words = log['message'].lower().split()
            all_words.extend(words)
    
    # Count word frequencies
    from collections import Counter
    word_counts = Counter(all_words)
    
    # Identify error and warning keywords
    error_keywords = [word for word, count in word_counts.items() 
                     if 'error' in word or 'fail' in word or 'exception' in word or 
                        'critical' in word or 'crash' in word]
    
    warning_keywords = [word for word, count in word_counts.items()
                       if 'warn' in word or 'slow' in word or 'timeout' in word or
                          'delay' in word]
    
    # Add common error and warning terms if not found in data
    common_error_terms = ['error', 'exception', 'fail', 'critical', 'crash', 'fatal']
    common_warning_terms = ['warning', 'warn', 'timeout', 'slow', 'high', 'low']
    
    for term in common_error_terms:
        if term not in error_keywords:
            error_keywords.append(term)
    
    for term in common_warning_terms:
        if term not in warning_keywords:
            warning_keywords.append(term)
    
    # Create model
    model = {
        'config': {
            'error_keywords': error_keywords,
            'warning_keywords': warning_keywords,
            'threshold': 0.7
        }
    }
    
    # Save model
    model_file = Path(model_path) / 'anomaly' / 'logs_model.npy'
    model_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(model_file, np.array([model]))
    
    logger.info(f"Logs model saved to {model_file}")
    return model

def main():
    """Main function."""
    args = parse_args()
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_path, exist_ok=True)
    
    # Train models
    train_metrics_model(args.model_path, args.data_path)
    train_logs_model(args.model_path, args.data_path)
    
    logger.info("All models trained successfully!")

if __name__ == '__main__':
    main()