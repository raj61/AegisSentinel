#!/usr/bin/env python3
"""
Fix Rules Model for Aegis Sentinel
=================================

This script fixes the rules model for the remediation learner.
"""

import os
import sys
import json
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def fix_rules_model():
    """Fix the rules model for the remediation learner."""
    model_path = Path("models/remediation/rules_model.json")
    
    if not model_path.exists():
        print(f"Rules model not found at {model_path}")
        return
    
    # Load the existing model
    with open(model_path, 'r') as f:
        model = json.load(f)
    
    # Add the config key if it doesn't exist
    if 'config' not in model:
        print("Adding config key to rules model")
        model['config'] = {
            'default_severity_threshold': 3,
            'enable_auto_remediation': True,
            'max_concurrent_remediations': 3
        }
    
    # Save the updated model
    with open(model_path, 'w') as f:
        json.dump(model, f, indent=2)
    
    print(f"Rules model updated at {model_path}")

if __name__ == '__main__':
    fix_rules_model()