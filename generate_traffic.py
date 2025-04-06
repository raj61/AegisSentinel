#!/usr/bin/env python3
"""
Generate Traffic to Kubernetes Services
======================================

This script generates traffic to the services in the Kubernetes cluster
to demonstrate real-time monitoring with Aegis Sentinel.
"""

import os
import sys
import time
import random
import logging
import argparse
import threading
import requests
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('traffic_generator.log')
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate traffic to Kubernetes services')
    parser.add_argument('--namespace', type=str, default='aegis-demo',
                        help='Kubernetes namespace where services are deployed')
    parser.add_argument('--duration', type=int, default=300,
                        help='Duration to generate traffic (in seconds)')
    parser.add_argument('--rate', type=float, default=5.0,
                        help='Requests per second per service')
    parser.add_argument('--error-rate', type=float, default=0.05,
                        help='Percentage of requests that should generate errors (0.0-1.0)')
    parser.add_argument('--port-forward', action='store_true',
                        help='Set up port forwarding to access services')
    return parser.parse_args()

def get_services(namespace):
    """Get services in the specified namespace."""
    try:
        import kubernetes.client
        from kubernetes.client import Configuration
        from kubernetes.client.api import core_v1_api
        from kubernetes.config import load_kube_config, load_incluster_config
        
        # Load Kubernetes configuration
        try:
            load_kube_config()
        except:
            try:
                load_incluster_config()
            except:
                logger.error("Could not load Kubernetes configuration")
                return []
        
        # Create API client
        api = core_v1_api.CoreV1Api()
        
        # Get services
        services = api.list_namespaced_service(namespace)
        
        return [
            {
                'name': svc.metadata.name,
                'namespace': svc.metadata.namespace,
                'cluster_ip': svc.spec.cluster_ip,
                'ports': [{'port': port.port, 'target_port': port.target_port} for port in svc.spec.ports]
            }
            for svc in services.items
        ]
    except ImportError:
        logger.error("Kubernetes client library not installed. Please install it with 'pip install kubernetes'")
        return []
    except Exception as e:
        logger.error(f"Error getting services: {e}")
        return []

def setup_port_forwarding(services):
    """Set up port forwarding for services."""
    import subprocess
    
    port_forwards = []
    service_urls = {}
    
    # Start from port 8081 to avoid conflicts with the Aegis web server
    base_port = 8081
    
    for service in services:
        if not service['ports']:
            continue
            
        service_port = service['ports'][0]['port']
        local_port = base_port
        base_port += 1
        
        logger.info(f"Setting up port forwarding for {service['name']} from local port {local_port} to service port {service_port}")
        
        # Start port forwarding
        cmd = [
            'kubectl', 'port-forward', 
            f"service/{service['name']}", 
            f"{local_port}:{service_port}",
            '-n', service['namespace']
        ]
        
        try:
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            
            port_forwards.append(process)
            service_urls[service['name']] = f"http://localhost:{local_port}"
            
            # Give it a moment to establish
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error setting up port forwarding for {service['name']}: {e}")
    
    return port_forwards, service_urls

def generate_traffic(service, service_url, duration, rate, error_rate):
    """Generate traffic to a service."""
    logger.info(f"Generating traffic to {service['name']} at {service_url}")
    
    start_time = time.time()
    request_count = 0
    error_count = 0
    
    # Calculate sleep time between requests
    sleep_time = 1.0 / rate if rate > 0 else 1.0
    
    # Track connection status
    connection_working = False
    connection_attempts = 0
    
    while time.time() - start_time < duration:
        try:
            # Determine if this request should generate an error
            should_error = random.random() < error_rate
            
            # Add a query parameter to potentially cause an error
            url = f"{service_url}{'?error=true' if should_error else ''}"
            
            # Only try to connect if we haven't determined it's not working
            # or every 10 requests to check if it's back up
            if connection_working or connection_attempts % 10 == 0:
                # Send the request with a short timeout
                response = requests.get(url, timeout=1)
                
                # If we get here, connection is working
                connection_working = True
                
                # Log the result
                if response.status_code >= 400:
                    logger.info(f"Request to {service['name']} returned error: {response.status_code}")
                    error_count += 1
                else:
                    logger.debug(f"Request to {service['name']} successful")
            else:
                # Simulate a request without actually making one
                if should_error:
                    error_count += 1
                    logger.debug(f"Simulated error request to {service['name']}")
                else:
                    logger.debug(f"Simulated successful request to {service['name']}")
            
            request_count += 1
            
        except requests.RequestException as e:
            # Connection not working, switch to simulation mode
            connection_working = False
            connection_attempts += 1
            
            if connection_attempts <= 3 or connection_attempts % 10 == 0:
                logger.info(f"Error connecting to {service['name']}: {e}")
                logger.info(f"Switching to traffic simulation mode for {service['name']}")
            
            # Still count this as a request and error
            error_count += 1
            request_count += 1
        
        # Sleep between requests
        time.sleep(sleep_time)
    
    logger.info(f"Traffic generation complete for {service['name']}")
    logger.info(f"Sent/simulated {request_count} requests with {error_count} errors")
    
    return request_count, error_count

def main():
    """Main function."""
    args = parse_args()
    
    # Get services
    logger.info(f"Getting services in namespace {args.namespace}")
    services = get_services(args.namespace)
    
    if not services:
        logger.error("No services found")
        sys.exit(1)
    
    logger.info(f"Found {len(services)} services: {', '.join(s['name'] for s in services)}")
    
    # Set up port forwarding if requested
    port_forwards = []
    service_urls = {}
    
    if args.port_forward:
        port_forwards, service_urls = setup_port_forwarding(services)
    else:
        # Use cluster IPs (this will only work if running inside the cluster)
        for service in services:
            if service['ports']:
                service_port = service['ports'][0]['port']
                service_urls[service['name']] = f"http://{service['cluster_ip']}:{service_port}"
    
    if not service_urls:
        logger.error("No service URLs available")
        sys.exit(1)
    
    # Start traffic generation threads
    threads = []
    
    for service in services:
        if service['name'] in service_urls:
            thread = threading.Thread(
                target=generate_traffic,
                args=(service, service_urls[service['name']], args.duration, args.rate, args.error_rate)
            )
            thread.daemon = True
            thread.start()
            threads.append(thread)
    
    # Wait for traffic generation to complete
    try:
        logger.info(f"Generating traffic for {args.duration} seconds")
        logger.info("Press Ctrl+C to stop early")
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        logger.info("Traffic generation complete")
    except KeyboardInterrupt:
        logger.info("Traffic generation interrupted by user")
    finally:
        # Clean up port forwarding processes
        for process in port_forwards:
            process.terminate()
    
    logger.info("Done")

if __name__ == '__main__':
    main()