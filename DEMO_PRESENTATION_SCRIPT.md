# Aegis Demo Presentation Script

## Introduction

Hello everyone! Today I'll be demonstrating Aegis, our intelligent Kubernetes monitoring and anomaly detection system. Aegis provides real-time visibility into your microservices architecture, detects anomalies, and helps resolve issues before they impact your users.

## Setup and Preparation

Before we begin the demo, let's make sure everything is set up correctly:

```bash
# 1. Make sure you're in the project directory
cd /path/to/CredHackathon

# 2. Check if Kubernetes cluster is running (if using real K8s)
kubectl get nodes

# 3. Start the demo with metrics enabled
python3 run_k8s_demo.py --use-real-k8s --namespace aegis-demo --enable-metrics
```

## Demo Flow

### 1. Service Graph Visualization

**UI Approach:**
- Point out the service graph visualization in the browser
- Explain: "This graph shows all services in our Kubernetes cluster and how they're connected"
- Demonstrate zooming and panning: "We can easily navigate through complex architectures"
- Click on a node: "Clicking on any service shows its details and metrics"

**Terminal Fallback:**
```bash
# Show service relationships via API
# If you have jq installed (for formatted JSON output):
curl http://localhost:8080/api/graph | jq
# If you don't have jq:
curl http://localhost:8080/api/graph
echo "As you can see, we have services like frontend, backend, database, etc., with clear relationships between them"
```

### 2. Service Metrics Monitoring

**UI Approach:**
- Click on a service node: "Let's look at the metrics for our frontend service"
- Point out the metrics panel: "We can see CPU, memory, latency, and error rates"
- Highlight the charts: "These charts show historical data, helping us identify trends"

**Terminal Fallback:**
```bash
# Show metrics for a specific service
# With jq (if installed):
curl http://localhost:8080/api/metrics/service/aegis-demo/frontend | jq
# Without jq:
curl http://localhost:8080/api/metrics/service/aegis-demo/frontend
echo "These metrics show the current state of our frontend service"

# Show historical metrics
# With jq (if installed):
curl http://localhost:8080/api/metrics/historical/aegis-demo/frontend/cpu_usage | jq
# Without jq:
curl http://localhost:8080/api/metrics/historical/aegis-demo/frontend/cpu_usage
echo "We can track how metrics change over time to identify trends"
```

### 3. Anomaly Detection

**UI Approach:**
- Point to the healthy services: "All services are currently healthy, as indicated by the green status"
- Click the "Inject Anomaly" button: "Let's simulate an issue to demonstrate detection capabilities"
- Wait for the UI to update: "Notice how the affected service turns red, indicating a critical issue"
- Point out the metrics spike: "We can see corresponding spikes in CPU usage and error rates"

**Terminal Fallback:**
```bash
# Inject an anomaly
# With jq (if installed):
curl -X POST http://localhost:8080/api/inject-anomaly -H "Content-Type: application/json" -d '{"type":"random"}' | jq
# Without jq:
curl -X POST http://localhost:8080/api/inject-anomaly -H "Content-Type: application/json" -d '{"type":"random"}'
echo "We've just injected an anomaly into the system"

# Check the status after injection
# With jq (if installed):
curl http://localhost:8080/api/issues | jq
# Without jq:
curl http://localhost:8080/api/issues
echo "As you can see, Aegis has detected the issue and identified the affected services"
```

### 4. Root Cause Analysis

**UI Approach:**
- Point to the affected service: "Aegis not only detects the anomaly but identifies the root cause"
- Show the connected services: "It also shows which dependent services might be affected"
- Explain: "This helps teams quickly understand the blast radius of an incident"

**Terminal Fallback:**
```bash
# Get detailed issue information
# With jq (if installed):
curl http://localhost:8080/api/issues | jq '.issues[0]'
# Without jq (will show all issues, you'll need to look at the first one):
curl http://localhost:8080/api/issues
echo "Aegis provides detailed information about the issue, including affected nodes and potential root causes"
```

### 5. Automatic Resolution

**UI Approach:**
- Wait for 60 seconds: "Aegis can automatically resolve certain issues"
- Point out the service returning to green: "Notice how the service has returned to a healthy state"
- Show the metrics returning to normal: "The metrics have also normalized"

**Terminal Fallback:**
```bash
# Wait for resolution
echo "Waiting for automatic resolution..."
sleep 60

# Check status after resolution
# With jq (if installed):
curl http://localhost:8080/api/issues | jq
# Without jq:
curl http://localhost:8080/api/issues
echo "The issue has been automatically resolved by the system"

# Verify metrics have normalized
# With jq (if installed):
curl http://localhost:8080/api/metrics/service/aegis-demo/frontend | jq
# Without jq:
curl http://localhost:8080/api/metrics/service/aegis-demo/frontend
echo "Metrics have returned to normal levels"
```

### 6. Advanced Features (Optional)

**ML-Based Detection:**
```bash
# Restart with ML enabled
python3 run_k8s_demo.py --use-real-k8s --namespace aegis-demo --enable-metrics --enable-ml

# Show ML-enhanced detection
curl http://localhost:8080/api/issues | jq
echo "With ML enabled, Aegis can detect more subtle anomalies and predict potential issues before they occur"
```

**Infrastructure as Code Analysis:**
```bash
# Analyze Terraform code with required arguments
python3 examples/analyze_terraform.py --source examples/terraform/aws-microservices --detect --verbose

# For more detailed analysis with resolution suggestions
python3 examples/analyze_terraform.py --source examples/terraform/aws-microservices --detect --resolve --verbose

# Show results
echo "Aegis can also analyze your infrastructure as code to identify potential issues before deployment"
```

## Conclusion

Aegis provides comprehensive monitoring, detection, and resolution capabilities for Kubernetes environments:

1. **Real-time visibility** into your service architecture
2. **Proactive anomaly detection** to catch issues early
3. **Automatic resolution** to reduce mean time to recovery
4. **ML-enhanced analysis** for predictive capabilities

This helps teams:
- Reduce downtime
- Improve reliability
- Decrease mean time to resolution
- Focus on building features instead of firefighting

## Q&A

Thank you for your attention! I'm happy to answer any questions you might have about Aegis.

## Additional Demo Tips

- If the graph visualization isn't working properly, focus on the metrics and terminal outputs
- Have a pre-recorded video backup of the UI in case of technical difficulties
- Practice the flow several times to ensure smooth transitions
- Be prepared to explain technical concepts in simple terms for non-technical audience members