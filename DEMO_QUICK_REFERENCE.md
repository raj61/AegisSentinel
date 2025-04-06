# Aegis Demo Quick Reference

## Startup Commands

```bash
# Basic demo
python3 run_k8s_demo.py

# With real Kubernetes
python3 run_k8s_demo.py --use-real-k8s --namespace aegis-demo

# With metrics
python3 run_k8s_demo.py --enable-metrics

# Full featured demo
python3 run_k8s_demo.py --use-real-k8s --namespace aegis-demo --enable-metrics --enable-ml
```

## Essential API Commands

> **Note**: Commands are shown with and without `jq` (a JSON formatter). If you don't have jq installed, use the commands without the `| jq` part.

```bash
# Service graph
curl http://localhost:8080/api/graph | jq
# Without jq:
curl http://localhost:8080/api/graph

# Service metrics
curl http://localhost:8080/api/metrics/service/aegis-demo/frontend | jq
# Without jq:
curl http://localhost:8080/api/metrics/service/aegis-demo/frontend

# Inject anomaly
curl -X POST http://localhost:8080/api/inject-anomaly -H "Content-Type: application/json" -d '{"type":"random"}' | jq
# Without jq:
curl -X POST http://localhost:8080/api/inject-anomaly -H "Content-Type: application/json" -d '{"type":"random"}'

# Check issues
curl http://localhost:8080/api/issues | jq
# Without jq:
curl http://localhost:8080/api/issues
```

### Installing jq (Optional)

```bash
# Ubuntu/Debian
sudo apt-get install jq

# macOS
brew install jq

# Windows (with chocolatey)
choco install jq
```

## Demo Flow

1. **Introduction (1-2 min)**
   - "Aegis provides real-time visibility into Kubernetes environments"
   - "Detects anomalies and helps resolve issues before they impact users"

2. **Service Graph (2-3 min)**
   - Show the graph visualization
   - Point out different service types and relationships
   - Demonstrate navigation (zoom, pan, click)

3. **Metrics Monitoring (2-3 min)**
   - Click on a service to show metrics
   - Explain the different metrics (CPU, memory, latency, errors)
   - Show historical data charts

4. **Anomaly Detection (3-4 min)**
   - Inject an anomaly using the UI button or API
   - Point out the visual changes (service turns red)
   - Show the metrics spike
   - Explain how Aegis detected the issue

5. **Automatic Resolution (2-3 min)**
   - Wait for the system to resolve the issue
   - Point out the service returning to normal
   - Explain the resolution process

6. **Conclusion (1-2 min)**
   - Summarize key benefits
   - Invite questions

## Additional Commands

### Infrastructure as Code Analysis

```bash
# Analyze Terraform code (basic)
python3 examples/analyze_terraform.py --source examples/terraform/aws-microservices --detect --verbose

# With resolution suggestions
python3 examples/analyze_terraform.py --source examples/terraform/aws-microservices --detect --resolve --verbose
```

## Key Talking Points

- **Visibility**: "Complete view of your microservices architecture"
- **Proactive**: "Detect issues before they impact users"
- **Efficiency**: "Reduce mean time to resolution"
- **Intelligence**: "ML-enhanced analysis for subtle patterns"
- **Prevention**: "Analyze infrastructure as code to prevent issues"

## UI Fallback Plan

If the UI isn't working properly:

1. Use terminal commands to show the same information
2. Focus on the API responses and explain what they mean
3. Use diagrams or screenshots as visual aids

## Common Questions & Answers

**Q: How does Aegis detect anomalies?**
A: "Aegis uses a combination of rule-based detection and machine learning models that analyze metrics and logs to identify patterns that deviate from normal behavior."

**Q: Can Aegis integrate with existing monitoring tools?**
A: "Yes, Aegis can integrate with Prometheus for metrics collection and can complement existing monitoring solutions by providing deeper insights and automated resolution."

**Q: How much overhead does Aegis add to my cluster?**
A: "Aegis is designed to be lightweight, with minimal resource requirements. The core components typically use less than 100MB of memory and negligible CPU when idle."

**Q: Can Aegis work in multi-cluster environments?**
A: "Yes, Aegis can monitor multiple Kubernetes clusters simultaneously, providing a unified view across your entire infrastructure."

**Q: Is Aegis open source?**
A: "The core components of Aegis are open source, with enterprise features available for larger deployments."
