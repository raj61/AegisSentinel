// MetricsPanel Component for Aegis Sentinel

const MetricsPanel = ({ darkMode, loading: globalLoading, selectedNode }) => {
    const chartsRef = React.useRef({});
    const [localLoading, setLocalLoading] = React.useState(false);
    const [metricsData, setMetricsData] = React.useState(null);
    const [historicalData, setHistoricalData] = React.useState({
        cpu: [],
        memory: [],
        latency: [],
        error_rate: []
    });
    
    // Initialize charts when selected node changes
    React.useEffect(() => {
        if (selectedNode) {
            fetchMetricsData();
        }
    }, [selectedNode]);
    
    // Fetch metrics data for the selected node
    const fetchMetricsData = async () => {
        if (!selectedNode) return;
        
        setLocalLoading(true);
        
        try {
            // Fetch current metrics
            // Add namespace to service ID if not already included
            const serviceId = selectedNode.id.includes('/') ?
                selectedNode.id :
                `aegis-demo/${selectedNode.id}`;
            
            const response = await fetch(`/api/metrics/service/${serviceId}`);
            if (!response.ok) {
                throw new Error(`Failed to fetch metrics: ${response.statusText}`);
            }
            
            const data = await response.json();
            setMetricsData(data.metrics || {});
            
            // Fetch historical metrics for CPU
            if (data.metrics && data.metrics.cpu_usage !== undefined) {
                fetchHistoricalMetrics('cpu_usage');
            }
            
            // Fetch historical metrics for memory
            if (data.metrics && data.metrics.memory_usage !== undefined) {
                fetchHistoricalMetrics('memory_usage');
            }
            
            // Fetch historical metrics for latency
            if (data.metrics && data.metrics.latency_p95 !== undefined) {
                fetchHistoricalMetrics('latency_p95');
            }
            
            // Fetch historical metrics for error rate
            if (data.metrics && data.metrics.error_rate !== undefined) {
                fetchHistoricalMetrics('error_rate');
            }
            
            // Initialize charts after a short delay
            setTimeout(() => {
                initMetricsCharts();
                setLocalLoading(false);
            }, 100);
        } catch (error) {
            console.error('Error fetching metrics:', error);
            setLocalLoading(false);
            
            // Use fallback random data for demo purposes
            generateFallbackData();
        }
    };
    
    // Fetch historical metrics for a specific metric
    const fetchHistoricalMetrics = async (metric) => {
        try {
            // Add namespace to service ID if not already included
            const serviceId = selectedNode.id.includes('/') ?
                selectedNode.id :
                `aegis-demo/${selectedNode.id}`;
                
            const response = await fetch(`/api/metrics/historical/${serviceId}/${metric}?step=5m`);
            if (!response.ok) {
                throw new Error(`Failed to fetch historical metrics: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            // Map the metrics to the appropriate chart data
            let chartKey;
            switch (metric) {
                case 'cpu_usage':
                    chartKey = 'cpu';
                    break;
                case 'memory_usage':
                    chartKey = 'memory';
                    break;
                case 'latency_p95':
                    chartKey = 'latency';
                    break;
                case 'error_rate':
                    chartKey = 'error_rate';
                    break;
                default:
                    return;
            }
            
            setHistoricalData(prev => ({
                ...prev,
                [chartKey]: data.data_points || []
            }));
        } catch (error) {
            console.error(`Error fetching historical metrics for ${metric}:`, error);
        }
    };
    
    // Generate fallback data for demo purposes
    const generateFallbackData = () => {
        const generateData = (count, base, variance) => {
            return Array.from({length: count}, () => base + (Math.random() - 0.5) * variance);
        };
        
        const metrics = {
            cpu_usage: Math.random() * 100,
            memory_usage: Math.random() * 100,
            latency_p95: Math.random() * 500,
            error_rate: Math.random() * 0.1
        };
        
        setMetricsData(metrics);
        
        // Generate historical data
        const timestamps = Array.from({length: 10}, (_, i) => Date.now() - (9 - i) * 60000);
        
        setHistoricalData({
            cpu: timestamps.map((timestamp, i) => ({
                timestamp,
                value: metrics.cpu_usage + (Math.random() - 0.5) * 20
            })),
            memory: timestamps.map((timestamp, i) => ({
                timestamp,
                value: metrics.memory_usage + (Math.random() - 0.5) * 15
            })),
            latency: timestamps.map((timestamp, i) => ({
                timestamp,
                value: metrics.latency_p95 + (Math.random() - 0.5) * 100
            })),
            error_rate: timestamps.map((timestamp, i) => ({
                timestamp,
                value: metrics.error_rate + (Math.random() - 0.5) * 0.05
            }))
        });
        
        // Initialize charts after a short delay
        setTimeout(() => {
            initMetricsCharts();
            setLocalLoading(false);
        }, 100);
    };
    
    // Initialize metrics charts
    const initMetricsCharts = () => {
        // Common chart options
        const commonOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: darkMode ? '#1f2937' : 'white',
                    titleColor: darkMode ? 'white' : '#1f2937',
                    bodyColor: darkMode ? '#cbd5e1' : '#4b5563',
                    borderColor: darkMode ? '#374151' : '#e5e7eb',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    display: true,
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: darkMode ? '#cbd5e1' : '#64748b',
                        maxRotation: 0,
                        autoSkip: true,
                        maxTicksLimit: 5
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: darkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'
                    },
                    ticks: {
                        color: darkMode ? '#cbd5e1' : '#64748b'
                    }
                }
            }
        };
        
        // Destroy existing charts
        Object.values(chartsRef.current).forEach(chart => {
            if (chart) chart.destroy();
        });
        chartsRef.current = {};
        
        // Format timestamps for labels
        const formatTimestamp = (timestamp) => {
            const date = new Date(timestamp * 1000);
            return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        };
        
        // CPU chart
        const cpuCtx = document.getElementById('cpuChart');
        if (cpuCtx) {
            const labels = historicalData.cpu.map(point => formatTimestamp(point.timestamp));
            const data = historicalData.cpu.map(point => point.value);
            
            chartsRef.current.cpu = new Chart(cpuCtx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'CPU Usage (%)',
                        data: data,
                        borderColor: '#3B82F6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: commonOptions
            });
        }
        
        // Memory chart
        const memoryCtx = document.getElementById('memoryChart');
        if (memoryCtx) {
            const labels = historicalData.memory.map(point => formatTimestamp(point.timestamp));
            const data = historicalData.memory.map(point => point.value);
            
            chartsRef.current.memory = new Chart(memoryCtx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Memory Usage (MB)',
                        data: data,
                        borderColor: '#10B981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: commonOptions
            });
        }
        
        // Latency chart
        const latencyCtx = document.getElementById('latencyChart');
        if (latencyCtx) {
            const labels = historicalData.latency.map(point => formatTimestamp(point.timestamp));
            const data = historicalData.latency.map(point => point.value);
            
            chartsRef.current.latency = new Chart(latencyCtx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Latency (ms)',
                        data: data,
                        borderColor: '#F59E0B',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: commonOptions
            });
        }
        
        // Error rate chart
        const errorCtx = document.getElementById('errorChart');
        if (errorCtx) {
            const labels = historicalData.error_rate.map(point => formatTimestamp(point.timestamp));
            const data = historicalData.error_rate.map(point => point.value);
            
            chartsRef.current.error = new Chart(errorCtx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Error Rate (%)',
                        data: data.map(val => val * 100), // Convert to percentage
                        borderColor: '#EF4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: commonOptions
            });
        }
    };
    
    // Get health class for badge
    const getHealthClass = (status) => {
        switch (status) {
            case 'healthy':
                return 'bg-green-100 text-green-800';
            case 'warning':
                return 'bg-yellow-100 text-yellow-800';
            case 'critical':
                return 'bg-red-100 text-red-800';
            default:
                return 'bg-gray-100 text-gray-800';
        }
    };
    
    // Format metric value for display
    const formatMetricValue = (metric, value) => {
        if (value === undefined || value === null) return 'N/A';
        
        switch (metric) {
            case 'cpu_usage':
                return `${value.toFixed(2)}%`;
            case 'memory_usage':
                return `${value.toFixed(2)} MB`;
            case 'latency_p95':
                return `${value.toFixed(2)} ms`;
            case 'error_rate':
                return `${(value * 100).toFixed(2)}%`;
            default:
                return value.toString();
        }
    };
    
    return (
        <div className="mt-6 bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden">
            <div className="p-4 bg-gray-50 dark:bg-gray-700 border-b dark:border-gray-600">
                <h2 className="text-lg font-semibold text-gray-800 dark:text-white">
                    {selectedNode ? `${selectedNode.name || selectedNode.id} Metrics` : 'Service Metrics'}
                </h2>
            </div>
            <div className="p-4">
                {globalLoading || localLoading ? (
                    <div className="flex justify-center py-8">
                        <div className="flex flex-col items-center">
                            <svg className="animate-spin h-8 w-8 text-blue-600 dark:text-blue-400 mb-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            <p className="text-gray-600 dark:text-gray-300">Loading metrics...</p>
                        </div>
                    </div>
                ) : !selectedNode ? (
                    <div className="text-center py-8">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 mx-auto text-gray-400 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                        </svg>
                        <p className="text-gray-500 dark:text-gray-400">Select a service to view detailed metrics</p>
                    </div>
                ) : (
                    <>
                        <div className="mb-4 flex flex-wrap items-center gap-2">
                            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200">
                                {selectedNode.kind || 'Service'}
                            </span>
                            {selectedNode.category && (
                                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
                                    {selectedNode.category}
                                </span>
                            )}
                            {selectedNode.health_status && (
                                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getHealthClass(selectedNode.health_status)}`}>
                                    {selectedNode.health_status}
                                </span>
                            )}
                        </div>
                        
                        {metricsData && (
                            <div className="grid grid-cols-2 gap-4 mb-6">
                                <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                                    <div className="text-sm text-gray-500 dark:text-gray-400">CPU Usage</div>
                                    <div className="text-xl font-semibold mt-1">{formatMetricValue('cpu_usage', metricsData.cpu_usage)}</div>
                                </div>
                                <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                                    <div className="text-sm text-gray-500 dark:text-gray-400">Memory Usage</div>
                                    <div className="text-xl font-semibold mt-1">{formatMetricValue('memory_usage', metricsData.memory_usage)}</div>
                                </div>
                                <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                                    <div className="text-sm text-gray-500 dark:text-gray-400">Response Time (p95)</div>
                                    <div className="text-xl font-semibold mt-1">{formatMetricValue('latency_p95', metricsData.latency_p95)}</div>
                                </div>
                                <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                                    <div className="text-sm text-gray-500 dark:text-gray-400">Error Rate</div>
                                    <div className="text-xl font-semibold mt-1">{formatMetricValue('error_rate', metricsData.error_rate)}</div>
                                </div>
                            </div>
                        )}
                        
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
                                <h3 className="text-sm font-medium text-gray-500 dark:text-gray-300 mb-2">CPU Utilization</h3>
                                <div className="h-48">
                                    <canvas id="cpuChart"></canvas>
                                </div>
                            </div>
                            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
                                <h3 className="text-sm font-medium text-gray-500 dark:text-gray-300 mb-2">Memory Usage</h3>
                                <div className="h-48">
                                    <canvas id="memoryChart"></canvas>
                                </div>
                            </div>
                            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
                                <h3 className="text-sm font-medium text-gray-500 dark:text-gray-300 mb-2">Response Time</h3>
                                <div className="h-48">
                                    <canvas id="latencyChart"></canvas>
                                </div>
                            </div>
                            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
                                <h3 className="text-sm font-medium text-gray-500 dark:text-gray-300 mb-2">Error Rate</h3>
                                <div className="h-48">
                                    <canvas id="errorChart"></canvas>
                                </div>
                            </div>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
};