// MLGraphPanel Component for Aegis Sentinel

const MLGraphPanel = ({ darkMode, loading: globalLoading, serviceGraph }) => {
    const [localLoading, setLocalLoading] = React.useState(false);
    const [graphData, setGraphData] = React.useState(null);
    const [graphStats, setGraphStats] = React.useState({
        nodes: 0,
        edges: 0,
        mlEdges: 0,
        confidence: 0
    });
    const graphRef = React.useRef(null);
    const networkRef = React.useRef(null);
    
    // Initialize graph when service graph data changes
    React.useEffect(() => {
        if (serviceGraph && !globalLoading) {
            processGraphData();
        }
    }, [serviceGraph, globalLoading]);
    
    // Process graph data for visualization
    const processGraphData = () => {
        setLocalLoading(true);
        
        try {
            // Extract nodes and edges from service graph
            const nodes = serviceGraph.nodes.map(node => ({
                id: node.id,
                label: node.name || node.id,
                title: createNodeTooltip(node),
                group: getNodeGroup(node),
                shape: getNodeShape(node),
                size: getNodeSize(node),
                font: {
                    color: darkMode ? '#f3f4f6' : '#1f2937',
                    size: 12
                },
                borderWidth: 2,
                borderWidthSelected: 4,
                color: {
                    border: getNodeBorderColor(node),
                    background: getNodeColor(node),
                    highlight: {
                        border: '#3B82F6',
                        background: getNodeColor(node)
                    }
                }
            }));
            
            const edges = serviceGraph.edges.map(edge => ({
                from: edge.source,
                to: edge.target,
                title: createEdgeTooltip(edge),
                width: getEdgeWidth(edge),
                dashes: edge.type?.includes('ml-inferred') || edge.type?.includes('inferred'),
                arrows: {
                    to: {
                        enabled: true,
                        scaleFactor: 0.5
                    }
                },
                color: {
                    color: getEdgeColor(edge),
                    highlight: '#3B82F6',
                    opacity: getEdgeOpacity(edge)
                },
                smooth: {
                    type: 'curvedCW',
                    roundness: 0.2
                }
            }));
            
            setGraphData({ nodes, edges });
            
            // Calculate graph statistics
            const mlEdges = serviceGraph.edges.filter(edge => 
                edge.type?.includes('ml-inferred') || 
                edge.type?.includes('traffic-inferred') || 
                edge.type?.includes('error-inferred')
            ).length;
            
            const avgConfidence = serviceGraph.edges.reduce((sum, edge) => 
                sum + (edge.confidence || 0), 0) / serviceGraph.edges.length || 0;
            
            setGraphStats({
                nodes: serviceGraph.nodes.length,
                edges: serviceGraph.edges.length,
                mlEdges,
                confidence: avgConfidence
            });
            
            // Initialize the graph visualization after a longer delay to ensure DOM is ready
            console.log("Scheduling graph initialization with delay...");
            setTimeout(() => {
                try {
                    console.log("Initializing graph after delay...");
                    initGraph();
                } catch (error) {
                    console.error("Error in delayed graph initialization:", error);
                } finally {
                    setLocalLoading(false);
                }
            }, 1000); // Increased delay to ensure DOM is fully ready
        } catch (error) {
            console.error('Error processing graph data:', error);
            setLocalLoading(false);
        }
    };
    
    // Initialize the graph visualization
    const initGraph = () => {
        if (!graphRef.current || !graphData) return;
        
        try {
            console.log("Initializing graph visualization...");
            
            // Destroy existing network
            if (networkRef.current) {
                networkRef.current.destroy();
                networkRef.current = null;
            }
            
            // Create vis.js data sets
            const nodes = new vis.DataSet(graphData.nodes);
            const edges = new vis.DataSet(graphData.edges);
            
            // Create vis.js network
            const container = graphRef.current;
            const data = { nodes, edges };
            
            console.log("Graph data prepared:", {
                nodeCount: graphData.nodes.length,
                edgeCount: graphData.edges.length
            });
        } catch (error) {
            console.error("Error initializing graph:", error);
            setLocalLoading(false);
        }
        try {
            const options = {
                physics: {
                    enabled: true,
                    barnesHut: {
                        gravitationalConstant: -2000,
                        centralGravity: 0.3,
                        springLength: 150,
                        springConstant: 0.04,
                        damping: 0.09
                    },
                    stabilization: {
                        iterations: 100
                    }
                },
                layout: {
                    improvedLayout: true,
                    hierarchical: {
                        enabled: false
                    }
                },
                interaction: {
                    hover: true,
                    tooltipDelay: 300,
                    zoomView: true,
                    dragView: true,
                    navigationButtons: true,
                    keyboard: true
                },
                groups: {
                    frontend: {
                        color: { background: '#93C5FD', border: '#3B82F6' },
                        shape: 'box'
                    },
                    backend: {
                        color: { background: '#A7F3D0', border: '#10B981' },
                        shape: 'box'
                    },
                    database: {
                        color: { background: '#FDE68A', border: '#F59E0B' },
                        shape: 'database'
                    },
                    cache: {
                        color: { background: '#C4B5FD', border: '#8B5CF6' },
                        shape: 'hexagon'
                    },
                    queue: {
                        color: { background: '#FECACA', border: '#EF4444' },
                        shape: 'diamond'
                    },
                    api: {
                        color: { background: '#BFDBFE', border: '#3B82F6' },
                        shape: 'triangle'
                    },
                    other: {
                        color: { background: '#E5E7EB', border: '#9CA3AF' },
                        shape: 'dot'
                    }
                }
            };
            
            console.log("Creating network with options:", options);
            
            try {
                // Create network
                networkRef.current = new vis.Network(container, data, options);
                console.log("Network created successfully");
            } catch (e) {
                console.error("Error creating vis.Network:", e);
                // Try with simpler options as fallback
                const simpleOptions = {
                    physics: false,
                    interaction: {
                        hover: true,
                        tooltipDelay: 300,
                        zoomView: true,
                        dragView: true
                    }
                };
                console.log("Trying with simpler options:", simpleOptions);
                networkRef.current = new vis.Network(container, data, simpleOptions);
                console.log("Network created with fallback options");
            }
            
            // Add event listeners
            networkRef.current.on('click', function(params) {
                if (params.nodes.length > 0) {
                    const nodeId = params.nodes[0];
                    const node = nodes.get(nodeId);
                    console.log('Selected node:', node);
                }
            });
            
            console.log("Event listeners added");
        } catch (error) {
            console.error("Error creating network:", error);
        }
    };
    
    // Helper functions for node and edge styling
    
    const getNodeGroup = (node) => {
        const category = (node.category || '').toLowerCase();
        if (category.includes('frontend') || category.includes('ui')) return 'frontend';
        if (category.includes('backend') || category.includes('service')) return 'backend';
        if (category.includes('database') || category.includes('db')) return 'database';
        if (category.includes('cache') || category.includes('redis')) return 'cache';
        if (category.includes('queue') || category.includes('kafka')) return 'queue';
        if (category.includes('api') || category.includes('gateway')) return 'api';
        return 'other';
    };
    
    const getNodeShape = (node) => {
        const kind = (node.kind || '').toLowerCase();
        if (kind === 'deployment') return 'box';
        if (kind === 'statefulset') return 'database';
        if (kind === 'service') return 'dot';
        if (kind === 'pod') return 'dot';
        if (kind === 'ingress') return 'triangle';
        return 'dot';
    };
    
    const getNodeSize = (node) => {
        // Base size
        let size = 20;
        
        // Adjust based on importance
        if (node.kind === 'Deployment' || node.kind === 'StatefulSet') size += 5;
        if (node.category === 'api' || node.category === 'gateway') size += 5;
        
        // Adjust based on metrics if available
        if (node.request_rate) size += Math.min(node.request_rate / 10, 10);
        
        return size;
    };
    
    const getNodeColor = (node) => {
        // Base color based on health status
        const healthStatus = (node.health_status || '').toLowerCase();
        if (healthStatus === 'critical') return '#FEE2E2';
        if (healthStatus === 'warning') return '#FEF3C7';
        if (healthStatus === 'healthy') return '#D1FAE5';
        
        // Default color based on category
        const category = (node.category || '').toLowerCase();
        if (category.includes('frontend')) return '#DBEAFE';
        if (category.includes('backend')) return '#D1FAE5';
        if (category.includes('database')) return '#FEF3C7';
        if (category.includes('cache')) return '#EDE9FE';
        if (category.includes('queue')) return '#FEE2E2';
        if (category.includes('api')) return '#BFDBFE';
        
        return darkMode ? '#374151' : '#F3F4F6';
    };
    
    const getNodeBorderColor = (node) => {
        // Border color based on health status
        const healthStatus = (node.health_status || '').toLowerCase();
        if (healthStatus === 'critical') return '#EF4444';
        if (healthStatus === 'warning') return '#F59E0B';
        if (healthStatus === 'healthy') return '#10B981';
        
        // Default border color
        return darkMode ? '#6B7280' : '#9CA3AF';
    };
    
    const getEdgeColor = (edge) => {
        // Color based on edge type
        const type = (edge.type || '').toLowerCase();
        if (type.includes('direct')) return '#3B82F6';
        if (type.includes('ml-inferred')) return '#8B5CF6';
        if (type.includes('traffic')) return '#F59E0B';
        if (type.includes('error')) return '#EF4444';
        if (type.includes('deployment')) return '#10B981';
        if (type.includes('inferred')) return '#6B7280';
        
        return darkMode ? '#6B7280' : '#9CA3AF';
    };
    
    const getEdgeWidth = (edge) => {
        // Base width
        let width = 1;
        
        // Adjust based on confidence
        if (edge.confidence) width += edge.confidence * 2;
        
        // Adjust based on type
        const type = (edge.type || '').toLowerCase();
        if (type.includes('direct')) width += 1;
        if (type.includes('ml-inferred')) width += 0.5;
        
        return width;
    };
    
    const getEdgeOpacity = (edge) => {
        // Base opacity
        let opacity = 0.6;
        
        // Adjust based on confidence
        if (edge.confidence) opacity = 0.4 + edge.confidence * 0.6;
        
        // Adjust based on type
        const type = (edge.type || '').toLowerCase();
        if (type.includes('direct')) opacity = 1;
        if (type.includes('inferred')) opacity = 0.7;
        
        return opacity;
    };
    
    const createNodeTooltip = (node) => {
        let tooltip = `<div class="p-2">
            <div class="font-bold">${node.name || node.id}</div>
            <div class="text-sm">Kind: ${node.kind || 'Unknown'}</div>`;
        
        if (node.category) {
            tooltip += `<div class="text-sm">Category: ${node.category}</div>`;
        }
        
        if (node.health_status) {
            tooltip += `<div class="text-sm">Health: ${node.health_status}</div>`;
        }
        
        // Add metrics if available
        if (node.cpu_usage !== undefined) {
            tooltip += `<div class="text-sm">CPU: ${node.cpu_usage.toFixed(2)}%</div>`;
        }
        
        if (node.memory_usage !== undefined) {
            tooltip += `<div class="text-sm">Memory: ${node.memory_usage.toFixed(2)} MB</div>`;
        }
        
        if (node.latency_p95 !== undefined) {
            tooltip += `<div class="text-sm">Latency (p95): ${node.latency_p95.toFixed(2)} ms</div>`;
        }
        
        if (node.error_rate !== undefined) {
            tooltip += `<div class="text-sm">Error Rate: ${(node.error_rate * 100).toFixed(2)}%</div>`;
        }
        
        tooltip += '</div>';
        return tooltip;
    };
    
    const createEdgeTooltip = (edge) => {
        let tooltip = `<div class="p-2">
            <div class="font-bold">${edge.source} → ${edge.target}</div>
            <div class="text-sm">Type: ${edge.type || 'Unknown'}</div>`;
        
        if (edge.confidence !== undefined) {
            tooltip += `<div class="text-sm">Confidence: ${(edge.confidence * 100).toFixed(1)}%</div>`;
        }
        
        if (edge.similarity !== undefined) {
            tooltip += `<div class="text-sm">Similarity: ${(edge.similarity * 100).toFixed(1)}%</div>`;
        }
        
        tooltip += '</div>';
        return tooltip;
    };
    
    return (
        <div className="mt-6 bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden">
            <div className="p-4 bg-gray-50 dark:bg-gray-700 border-b dark:border-gray-600">
                <h2 className="text-lg font-semibold text-gray-800 dark:text-white">
                    ML-Enhanced Service Graph
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
                            <p className="text-gray-600 dark:text-gray-300">Loading service graph...</p>
                        </div>
                    </div>
                ) : !serviceGraph ? (
                    <div className="text-center py-8">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 mx-auto text-gray-400 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
                        </svg>
                        <p className="text-gray-500 dark:text-gray-400">No service graph data available</p>
                    </div>
                ) : (
                    <>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                            <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                                <div className="text-sm text-gray-500 dark:text-gray-400">Services</div>
                                <div className="text-xl font-semibold mt-1">{graphStats.nodes}</div>
                            </div>
                            <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                                <div className="text-sm text-gray-500 dark:text-gray-400">Dependencies</div>
                                <div className="text-xl font-semibold mt-1">{graphStats.edges}</div>
                            </div>
                            <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                                <div className="text-sm text-gray-500 dark:text-gray-400">ML-Inferred</div>
                                <div className="text-xl font-semibold mt-1">{graphStats.mlEdges}</div>
                            </div>
                            <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                                <div className="text-sm text-gray-500 dark:text-gray-400">Avg. Confidence</div>
                                <div className="text-xl font-semibold mt-1">{(graphStats.confidence * 100).toFixed(1)}%</div>
                            </div>
                        </div>
                        
                        <div className="bg-gray-50 dark:bg-gray-700 p-2 rounded-lg mb-4">
                            <div className="flex flex-wrap gap-2 text-xs">
                                <div className="flex items-center">
                                    <span className="inline-block w-3 h-3 bg-blue-500 rounded-full mr-1"></span>
                                    <span>Direct</span>
                                </div>
                                <div className="flex items-center">
                                    <span className="inline-block w-3 h-3 bg-purple-500 rounded-full mr-1"></span>
                                    <span>ML-Inferred</span>
                                </div>
                                <div className="flex items-center">
                                    <span className="inline-block w-3 h-3 bg-amber-500 rounded-full mr-1"></span>
                                    <span>Traffic-Based</span>
                                </div>
                                <div className="flex items-center">
                                    <span className="inline-block w-3 h-3 bg-red-500 rounded-full mr-1"></span>
                                    <span>Error-Based</span>
                                </div>
                                <div className="flex items-center">
                                    <span className="inline-block w-3 h-3 bg-green-500 rounded-full mr-1"></span>
                                    <span>Deployment</span>
                                </div>
                                <div className="flex items-center">
                                    <span className="inline-block w-3 h-3 bg-gray-500 rounded-full mr-1"></span>
                                    <span>Inferred</span>
                                </div>
                            </div>
                        </div>
                        
                        <div className="h-[500px] bg-gray-50 dark:bg-gray-700 rounded-lg border border-gray-200 dark:border-gray-600">
                            <div ref={graphRef} className="w-full h-full"></div>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
};