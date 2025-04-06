// ServiceGraph Component for Aegis Sentinel

const ServiceGraph = ({ darkMode, loading, serviceGraph, issues, onNodeSelect }) => {
    const graphRef = React.useRef(null);
    const simulationRef = React.useRef(null);
    const zoomRef = React.useRef(null);
    
    // Initialize graph when data is available
    React.useEffect(() => {
        if (serviceGraph && !loading) {
            renderGraph();
        }
    }, [serviceGraph, loading, darkMode]);
    
    // Update node colors when issues change
    React.useEffect(() => {
        if (issues && issues.length > 0) {
            updateNodeColors();
        }
    }, [issues]);
    
    // Zoom functions
    const zoomIn = () => {
        if (zoomRef.current) {
            d3.select(graphRef.current).select('svg')
                .transition()
                .duration(300)
                .call(zoomRef.current.scaleBy, 1.3);
        }
    };
    
    const zoomOut = () => {
        if (zoomRef.current) {
            d3.select(graphRef.current).select('svg')
                .transition()
                .duration(300)
                .call(zoomRef.current.scaleBy, 0.7);
        }
    };
    
    const resetZoom = () => {
        if (zoomRef.current) {
            d3.select(graphRef.current).select('svg')
                .transition()
                .duration(300)
                .call(zoomRef.current.transform, d3.zoomIdentity);
        }
    };
    
    // Get node color based on status and category
    const getNodeColor = (node) => {
        // Check if this node is affected by any issues
        const isAffectedByIssue = issues && issues.some(issue =>
            issue.affected_nodes && issue.affected_nodes.includes(node.id)
        );
        
        // If node is affected by an issue, mark it as a bottleneck
        if (isAffectedByIssue) {
            return '#FF5252'; // Red for bottleneck/problematic nodes
        }
        
        // If node has a health status, use that
        if (node.health_status === 'healthy') {
            return '#10B981'; // Green for healthy nodes
        } else if (node.health_status === 'warning') {
            return '#F59E0B'; // Yellow for warning nodes
        } else if (node.health_status === 'critical') {
            return '#EF4444'; // Red for critical nodes
        }
        
        // Otherwise use category colors
        const categoryColors = {
            'compute': '#3B82F6',
            'serverless': '#10B981',
            'container': '#F59E0B',
            'kubernetes': '#EF4444',
            'api': '#3B82F6',
            'database': '#F59E0B',
            'cache': '#8B5CF6',
            'queue': '#F97316',
            'other': '#6B7280'
        };
        return categoryColors[node.category] || categoryColors.other;
    };
    
    // Get health marker color
    const getHealthMarkerColor = (node) => {
        if (node.health_status === 'healthy') {
            return '#10B981'; // Green for healthy nodes
        } else if (node.health_status === 'warning') {
            return '#F59E0B'; // Yellow for warning nodes
        } else if (node.health_status === 'critical') {
            return '#EF4444'; // Red for critical nodes
        }
        return '#6B7280'; // Grey for unknown health
    };
    
    // Get health status class
    const getHealthStatus = (node) => {
        if (node.health_status === 'healthy') {
            return 'health-healthy';
        } else if (node.health_status === 'warning') {
            return 'health-warning';
        } else if (node.health_status === 'critical') {
            return 'health-critical';
        }
        return 'health-unknown';
    };
    
    // Update node colors based on issues
    const updateNodeColors = () => {
        if (!issues || !graphRef.current) return;
        
        // Update node colors based on affected nodes
        d3.selectAll('.node-circle').each(function(d) {
            const isAffected = issues.some(issue => 
                issue.affected_nodes && issue.affected_nodes.includes(d.id)
            );
            
            if (isAffected) {
                d3.select(this).transition().duration(500).attr('fill', '#FF5252');
                
                // Also update health marker
                const healthMarkerId = `.health-marker-${d.id.replace(/[\/\.]/g, '-')}`;
                d3.select(healthMarkerId)
                    .transition()
                    .duration(500)
                    .attr('fill', '#F44336')
                    .attr('class', function() {
                        return this.getAttribute('class').replace(/healthy|warning|unknown/, 'critical');
                    });
            }
        });
    };
    
    // Render the graph
    const renderGraph = () => {
        if (!graphRef.current || !serviceGraph) return;
        
        // Clear previous graph
        d3.select(graphRef.current).selectAll("*").remove();
        
        const width = graphRef.current.clientWidth;
        const height = graphRef.current.clientHeight;
        
        // Create SVG with better dimensions
        const svg = d3.select(graphRef.current)
            .append('svg')
            .attr('width', '100%')
            .attr('height', '100%')
            .attr('viewBox', `0 0 ${width} ${height}`)
            .attr('preserveAspectRatio', 'xMidYMid meet');
        
        // Create the group first
        const g = svg.append('g');
        
        // Add zoom behavior with smoother transitions
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => {
                g.attr('transform', event.transform);
            });
        
        zoomRef.current = zoom;
        svg.call(zoom);
        
        // Apply initial transform after the group is created
        setTimeout(() => {
            // Calculate the bounding box of all nodes
            let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
            serviceGraph.nodes.forEach(node => {
                minX = Math.min(minX, node.x || 0);
                minY = Math.min(minY, node.y || 0);
                maxX = Math.max(maxX, node.x || 0);
                maxY = Math.max(maxY, node.y || 0);
            });
            
            // Add padding
            const padding = 50;
            minX -= padding;
            minY -= padding;
            maxX += padding;
            maxY += padding;
            
            // Calculate the scale to fit all nodes
            const graphWidth = maxX - minX;
            const graphHeight = maxY - minY;
            const scale = Math.min(
                width / graphWidth,
                height / graphHeight,
                1.0 // Cap the scale to avoid too much zoom
            ) * 0.9; // 90% to add some margin
            
            // Calculate the center of the graph
            const centerX = (minX + maxX) / 2;
            const centerY = (minY + maxY) / 2;
            
            // Calculate the translation to center the graph
            const translateX = width / 2 - centerX * scale;
            const translateY = height / 2 - centerY * scale;
            
            // Apply the transform
            svg.call(zoom.transform, d3.zoomIdentity
                .translate(translateX, translateY)
                .scale(scale));
        }, 100);
        
        // Create a defs section for gradient fills
        const defs = svg.append('defs');
        
        // Add gradients for node colors
        const gradients = {
            'compute': createGradient(defs, 'compute-gradient', '#3B82F6', '#2563EB'),
            'serverless': createGradient(defs, 'serverless-gradient', '#10B981', '#059669'),
            'container': createGradient(defs, 'container-gradient', '#F59E0B', '#D97706'),
            'kubernetes': createGradient(defs, 'kubernetes-gradient', '#EF4444', '#DC2626'),
            'api': createGradient(defs, 'api-gradient', '#3B82F6', '#2563EB'),
            'database': createGradient(defs, 'database-gradient', '#F59E0B', '#D97706'),
            'cache': createGradient(defs, 'cache-gradient', '#8B5CF6', '#7C3AED'),
            'queue': createGradient(defs, 'queue-gradient', '#F97316', '#EA580C'),
            'storage': createGradient(defs, 'storage-gradient', '#06B6D4', '#0891B2'),
            'other': createGradient(defs, 'other-gradient', '#6B7280', '#4B5563')
        };
        
        // Helper function to create gradient
        function createGradient(defs, id, color1, color2) {
            const gradient = defs.append('linearGradient')
                .attr('id', id)
                .attr('x1', '0%')
                .attr('y1', '0%')
                .attr('x2', '0%')
                .attr('y2', '100%');
                
            gradient.append('stop')
                .attr('offset', '0%')
                .attr('stop-color', color1);
                
            gradient.append('stop')
                .attr('offset', '100%')
                .attr('stop-color', color2);
                
            return `url(#${id})`;
        }
        
        // Create a map of nodes by ID for faster lookups
        const nodeMap = {};
        serviceGraph.nodes.forEach(node => {
            // Ensure each node has initial positions if not already set
            if (node.x === undefined) node.x = Math.random() * width;
            if (node.y === undefined) node.y = Math.random() * height;
            nodeMap[node.id] = node;
        });
        
        // Ensure edges have proper source and target references
        const edges = serviceGraph.edges.map(edge => {
            // Handle both string IDs and object references
            let source, target;
            
            if (typeof edge.source === 'string') {
                source = nodeMap[edge.source];
                if (!source) {
                    console.warn(`Source node not found: ${edge.source}`);
                    return null;
                }
            } else {
                source = edge.source;
            }
            
            if (typeof edge.target === 'string') {
                target = nodeMap[edge.target];
                if (!target) {
                    console.warn(`Target node not found: ${edge.target}`);
                    return null;
                }
            } else {
                target = edge.target;
            }
            
            return {...edge, source, target};
        }).filter(edge => edge && edge.source && edge.target);
        
        // Log the processed edges for debugging
        console.log('Processed edges:', edges.map(e => ({
            source: typeof e.source === 'object' ? e.source.id : e.source,
            target: typeof e.target === 'object' ? e.target.id : e.target
        })));
        
        // Log the processed graph data for debugging
        console.log('Processed graph data:', {
            nodes: serviceGraph.nodes.length,
            edges: edges.length
        });
        
        // Create simulation with better forces for proper node positioning
        const simulation = d3.forceSimulation(serviceGraph.nodes)
            // Link force with increased distance to spread nodes out more
            .force('link', d3.forceLink(edges).id(d => d.id).distance(150))
            // Stronger repulsive force to push nodes apart
            .force('charge', d3.forceManyBody().strength(-800))
            // Center force to keep the graph centered
            .force('center', d3.forceCenter(width / 2, height / 2))
            // Collision detection to prevent node overlap
            .force('collision', d3.forceCollide().radius(40))
            // X and Y forces to pull nodes toward the center
            .force('x', d3.forceX(width / 2).strength(0.1))
            .force('y', d3.forceY(height / 2).strength(0.1));
        
        // Run the simulation for more iterations to stabilize the layout
        // Use a higher alpha value to make the simulation more energetic
        simulation.alpha(1);
        for (let i = 0; i < 500; i++) {
            simulation.tick();
        }
        
        simulationRef.current = simulation;
        
        // Create arrow markers with better positioning
        svg.append('defs').selectAll('marker')
            .data(['end'])
            .enter().append('marker')
            .attr('id', d => d)
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 20) // Adjusted to better position the arrow at the edge of the node
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', darkMode ? '#888' : '#999');
        
        // Create links with better styling and visibility
        const link = g.append('g')
            .selectAll('path')
            .data(edges)
            .enter()
            .append('path')
            .attr('stroke', darkMode ? '#aaa' : '#666') // Darker color for better visibility
            .attr('stroke-width', 2.5) // Thicker lines
            .attr('stroke-opacity', 0.8) // More opaque
            .attr('fill', 'none')
            .attr('marker-end', 'url(#end)')
            .style('stroke-dasharray', d => d.type === 'async' ? '5, 5' : 'none')
            .style('transition', 'stroke-width 0.3s, stroke-opacity 0.3s');
        
        // Create node groups
        const nodeGroups = g.append('g')
            .selectAll('.node-group')
            .data(serviceGraph.nodes)
            .enter()
            .append('g')
            .attr('class', 'node-group');
        
        // Create nodes with improved styling
        const node = nodeGroups
            .append('circle')
            .attr('r', 20) // Smaller radius
            .attr('fill', d => {
                // Use gradient based on category if not affected by issues
                const isAffectedByIssue = issues && issues.some(issue =>
                    issue.affected_nodes && issue.affected_nodes.includes(d.id)
                );
                
                if (isAffectedByIssue) {
                    return '#FF5252';
                }
                
                if (d.health_status === 'critical') {
                    return '#EF4444';
                } else if (d.health_status === 'warning') {
                    return '#F59E0B';
                } else if (d.health_status === 'healthy') {
                    return '#10B981';
                }
                
                const category = d.category || 'other';
                return gradients[category] || gradients.other;
            })
            .attr('stroke', darkMode ? '#444' : '#fff')
            .attr('stroke-width', 2)
            .attr('class', d => `node-circle node-${d.id.replace(/[\/\.]/g, '-')}`)
            .style('filter', 'drop-shadow(0px 2px 3px rgba(0, 0, 0, 0.15))')
            .on('mouseover', function(event, d) {
                // Remove any existing tooltips first
                d3.selectAll('.tooltip').remove();
                
                // Show tooltip
                const tooltip = d3.select('body')
                    .append('div')
                    .attr('class', 'tooltip')
                    .attr('id', `tooltip-${d.id.replace(/[\/\.]/g, '-')}`)
                    .style('position', 'absolute')
                    .style('background-color', darkMode ? '#374151' : 'white')
                    .style('color', darkMode ? 'white' : '#333')
                    .style('border', darkMode ? '1px solid #4B5563' : '1px solid #E5E7EB')
                    .style('border-radius', '0.375rem')
                    .style('padding', '0.5rem')
                    .style('box-shadow', '0 2px 5px rgba(0, 0, 0, 0.2)')
                    .style('pointer-events', 'none')
                    .style('z-index', '1000')
                    .style('opacity', 0);
                
                tooltip.transition()
                    .duration(200)
                    .style('opacity', 0.9);
                
                tooltip.html(`
                    <div class="p-2">
                        <div class="font-bold">${d.name || d.id}</div>
                        <div class="text-xs opacity-70">${d.kind || 'Service'}</div>
                        <div class="mt-1">
                            ${d.category ? `<span class="px-2 py-1 rounded text-xs bg-blue-100 text-blue-800">${d.category}</span>` : ''}
                            ${d.health_status ? `<span class="px-2 py-1 rounded text-xs bg-${d.health_status === 'healthy' ? 'green' : d.health_status === 'warning' ? 'yellow' : 'red'}-100 text-${d.health_status === 'healthy' ? 'green' : d.health_status === 'warning' ? 'yellow' : 'red'}-800">${d.health_status}</span>` : ''}
                        </div>
                    </div>
                `)
                    .style('left', (event.pageX + 10) + 'px')
                    .style('top', (event.pageY - 28) + 'px');
                
                // Highlight connected nodes and links
                const connectedNodeIds = new Set();
                edges.forEach(edge => {
                    if (edge.source.id === d.id) connectedNodeIds.add(edge.target.id);
                    if (edge.target.id === d.id) connectedNodeIds.add(edge.source.id);
                });
                
                d3.selectAll('.node-circle')
                    .attr('opacity', n => n.id === d.id || connectedNodeIds.has(n.id) ? 1 : 0.3);
                
                d3.selectAll('path')
                    .attr('opacity', e => e.source.id === d.id || e.target.id === d.id ? 1 : 0.1);
                    
                d3.selectAll('.node-label')
                    .attr('opacity', n => n.id === d.id || connectedNodeIds.has(n.id) ? 1 : 0.3);
                    
                d3.selectAll('.health-marker')
                    .attr('opacity', n => n.id === d.id || connectedNodeIds.has(n.id) ? 1 : 0.3);
            })
            .on('mouseout', function() {
                // Hide tooltip with a transition
                d3.selectAll('.tooltip')
                    .transition()
                    .duration(200)
                    .style('opacity', 0)
                    .remove();
                
                // Reset highlight
                d3.selectAll('.node-circle').attr('opacity', 1);
                d3.selectAll('path').attr('opacity', 1);
                d3.selectAll('.node-label').attr('opacity', 1);
                d3.selectAll('.health-marker').attr('opacity', 1);
            })
            .on('click', function(event, d) {
                // Select node
                if (onNodeSelect) {
                    onNodeSelect(d);
                }
                
                // Highlight selected node
                d3.selectAll('.node-circle')
                    .attr('stroke-width', n => n.id === d.id ? 5 : 3);
            });
        
        // Add health markers as part of the node (not separate elements)
        const healthMarkers = nodeGroups
            .append('circle')
            .attr('r', 6)
            .attr('cx', 15)
            .attr('cy', -15)
            .attr('class', d => `health-marker health-marker-${d.id.replace(/[\/\.]/g, '-')} ${getHealthStatus(d)}`)
            .attr('fill', d => getHealthMarkerColor(d))
            .attr('stroke', darkMode ? '#444' : '#fff')
            .attr('stroke-width', 1.5);
        
        // Add labels with better visibility
        const label = nodeGroups
            .append('text')
            .text(d => {
                // Use a shorter name if available, or truncate the ID
                const displayText = d.name || d.id;
                return displayText.length > 15 ? displayText.substring(0, 12) + '...' : displayText;
            })
            .attr('class', 'node-label')
            .attr('font-size', 12)
            .attr('font-weight', 'bold')
            .attr('text-anchor', 'middle')
            .attr('dy', 35)
            .attr('fill', darkMode ? '#fff' : '#333')
            .attr('stroke', darkMode ? '#1f2937' : 'white')
            .attr('stroke-width', 0.5)
            .attr('paint-order', 'stroke');
        
        // Update positions on simulation tick
        simulation.on('tick', () => {
            // Keep nodes within bounds
            node
                .attr('cx', d => d.x = Math.max(30, Math.min(width - 30, d.x)))
                .attr('cy', d => d.y = Math.max(30, Math.min(height - 30, d.y)));
        
            // Update node group positions first
            nodeGroups
                .attr('transform', d => `translate(${d.x}, ${d.y})`);
                
            // Update link positions with straight lines
            link.attr('d', d => {
                if (!d.source.x || !d.target.x) return '';
                
                // Calculate the direction vector
                const dx = d.target.x - d.source.x;
                const dy = d.target.y - d.source.y;
                
                // Calculate the length of the vector
                const length = Math.sqrt(dx * dx + dy * dy);
                if (length === 0) return '';
                
                // Normalize the vector
                const nx = dx / length;
                const ny = dy / length;
                
                // Calculate start and end points (offset from node centers by node radius)
                const nodeRadius = 20; // Same as the node radius defined earlier
                const startX = d.source.x + nx * nodeRadius;
                const startY = d.source.y + ny * nodeRadius;
                const endX = d.target.x - nx * nodeRadius;
                const endY = d.target.y - ny * nodeRadius;
                
                // Return a straight line path
                return `M${startX},${startY}L${endX},${endY}`;
            });
        });
        
        // Add drag behavior
        node.call(d3.drag()
            .on('start', (event, d) => {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            })
            .on('drag', (event, d) => {
                d.fx = event.x;
                d.fy = event.y;
            })
            .on('end', (event, d) => {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }));
    };
    
    return (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden">
            <div className="p-4 bg-gray-50 dark:bg-gray-700 border-b dark:border-gray-600 flex justify-between items-center">
                <h2 className="text-lg font-semibold text-gray-800 dark:text-white">Service Dependency Graph</h2>
                <div className="flex space-x-2">
                    <button onClick={zoomIn} className="bg-gray-200 dark:bg-gray-600 hover:bg-gray-300 dark:hover:bg-gray-500 rounded p-1 transition-colors">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-700 dark:text-gray-200" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M10 5a1 1 0 011 1v3h3a1 1 0 110 2h-3v3a1 1 0 11-2 0v-3H6a1 1 0 110-2h3V6a1 1 0 011-1z" clipRule="evenodd" />
                        </svg>
                    </button>
                    <button onClick={zoomOut} className="bg-gray-200 dark:bg-gray-600 hover:bg-gray-300 dark:hover:bg-gray-500 rounded p-1 transition-colors">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-700 dark:text-gray-200" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M5 10a1 1 0 011-1h8a1 1 0 110 2H6a1 1 0 01-1-1z" clipRule="evenodd" />
                        </svg>
                    </button>
                    <button onClick={resetZoom} className="bg-gray-200 dark:bg-gray-600 hover:bg-gray-300 dark:hover:bg-gray-500 rounded p-1 transition-colors">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-700 dark:text-gray-200" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clipRule="evenodd" />
                        </svg>
                    </button>
                    <button
                        onClick={() => {
                            // Inject anomaly via API
                            fetch('/api/inject-anomaly', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json'
                                },
                                body: JSON.stringify({ type: 'random' })
                            })
                            .then(response => response.json())
                            .then(data => {
                                if (data.success) {
                                    console.log('Injected anomaly:', data);
                                    // Show a notification
                                    alert(`Injected anomaly in service: ${data.affected_service}. It will be automatically resolved in 60 seconds.`);
                                } else {
                                    console.error('Failed to inject anomaly:', data.error);
                                    alert('Failed to inject anomaly: ' + data.error);
                                }
                            })
                            .catch(error => {
                                console.error('Error injecting anomaly:', error);
                                alert('Error injecting anomaly: ' + error.message);
                            });
                        }}
                        className="bg-red-100 text-red-700 dark:bg-red-700 dark:text-red-100 hover:bg-red-200 dark:hover:bg-red-600 rounded p-1 transition-colors ml-4"
                        title="Inject a random anomaly for demonstration purposes"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                        </svg>
                    </button>
                </div>
            </div>
            <div className="relative" style={{ height: '500px' }}>
                {loading && (
                    <div className="absolute inset-0 flex items-center justify-center bg-white dark:bg-gray-800 bg-opacity-80 dark:bg-opacity-80 z-10">
                        <div className="flex flex-col items-center">
                            <svg className="animate-spin h-10 w-10 text-blue-600 dark:text-blue-400 mb-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            <p className="text-gray-600 dark:text-gray-300">Loading service graph...</p>
                        </div>
                    </div>
                )}
                <div ref={graphRef} className="w-full h-full"></div>
            </div>
        </div>
    );
};