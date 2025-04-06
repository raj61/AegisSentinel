// Main App for Aegis Sentinel

// Main App Component
const App = () => {
    // State
    const [darkMode, setDarkMode] = React.useState(false);
    const [loading, setLoading] = React.useState({
        graph: true,
        issues: true,
        metrics: true
    });
    const [serviceGraph, setServiceGraph] = React.useState(null);
    const [issues, setIssues] = React.useState([]);
    const [selectedNode, setSelectedNode] = React.useState(null);
    const [overallStatus, setOverallStatus] = React.useState('healthy');
    const [statusMessage, setStatusMessage] = React.useState('All systems operational');
    const [notifications, setNotifications] = React.useState([]);
    const [mlStatus, setMlStatus] = React.useState({
        anomalyDetection: 'error',
        rootCauseAnalysis: 'active',
        remediationLearning: 'error',
        predictionEngine: 'active'
    });
    const [mlErrors, setMlErrors] = React.useState([
        'Error in detector metrics: Model must be trained before detection',
        'Error in detector logs: Model must be trained before detection'
    ]);
    const [showMlGraph, setShowMlGraph] = React.useState(false);
    
    // Toggle dark mode
    const toggleDarkMode = () => {
        const newMode = !darkMode;
        setDarkMode(newMode);
        document.body.className = newMode ? 'dark-mode' : 'light-mode';
    };
    
    // Toggle ML graph visualization
    const toggleMlGraph = () => {
        setShowMlGraph(prev => !prev);
        showNotification(
            showMlGraph ? 'ML Graph visualization disabled' : 'ML Graph visualization enabled',
            'info'
        );
    };
    
    // Fetch data
    React.useEffect(() => {
        fetchGraph();
        fetchIssues();
        
        // Set up polling
        const interval = setInterval(() => {
            fetchIssues();
        }, 5000);
        
        return () => clearInterval(interval);
    }, []);
    
    // Fetch graph data
    const fetchGraph = async () => {
        setLoading(prev => ({ ...prev, graph: true }));
        try {
            const response = await fetch('/api/graph');
            if (!response.ok) {
                throw new Error('Failed to fetch graph data');
            }
            const data = await response.json();
            setServiceGraph(data);
        } catch (error) {
            console.error('Error fetching graph:', error);
            showNotification('Error loading graph: ' + error.message, 'error');
        } finally {
            setLoading(prev => ({ ...prev, graph: false }));
        }
    };
    
    // Fetch issues data with incremental updates
    const fetchIssues = async () => {
        setLoading(prev => ({ ...prev, issues: true }));
        try {
            const response = await fetch('/api/issues');
            if (!response.ok) {
                throw new Error('Failed to fetch issues');
            }
            const data = await response.json();
            const apiIssues = data.issues || [];
            
            // Process new issues
            const processedIssues = apiIssues.map(issue => ({
                ...issue,
                showDetails: false,
                progress: issue.status === 'mitigating' ? Math.floor(Math.random() * 80) + 10 : 0,
                id: issue.type + '-' + (issue.detected_at || Date.now())
            }));
            
            // Merge with existing issues, preserving UI state
            setIssues(prevIssues => {
                // Create a map of existing issues by ID for quick lookup
                const existingIssuesMap = new Map(prevIssues.map(issue => [issue.id, issue]));
                
                // Process each new issue
                const mergedIssues = processedIssues.map(newIssue => {
                    const existingIssue = existingIssuesMap.get(newIssue.id);
                    
                    // If issue exists, preserve UI state like showDetails
                    if (existingIssue) {
                        return {
                            ...newIssue,
                            showDetails: existingIssue.showDetails,
                            // For mitigating issues, either keep increasing progress or preserve it
                            progress: newIssue.status === 'mitigating'
                                ? Math.min(95, existingIssue.progress + 5)
                                : newIssue.progress
                        };
                    }
                    
                    // If it's a new issue, notify
                    showNotification(`New issue detected: ${newIssue.type.replace(/_/g, ' ')}`, 'warning');
                    return newIssue;
                });
                
                // Update overall status with the merged issues
                updateOverallStatus(mergedIssues);
                
                return mergedIssues;
            });
        } catch (error) {
            console.error('Error fetching issues:', error);
        } finally {
            setLoading(prev => ({ ...prev, issues: false }));
        }
    };
    
    // Update overall status based on issues
    const updateOverallStatus = (issues) => {
        const criticalIssues = issues.filter(i => i.severity >= 4 && i.status !== 'mitigated').length;
        const warningIssues = issues.filter(i => i.severity < 4 && i.severity >= 2 && i.status !== 'mitigated').length;
        const mitigatedIssues = issues.filter(i => i.status === 'mitigated').length;
        
        let status = 'healthy';
        let message = 'All systems operational';
        
        if (criticalIssues > 0) {
            status = 'critical';
            message = `${criticalIssues} critical issue${criticalIssues > 1 ? 's' : ''} detected`;
        } else if (warningIssues > 0) {
            status = 'warning';
            message = `${warningIssues} warning issue${warningIssues > 1 ? 's' : ''} detected`;
        } else if (mitigatedIssues > 0) {
            message = `${mitigatedIssues} issue${mitigatedIssues > 1 ? 's' : ''} mitigated`;
        }
        
        setOverallStatus(status);
        setStatusMessage(message);
    };
    
    // Show notification
    const showNotification = (message, type = 'info') => {
        const id = Date.now();
        const notification = { id, message, type };
        setNotifications(prev => [...prev, notification]);
        
        // Remove notification after 5 seconds
        setTimeout(() => {
            setNotifications(prev => prev.filter(n => n.id !== id));
        }, 5000);
    };
    
    // Train ML models
    const trainModels = () => {
        showNotification('Training ML models...', 'info');
        
        // Simulate API call to train models
        setTimeout(() => {
            setMlStatus(prev => ({
                ...prev,
                anomalyDetection: 'training',
                remediationLearning: 'training'
            }));
            
            // Simulate training completion
            setTimeout(() => {
                setMlStatus(prev => ({
                    ...prev,
                    anomalyDetection: 'active',
                    remediationLearning: 'active'
                }));
                setMlErrors([]);
                showNotification('ML models trained successfully!', 'success');
            }, 3000);
        }, 1000);
    };
    
    return (
        <div className={darkMode ? 'dark-mode' : 'light-mode'}>
            {/* Notifications */}
            <Notifications notifications={notifications} />
            
            {/* Header */}
            <Header 
                darkMode={darkMode} 
                toggleDarkMode={toggleDarkMode} 
                overallStatus={overallStatus} 
                statusMessage={statusMessage} 
            />
            
            {/* Main Content */}
            <main className="container mx-auto px-4 py-6">
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Left Column - Service Graph */}
                    <div className="lg:col-span-2">
                        <ServiceGraph
                            darkMode={darkMode}
                            loading={loading.graph}
                            serviceGraph={serviceGraph}
                            issues={issues}
                            onNodeSelect={setSelectedNode}
                        />
                        
                        {/* ML-Enhanced Graph Panel */}
                        {showMlGraph && (
                            <MLGraphPanel
                                darkMode={darkMode}
                                loading={loading.graph}
                                serviceGraph={serviceGraph}
                            />
                        )}
                        
                        {/* Metrics Panel */}
                        <MetricsPanel
                            darkMode={darkMode}
                            loading={loading.metrics}
                            selectedNode={selectedNode}
                        />
                    </div>
                    
                    {/* Right Column - Issues and ML Status */}
                    <div>
                        <IssuesPanel 
                            darkMode={darkMode}
                            loading={loading.issues}
                            issues={issues}
                        />
                        
                        <MLStatusPanel
                            darkMode={darkMode}
                            mlStatus={mlStatus}
                            mlErrors={mlErrors}
                            trainModels={trainModels}
                            showMlGraph={showMlGraph}
                            toggleMlGraph={toggleMlGraph}
                        />
                    </div>
                </div>
            </main>
            
            {/* Footer */}
            <footer className={`mt-12 py-6 ${darkMode ? 'bg-gray-900' : 'bg-gray-100'}`}>
                <div className="container mx-auto px-4">
                    <div className="flex flex-col md:flex-row justify-between items-center">
                        <div className="mb-4 md:mb-0">
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                                &copy; 2025 Aegis Sentinel - Advanced SRE Platform
                            </p>
                        </div>
                        <div className="flex space-x-4">
                            <a href="#" className="text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200">Documentation</a>
                            <a href="#" className="text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200">API</a>
                            <a href="#" className="text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200">Support</a>
                        </div>
                    </div>
                </div>
            </footer>
        </div>
    );
};

// Render the app
ReactDOM.render(<App />, document.getElementById('root'));
