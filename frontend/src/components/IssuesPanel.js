// IssuesPanel Component for Aegis Sentinel

const IssuesPanel = ({ darkMode, loading, issues }) => {
    // Format issue type for display
    const formatIssueType = (type) => {
        if (!type) return 'Unknown Issue';
        return type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    };
    
    // Format timestamp for display
    const formatTime = (timestamp) => {
        if (!timestamp) return 'N/A';
        try {
            return new Date(timestamp).toLocaleString();
        } catch (e) {
            return timestamp;
        }
    };
    
    // Format edges for display
    const formatEdges = (edges) => {
        if (!edges || !edges.length) return 'None';
        return edges.map(e => `${e[0]} → ${e[1]}`).join(', ');
    };
    
    // State for tracking which issues have expanded details
    const [expandedIssues, setExpandedIssues] = React.useState(new Set());
    
    // Toggle issue details
    const toggleDetails = (issueId) => {
        setExpandedIssues(prev => {
            const newSet = new Set(prev);
            if (newSet.has(issueId)) {
                newSet.delete(issueId);
            } else {
                newSet.add(issueId);
            }
            return newSet;
        });
    };
    
    // Get severity class
    const getSeverityClass = (severity) => {
        switch (severity) {
            case 5:
                return 'border-red-500 bg-red-50 dark:bg-red-900/20';
            case 4:
                return 'border-orange-500 bg-orange-50 dark:bg-orange-900/20';
            case 3:
                return 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20';
            case 2:
                return 'border-green-500 bg-green-50 dark:bg-green-900/20';
            case 1:
                return 'border-blue-500 bg-blue-50 dark:bg-blue-900/20';
            default:
                return 'border-gray-500 bg-gray-50 dark:bg-gray-700';
        }
    };
    
    // Get status badge class
    const getStatusBadgeClass = (status) => {
        switch (status) {
            case 'detected':
                return 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-200';
            case 'mitigating':
                return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-200';
            case 'mitigated':
                return 'bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-200';
            case 'failed':
                return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200';
            default:
                return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200';
        }
    };
    
    // Get progress bar class
    const getProgressBarClass = (status) => {
        switch (status) {
            case 'mitigated':
                return 'bg-green-500';
            case 'failed':
                return 'bg-red-500';
            case 'mitigating':
                return 'bg-yellow-500';
            default:
                return 'bg-blue-500';
        }
    };
    
    return (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden">
            <div className="p-4 bg-gray-50 dark:bg-gray-700 border-b dark:border-gray-600">
                <h2 className="text-lg font-semibold text-gray-800 dark:text-white">Issues & Alerts</h2>
            </div>
            <div className="p-4">
                {loading ? (
                    <div className="flex justify-center py-8">
                        <div className="flex flex-col items-center">
                            <svg className="animate-spin h-8 w-8 text-blue-600 dark:text-blue-400 mb-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            <p className="text-gray-600 dark:text-gray-300">Loading issues...</p>
                        </div>
                    </div>
                ) : !issues || issues.length === 0 ? (
                    <div className="text-center py-8">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 mx-auto text-green-500 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <p className="text-gray-600 dark:text-gray-300">No issues detected</p>
                        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">All systems are operating normally</p>
                    </div>
                ) : (
                    <div className="space-y-4 max-h-[600px] overflow-y-auto pr-2">
                        {issues.map(issue => (
                            <div 
                                key={issue.id || `${issue.type}-${issue.detected_at}`} 
                                className={`border-l-4 rounded-r-lg shadow-sm p-4 hover-card ${getSeverityClass(issue.severity)} ${issue.status === 'detected' && issue.severity >= 4 ? 'pulse' : ''}`}
                            >
                                <div className="flex justify-between items-start">
                                    <h3 className="font-medium text-gray-900 dark:text-gray-100">
                                        {formatIssueType(issue.type)}
                                    </h3>
                                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium`}>
                                        Severity {issue.severity}/5
                                    </span>
                                </div>
                                <p className="text-gray-600 dark:text-gray-300 text-sm mt-1">{issue.description}</p>
                                
                                <div className="mt-3 flex items-center">
                                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusBadgeClass(issue.status)}`}>
                                        {issue.status ? issue.status.toUpperCase() : 'DETECTED'}
                                    </span>
                                    {issue.status === 'mitigating' && (
                                        <div className="ml-2 flex-grow bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                                            <div 
                                                className={`${getProgressBarClass(issue.status)} h-2 rounded-full`} 
                                                style={{ width: `${issue.progress || 50}%` }}
                                            ></div>
                                        </div>
                                    )}
                                </div>
                                
                                <div className="mt-3 text-xs text-gray-500 dark:text-gray-400">
                                    <div className="flex justify-between">
                                        <span>Detected: {formatTime(issue.detected_at)}</span>
                                        {issue.mitigated_at && <span>Mitigated: {formatTime(issue.mitigated_at)}</span>}
                                    </div>
                                </div>
                                
                                <div className="mt-3">
                                    <button
                                        onClick={() => toggleDetails(issue.id)}
                                        className="text-xs flex items-center text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100"
                                    >
                                        <span>{expandedIssues.has(issue.id) ? 'Hide' : 'Show'} details</span>
                                        <svg
                                            xmlns="http://www.w3.org/2000/svg"
                                            className={`h-4 w-4 ml-1 transition-transform ${expandedIssues.has(issue.id) ? 'rotate-180' : ''}`}
                                            viewBox="0 0 20 20"
                                            fill="currentColor"
                                        >
                                            <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                                        </svg>
                                    </button>
                                    {expandedIssues.has(issue.id) && (
                                        <div className="mt-2 text-xs bg-white dark:bg-gray-700 bg-opacity-50 dark:bg-opacity-50 p-2 rounded">
                                            {issue.affected_nodes && issue.affected_nodes.length > 0 && (
                                                <div className="mb-1">
                                                    <span className="font-medium">Affected Services:</span>
                                                    <span className="ml-1">{issue.affected_nodes.join(', ')}</span>
                                                </div>
                                            )}
                                            {issue.affected_edges && issue.affected_edges.length > 0 && (
                                                <div className="mb-1">
                                                    <span className="font-medium">Affected Dependencies:</span>
                                                    <span className="ml-1">{formatEdges(issue.affected_edges)}</span>
                                                </div>
                                            )}
                                            {issue.mitigation_action && (
                                                <div className="mb-1">
                                                    <span className="font-medium">Mitigation Action:</span>
                                                    <span className="ml-1">{issue.mitigation_action}</span>
                                                </div>
                                            )}
                                        </div>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};