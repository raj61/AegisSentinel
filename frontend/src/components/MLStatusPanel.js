// MLStatusPanel Component for Aegis Sentinel

const MLStatusPanel = ({ darkMode, mlStatus, mlErrors, trainModels, showMlGraph, toggleMlGraph }) => {
    // Function to inject a synthetic anomaly
    const injectAnomaly = async () => {
        try {
            const response = await fetch('/api/inject-anomaly', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    type: 'synthetic'
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to inject anomaly');
            }
            
            // Show notification
            const notification = document.createElement('div');
            notification.className = `fixed top-4 right-4 bg-blue-500 text-white px-4 py-2 rounded shadow-lg z-50 ${darkMode ? 'dark' : ''}`;
            notification.textContent = 'Synthetic anomaly injected!';
            document.body.appendChild(notification);
            
            // Remove notification after 3 seconds
            setTimeout(() => {
                notification.remove();
            }, 3000);
        } catch (error) {
            console.error('Error injecting anomaly:', error);
        }
    };
    // Get status badge class
    const getStatusBadgeClass = (status) => {
        switch (status) {
            case 'active':
                return 'bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-200';
            case 'training':
                return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-200';
            case 'error':
                return 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-200';
            default:
                return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200';
        }
    };
    
    // Check if there are any ML errors
    const hasMLErrors = mlErrors && mlErrors.length > 0;
    
    return (
        <div className="mt-6 bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden">
            <div className="p-4 bg-gray-50 dark:bg-gray-700 border-b dark:border-gray-600">
                <h2 className="text-lg font-semibold text-gray-800 dark:text-white">ML System Status</h2>
            </div>
            <div className="p-4">
                <div className="space-y-4">
                    <div className="flex items-center justify-between">
                        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Anomaly Detection</span>
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusBadgeClass(mlStatus.anomalyDetection)}`}>
                            {mlStatus.anomalyDetection.toUpperCase()}
                        </span>
                    </div>
                    <div className="flex items-center justify-between">
                        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Root Cause Analysis</span>
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusBadgeClass(mlStatus.rootCauseAnalysis)}`}>
                            {mlStatus.rootCauseAnalysis.toUpperCase()}
                        </span>
                    </div>
                    <div className="flex items-center justify-between">
                        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Remediation Learning</span>
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusBadgeClass(mlStatus.remediationLearning)}`}>
                            {mlStatus.remediationLearning.toUpperCase()}
                        </span>
                    </div>
                    <div className="flex items-center justify-between">
                        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Prediction Engine</span>
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusBadgeClass(mlStatus.predictionEngine)}`}>
                            {mlStatus.predictionEngine.toUpperCase()}
                        </span>
                    </div>
                    
                    {hasMLErrors && (
                        <div className="mt-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
                            <div className="flex">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-red-400 dark:text-red-500 mr-2" viewBox="0 0 20 20" fill="currentColor">
                                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                                </svg>
                                <div>
                                    <p className="text-sm font-medium text-red-800 dark:text-red-200">ML System Warnings</p>
                                    <ul className="mt-1 text-xs text-red-700 dark:text-red-300 list-disc list-inside">
                                        {mlErrors.map((error, index) => (
                                            <li key={index}>{error}</li>
                                        ))}
                                    </ul>
                                    <button 
                                        onClick={trainModels} 
                                        className="mt-2 text-xs bg-red-100 hover:bg-red-200 dark:bg-red-800 dark:hover:bg-red-700 text-red-800 dark:text-red-200 font-medium py-1 px-2 rounded transition-colors"
                                    >
                                        Train Models
                                    </button>
                                </div>
                            </div>
                        </div>
                    )}
                    
                    <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-md">
                        <div className="flex justify-between items-center mb-2">
                            <h3 className="text-sm font-medium text-blue-800 dark:text-blue-200">ML System Capabilities</h3>
                            <div className="flex space-x-2">
                                <button
                                    onClick={injectAnomaly}
                                    className="text-xs bg-red-100 hover:bg-red-200 dark:bg-red-800 dark:hover:bg-red-700 text-red-800 dark:text-red-200 font-medium py-1 px-2 rounded transition-colors"
                                >
                                    Inject Anomaly
                                </button>
                                <button
                                    onClick={toggleMlGraph}
                                    className={`text-xs ${showMlGraph ? 'bg-blue-500 hover:bg-blue-600 text-white' : 'bg-blue-100 hover:bg-blue-200 text-blue-800 dark:bg-blue-800 dark:hover:bg-blue-700 dark:text-blue-200'} font-medium py-1 px-2 rounded transition-colors`}
                                >
                                    {showMlGraph ? 'Hide ML Graph' : 'Show ML Graph'}
                                </button>
                            </div>
                        </div>
                        <ul className="text-xs text-blue-700 dark:text-blue-300 space-y-1">
                            <li className="flex items-start">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1 mt-0.5 text-blue-500" viewBox="0 0 20 20" fill="currentColor">
                                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                                </svg>
                                <span>Anomaly detection in metrics and logs</span>
                            </li>
                            <li className="flex items-start">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1 mt-0.5 text-blue-500" viewBox="0 0 20 20" fill="currentColor">
                                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                                </svg>
                                <span>Automated root cause analysis</span>
                            </li>
                            <li className="flex items-start">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1 mt-0.5 text-blue-500" viewBox="0 0 20 20" fill="currentColor">
                                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                                </svg>
                                <span>Reinforcement learning for remediation</span>
                            </li>
                            <li className="flex items-start">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1 mt-0.5 text-blue-500" viewBox="0 0 20 20" fill="currentColor">
                                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                                </svg>
                                <span>Predictive analytics for proactive issue prevention</span>
                            </li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    );
};