// Header Component for Aegis Sentinel

const Header = ({ darkMode, toggleDarkMode, overallStatus, statusMessage }) => {
    // Get status color based on overall status
    const getStatusColor = (status) => {
        switch (status) {
            case 'healthy':
                return 'bg-green-500';
            case 'warning':
                return 'bg-yellow-500';
            case 'critical':
                return 'bg-red-500';
            default:
                return 'bg-gray-500';
        }
    };

    return (
        <header className={`${darkMode ? 'bg-gray-900' : 'bg-gradient-to-r from-blue-600 to-indigo-700'} text-white shadow-lg`}>
            <div className="container mx-auto px-4 py-4 flex justify-between items-center">
                <div className="flex items-center space-x-4">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M11.3 1.046A1 1 0 0112 2v5h4a1 1 0 01.82 1.573l-7 10A1 1 0 018 18v-5H4a1 1 0 01-.82-1.573l7-10a1 1 0 011.12-.38z" clipRule="evenodd" />
                    </svg>
                    <div>
                        <h1 className="text-2xl font-bold">Aegis Sentinel</h1>
                        <p className="text-sm text-blue-100">Advanced SRE Platform</p>
                    </div>
                </div>
                
                <div className="flex items-center space-x-4">
                    {/* Status indicator */}
                    <div className="flex items-center space-x-2 bg-white bg-opacity-20 rounded-full px-3 py-1">
                        <span className={`inline-block w-3 h-3 rounded-full ${getStatusColor(overallStatus)} ${overallStatus !== 'healthy' ? 'animate-pulse' : ''}`}></span>
                        <span className="text-sm font-medium">{statusMessage}</span>
                    </div>
                    
                    {/* Dark mode toggle */}
                    <button 
                        onClick={toggleDarkMode} 
                        className="bg-white bg-opacity-20 hover:bg-opacity-30 rounded-full p-2 transition-colors"
                        aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
                    >
                        {darkMode ? (
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                                <path fillRule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" clipRule="evenodd" />
                            </svg>
                        ) : (
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                                <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
                            </svg>
                        )}
                    </button>
                    
                    {/* Refresh button */}
                    <button 
                        onClick={() => window.location.reload()} 
                        className="bg-white bg-opacity-20 hover:bg-opacity-30 rounded-full p-2 transition-colors"
                        aria-label="Refresh data"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clipRule="evenodd" />
                        </svg>
                    </button>
                </div>
            </div>
            
            {/* Navigation */}
            <nav className={`${darkMode ? 'bg-gray-800' : 'bg-indigo-800'} px-4 py-2`}>
                <div className="container mx-auto flex space-x-6">
                    <a href="#" className="text-white hover:text-blue-200 font-medium px-2 py-1 rounded transition-colors">Dashboard</a>
                    <a href="#" className="text-blue-200 hover:text-white font-medium px-2 py-1 rounded transition-colors">Services</a>
                    <a href="#" className="text-blue-200 hover:text-white font-medium px-2 py-1 rounded transition-colors">Issues</a>
                    <a href="#" className="text-blue-200 hover:text-white font-medium px-2 py-1 rounded transition-colors">Analytics</a>
                    <a href="#" className="text-blue-200 hover:text-white font-medium px-2 py-1 rounded transition-colors">Settings</a>
                </div>
            </nav>
        </header>
    );
};