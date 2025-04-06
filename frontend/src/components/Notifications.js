// Notifications Component for Aegis Sentinel

const Notifications = ({ notifications }) => {
    // Get notification icon based on type
    const getNotificationIcon = (type) => {
        switch (type) {
            case 'error':
                return (
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                    </svg>
                );
            case 'success':
                return (
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-green-400" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                );
            case 'warning':
                return (
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                    </svg>
                );
            default:
                return (
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-blue-400" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                    </svg>
                );
        }
    };
    
    // Get notification background color based on type
    const getNotificationBgColor = (type) => {
        switch (type) {
            case 'error':
                return 'bg-red-50 dark:bg-red-900/30 border-red-400 dark:border-red-800';
            case 'success':
                return 'bg-green-50 dark:bg-green-900/30 border-green-400 dark:border-green-800';
            case 'warning':
                return 'bg-yellow-50 dark:bg-yellow-900/30 border-yellow-400 dark:border-yellow-800';
            default:
                return 'bg-blue-50 dark:bg-blue-900/30 border-blue-400 dark:border-blue-800';
        }
    };
    
    return (
        <div className="fixed top-4 right-4 z-50 space-y-3">
            {notifications.map(notification => (
                <div 
                    key={notification.id} 
                    className={`notification show border rounded-lg shadow-lg ${getNotificationBgColor(notification.type)}`}
                >
                    <div className="flex items-center p-3">
                        {getNotificationIcon(notification.type)}
                        <span className="ml-3 text-sm font-medium text-gray-800 dark:text-gray-200">{notification.message}</span>
                    </div>
                </div>
            ))}
        </div>
    );
};