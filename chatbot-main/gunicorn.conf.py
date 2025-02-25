import multiprocessing

# Server socket
bind = '0.0.0.0:5000'  # Bind to all IP addresses on port 5000
backlog = 2048  # The maximum number of pending connections


# Number of workers
workers = multiprocessing.cpu_count() * 2 + 1  # Automatically set number of workers to handle multiple requests
#worker_class = "uvicorn.workers.UvicornWorker"  # Asynchronous worker class for better performance

# Log configuration
accesslog = '-'  # Output access logs to stdout
errorlog = '-'  # Output error logs to stdout
loglevel = 'info'  # Set log level to 'info', 'debug', 'warning', or 'error'

# Timeout
timeout = 130  # Workers silent for more than this many seconds are killed and restarted

# Graceful timeout
graceful_timeout = 20  # Seconds to gracefully restart workers
