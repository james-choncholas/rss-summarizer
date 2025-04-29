import http.server
import socketserver
import threading
import os

# Import logger and constants from config
from config import logger, DEFAULT_OUTPUT_FEED_FILE # Need default file for log message

class QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler subclass that suppresses log messages."""
    def log_message(self, format, *args):
        # Override to do nothing, suppressing GET/HEAD request logs
        pass

def run_http_server(port, directory="."):
    """Runs a simple HTTP server in the specified directory."""
    # Ensure the target directory exists
    os.makedirs(directory, exist_ok=True)

    # Use a Handler that doesn't log every request to avoid clutter
    Handler = QuietHTTPRequestHandler

    # Allow address reuse to prevent errors on quick restarts
    socketserver.TCPServer.allow_reuse_address = True

    # Change directory *within the thread* if needed, or configure the handler
    # For SimpleHTTPRequestHandler, it serves from the CWD *when the server starts*.
    # A more robust solution would involve a custom handler or framework.
    # For now, assume the script runs from the desired serving directory.

    logger.info(f"Starting HTTP server on port {port} to serve files from '{os.path.abspath(directory)}'.")
    # Construct the likely feed URL for the log message
    # Note: This assumes the server runs on localhost. A more dynamic approach
    # might be needed if serving externally.
    feed_url = f"http://localhost:{port}/{DEFAULT_OUTPUT_FEED_FILE}"
    logger.info(f"Access the generated feed typically at {feed_url}")

    # Create the server instance
    # Need to use functools.partial if the handler needs arguments like directory
    # httpd = socketserver.TCPServer(("", port), Handler)
    # For SimpleHTTPRequestHandler, serving directory is handled differently.
    # We bind the handler to the directory *before* creating the server.
    import functools
    Handler = functools.partial(QuietHTTPRequestHandler, directory=directory)

    with socketserver.TCPServer(("", port), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            # This might not be caught here if the main thread catches it first
            logger.info("HTTP server received KeyboardInterrupt (likely from main thread). Shutting down...")
        except Exception as e:
            # Log unexpected errors in the server thread
            logger.error(f"HTTP server error: {e}", exc_info=True)
        finally:
            # Ensure cleanup happens
            httpd.server_close() # Close the server socket
            logger.info(f"HTTP server on port {port} stopped.")

# Example of how to run this server in a thread (similar to main.py)
# if __name__ == '__main__':
#     PORT = 8000
#     DIRECTORY = "."
#     server_thread = threading.Thread(
#         target=run_http_server,
#         args=(PORT, DIRECTORY),
#         daemon=True
#     )
#     server_thread.start()
#     print(f"Server started on port {PORT}. Press Ctrl+C to stop.")
#     try:
#         while server_thread.is_alive():
#             server_thread.join(timeout=1.0) # Wait for thread
#     except KeyboardInterrupt:
#         print("\nShutting down...") 