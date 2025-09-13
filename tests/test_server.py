import pytest
from unittest.mock import patch, MagicMock
import server
import threading
import time
import requests

@patch('server.socketserver.TCPServer')
def test_run_http_server(mock_tcp_server):
    mock_server_instance = MagicMock()
    mock_tcp_server.return_value.__enter__.return_value = mock_server_instance

    server_thread = threading.Thread(target=server.run_http_server, args=(8001, '.'))
    server_thread.daemon = True
    server_thread.start()
    time.sleep(0.1) # Give the thread time to start

    mock_tcp_server.assert_called_with(("", 8001), mock_server_instance.RequestHandlerClass)
    mock_server_instance.serve_forever.assert_called()

def test_server_actually_serves_file(tmp_path):
    # Create a dummy file to serve
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "hello.txt"
    p.write_text("hello world")

    port = 8002
    server_thread = threading.Thread(target=server.run_http_server, args=(port, d))
    server_thread.daemon = True
    server_thread.start()
    time.sleep(0.1) # Give the server time to start

    try:
        response = requests.get(f"http://localhost:{port}/hello.txt")
        assert response.status_code == 200
        assert response.text == "hello world"
    finally:
        # The server is a daemon thread, so it will shut down with the main thread.
        # No explicit shutdown needed for this test.
        pass
