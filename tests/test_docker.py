import unittest
import asyncio
from unittest.mock import MagicMock, patch, mock_open
from comind.environment.docker import Docker, DockerConfig
import os

class TestDocker(unittest.TestCase):

    def setUp(self):
        """Set up for the tests."""
        # Patch docker.from_env before initializing Docker manager
        self.patcher = patch('comind.environment.docker.docker.from_env')
        self.mock_from_env = self.patcher.start()
        
        # Configure the mock client that docker.from_env() will return
        self.mock_docker_client = MagicMock()
        self.mock_from_env.return_value = self.mock_docker_client

        self.mock_config = DockerConfig(
            image="python:3.9-slim",
            mounts=["./test_workspace:/app"],
            code_path="/app/test_script.py",
            timeout=60
        )

        # Create a dummy workspace directory for testing mounts
        if not os.path.exists("./test_workspace"):
            os.makedirs("./test_workspace")

        self.docker_manager = Docker(self.mock_config)

    def tearDown(self):
        """Tear down after tests."""
        self.patcher.stop()
        # Clean up the dummy workspace
        if os.path.exists("./test_workspace"):
            import shutil
            shutil.rmtree("./test_workspace")
        # Ensure container is stopped (will use mock if container was 'run')
        if self.docker_manager._container:
            self.docker_manager.stop_container()

    def test_initialization(self):
        """Test that the Docker manager initializes correctly."""
        self.assertIsNotNone(self.docker_manager)
        self.assertEqual(self.docker_manager.config, self.mock_config)
        self.mock_from_env.assert_called_once()
        self.assertEqual(self.docker_manager.client, self.mock_docker_client)

    def test_run_program_and_get_logs(self):
        """Test running a simple python program and retrieving logs."""
        
        async def run_test():
            # A simple python script
            code = "print('hello world')"
            
            # Mock the container and its methods
            mock_container = MagicMock()
            mock_container.id = "test_container_id"
            mock_container.status = 'running'

            # Configure the mocked docker client from setUp to simulate running a container
            self.mock_docker_client.containers.run.return_value = mock_container

            # Mock exec_create and exec_inspect on the 'api' attribute of the client
            exec_create_response = {'Id': 'exec_id'}
            self.mock_docker_client.api.exec_create.return_value = exec_create_response
            self.mock_docker_client.api.exec_inspect.side_effect = [
                {'Running': True},                  # First call inside the while loop
                {'Running': False},                 # Second call, breaks the loop
                {'ExitCode': 0}                     # Third call for the return value
            ]
            
            # Mock the log stream to be an iterator
            log_stream = iter([(b'hello world\n', None)])
            self.mock_docker_client.api.exec_start.return_value = log_stream
            
            # We need to mock open to avoid writing to the actual file system in the test
            with patch('builtins.open', mock_open()) as mock_file:
                exit_code = await self.docker_manager.run_program(code)
                logs = await self.docker_manager.get_logs()
                
                self.assertEqual(exit_code, 0)
                self.assertIn("hello world", logs)

        # Run the async test
        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main() 