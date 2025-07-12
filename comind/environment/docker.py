import asyncio
import docker
from docker.types import DeviceRequest, Mount
from typing import Optional, List
from comind.core.config import DockerConfig
from comind.core.logger import logger
import time
import os
from copy import deepcopy

class Docker:
    """
    Docker manager for running Python code in a persistent container.
    Only supports .py files. Uses Ubuntu's `timeout` for execution limits.
    """

    def __init__(self, config: DockerConfig):
        """
        Initialize Docker manager and validate configuration.
        """
        self._validate_config(config)
        self.config = deepcopy(config)
        self.client = docker.from_env()
        self._container = None
        self._log_buffer: List[str] = []
        self._log_task: Optional[asyncio.Task] = None
        self._log_event = asyncio.Event()
        self._container_ready = False

    def _validate_config(self, config: DockerConfig):
        """
        Validate DockerConfig fields for correctness.
        """
        if not config.image or not isinstance(config.image, str):
            logger.error('DockerConfig.image must be a non-empty string.')
            raise ValueError('DockerConfig.image must be a non-empty string.')
        if not isinstance(config.mounts, list):
            logger.error('DockerConfig.mounts must be a list.')
            raise ValueError('DockerConfig.mounts must be a list.')
        for m in config.mounts:
            if ':' not in m:
                logger.error(f'Invalid mount format: {m}. Should be "/host/path:/container/path".')
                raise ValueError(f'Invalid mount format: {m}. Should be "/host/path:/container/path".')
            
            # Handle Windows paths properly (e.g., C:/path:/container/path)
            host, container = self._parse_mount_string(m)
            
            if not os.path.exists(host):
                logger.error(f'Host mount path does not exist: {host}')
                raise ValueError(f'Host mount path does not exist: {host}')
        if not config.code_path or not isinstance(config.code_path, str):
            logger.error('DockerConfig.code_path must be a non-empty string.')
            raise ValueError('DockerConfig.code_path must be a non-empty string.')
        if not config.code_path.endswith('.py'):
            logger.error('DockerConfig.code_path must end with .py since only Python code is supported.')
            raise ValueError('DockerConfig.code_path must end with .py since only Python code is supported.')
        if config.cpu is not None:
            if not isinstance(config.cpu, list) or not all(isinstance(i, int) and i >= 0 for i in config.cpu):
                logger.error('DockerConfig.cpu must be a list of non-negative integers.')
                raise ValueError('DockerConfig.cpu must be a list of non-negative integers.')
        if config.gpu is not None:
            if not (isinstance(config.gpu, str) and (config.gpu == 'all' or config.gpu.isdigit())):
                logger.error('DockerConfig.gpu must be "all" or a string representing a non-negative integer.')
                raise ValueError('DockerConfig.gpu must be "all" or a string representing a non-negative integer.')

    def _parse_mount_string(self, mount_str: str) -> tuple[str, str]:
        """
        Parse mount string handling Windows paths properly.
        Examples:
        - "C:/path:/container/path" -> ("C:/path", "/container/path")
        - "/unix/path:/container/path" -> ("/unix/path", "/container/path")
        """
        # Find the last occurrence of ':' that's not part of a Windows drive letter
        # Windows drive letters are like "C:", "D:", etc.
        parts = mount_str.split(':')
        
        if len(parts) < 2:
            raise ValueError(f'Invalid mount format: {mount_str}')
        
        # Check if this looks like a Windows path (drive letter followed by path)
        if len(parts) >= 3 and len(parts[0]) == 1 and parts[0].isalpha():
            # This is likely a Windows path like "C:/path:/container/path"
            host = parts[0] + ':' + parts[1]  # "C:/path"
            container = ':'.join(parts[2:])   # "/container/path"
        else:
            # Unix-style path like "/unix/path:/container/path"
            host = parts[0]
            container = ':'.join(parts[1:])
        
        return host, container

    def _ensure_container(self):
        """
        Ensure the persistent container is running, create if needed.
        """
        if self._container is not None:
            try:
                self._container.reload()
                if self._container.status == 'running':
                    logger.debug('Container is already running.')
                    return
            except Exception as e:
                logger.warning(f'Failed to reload container: {e}')
        logger.info('Starting a new container...')
        mounts = []
        for m in self.config.mounts:
            host, container = self._parse_mount_string(m)
            mounts.append(Mount(target=container, source=host, type='bind'))
        device_requests = None

        if self.config.gpu:
            if self.config.gpu == "all":
                device_requests = [
                    DeviceRequest(count=-1, capabilities=[["gpu"]])
                ]
            else:
                device_requests = [
                    DeviceRequest(device_ids=[self.config.gpu], capabilities=[["gpu"]])
                ]

        host_config = {}
        if self.config.cpu:
            host_config['cpuset_cpus'] = ','.join(str(i) for i in self.config.cpu)
        self._container = self.client.containers.run(
            self.config.image,
            command="tail -f /dev/null",  # keep container running
            detach=True,
            stdin_open=True,
            tty=True,
            mounts=mounts,
            device_requests=device_requests,
            **host_config
        )
        self._container_ready = True
        logger.info(f'Container started: {self._container.id}')

    def stop_container(self):
        """
        Stop and remove the persistent container.
        """
        if self._container:
            logger.info('Stopping container...')
            self._container.stop()
            self._container.remove(force=True)
            self._container = None
            self._container_ready = False
            logger.info('Container stopped and removed.')

    async def run_program(self, code: str, timeout: int = None):
        """
        Run Python code inside the persistent container, using Ubuntu's `timeout` command.
        Only .py files are supported. Logs and returns the exit code.
        :param code: Python code as a string
        :param timeout: Maximum run time in seconds
        :return: Exit code of the execution
        """
        if not self.config.code_path.endswith('.py'):
            logger.error('Only Python (.py) files are supported for execution in this container.')
            raise ValueError('Only Python (.py) files are supported for execution in this container.')
        
        if timeout is None:
            timeout = self.config.timeout

        self._log_buffer = []
        self._log_event.clear()
        self._ensure_container()

        # Save code to the mounted host directory
        host_code_path = None
        for m in self.config.mounts:
            host, container = self._parse_mount_string(m)
            if container == self.config.code_path.rsplit('/', 1)[0]:
                host_code_path = host
                break

        if not host_code_path:
            logger.error('Code path is not correctly mounted to the host.')
            raise RuntimeError('Code path is not correctly mounted to the host.')

        code_file = self.config.code_path
        with open(f"{host_code_path}/{code_file.rsplit('/',1)[-1]}", 'w', encoding='utf-8') as f:
            f.write(code)
        
        code_name = code_file.rsplit('/',1)[-1]
        logger.info(f'Code written to {host_code_path}/{code_name}')

        # Build the execution command with timeout
        exec_cmd = self._get_exec_command(code_file)
        timeout_cmd = f"timeout {timeout}s {exec_cmd}"
        logger.info(f'Execution command: {timeout_cmd}')
        
        # Set working directory to the directory containing the code file
        working_dir = self.config.code_path.rsplit('/', 1)[0]  # e.g., "/workspace" from "/workspace/main.py"
        exec_instance = self.client.api.exec_create(
            self._container.id, 
            timeout_cmd, 
            stdout=True, 
            stderr=True,
            workdir=working_dir
        )
        
        # Start execution and log collection
        start_time = time.time()
        logger.info(f"Starting execution and log collection...")
        
        # Start execution and collect logs with better timeout handling
        try:
            # Start the execution and get the stream
            exec_stream = self.client.api.exec_start(exec_instance['Id'], stream=True, demux=True)
            
            # Collect logs with timeout handling
            log_collection_task = asyncio.create_task(self._collect_logs_from_stream(exec_stream))
            
            # Wait for the log collection to complete, but with a timeout buffer
            try:
                await asyncio.wait_for(log_collection_task, timeout=timeout + 2)
            except asyncio.TimeoutError:
                logger.warning("Log collection timed out, canceling task")
                log_collection_task.cancel()
                try:
                    await log_collection_task
                except asyncio.CancelledError:
                    pass
            
            logger.debug(f"Log collection completed, collected {len(self._log_buffer)} lines")
            
        except Exception as e:
            logger.warning(f"Error during execution/log collection: {e}")
        
        # Get the final execution result
        elapsed = time.time() - start_time
        exit_code = 1  # Default error code
        try:
            inspect_result = self.client.api.exec_inspect(exec_instance['Id'])
            exit_code = inspect_result['ExitCode']
            logger.info(f"Execution completed in {elapsed:.1f}s with exit code: {exit_code}")
        except Exception as e:
            logger.error(f"Error getting exit code: {e}")

        # Ensure the log event is set so get_logs() won't block
        self._log_event.set()
        logger.debug(f"Execution finished, final log buffer has {len(self._log_buffer)} lines")

        return exit_code

    def _get_exec_command(self, code_file: str) -> str:
        """
        Return the python execution command for the given file.
        Only .py files are supported.
        """
        if not code_file.endswith('.py'):
            logger.error('Only Python (.py) files are supported for execution in this container.')
            raise ValueError('Only Python (.py) files are supported for execution in this container.')
        return f"python -u {code_file}"

    async def get_logs(self) -> str:
        """
        Asynchronously get all logs collected from the most recent execution.
        """
        try:
            # Wait for logs with a timeout to avoid infinite waiting
            await asyncio.wait_for(self._log_event.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            # If timeout, just return what we have so far
            logger.debug("get_logs() timeout, returning current buffer")
        
        # Don't clear the event here since it might be needed by other calls
        return ''.join(self._log_buffer)

    async def _collect_logs_from_stream(self, exec_stream):
        """
        Collect logs from execution stream with better handling of timeout scenarios.
        """
        try:
            # Collect all logs from the stream
            async for out, err in self._async_log_stream(exec_stream):
                if out:
                    line = out.decode(errors='ignore')
                    self._log_buffer.append(line)
                    logger.debug(f"STDOUT: {line.strip()}")
                if err:
                    line = err.decode(errors='ignore')
                    self._log_buffer.append(line)
                    logger.debug(f"STDERR: {line.strip()}")
                self._log_event.set()
            
            logger.debug("Stream log collection completed normally")
            
        except Exception as e:
            logger.warning(f"Error in stream log collection: {e}")
        finally:
            # Always set the event
            self._log_event.set()

    async def _async_log_stream(self, log_stream):
        """
        Helper to make a blocking log_stream async.
        """
        loop = asyncio.get_event_loop()
        items_yielded = 0
        try:
            while True:
                try:
                    # Read from the stream without timeout (handled at higher level)
                    item = await loop.run_in_executor(None, next, log_stream, None)
                    if item is None:
                        logger.debug(f"Log stream ended naturally after {items_yielded} items")
                        break
                    yield item
                    items_yielded += 1
                except StopIteration:
                    logger.debug(f"Log stream StopIteration after {items_yielded} items")
                    break
                except Exception as e:
                    logger.warning(f"Error in log stream after {items_yielded} items: {e}")
                    break
        finally:
            # Ensure the stream is properly closed
            try:
                log_stream.close()
                logger.debug("Log stream closed")
            except Exception as e:
                logger.debug(f"Error closing log stream: {e}")
