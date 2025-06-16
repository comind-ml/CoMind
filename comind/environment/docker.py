import asyncio
import docker
from docker.types import DeviceRequest, Mount
from typing import Optional, List
from comind.core.config import DockerConfig
from comind.core.logger import logger
import time
import os

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
        self.config = config
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
            host, container = m.split(':', 1)
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
        mounts = [Mount(target=m.split(':')[1], source=m.split(':')[0], type='bind') for m in self.config.mounts]
        device_requests = None

        if self.config.gpu:
            device_requests = [
                DeviceRequest(
                    count=-1 if self.config.gpu == 'all' else 1, 
                    capabilities=[["gpu"]], 
                    device_ids=None if self.config.gpu == 'all' else [self.config.gpu]
                )
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
            host, container = m.split(':')
            if container == self.config.code_path.rsplit('/', 1)[0]:
                host_code_path = host
                break

        if not host_code_path:
            logger.error('Code path is not correctly mounted to the host.')
            raise RuntimeError('Code path is not correctly mounted to the host.')

        code_file = self.config.code_path
        with open(f"{host_code_path}/{code_file.rsplit('/',1)[-1]}", 'w', encoding='utf-8') as f:
            f.write(code)
        logger.info(f'Code written to {host_code_path}/{code_file.rsplit('/',1)[-1]}')

        # Build the execution command with timeout
        exec_cmd = self._get_exec_command(code_file)
        timeout_cmd = f"timeout {timeout}s {exec_cmd}"
        logger.info(f'Execution command: {timeout_cmd}')
        exec_instance = self.client.api.exec_create(self._container.id, timeout_cmd, stdout=True, stderr=True)
        self._log_task = asyncio.create_task(self._collect_exec_logs(exec_instance['Id']))

        # Wait for execution to finish
        while True:
            inspect = self.client.api.exec_inspect(exec_instance['Id'])
            if not inspect['Running']:
                break
            await asyncio.sleep(0.5)

        await self._log_task
        return self.client.api.exec_inspect(exec_instance['Id'])['ExitCode']

    def _get_exec_command(self, code_file: str) -> str:
        """
        Return the python execution command for the given file.
        Only .py files are supported.
        """
        if not code_file.endswith('.py'):
            logger.error('Only Python (.py) files are supported for execution in this container.')
            raise ValueError('Only Python (.py) files are supported for execution in this container.')
        return f"python {code_file}"

    async def get_logs(self) -> str:
        """
        Asynchronously get all logs collected from the most recent execution.
        """
        await self._log_event.wait()
        self._log_event.clear()
        return ''.join(self._log_buffer)

    async def _collect_exec_logs(self, exec_id):
        """
        Collect logs from the running exec instance asynchronously.
        """
        log_stream = self.client.api.exec_start(exec_id, stream=True, demux=True)
        async for out, err in self._async_log_stream(log_stream):
            if out:
                line = out.decode(errors='ignore')
                self._log_buffer.append(line)
                logger.debug(line.strip())
            if err:
                line = err.decode(errors='ignore')
                self._log_buffer.append(line)
                logger.error(line.strip())
            self._log_event.set()
            await asyncio.sleep(0)

    async def _async_log_stream(self, log_stream):
        """
        Helper to make a blocking log_stream async.
        """
        loop = asyncio.get_event_loop()
        while True:
            try:
                item = await loop.run_in_executor(None, next, log_stream, None)
                if item is None:
                    break
                yield item
            except StopIteration:
                break
