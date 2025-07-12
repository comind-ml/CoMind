import asyncio
import tempfile
import os
import shutil
import select
import pty
import time
from typing import Optional, List, Dict, Tuple
from pathlib import Path

class Executor:
    def __init__(self, working_dir: str, timeout: int = 3600, use_pty: bool = False):
        self.working_dir = working_dir
        self.timeout = timeout
        self.use_pty = use_pty
        os.makedirs(self.working_dir, exist_ok=True)

    async def run(self, code: str, agent_file_name: str = None) -> Dict[str, Optional[str | List[str]]]:
        file_path = self._write_temp_code(code, agent_file_name)
        start_time = time.time()

        try:
            if self.use_pty:
                result = await self._run_with_pty(file_path)
            else:
                result = await self._run_with_subprocess(file_path)
            end_time = time.time()
            result["execution_time"] = end_time - start_time
            return result
        finally:
            self._cleanup(file_path)

    async def _run_with_subprocess(self, file_path: str) -> Dict[str, Optional[str | List[str]]]:
        env = os.environ.copy()

        process = await asyncio.create_subprocess_exec(
            "python", file_path,
            cwd=self.working_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )

        output_lines = []

        async def read_stream(stream, prefix):
            while True:
                line = await stream.readline()
                if line:
                    text = line.decode().rstrip()
                    output_lines.append(text)
                else:
                    break

        try:
            await asyncio.wait_for(
                asyncio.gather(
                    read_stream(process.stdout, "[OUT] "),
                    read_stream(process.stderr, "[ERR] "),
                    process.wait()
                ),
                timeout=self.timeout
            )

            success = process.returncode == 0
            return {
                "success": success,
                "output": "\n".join(output_lines) if success else "",
                "error": None if success else "RuntimeError"
            }
        except asyncio.TimeoutError:
            process.kill()
            return {
                "success": False,
                "output": "\n".join(output_lines),
                "error": "Timeout"
            }

    async def _run_with_pty(self, file_path: str) -> Dict[str, Optional[str | List[str]]]:
        loop = asyncio.get_event_loop()
        status, output = await loop.run_in_executor(None, self._sync_pty_exec, file_path)
        return {
            "success": True if status == 'Exited' else False,
            "output": output,
            "error": None if status == 'Exited' else status
        }

    def _sync_pty_exec(self, file_path: str) -> Tuple[str, str]:
        master_fd, slave_fd = pty.openpty()
        pid = os.fork()
        if pid == 0:
            os.setsid()
            os.dup2(slave_fd, 0)
            os.dup2(slave_fd, 1)
            os.dup2(slave_fd, 2)
            os.chdir(self.working_dir)
            os.execvp("python", ["python", file_path])
        else:
            os.close(slave_fd)
            output, status = "", ""
            try:
                deadline = time.time() + self.timeout
                while True:
                    if time.time() > deadline:
                        os.kill(pid, 9)
                        status = "Timeout"
                        break
                    r, _, _ = select.select([master_fd], [], [], 10)
                    if r:
                        try:
                            data = os.read(master_fd, 1024).decode(errors="ignore")
                            output += data
                        except OSError:
                            pass

                    pid_ret, code = os.waitpid(pid, os.WNOHANG)
                    if pid_ret != 0:
                        status = "Exited" if os.WIFEXITED(code) else "RuntimeError"
                        break
            finally:
                os.close(master_fd)

            return status, output

    def _write_temp_code(self, code: str, agent_file_name: str = None) -> str:
        if agent_file_name:
            with open(Path(self.working_dir) / agent_file_name, 'w') as f:
                f.write(code)
            return agent_file_name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', dir=self.working_dir, delete=False) as f:
            f.write(code)
            return f.name

    def _indent_code(self, code: str, indent: str = "    ") -> str:
        return "\n".join(indent + line if line.strip() else line for line in code.splitlines())

    def _cleanup(self, file_path: str):
        try:
            os.remove(file_path)
        except Exception:
            pass

def main():
    executor = Executor(working_dir="/usr1/data/sijiel/graphml/workspaces/brave-splendid-ape", timeout=2, use_pty=True)
    code = """
from tqdm import tqdm
t = 0
for i in tqdm(range(100)):
    t += i
print(t)
"""
    result = asyncio.run(executor.run(code, agent_file_name="train.py"))
    print(result)


if __name__ == "__main__":
    main()
