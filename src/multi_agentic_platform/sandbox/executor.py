import asyncio
import subprocess
from dataclasses import dataclass


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    return_code: int


class SandboxExecutor:
    def __init__(self, timeout_seconds: int = 8) -> None:
        self.timeout_seconds = timeout_seconds

    async def run_python(self, code: str) -> SandboxResult:
        process = await asyncio.create_subprocess_exec(
            "python",
            "-c",
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return SandboxResult("", "Execution timed out.", 124)

        return SandboxResult(
            stdout.decode("utf-8"),
            stderr.decode("utf-8"),
            process.returncode or 0,
        )

    @staticmethod
    def safety_notice() -> str:
        return (
            "This sandbox is a lightweight subprocess runner. For production workloads, "
            "run code in locked-down containers or VMs with network and filesystem controls."
        )
