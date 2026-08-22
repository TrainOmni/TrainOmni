from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

FRAMEWORK_ROOT = Path(__file__).resolve().parents[3]
WINDOWS_LAUNCHER = FRAMEWORK_ROOT / "launch" / "windows" / "trainomni.ps1"
LINUX_LAUNCHER = FRAMEWORK_ROOT / "launch" / "linux" / "trainomni.sh"
WINDOWS_TORCHRUN = (
    FRAMEWORK_ROOT / "launch" / "windows" / "distributed" / "torchrun.ps1"
)
LINUX_TORCHRUN = FRAMEWORK_ROOT / "launch" / "linux" / "distributed" / "torchrun.sh"


def _launcher_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["TRAINOMNI_PYTHON"] = str(Path(sys.executable).resolve())
    source_root = str(FRAMEWORK_ROOT / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root if not existing else os.pathsep.join((source_root, existing))
    )
    return environment


def test_platform_launchers_are_thin_and_keep_task_run_semantics_out() -> None:
    windows = WINDOWS_LAUNCHER.read_text(encoding="utf-8")
    linux = LINUX_LAUNCHER.read_text(encoding="utf-8")

    for source in (windows, linux):
        assert "TRAINOMNI_PYTHON" in source
        assert "trainomni" in source
        assert "task.yaml" not in source
        assert "run.yaml" not in source
        assert "torchrun" not in source
        assert "pip install" not in source
        assert "conda activate" not in source
        assert "CUDA_VISIBLE_DEVICES" not in source

    assert "-m trainomni @TrainOmniArguments" in windows
    assert "exit $processExitCode" in windows
    assert 'exec "$TRAINOMNI_PYTHON" -m trainomni "$@"' in linux


@pytest.mark.skipif(os.name != "nt", reason="Windows adapter is validated on Windows")
def test_windows_launcher_forwards_cli_and_exit_code() -> None:
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is unavailable")
    result = subprocess.run(
        [
            powershell,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(WINDOWS_LAUNCHER),
            "--help",
        ],
        env=_launcher_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage: trainomni" in result.stdout


@pytest.mark.skipif(os.name == "nt", reason="Linux adapter is validated on Linux")
def test_linux_launcher_forwards_cli_and_exit_code() -> None:
    result = subprocess.run(
        ["/bin/sh", str(LINUX_LAUNCHER), "--help"],
        env=_launcher_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage: trainomni" in result.stdout


def test_active_platform_launcher_rejects_implicit_python_selection() -> None:
    environment = os.environ.copy()
    environment.pop("TRAINOMNI_PYTHON", None)
    if os.name == "nt":
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if powershell is None:
            pytest.skip("PowerShell is unavailable")
        command = [
            powershell,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(WINDOWS_LAUNCHER),
            "--help",
        ]
    else:
        command = ["/bin/sh", str(LINUX_LAUNCHER), "--help"]
    result = subprocess.run(
        command,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode != 0
    assert "TRAINOMNI_PYTHON" in result.stderr


def test_distributed_launchers_only_translate_host_topology() -> None:
    windows = WINDOWS_TORCHRUN.read_text(encoding="utf-8")
    linux = LINUX_TORCHRUN.read_text(encoding="utf-8")
    for source in (windows, linux):
        assert "TRAINOMNI_PYTHON" in source
        assert "trainomni" in source
        assert "task.json" not in source
        assert "run.json" not in source
        assert "pip install" not in source
        assert "CUDA_VISIBLE_DEVICES" not in source
    assert "torch.distributed.run" in linux
    assert "certified only for one process" in windows


@pytest.mark.skipif(os.name != "nt", reason="Windows adapter is validated on Windows")
def test_windows_torchrun_world_size_one_forwards_cli() -> None:
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is unavailable")
    result = subprocess.run(
        [
            powershell,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(WINDOWS_TORCHRUN),
            "-NProcPerNode",
            "1",
            "--help",
        ],
        env=_launcher_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage: trainomni" in result.stdout
