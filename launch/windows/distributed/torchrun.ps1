[CmdletBinding()]
param(
    [ValidateRange(1, 1024)]
    [int] $NProcPerNode = 1,

    [ValidateRange(1, 1024)]
    [int] $NNodes = 1,

    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]] $TrainOmniArguments
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$pythonExecutable = $env:TRAINOMNI_PYTHON
if ([string]::IsNullOrWhiteSpace($pythonExecutable)) {
    throw 'TRAINOMNI_PYTHON must be an absolute path to the intended Python executable.'
}
if (-not [System.IO.Path]::IsPathFullyQualified($pythonExecutable)) {
    throw 'TRAINOMNI_PYTHON must be an absolute path.'
}
if (-not (Test-Path -LiteralPath $pythonExecutable -PathType Leaf)) {
    throw "TRAINOMNI_PYTHON does not exist: $pythonExecutable"
}
if ($NProcPerNode -ne 1 -or $NNodes -ne 1) {
    throw ('Native Windows distributed launch is certified only for one process. ' +
        'Use launch/linux/distributed/torchrun.sh for CUDA multi-device execution.')
}

# The validated Windows PyTorch wheel has Gloo but no NCCL/libuv. The Python
# runtime creates a real world-size-one process group without torchrun.
& $pythonExecutable -m trainomni @TrainOmniArguments
$processExitCode = $LASTEXITCODE
if ($null -eq $processExitCode) {
    throw 'TrainOmni worker did not provide an exit code.'
}
exit $processExitCode
