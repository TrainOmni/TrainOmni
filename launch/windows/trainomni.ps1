[CmdletBinding()]
param(
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

& $pythonExecutable -m trainomni @TrainOmniArguments
$processExitCode = $LASTEXITCODE
if ($null -eq $processExitCode) {
    throw 'Python process did not provide an exit code.'
}
exit $processExitCode
