param(
    [string]$RepoRoot = 'D:\Codex\TrainOmniTemp\framework-upstream-references-20260821\upstreams\VLMEvalKit',
    [string]$Python = 'D:\Codex\TrainOmni\Framework\.venv\Scripts\python.exe'
)

$ErrorActionPreference = 'Stop'
$expectedCommit = 'e8e78f05f3080fe28154f2130321f17951c3be94'
$patchPath = Join-Path $PSScriptRoot 'patches\vlmevalkit-windows.patch'

if (-not (Test-Path -LiteralPath $RepoRoot -PathType Container)) {
    throw "VLMEvalKit checkout is missing: $RepoRoot"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable is missing: $Python"
}

$actualCommit = (& git -C $RepoRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $expectedCommit) {
    throw "Expected VLMEvalKit $expectedCommit, found $actualCommit"
}

& git -C $RepoRoot apply --check $patchPath 2>$null
if ($LASTEXITCODE -eq 0) {
    & git -C $RepoRoot apply $patchPath
    if ($LASTEXITCODE -ne 0) { throw 'Failed to apply the Windows compatibility patch.' }
} else {
    & git -C $RepoRoot apply --reverse --check $patchPath 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw 'The checkout is neither clean nor patched with the expected Windows overlay.'
    }
}

& $Python -m pip install -e $RepoRoot --no-deps
if ($LASTEXITCODE -ne 0) { throw 'Failed to install VLMEvalKit itself.' }

$requirements = Get-Content -LiteralPath (Join-Path $RepoRoot 'requirements.txt') |
    Where-Object { $_ -and -not $_.StartsWith('#') -and -not $_.StartsWith('polygon3') }
& $Python -m pip install $requirements
if ($LASTEXITCODE -ne 0) { throw 'Failed to install VLMEvalKit runtime dependencies.' }

# Imported by an eagerly loaded dataset module but missing from upstream requirements.txt.
& $Python -m pip install rouge-score
if ($LASTEXITCODE -ne 0) { throw 'Failed to install rouge-score.' }

& $Python -c "import torch, vlmeval; assert torch.cuda.is_available(); print(vlmeval.__version__, torch.__version__, torch.version.cuda)"
if ($LASTEXITCODE -ne 0) { throw 'VLMEvalKit import/CUDA verification failed.' }
