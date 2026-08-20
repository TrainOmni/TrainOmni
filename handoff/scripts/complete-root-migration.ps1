$ErrorActionPreference = 'Stop'

$sourceRoot = 'D:\Codex\TrainVLM'
$targetRoot = 'D:\Codex\TrainOmni'
$python = 'C:\Users\liubyfly\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$deadline = [DateTime]::UtcNow.AddMinutes(30)

function Test-LegacyJunction {
    if (-not (Test-Path -LiteralPath $sourceRoot)) {
        return $false
    }
    $item = Get-Item -LiteralPath $sourceRoot -Force
    return $item.LinkType -eq 'Junction' -and @($item.Target) -contains $targetRoot
}

while ([DateTime]::UtcNow -lt $deadline) {
    try {
        if (-not (Test-Path -LiteralPath $targetRoot)) {
            $sourceItem = Get-Item -LiteralPath $sourceRoot -Force
            if (-not $sourceItem.PSIsContainer -or $sourceItem.LinkType) {
                throw "Unexpected source item: $sourceRoot"
            }
            Move-Item -LiteralPath $sourceRoot -Destination $targetRoot
        }

        if (-not (Test-LegacyJunction)) {
            if (Test-Path -LiteralPath $sourceRoot) {
                throw "Both roots exist and the legacy root is not the expected junction"
            }
            New-Item -ItemType Junction -Path $sourceRoot -Target $targetRoot | Out-Null
        }

        & $python "$targetRoot\handoff\skills\handoff\scripts\registry.py" migrate-root --from-root $sourceRoot --to-root $targetRoot
        if ($LASTEXITCODE -ne 0) {
            throw 'Registry root migration failed'
        }
        exit 0
    }
    catch {
        Start-Sleep -Milliseconds 500
    }
}

exit 2

