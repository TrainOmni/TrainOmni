$ErrorActionPreference = 'Stop'

$env:HTTP_PROXY = 'http://127.0.0.1:7897'
$env:HTTPS_PROXY = 'http://127.0.0.1:7897'
$env:HF_HOME = 'D:\Models\_cache\huggingface'
$env:HF_HUB_DOWNLOAD_TIMEOUT = '120'
$env:HF_HUB_ETAG_TIMEOUT = '30'

$python = 'D:\Applications\Anaconda\python.exe'
$downloadScript = Join-Path $PSScriptRoot 'resume_minicpm5_download.py'
& $python $downloadScript
