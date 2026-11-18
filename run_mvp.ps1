param(
    [string]$BindAddress = '127.0.0.1',
    [ValidateRange(1,65535)][int]$Port = 8000
)

$ErrorActionPreference = 'Stop'
$anprPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $anprPython -PathType Leaf)) {
    throw 'Create the project .venv and install the documented requirements before running the MVP.'
}
Push-Location -LiteralPath $PSScriptRoot
try {
    & $anprPython -m uvicorn app.main:app --host $BindAddress --port $Port
    $anprExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $anprExitCode
