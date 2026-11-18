param(
    [string]$BindAddress,
    [ValidateRange(1,65535)][int]$Port,
    [switch]$RequireModels
)

$ErrorActionPreference = 'Stop'
$anprPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $anprPython -PathType Leaf)) {
    throw 'Create the project .venv and install the documented requirements before running the MVP.'
}
Push-Location -LiteralPath $PSScriptRoot
try {
    $anprArguments = @()
    if ($PSBoundParameters.ContainsKey('BindAddress')) { $anprArguments += @('--host', $BindAddress) }
    if ($PSBoundParameters.ContainsKey('Port')) { $anprArguments += @('--port', $Port) }
    if ($RequireModels) { $anprArguments += '--require-models' }
    & $anprPython -m app.serve @anprArguments
    $anprExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $anprExitCode
