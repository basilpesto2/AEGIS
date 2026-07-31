[CmdletBinding()]
param(
    [ValidateSet("llava05b", "qwen25vl3b")]
    [string]$Target = "llava05b",

    [ValidateRange(1, 3600)]
    [int]$ReadyTimeoutSeconds = 900
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-DotEnvValue {
    param(
        [Parameter(Mandatory)]
        [string]$Path,

        [Parameter(Mandatory)]
        [string]$Name
    )

    $escapedName = [Regex]::Escape($Name)
    $foundValue = $null
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -notmatch "^\s*$escapedName\s*=(.*)$") {
            continue
        }

        $value = $Matches[1].Trim()
        if ($value.Length -ge 2) {
            $firstCharacter = $value[0]
            $lastCharacter = $value[$value.Length - 1]
            if (
                ($firstCharacter -eq '"' -and $lastCharacter -eq '"') -or
                ($firstCharacter -eq "'" -and $lastCharacter -eq "'")
            ) {
                $value = $value.Substring(1, $value.Length - 2)
            }
        }
        $foundValue = $value
    }

    return $foundValue
}

function Invoke-CheckedDocker {
    param(
        [Parameter(Mandatory)]
        [string[]]$Arguments,

        [Parameter(Mandatory)]
        [string]$FailureMessage
    )

    & docker @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FailureMessage (docker exit code $LASTEXITCODE)."
    }
}

try {
    $projectRoot = $PSScriptRoot
    $composeFile = Join-Path $projectRoot "compose.yaml"
    $environmentFile = Join-Path $projectRoot ".env"
    $aegisExecutable = Join-Path $projectRoot ".venv\Scripts\aegis.exe"

    if (-not (Test-Path -LiteralPath $composeFile -PathType Leaf)) {
        throw "compose.yaml was not found beside this script."
    }
    if (-not (Test-Path -LiteralPath $environmentFile -PathType Leaf)) {
        throw (
            "The .env file is missing. Copy .env.example to .env and " +
            "configure its secrets first."
        )
    }
    if (-not (Test-Path -LiteralPath $aegisExecutable -PathType Leaf)) {
        throw (
            "The GUI is not installed in .venv. Complete the README's " +
            "first-time GUI installation first."
        )
    }
    try {
        & $aegisExecutable --version *> $null
        if ($LASTEXITCODE -ne 0) {
            throw "aegis exited with code $LASTEXITCODE."
        }
    }
    catch {
        throw (
            "The AEGIS virtual environment is not runnable. Recreate .venv " +
            "with an installed Python 3.10 or newer, then reinstall AEGIS."
        )
    }
    if ($null -eq (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw (
            "Docker was not found. Install and start Docker Desktop, then " +
            "open a new PowerShell window."
        )
    }

    & docker info --format "{{.ServerVersion}}" *> $null
    if ($LASTEXITCODE -ne 0) {
        throw (
            "The Docker engine is not running. Start Docker Desktop and " +
            "wait for it to become ready."
        )
    }
    & docker compose version *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Compose v2 is not available."
    }

    $apiToken = Get-DotEnvValue -Path $environmentFile -Name "AEGIS_API_TOKEN"
    if ([string]::IsNullOrWhiteSpace($apiToken)) {
        throw "AEGIS_API_TOKEN is missing or empty in .env."
    }
    if ($apiToken -like "replace-with-*") {
        throw (
            "AEGIS_API_TOKEN still contains the example placeholder. Set " +
            "a private token in .env first."
        )
    }
    $fingerprintKey = Get-DotEnvValue `
        -Path $environmentFile `
        -Name "AEGIS_FINGERPRINT_KEY"
    if ([string]::IsNullOrWhiteSpace($fingerprintKey)) {
        throw "AEGIS_FINGERPRINT_KEY is missing or empty in .env."
    }
    if ($fingerprintKey -like "replace-with-*") {
        throw (
            "AEGIS_FINGERPRINT_KEY still contains the example placeholder. " +
            "Set a private key in .env first."
        )
    }

    $targets = @{
        llava05b = @{
            Profile = "llava"
            Service = "aegis-llava"
            OtherProfile = "qwen"
            OtherService = "aegis-qwen"
        }
        qwen25vl3b = @{
            Profile = "qwen"
            Service = "aegis-qwen"
            OtherProfile = "llava"
            OtherService = "aegis-llava"
        }
    }
    $selected = $targets[$Target]
    $composePrefix = @(
        "compose",
        "--project-directory", $projectRoot,
        "--file", $composeFile
    )

    Write-Host "Stopping the alternate AEGIS target, if it is running..."
    $stopArguments = $composePrefix + @(
        "--profile", $selected.OtherProfile,
        "stop", $selected.OtherService
    )
    Invoke-CheckedDocker `
        -Arguments $stopArguments `
        -FailureMessage "Could not stop $($selected.OtherService)"

    Write-Host "Starting $($selected.Service)..."
    $startArguments = $composePrefix + @(
        "--profile", $selected.Profile,
        "up", "-d", "--no-deps", $selected.Service
    )
    Invoke-CheckedDocker `
        -Arguments $startArguments `
        -FailureMessage "Could not start $($selected.Service)"

    Write-Host "Waiting for $Target to report ready (timeout: $ReadyTimeoutSeconds seconds)..."
    $stopwatch = [Diagnostics.Stopwatch]::StartNew()
    $ready = $false
    while ($stopwatch.Elapsed.TotalSeconds -lt $ReadyTimeoutSeconds) {
        try {
            $readiness = Invoke-RestMethod `
                -Uri "http://127.0.0.1:8766/readyz" `
                -Method Get `
                -TimeoutSec 5
            if ($readiness.ok -eq $true -and $readiness.target_profile -eq $Target) {
                $ready = $true
                break
            }
        }
        catch {
            # Connection failures are expected while the model is loading.
        }
        Start-Sleep -Seconds 2
    }

    if (-not $ready) {
        Write-Host "Recent container logs:"
        $logArguments = $composePrefix + @(
            "--profile", $selected.Profile,
            "logs", "--tail=120", $selected.Service
        )
        & docker @logArguments
        throw "$Target did not report ready within $ReadyTimeoutSeconds seconds."
    }

    $env:AEGIS_API_TOKEN = $apiToken
    Write-Host "$Target is ready. Starting the AEGIS dashboard..."
    Push-Location $projectRoot
    try {
        & $aegisExecutable gui --project-root $projectRoot --open-browser
        if ($LASTEXITCODE -ne 0) {
            throw "The AEGIS dashboard exited with code $LASTEXITCODE."
        }
    }
    finally {
        Pop-Location
    }
}
catch {
    Write-Error $_.Exception.Message -ErrorAction Continue
    exit 1
}
