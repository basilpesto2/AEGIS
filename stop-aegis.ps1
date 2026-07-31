[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$GuiPort = 8767,

    [ValidateRange(1, 600)]
    [int]$GuiTimeoutSeconds = 180,

    [ValidateRange(1, 300)]
    [int]$ContainerTimeoutSeconds = 30
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

function Test-LocalTcpPort {
    param(
        [Parameter(Mandatory)]
        [int]$Port,

        [int]$TimeoutMilliseconds = 1000
    )

    $client = [Net.Sockets.TcpClient]::new()
    $connection = $null
    try {
        $connection = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
        if (-not $connection.AsyncWaitHandle.WaitOne($TimeoutMilliseconds)) {
            return $false
        }
        $client.EndConnect($connection)
        return $true
    }
    catch {
        return $false
    }
    finally {
        if ($null -ne $connection) {
            $connection.AsyncWaitHandle.Close()
        }
        $client.Dispose()
    }
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

    if (-not (Test-Path -LiteralPath $composeFile -PathType Leaf)) {
        throw "compose.yaml was not found beside this script."
    }
    if (-not (Test-Path -LiteralPath $environmentFile -PathType Leaf)) {
        throw (
            "The .env file is missing. It is required to authenticate the " +
            "dashboard shutdown request."
        )
    }
    if ($null -eq (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw (
            "Docker was not found. Open a PowerShell window where Docker " +
            "Desktop is available."
        )
    }

    & docker info --format "{{.ServerVersion}}" *> $null
    if ($LASTEXITCODE -ne 0) {
        throw (
            "The Docker engine is not running, so the containers cannot be " +
            "marked as stopped."
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
            "AEGIS_API_TOKEN still contains the example placeholder. The " +
            "running GUI cannot be authenticated safely."
        )
    }
    $fingerprintKey = Get-DotEnvValue `
        -Path $environmentFile `
        -Name "AEGIS_FINGERPRINT_KEY"
    if ([string]::IsNullOrWhiteSpace($fingerprintKey)) {
        throw "AEGIS_FINGERPRINT_KEY is missing or empty in .env."
    }

    if (Test-LocalTcpPort -Port $GuiPort) {
        Write-Host "Requesting a graceful dashboard shutdown..."
        $headers = @{
            "Authorization" = "Bearer $apiToken"
            "X-AEGIS-Confirmation" = "shutdown-gui"
        }
        try {
            $response = Invoke-WebRequest `
                -Uri "http://127.0.0.1:$GuiPort/api/shutdown" `
                -Method Post `
                -Headers $headers `
                -TimeoutSec 10 `
                -UseBasicParsing
        }
        catch {
            throw (
                "A service is listening on dashboard port $GuiPort but did " +
                "not accept the authenticated AEGIS shutdown request. Stop " +
                "the GUI with Ctrl+C before retrying."
            )
        }
        if ([int]$response.StatusCode -ne 202) {
            throw "The AEGIS GUI returned HTTP $($response.StatusCode) during shutdown."
        }

        $stopwatch = [Diagnostics.Stopwatch]::StartNew()
        while (
            (Test-LocalTcpPort -Port $GuiPort) -and
            $stopwatch.Elapsed.TotalSeconds -lt $GuiTimeoutSeconds
        ) {
            Start-Sleep -Milliseconds 250
        }
        if (Test-LocalTcpPort -Port $GuiPort) {
            throw (
                "The GUI did not stop within $GuiTimeoutSeconds seconds. " +
                "Its current work may still be running; use Ctrl+C in its " +
                "terminal before retrying."
            )
        }
        Write-Host "Dashboard stopped."
    }
    else {
        Write-Host "Dashboard is not running."
    }

    Write-Host "Stopping AEGIS Docker services..."
    $stopArguments = @(
        "compose",
        "--project-directory", $projectRoot,
        "--file", $composeFile,
        "--profile", "llava",
        "--profile", "qwen",
        "stop",
        "--timeout", "$ContainerTimeoutSeconds",
        "aegis-llava",
        "aegis-qwen"
    )
    Invoke-CheckedDocker `
        -Arguments $stopArguments `
        -FailureMessage "Could not stop the AEGIS Docker services"

    Write-Host (
        "AEGIS is stopped. Containers, downloaded models, audit data, and " +
        "GUI history were retained."
    )
}
catch {
    Write-Error $_.Exception.Message -ErrorAction Continue
    exit 1
}
