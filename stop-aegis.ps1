[CmdletBinding()]
param(
    [string]$GuiHost = "127.0.0.1",

    [ValidateRange(1, 65535)]
    [int]$GuiPort = 8767,

    [string]$GuiStateFile = "",

    [ValidateRange(1, 600)]
    [int]$GuiTimeoutSeconds = 180,

    [ValidateRange(1, 300)]
    [int]$ContainerTimeoutSeconds = 180
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Restore-ProcessEnvironmentVariable {
    param(
        [Parameter(Mandatory)]
        [string]$Name,

        [Parameter(Mandatory)]
        [bool]$Exists,

        [AllowNull()]
        [AllowEmptyString()]
        [string]$Value
    )

    if ($Exists) {
        [Environment]::SetEnvironmentVariable(
            $Name,
            $Value,
            [EnvironmentVariableTarget]::Process
        )
        return
    }

    # On PowerShell 7 for Windows, SetEnvironmentVariable(..., $null, "Process")
    # leaves an empty environment entry. Remove it through the environment provider
    # so Docker Compose can fall back to the value in .env after this script exits.
    Remove-Item -LiteralPath "Env:$Name" -ErrorAction SilentlyContinue
}

function Test-LocalTcpPort {
    param(
        [Parameter(Mandatory)]
        [string]$HostName,

        [Parameter(Mandatory)]
        [int]$Port,

        [int]$TimeoutMilliseconds = 1000
    )

    $client = [Net.Sockets.TcpClient]::new()
    $connection = $null
    try {
        $connection = $client.BeginConnect($HostName, $Port, $null, $null)
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

function Invoke-DirectLoopbackShutdown {
    param(
        [Parameter(Mandatory)]
        [string]$Uri,

        [Parameter(Mandatory)]
        [string]$ShutdownToken
    )

    Add-Type -AssemblyName System.Net.Http
    $handler = [Net.Http.HttpClientHandler]::new()
    $handler.UseProxy = $false
    $client = [Net.Http.HttpClient]::new($handler)
    $request = [Net.Http.HttpRequestMessage]::new(
        [Net.Http.HttpMethod]::Post,
        $Uri
    )
    $response = $null
    try {
        $client.Timeout = [TimeSpan]::FromSeconds(10)
        $request.Headers.Authorization = [Net.Http.Headers.AuthenticationHeaderValue]::new(
            "Bearer",
            $ShutdownToken
        )
        [void]$request.Headers.Add("X-AEGIS-Confirmation", "shutdown-gui")
        $request.Content = [Net.Http.ByteArrayContent]::new([byte[]]::new(0))
        $response = $client.SendAsync($request).GetAwaiter().GetResult()
        return [int]$response.StatusCode
    }
    finally {
        if ($null -ne $response) {
            $response.Dispose()
        }
        $request.Dispose()
        $client.Dispose()
        $handler.Dispose()
    }
}

function Invoke-CheckedDocker {
    param(
        [Parameter(Mandatory)]
        [string[]]$Arguments,

        [Parameter(Mandatory)]
        [string]$FailureMessage
    )

    & $script:DockerExecutable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FailureMessage (docker exit code $LASTEXITCODE)."
    }
}

try {
    $projectRoot = $PSScriptRoot
    $composeFile = Join-Path $projectRoot "compose.yaml"

    $statePath = if ([string]::IsNullOrWhiteSpace($GuiStateFile)) {
        Join-Path ([Environment]::GetFolderPath("UserProfile")) ".aegis\gui-state.json"
    }
    else {
        [IO.Path]::GetFullPath($GuiStateFile)
    }
    $shutdownToken = $null
    $statePid = $null
    if (Test-Path -LiteralPath $statePath -PathType Leaf) {
        try {
            $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
            $stateFields = @($state.PSObject.Properties.Name | Sort-Object)
            $expectedStateFields = @(
                "host", "pid", "port", "schema_version", "shutdown_token"
            ) | Sort-Object
            if (($stateFields -join "`n") -cne ($expectedStateFields -join "`n")) {
                throw "unexpected fields"
            }
            if ([int]$state.schema_version -ne 1) {
                throw "unsupported schema"
            }
            $statePid = [int]$state.pid
            if ($statePid -le 0) {
                throw "invalid PID"
            }
            $statePort = [int]$state.port
            if ($statePort -lt 1 -or $statePort -gt 65535) {
                throw "invalid port"
            }
            $stateHost = [string]$state.host
            $parsedStateAddress = $null
            if (
                -not [Net.IPAddress]::TryParse($stateHost, [ref]$parsedStateAddress) -or
                -not [Net.IPAddress]::IsLoopback($parsedStateAddress)
            ) {
                throw "invalid host"
            }
            $shutdownToken = [string]$state.shutdown_token
            if (
                $shutdownToken.Length -lt 32 -or
                $shutdownToken -cnotmatch '^[A-Za-z0-9_-]+$'
            ) {
                throw "invalid shutdown token"
            }
            $null = Get-Process -Id $statePid -ErrorAction Stop
        }
        catch {
            throw (
                "The GUI state file '$statePath' is invalid or stale. No " +
                "credential was sent. Stop the dashboard with Ctrl+C, remove " +
                "the stale state file, and retry."
            )
        }
        if ($PSBoundParameters.ContainsKey("GuiPort") -and $GuiPort -ne $statePort) {
            throw "GuiPort does not match the active GUI state file."
        }
        if (
            $PSBoundParameters.ContainsKey("GuiHost") -and
            $GuiHost -cne "localhost" -and
            $GuiHost -cne $stateHost
        ) {
            throw "GuiHost does not match the active GUI state file."
        }
        $GuiHost = $stateHost
        $GuiPort = $statePort
    }

    $guiAddress = $null
    $parsedAddress = $null
    if ($GuiHost -ceq "localhost") {
        $guiAddress = "localhost"
        $guiUriHost = "localhost"
    }
    elseif (
        [Net.IPAddress]::TryParse($GuiHost, [ref]$parsedAddress) -and
        [Net.IPAddress]::IsLoopback($parsedAddress)
    ) {
        $guiAddress = $parsedAddress.ToString()
        $guiUriHost = if (
            $parsedAddress.AddressFamily -eq [Net.Sockets.AddressFamily]::InterNetworkV6
        ) { "[$guiAddress]" } else { $guiAddress }
    }
    else {
        throw "GuiHost must be localhost or a numeric loopback IP address."
    }

    if (-not (Test-Path -LiteralPath $composeFile -PathType Leaf)) {
        throw "compose.yaml was not found beside this script."
    }
    if (Test-LocalTcpPort -HostName $guiAddress -Port $GuiPort) {
        if ([string]::IsNullOrWhiteSpace($shutdownToken)) {
            throw (
                "A service is listening on dashboard port $GuiPort, but the " +
                "private GUI state file '$statePath' is unavailable. No reusable " +
                "AEGIS credential was sent. Stop the dashboard with Ctrl+C."
            )
        }
        Write-Host "Requesting a graceful dashboard shutdown..."
        try {
            $shutdownStatus = Invoke-DirectLoopbackShutdown `
                -Uri "http://${guiUriHost}:$GuiPort/api/shutdown" `
                -ShutdownToken $shutdownToken
        }
        catch {
            throw (
                "A service is listening on dashboard port $GuiPort but did " +
                "not accept the authenticated AEGIS shutdown request. Stop " +
                "the GUI with Ctrl+C before retrying."
            )
        }
        if ($shutdownStatus -ne 202) {
            throw "The AEGIS GUI returned HTTP $shutdownStatus during shutdown."
        }

        $stopwatch = [Diagnostics.Stopwatch]::StartNew()
        while (
            (Test-LocalTcpPort -HostName $guiAddress -Port $GuiPort) -and
            $stopwatch.Elapsed.TotalSeconds -lt $GuiTimeoutSeconds
        ) {
            Start-Sleep -Milliseconds 250
        }
        if (Test-LocalTcpPort -HostName $guiAddress -Port $GuiPort) {
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

    $script:DockerExecutable = $null
    $dockerCommand = Get-Command docker -ErrorAction SilentlyContinue
    if ($null -ne $dockerCommand) {
        $script:DockerExecutable = $dockerCommand.Source
    }
    else {
        $dockerCandidates = @(
            (Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\resources\bin\docker.exe'),
            (Join-Path $env:ProgramFiles 'Docker\Docker\resources\bin\docker.exe')
        )
        foreach ($dockerDesktopDocker in $dockerCandidates) {
            if (Test-Path -LiteralPath $dockerDesktopDocker -PathType Leaf) {
                $script:DockerExecutable = $dockerDesktopDocker
                break
            }
        }
    }
    if ([string]::IsNullOrWhiteSpace($script:DockerExecutable)) {
        throw (
            "The dashboard is stopped, but Docker was not found, so the " +
            "containers could not be marked as stopped. Install Docker Desktop " +
            "or restore its command, then rerun this script."
        )
    }

    & $script:DockerExecutable info --format "{{.ServerVersion}}" *> $null
    if ($LASTEXITCODE -ne 0) {
        throw (
            "The dashboard is stopped and Docker services are currently " +
            "inaccessible, but the Docker engine is not running, so the " +
            "containers could not be marked as stopped and may restart with " +
            "Docker. Start Docker Desktop, then rerun this script."
        )
    }
    & $script:DockerExecutable compose version *> $null
    if ($LASTEXITCODE -ne 0) {
        throw (
            "The dashboard is stopped, but the Docker Compose CLI plugin " +
            "('docker compose') is not available, " +
            "so the containers could not be marked as stopped."
        )
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
    $previousApiToken = [Environment]::GetEnvironmentVariable(
        "AEGIS_API_TOKEN",
        "Process"
    )
    $previousApiTokenExists = Test-Path -LiteralPath Env:AEGIS_API_TOKEN
    $previousFingerprintKey = [Environment]::GetEnvironmentVariable(
        "AEGIS_FINGERPRINT_KEY",
        "Process"
    )
    $previousFingerprintKeyExists = Test-Path `
        -LiteralPath Env:AEGIS_FINGERPRINT_KEY
    try {
        # Compose expands required runtime variables even for `stop`. These fixed,
        # non-secret values are used only while resolving the stop command and are
        # never written into or used to recreate a container.
        [Environment]::SetEnvironmentVariable(
            "AEGIS_API_TOKEN",
            "aegis-stop-interpolation-only-api-value-0001",
            "Process"
        )
        [Environment]::SetEnvironmentVariable(
            "AEGIS_FINGERPRINT_KEY",
            "aegis-stop-interpolation-only-hmac-value-0002",
            "Process"
        )
        Invoke-CheckedDocker `
            -Arguments $stopArguments `
            -FailureMessage "Could not stop the AEGIS Docker services"
    }
    finally {
        Restore-ProcessEnvironmentVariable `
            -Name "AEGIS_API_TOKEN" `
            -Exists $previousApiTokenExists `
            -Value $previousApiToken
        Restore-ProcessEnvironmentVariable `
            -Name "AEGIS_FINGERPRINT_KEY" `
            -Exists $previousFingerprintKeyExists `
            -Value $previousFingerprintKey
    }

    Write-Host (
        "AEGIS is stopped. Containers, downloaded models, audit data, and " +
        "GUI history were retained."
    )
}
catch {
    Write-Error $_.Exception.Message -ErrorAction Continue
    exit 1
}
