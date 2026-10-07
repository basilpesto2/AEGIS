[CmdletBinding()]
param(
    [ValidateSet("llava05b", "qwen25vl3b")]
    [string]$Target = "llava05b",

    [ValidateRange(1, 3600)]
    [int]$ReadyTimeoutSeconds = 1200
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

    & $script:DockerExecutable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FailureMessage (docker exit code $LASTEXITCODE)."
    }
}

function Assert-ComposeServiceOwnsGuardPort {
    param(
        [Parameter(Mandatory)]
        [string[]]$ComposePrefix,

        [Parameter(Mandatory)]
        [string]$Profile,

        [Parameter(Mandatory)]
        [string]$Service
    )

    $psArguments = $ComposePrefix + @(
        "--profile", $Profile,
        "ps", "--status", "running", "-q", $Service
    )
    $containerIds = @(
        & $script:DockerExecutable @psArguments |
            ForEach-Object { ([string]$_).Trim() } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
    if ($LASTEXITCODE -ne 0 -or $containerIds.Count -ne 1) {
        throw (
            "Could not prove that exactly one running Compose service '$Service' " +
            "owns the AEGIS guard port. No bearer credential was sent."
        )
    }
    $containerId = $containerIds[0]
    if ($containerId -notmatch '^[0-9a-f]{12,64}$') {
        throw "Docker Compose returned an invalid container identifier for '$Service'."
    }

    $portJson = & $script:DockerExecutable inspect `
        --format "{{json .NetworkSettings.Ports}}" `
        $containerId
    if ($LASTEXITCODE -ne 0) {
        throw "Could not inspect the published port for Compose service '$Service'."
    }
    try {
        $ports = ($portJson -join "`n") | ConvertFrom-Json
        $property = $ports.PSObject.Properties['8766/tcp']
        $bindings = @(
            if ($null -ne $property) {
                $property.Value
            }
        )
    }
    catch {
        throw "Docker returned invalid port metadata for Compose service '$Service'."
    }
    if (
        $bindings.Count -ne 1 -or
        [string]$bindings[0].HostIp -cne "127.0.0.1" -or
        [string]$bindings[0].HostPort -cne "8766"
    ) {
        throw (
            "Compose service '$Service' does not exclusively publish container " +
            "port 8766 at 127.0.0.1:8766. No bearer credential was sent."
        )
    }
    return $containerId
}

function Invoke-DirectLoopbackJsonGet {
    param(
        [Parameter(Mandatory)]
        [string]$Uri,

        [Parameter(Mandatory)]
        [int]$TimeoutSeconds
    )

    Add-Type -AssemblyName System.Net.Http
    $handler = [Net.Http.HttpClientHandler]::new()
    $handler.UseProxy = $false
    $client = [Net.Http.HttpClient]::new($handler)
    $response = $null
    try {
        $client.Timeout = [TimeSpan]::FromSeconds($TimeoutSeconds)
        $response = $client.GetAsync($Uri).GetAwaiter().GetResult()
        if (-not $response.IsSuccessStatusCode) {
            throw "Loopback service returned HTTP $([int]$response.StatusCode)."
        }
        $body = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        return $body | ConvertFrom-Json
    }
    finally {
        if ($null -ne $response) {
            $response.Dispose()
        }
        $client.Dispose()
        $handler.Dispose()
    }
}

function Invoke-DirectLoopbackAuthenticatedJsonGet {
    param(
        [Parameter(Mandatory)]
        [string]$Uri,

        [Parameter(Mandatory)]
        [string]$Token,

        [Parameter(Mandatory)]
        [int]$TimeoutSeconds
    )

    Add-Type -AssemblyName System.Net.Http
    $handler = [Net.Http.HttpClientHandler]::new()
    $handler.UseProxy = $false
    $client = [Net.Http.HttpClient]::new($handler)
    $request = [Net.Http.HttpRequestMessage]::new(
        [Net.Http.HttpMethod]::Get,
        $Uri
    )
    $response = $null
    try {
        $client.Timeout = [TimeSpan]::FromSeconds($TimeoutSeconds)
        $request.Headers.Authorization = [Net.Http.Headers.AuthenticationHeaderValue]::new(
            "Bearer",
            $Token
        )
        $response = $client.SendAsync($request).GetAwaiter().GetResult()
        if (-not $response.IsSuccessStatusCode) {
            throw "Loopback status endpoint returned HTTP $([int]$response.StatusCode)."
        }
        $body = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        return $body | ConvertFrom-Json
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

function Get-TargetProfileDetectorSha256 {
    param(
        [Parameter(Mandatory)]
        [string]$AegisExecutable,

        [Parameter(Mandatory)]
        [string]$TargetProfile
    )

    $profileOutput = @(& $AegisExecutable targets)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read the bundled AEGIS target profiles."
    }
    try {
        $profileReport = ($profileOutput -join "`n") | ConvertFrom-Json
        $profilesProperty = $profileReport.PSObject.Properties['profiles']
        if ($null -eq $profilesProperty) {
            throw "The target-profile report does not contain profiles."
        }
        $matchingProfiles = @(
            $profilesProperty.Value |
                Where-Object { [string]$_.name -ceq $TargetProfile }
        )
    }
    catch {
        throw "The bundled AEGIS target-profile report is invalid."
    }
    if ($matchingProfiles.Count -ne 1) {
        throw "The target-profile report did not uniquely identify '$TargetProfile'."
    }
    $digestProperty = $matchingProfiles[0].PSObject.Properties['detector_identity_sha256']
    if ($null -eq $digestProperty) {
        $digestProperty = $matchingProfiles[0].PSObject.Properties['detector_sha256']
    }
    $digest = if ($null -eq $digestProperty) {
        ""
    }
    else {
        [string]$digestProperty.Value
    }
    if ($digest -cnotmatch '^[0-9a-fA-F]{64}$') {
        throw "The target profile '$TargetProfile' has an invalid detector digest."
    }
    return $digest.ToLowerInvariant()
}

function Test-RuntimeDetectorMatchesProfile {
    param(
        [Parameter(Mandatory)]
        [AllowNull()]
        [object]$Status,

        [Parameter(Mandatory)]
        [string]$TargetProfile,

        [Parameter(Mandatory)]
        [string]$ExpectedSha256
    )

    if ($null -eq $Status -or $Status -isnot [PSObject]) {
        return $false
    }
    $digestProperty = $Status.PSObject.Properties['detector_identity_sha256']
    if ($null -eq $digestProperty) {
        $digestProperty = $Status.PSObject.Properties['detector_sha256']
    }
    $targetProperty = $Status.PSObject.Properties['target_profile']
    $readyProperty = $Status.PSObject.Properties['ok']
    if (
        $null -eq $digestProperty -or
        $null -eq $targetProperty -or
        $null -eq $readyProperty
    ) {
        return $false
    }
    $reportedDigest = [string]$digestProperty.Value
    return (
        $readyProperty.Value -eq $true -and
        [string]$targetProperty.Value -ceq $TargetProfile -and
        $reportedDigest -cmatch '^[0-9a-fA-F]{64}$' -and
        $reportedDigest.ToLowerInvariant() -ceq $ExpectedSha256.ToLowerInvariant()
    )
}

function Invoke-DirectLoopbackTrafficModePut {
    param(
        [Parameter(Mandatory)]
        [string]$Uri,

        [Parameter(Mandatory)]
        [string]$Token,

        [Parameter(Mandatory)]
        [ValidateSet("shadow", "review", "enforce")]
        [string]$TrafficMode,

        [Parameter(Mandatory)]
        [int]$TimeoutSeconds
    )

    Add-Type -AssemblyName System.Net.Http
    $handler = [Net.Http.HttpClientHandler]::new()
    $handler.UseProxy = $false
    $client = [Net.Http.HttpClient]::new($handler)
    $request = [Net.Http.HttpRequestMessage]::new(
        [Net.Http.HttpMethod]::Put,
        $Uri
    )
    $response = $null
    try {
        $client.Timeout = [TimeSpan]::FromSeconds($TimeoutSeconds)
        $request.Headers.Authorization = [Net.Http.Headers.AuthenticationHeaderValue]::new(
            "Bearer",
            $Token
        )
        $body = @{ traffic_mode = $TrafficMode } | ConvertTo-Json -Compress
        $request.Content = [Net.Http.StringContent]::new(
            $body,
            [Text.Encoding]::UTF8,
            "application/json"
        )
        $response = $client.SendAsync($request).GetAwaiter().GetResult()
        if (-not $response.IsSuccessStatusCode) {
            throw "Loopback administration endpoint returned HTTP $([int]$response.StatusCode)."
        }
        $responseBody = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        return $responseBody | ConvertFrom-Json
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

function Get-ValidatedTrafficMode {
    param(
        [Parameter(Mandatory)]
        [AllowNull()]
        [object]$Value
    )

    if (
        $Value -isnot [string] -or
        @("shadow", "review", "enforce") -cnotcontains $Value
    ) {
        throw "The ready service reported an invalid traffic mode."
    }
    return [string]$Value
}

function Wait-ForTargetReadiness {
    param(
        [Parameter(Mandatory)]
        [string]$TargetProfile,

        [Parameter(Mandatory)]
        [int]$TimeoutSeconds
    )

    $stopwatch = [Diagnostics.Stopwatch]::StartNew()
    while ($stopwatch.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        try {
            $readiness = Invoke-DirectLoopbackJsonGet `
                -Uri "http://127.0.0.1:8766/readyz" `
                -TimeoutSeconds 5
            if (
                $readiness.ok -eq $true -and
                $readiness.target_profile -ceq $TargetProfile
            ) {
                return $readiness
            }
        }
        catch {
            # Connection failures are expected while the model is loading.
        }
        Start-Sleep -Seconds 2
    }
    return $null
}

function Assert-StrongSecret {
    param(
        [Parameter(Mandatory)]
        [string]$Name,

        [Parameter(Mandatory)]
        [AllowNull()]
        [AllowEmptyString()]
        [string]$Value
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        throw "$Name is missing or empty in .env."
    }
    if ($Value.Length -lt 32) {
        throw "$Name must contain at least 32 characters. Generate a private random value."
    }
    foreach ($character in $Value.ToCharArray()) {
        $codePoint = [int]$character
        if ($codePoint -lt 0x21 -or $codePoint -gt 0x7e) {
            throw "$Name must contain only visible ASCII characters."
        }
    }
    $normalized = $Value.ToLowerInvariant() -replace '[^a-z0-9]', ''
    if ($normalized -match '^(replacewith|changeme|example|default|password|secret|token)') {
        throw "$Name contains an example or common placeholder value."
    }
    $characters = @($Value.ToCharArray())
    $distinctCharacters = @($characters | Sort-Object -Unique).Count
    if ($distinctCharacters -lt 12) {
        throw "$Name does not contain enough distinct characters. Generate a random value."
    }
    $largestCharacterCount = @(
        $characters | Group-Object | Sort-Object Count -Descending
    )[0].Count
    if (($largestCharacterCount * 2) -gt $Value.Length) {
        throw "$Name is dominated by one character. Generate a random value."
    }
    if ($Value -cmatch '(.)\1{7}') {
        throw "$Name contains a long repeated-character run. Generate a random value."
    }
    $sequenceLength = 1
    $sequenceDirection = 0
    for ($index = 1; $index -lt $Value.Length; $index++) {
        $previous = $Value[$index - 1]
        $current = $Value[$index]
        $delta = ([int]$current) - ([int]$previous)
        $alphanumeric = (
            [char]::IsLetterOrDigit($previous) -and
            [char]::IsLetterOrDigit($current) -and
            [int]$previous -le 0x7e -and
            [int]$current -le 0x7e
        )
        if ($alphanumeric -and ($delta -eq 1 -or $delta -eq -1)) {
            if ($delta -eq $sequenceDirection) {
                $sequenceLength++
            }
            else {
                $sequenceDirection = $delta
                $sequenceLength = 2
            }
            if ($sequenceLength -ge 8) {
                throw "$Name contains a simple character sequence. Generate a random value."
            }
        }
        else {
            $sequenceLength = 1
            $sequenceDirection = 0
        }
    }
    $maximumPatternLength = [Math]::Min(16, [Math]::Floor($Value.Length / 2))
    for ($patternLength = 1; $patternLength -le $maximumPatternLength; $patternLength++) {
        if (($Value.Length % $patternLength) -ne 0) {
            continue
        }
        $repetitions = [int]($Value.Length / $patternLength)
        if (($Value.Substring(0, $patternLength) * $repetitions) -ceq $Value) {
            throw "$Name is a repeated pattern. Generate a random value."
        }
    }
}

$managedSecretNames = @(
    "AEGIS_API_TOKEN",
    "AEGIS_ADMIN_TOKEN",
    "AEGIS_FINGERPRINT_KEY"
)
$priorProcessSecrets = @{}
foreach ($secretName in $managedSecretNames) {
    $priorValue = [Environment]::GetEnvironmentVariable(
        $secretName,
        [EnvironmentVariableTarget]::Process
    )
    $priorProcessSecrets[$secretName] = @{
        Exists = $null -ne $priorValue
        Value = $priorValue
    }
}
$launcherExitCode = 0

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
            "Docker was not found. Install and start Docker Desktop, then " +
            "open a new PowerShell window."
        )
    }

    & $script:DockerExecutable info --format "{{.ServerVersion}}" *> $null
    if ($LASTEXITCODE -ne 0) {
        throw (
            "The Docker engine is not running. Start Docker Desktop and " +
            "wait for it to become ready."
        )
    }
    & $script:DockerExecutable compose version *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "The Docker Compose CLI plugin ('docker compose') is not available."
    }

    $apiToken = Get-DotEnvValue -Path $environmentFile -Name "AEGIS_API_TOKEN"
    Assert-StrongSecret -Name "AEGIS_API_TOKEN" -Value $apiToken
    $fingerprintKey = Get-DotEnvValue `
        -Path $environmentFile `
        -Name "AEGIS_FINGERPRINT_KEY"
    Assert-StrongSecret -Name "AEGIS_FINGERPRINT_KEY" -Value $fingerprintKey
    if ($apiToken -ceq $fingerprintKey) {
        throw "AEGIS_API_TOKEN and AEGIS_FINGERPRINT_KEY must be different secrets."
    }
    $adminToken = Get-DotEnvValue -Path $environmentFile -Name "AEGIS_ADMIN_TOKEN"
    if (-not [string]::IsNullOrWhiteSpace($adminToken)) {
        Assert-StrongSecret -Name "AEGIS_ADMIN_TOKEN" -Value $adminToken
        if ($adminToken -ceq $apiToken -or $adminToken -ceq $fingerprintKey) {
            throw "AEGIS_ADMIN_TOKEN must be different from the API token and fingerprint key."
        }
    }
    $env:AEGIS_API_TOKEN = $apiToken
    $env:AEGIS_FINGERPRINT_KEY = $fingerprintKey
    if ([string]::IsNullOrWhiteSpace($adminToken)) {
        Remove-Item Env:AEGIS_ADMIN_TOKEN -ErrorAction SilentlyContinue
    }
    else {
        $env:AEGIS_ADMIN_TOKEN = $adminToken
    }

    $targets = @{
        llava05b = @{
            Profile = "llava"
            Service = "aegis-llava"
            OtherTarget = "qwen25vl3b"
            OtherProfile = "qwen"
            OtherService = "aegis-qwen"
        }
        qwen25vl3b = @{
            Profile = "qwen"
            Service = "aegis-qwen"
            OtherTarget = "llava05b"
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
    $expectedDetectorSha256 = Get-TargetProfileDetectorSha256 `
        -AegisExecutable $aegisExecutable `
        -TargetProfile $Target

    $alternateReadiness = $null
    try {
        $alternateReadiness = Invoke-DirectLoopbackJsonGet `
            -Uri "http://127.0.0.1:8766/readyz" `
            -TimeoutSeconds 5
    }
    catch {
        # No rollback state is recorded unless the alternate is positively ready.
    }
    $previousHealthyTarget = $null
    $selectedAlreadyReady = $false
    if ($null -ne $alternateReadiness -and $alternateReadiness.ok -eq $true) {
        $reportedTarget = [string]$alternateReadiness.target_profile
        if ($reportedTarget -ceq $selected.OtherTarget) {
            $previousTrafficMode = Get-ValidatedTrafficMode `
                -Value $alternateReadiness.traffic_mode
            $previousHealthyTarget = [PSCustomObject]@{
                Target = $selected.OtherTarget
                TrafficMode = $previousTrafficMode
                Profile = $selected.OtherProfile
                Service = $selected.OtherService
            }
        }
        elseif ($reportedTarget -ceq $Target) {
            # Validate the full readiness identity before reusing the service. A
            # healthy requested target should not be rebuilt or recreated: doing so
            # would turn an unrelated build failure into an avoidable outage.
            $null = Get-ValidatedTrafficMode -Value $alternateReadiness.traffic_mode
            $selectedAlreadyReady = $true
        }
        else {
            throw "The ready service reported unexpected target profile '$reportedTarget'."
        }
    }

    if ($null -ne $previousHealthyTarget) {
        $previousContainerId = Assert-ComposeServiceOwnsGuardPort `
            -ComposePrefix $composePrefix `
            -Profile $previousHealthyTarget.Profile `
            -Service $previousHealthyTarget.Service
        $previousHealthyTarget | Add-Member `
            -NotePropertyName ContainerId `
            -NotePropertyValue $previousContainerId
        if ([string]::IsNullOrWhiteSpace($adminToken)) {
            throw (
                "$($previousHealthyTarget.Target) is healthy, but AEGIS_ADMIN_TOKEN " +
                "is not configured. Refusing to stop it because its exact traffic " +
                "mode could not be restored after a failed switch."
            )
        }
        $authenticationResult = Invoke-DirectLoopbackTrafficModePut `
            -Uri "http://127.0.0.1:8766/v1/admin/traffic-mode" `
            -Token $adminToken `
            -TrafficMode $previousHealthyTarget.TrafficMode `
            -TimeoutSeconds 5
        if (
            $authenticationResult.target_profile -cne $previousHealthyTarget.Target -or
            $authenticationResult.traffic_mode -cne $previousHealthyTarget.TrafficMode
        ) {
            throw (
                "The healthy alternate target changed while its administration " +
                "credential was being verified. It was not stopped."
            )
        }
    }

    if ($selectedAlreadyReady) {
        $selectedContainerId = Assert-ComposeServiceOwnsGuardPort `
            -ComposePrefix $composePrefix `
            -Profile $selected.Profile `
            -Service $selected.Service
        # The API bearer is attached only after the existing Compose service has
        # been proven to own the fixed loopback guard port.
        $authenticatedStatus = Invoke-DirectLoopbackAuthenticatedJsonGet `
            -Uri "http://127.0.0.1:8766/v1/status" `
            -Token $apiToken `
            -TimeoutSeconds 5
        if (
            Test-RuntimeDetectorMatchesProfile `
                -Status $authenticatedStatus `
                -TargetProfile $Target `
                -ExpectedSha256 $expectedDetectorSha256
        ) {
            Write-Host "$Target is already ready with the current detector; reusing the running service."
        }
        else {
            throw (
                "$Target is healthy, but its detector identity is stale. The " +
                "running service was left untouched because an in-place Compose " +
                "recreation cannot guarantee rollback to the retained container. " +
                "Stop it explicitly, then run start-aegis.ps1 again."
            )
        }
    }
    if ($selectedAlreadyReady) {
        Write-Host "Stopping the alternate AEGIS target, if it is running..."
        $stopArguments = $composePrefix + @(
            "--profile", $selected.OtherProfile,
            "stop", $selected.OtherService
        )
        Invoke-CheckedDocker `
            -Arguments $stopArguments `
            -FailureMessage "Could not stop $($selected.OtherService)"
    }
    if (-not $selectedAlreadyReady) {
      try {
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
            "up", "-d", "--no-deps", "--build", $selected.Service
        )
        Invoke-CheckedDocker `
            -Arguments $startArguments `
            -FailureMessage "Could not start $($selected.Service)"

        Write-Host "Waiting for $Target to report ready (timeout: $ReadyTimeoutSeconds seconds)..."
        $readiness = Wait-ForTargetReadiness `
            -TargetProfile $Target `
            -TimeoutSeconds $ReadyTimeoutSeconds
        if ($null -eq $readiness) {
            Write-Host "Recent container logs:"
            $logArguments = $composePrefix + @(
                "--profile", $selected.Profile,
                "logs", "--tail=120", $selected.Service
            )
            & $script:DockerExecutable @logArguments
            throw "$Target did not report ready within $ReadyTimeoutSeconds seconds."
        }
        $null = Assert-ComposeServiceOwnsGuardPort `
            -ComposePrefix $composePrefix `
            -Profile $selected.Profile `
            -Service $selected.Service
        # Public readiness proves availability, not which detector bytes the new
        # container loaded. Bind the freshly started service to the host profile
        # through the authenticated status endpoint before declaring success.
        $authenticatedStatus = Invoke-DirectLoopbackAuthenticatedJsonGet `
            -Uri "http://127.0.0.1:8766/v1/status" `
            -Token $apiToken `
            -TimeoutSeconds 5
        if (
            -not (Test-RuntimeDetectorMatchesProfile `
                -Status $authenticatedStatus `
                -TargetProfile $Target `
                -ExpectedSha256 $expectedDetectorSha256)
        ) {
            throw (
                "$Target started, but its authenticated detector identity does " +
                "not match the bundled target profile."
            )
        }
    }
    catch {
        $activationFailure = $_.Exception.Message
        $rollbackFailures = [Collections.Generic.List[string]]::new()

        Write-Host "Stopping the target that failed to become ready..."
        $cleanupArguments = $composePrefix + @(
            "--profile", $selected.Profile,
            "stop", $selected.Service
        )
        try {
            Invoke-CheckedDocker `
                -Arguments $cleanupArguments `
                -FailureMessage "Could not stop failed target $($selected.Service)"
        }
        catch {
            $rollbackFailures.Add($_.Exception.Message)
        }

        if ($null -ne $previousHealthyTarget) {
            try {
                Write-Host "Restoring previously healthy $($previousHealthyTarget.Target)..."
                # The selected build updates a shared image tag. Restart the exact
                # retained container ID so rollback cannot silently recreate the
                # previously healthy service from the newly built image.
                $restoreArguments = @("start", $previousHealthyTarget.ContainerId)
                Invoke-CheckedDocker `
                    -Arguments $restoreArguments `
                    -FailureMessage (
                        "Could not restart the exact retained container for " +
                        $previousHealthyTarget.Service
                    )
                $restoredReadiness = Wait-ForTargetReadiness `
                    -TargetProfile $previousHealthyTarget.Target `
                    -TimeoutSeconds $ReadyTimeoutSeconds
                if ($null -eq $restoredReadiness) {
                    throw (
                        "$($previousHealthyTarget.Target) did not report ready within " +
                        "$ReadyTimeoutSeconds seconds during rollback."
                    )
                }
                $restoredContainerId = Assert-ComposeServiceOwnsGuardPort `
                    -ComposePrefix $composePrefix `
                    -Profile $previousHealthyTarget.Profile `
                    -Service $previousHealthyTarget.Service
                if ($restoredContainerId -cne $previousHealthyTarget.ContainerId) {
                    throw (
                        "Rollback started a different container instead of the " +
                        "previously validated instance."
                    )
                }
                $modeResult = Invoke-DirectLoopbackTrafficModePut `
                    -Uri "http://127.0.0.1:8766/v1/admin/traffic-mode" `
                    -Token $adminToken `
                    -TrafficMode $previousHealthyTarget.TrafficMode `
                    -TimeoutSeconds 5
                if (
                    $modeResult.target_profile -cne $previousHealthyTarget.Target -or
                    $modeResult.traffic_mode -cne $previousHealthyTarget.TrafficMode
                ) {
                    throw "The restored target did not confirm its previous traffic mode."
                }
                $restoredReadiness = Wait-ForTargetReadiness `
                    -TargetProfile $previousHealthyTarget.Target `
                    -TimeoutSeconds 10
                if (
                    $null -eq $restoredReadiness -or
                    $restoredReadiness.traffic_mode -cne $previousHealthyTarget.TrafficMode
                ) {
                    throw "The restored target did not retain its previous traffic mode."
                }
            }
            catch {
                $rollbackFailures.Add($_.Exception.Message)
            }
        }

        if ($rollbackFailures.Count -gt 0) {
            throw (
                "$activationFailure Rollback also failed: " +
                ($rollbackFailures -join " ")
            )
        }
        if ($null -ne $previousHealthyTarget) {
            throw (
                "$activationFailure The previously healthy target " +
                "$($previousHealthyTarget.Target) and its exact traffic mode " +
                "$($previousHealthyTarget.TrafficMode) were restored."
            )
        }
        throw "$activationFailure The failed target was stopped; no positively ready alternate target had been recorded for rollback."
      }
    }

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
    $launcherExitCode = 1
}
finally {
    foreach ($secretName in $managedSecretNames) {
        $priorSecret = $priorProcessSecrets[$secretName]
        Restore-ProcessEnvironmentVariable `
            -Name $secretName `
            -Exists ([bool]$priorSecret.Exists) `
            -Value $priorSecret.Value
    }
}

if ($launcherExitCode -ne 0) {
    exit $launcherExitCode
}
