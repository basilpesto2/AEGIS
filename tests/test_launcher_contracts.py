from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _run_powershell_script(powershell: str, script: str) -> subprocess.CompletedProcess:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "launcher-contract.ps1"
        path.write_text(script, encoding="utf-8-sig")
        return subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-File", str(path)],
            check=False,
            capture_output=True,
            timeout=15,
        )


class LauncherContractTests(unittest.TestCase):
    def test_process_environment_restore_removes_absent_variables(self) -> None:
        # Windows PowerShell 5.1 removes a variable when .NET receives $null,
        # while PowerShell 7 binds it as an empty string. Exercise the runtime
        # that exposed the Compose-precedence regression instead of masking it.
        powershell = shutil.which("pwsh")
        if powershell is None:
            self.skipTest("PowerShell 7 is unavailable")
        launcher = (ROOT / "start-aegis.ps1").read_text(encoding="utf-8")
        functions = launcher.split("function Get-DotEnvValue", 1)[0]
        command = functions + r'''
$name = "AEGIS_LAUNCHER_RESTORE_TEST"
Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
[Environment]::SetEnvironmentVariable($name, "temporary", "Process")
Restore-ProcessEnvironmentVariable -Name $name -Exists $false -Value $null
if (Test-Path -LiteralPath "Env:$name") { exit 35 }

[Environment]::SetEnvironmentVariable($name, "prior", "Process")
Restore-ProcessEnvironmentVariable -Name $name -Exists $true -Value "prior"
if ((Get-Item -LiteralPath "Env:$name").Value -cne "prior") { exit 36 }
Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
exit 0
'''
        completed = _run_powershell_script(powershell, command)
        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr.decode(errors="replace"),
        )

        stop_script = (ROOT / "stop-aegis.ps1").read_text(encoding="utf-8")
        self.assertIn("Restore-ProcessEnvironmentVariable", stop_script)
        self.assertIn("$previousApiTokenExists", stop_script)
        self.assertIn("$previousFingerprintKeyExists", stop_script)

    def test_windows_launcher_accepts_single_guard_port_binding(self) -> None:
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        if powershell is None:
            self.skipTest("PowerShell is unavailable")
        script = (ROOT / "start-aegis.ps1").read_text(encoding="utf-8")
        functions = script.split("$managedSecretNames", 1)[0]
        command = functions + r'''
function Mock-Docker {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [object[]]$DockerArgs
    )
    if ($DockerArgs[0] -eq "inspect") {
        $global:LASTEXITCODE = 0
        '{"8766/tcp":[{"HostIp":"127.0.0.1","HostPort":"8766"}]}'
        return
    }
    $global:LASTEXITCODE = 0
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
}
$script:DockerExecutable = "Mock-Docker"
try {
    $containerId = Assert-ComposeServiceOwnsGuardPort `
        -ComposePrefix @("compose") `
        -Profile "llava" `
        -Service "aegis-llava"
}
catch {
    Write-Error $_
    exit 25
}
if (
    $containerId -cne
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
) {
    exit 26
}
exit 0
'''
        completed = _run_powershell_script(powershell, command)
        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr.decode(errors="replace"),
        )

    def test_windows_launcher_rejects_character_dominated_secret(self) -> None:
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        if powershell is None:
            self.skipTest("PowerShell is unavailable")
        script = (ROOT / "start-aegis.ps1").read_text(encoding="utf-8")
        functions = script.split("$managedSecretNames", 1)[0]
        command = functions + r'''
$rejected = $false
try {
    Assert-StrongSecret -Name "test" -Value (("a" * 28) + "bcde")
}
catch {
    $rejected = $true
}
if (-not $rejected) { exit 23 }
try {
    Assert-StrongSecret -Name "test" -Value "p5c_8XJvA1-Ns7Wq2kMz4Hr9Ty6Ld0Be"
}
catch {
    exit 24
}
exit 0
'''
        completed = _run_powershell_script(powershell, command)
        self.assertEqual(completed.returncode, 0)

    def test_windows_launcher_detector_identity_is_exact_and_fail_closed(
        self,
    ) -> None:
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        if powershell is None:
            self.skipTest("PowerShell is unavailable")
        script = (ROOT / "start-aegis.ps1").read_text(encoding="utf-8")
        functions = script.split("$managedSecretNames", 1)[0]
        digest = "a" * 64
        command = functions + rf'''
$matching = [PSCustomObject]@{{
    ok = $true
    target_profile = "llava05b"
    detector_sha256 = "{digest}"
}}
$matchingIdentity = [PSCustomObject]@{{
    ok = $true
    target_profile = "llava05b"
    detector_identity_sha256 = "{digest}"
    detector_sha256 = ("b" * 64)
}}
function Mock-Aegis {{
    $global:LASTEXITCODE = 0
    '{{"profiles":[' +
        '{{"name":"llava05b","detector_sha256":"{digest}"}},' +
        '{{"name":"qwen25vl3b","detector_identity_sha256":"{digest}",' +
        '"detector_sha256":"' + ('b' * 64) + '"}}]}}'
}}
$profileDigest = Get-TargetProfileDetectorSha256 `
    -AegisExecutable "Mock-Aegis" `
    -TargetProfile "llava05b"
if ($profileDigest -cne "{digest}") {{ exit 30 }}
$dualProfileDigest = Get-TargetProfileDetectorSha256 `
    -AegisExecutable "Mock-Aegis" `
    -TargetProfile "qwen25vl3b"
if ($dualProfileDigest -cne "{digest}") {{ exit 34 }}
if (-not (Test-RuntimeDetectorMatchesProfile `
    -Status $matching -TargetProfile "llava05b" -ExpectedSha256 "{digest}")) {{
    exit 31
}}
if (-not (Test-RuntimeDetectorMatchesProfile `
    -Status $matchingIdentity -TargetProfile "llava05b" -ExpectedSha256 "{digest}")) {{
    exit 33
}}
$cases = @(
    [PSCustomObject]@{{ ok = $true; target_profile = "llava05b" }},
    [PSCustomObject]@{{
        ok = $true
        target_profile = "llava05b"
        detector_sha256 = "bad"
    }},
    [PSCustomObject]@{{
        ok = $true
        target_profile = "llava05b"
        detector_sha256 = ("b" * 64)
    }}
)
foreach ($case in $cases) {{
    if (Test-RuntimeDetectorMatchesProfile `
        -Status $case -TargetProfile "llava05b" -ExpectedSha256 "{digest}") {{
        exit 32
    }}
}}
exit 0
'''
        completed = _run_powershell_script(powershell, command)
        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr.decode(errors="replace"),
        )

    def test_windows_launcher_proves_owner_before_authenticated_status(self) -> None:
        script = (ROOT / "start-aegis.ps1").read_text(encoding="utf-8")
        reuse_branch = script.split("if ($selectedAlreadyReady) {", 1)[1].split(
            "if (-not $selectedAlreadyReady) {", 1
        )[0]
        owner = reuse_branch.index("Assert-ComposeServiceOwnsGuardPort")
        authenticated_status = reuse_branch.index(
            "Invoke-DirectLoopbackAuthenticatedJsonGet"
        )
        identity_check = reuse_branch.index("Test-RuntimeDetectorMatchesProfile")
        self.assertLess(owner, authenticated_status)
        self.assertLess(authenticated_status, identity_check)
        self.assertIn("running service was left untouched", reuse_branch)
        self.assertNotIn("Rebuilding and recreating", reuse_branch)

    def test_windows_launcher_authenticates_new_detector_before_success(self) -> None:
        script = (ROOT / "start-aegis.ps1").read_text(encoding="utf-8")
        activation = script.split("if (-not $selectedAlreadyReady) {", 1)[1]
        protected = activation.split("catch {", 1)[0]
        readiness = protected.index("Wait-ForTargetReadiness")
        owner = protected.index("Assert-ComposeServiceOwnsGuardPort", readiness)
        authenticated_status = protected.index(
            "Invoke-DirectLoopbackAuthenticatedJsonGet", owner
        )
        identity_check = protected.index(
            "Test-RuntimeDetectorMatchesProfile", authenticated_status
        )
        self.assertLess(readiness, owner)
        self.assertLess(owner, authenticated_status)
        self.assertLess(authenticated_status, identity_check)
        self.assertIn("authenticated detector identity", protected)

    def test_windows_rollback_restarts_exact_retained_container(self) -> None:
        script = (ROOT / "start-aegis.ps1").read_text(encoding="utf-8")
        marker = "# The selected build updates a shared image tag."
        self.assertIn(marker, script)
        rollback = script.split(marker, 1)[1].split(
            "$modeResult = Invoke-DirectLoopbackTrafficModePut", 1
        )[0]
        self.assertIn(
            '$restoreArguments = @("start", $previousHealthyTarget.ContainerId)',
            rollback,
        )
        self.assertNotIn('"up", "-d"', rollback)
        self.assertIn(
            "$restoredContainerId -cne $previousHealthyTarget.ContainerId",
            rollback,
        )

    def test_switch_stop_is_inside_activation_rollback_boundary(self) -> None:
        script = (ROOT / "start-aegis.ps1").read_text(encoding="utf-8")
        activation = script.split("if (-not $selectedAlreadyReady) {", 1)[1]
        protected = activation.split("catch {", 1)[0]
        self.assertLess(protected.index("try {"), protected.index("stopArguments"))

    def test_admin_token_requirement_for_healthy_target_switch_is_documented(
        self,
    ) -> None:
        script = (ROOT / "start-aegis.ps1").read_text(encoding="utf-8")
        switch_gate = script.split(
            "if ($null -ne $previousHealthyTarget) {", 1
        )[1].split("if ($selectedAlreadyReady) {", 1)[0]
        owner = switch_gate.index("Assert-ComposeServiceOwnsGuardPort")
        admin_gate = switch_gate.index("IsNullOrWhiteSpace($adminToken)")
        authenticated_mode = switch_gate.index(
            "Invoke-DirectLoopbackTrafficModePut"
        )
        self.assertLess(owner, admin_gate)
        self.assertLess(admin_gate, authenticated_mode)
        self.assertIn("AEGIS_ADMIN_TOKEN", switch_gate)
        self.assertIn("Refusing to stop it", switch_gate)

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        windows_launcher = readme.split(
            "On Windows PowerShell, use the one-command", 1
        )[1].split("On Linux, start the prepared service", 1)[0]
        self.assertIn("other target is already healthy", windows_launcher)
        self.assertIn("AEGIS_ADMIN_TOKEN", windows_launcher)
        self.assertIn("refuses the switch", windows_launcher)
        self.assertIn("restore it exactly", windows_launcher)

    def test_stop_launcher_never_reads_reusable_aegis_api_token(self) -> None:
        script = (ROOT / "stop-aegis.ps1").read_text(encoding="utf-8")
        shutdown_prefix = script.split("Write-Host \"Stopping AEGIS Docker services", 1)[0]
        self.assertNotIn("AEGIS_API_TOKEN", shutdown_prefix)
        self.assertIn("shutdown_token", shutdown_prefix)
        self.assertIn("X-AEGIS-Confirmation", shutdown_prefix)


if __name__ == "__main__":
    unittest.main()
