# AEGIS

AEGIS is a self-hosted guardrail for prompts sent to multimodal language models. It
runs before the protected model, converts text and images into hidden-state features,
scores them with a small detector, and returns an `allow`, `review`, or `block`
recommendation through an HTTP API.

```text
User text/image -> AEGIS -> recommendation -> your application -> protected model
```

AEGIS does not answer the prompt or automatically forward it to another model. Your
application sends the prompt to AEGIS first, reads the decision, and decides what to
do next.

The deliverable artifacts—pipeline, red-teaming tools, annotated benchmark,
and ablation report—are organized under [`deliverables/`](deliverables/README.md).

## Start here if you are new

### What is Docker?

[Docker](https://www.docker.com/) runs an application and its dependencies inside an
isolated package called a *container*. AEGIS uses Docker so that its Python packages,
model runtime, system tools, and startup command are installed consistently. You do
not need a local Python installation to run the full AEGIS service with Docker.

Docker Compose reads [`compose.yaml`](compose.yaml) and manages the AEGIS container,
downloaded-model volume, GPU access, network port, and audit volume for you. In the
commands below:

- `docker compose build` creates the AEGIS container image.
- `docker compose run` performs a one-time task such as downloading a model.
- `docker compose up -d` starts AEGIS in the background.
- `docker compose logs` shows what the service is doing.
- `docker compose stop` stops it without deleting downloaded models.

### Choose a target

AEGIS ships with two model-specific targets:

| Target | Accepted input | Minimum host resources | Recommended for |
| --- | --- | --- | --- |
| `llava05b` | One image plus text | 8 GB RAM, about 2.5 GiB GPU memory, 4 GB free disk | First-time setup |
| `qwen25vl3b` | Text, or one image plus text | 16 GiB RAM, about 7 GiB GPU memory, 12 GB free disk | Text-only or larger experiments |

Start with `llava05b`. On Windows, a computer sold with 8 GB of RAM can expose less
than 8 GiB after hardware reservations. A 16 GB host is strongly recommended even
for LLaVA so that Windows, Docker, and model loading have enough working memory.

Both targets require an NVIDIA GPU. They are research candidates, not validated
production blockers.

The bundled LLaVA target uses the packaged tuned-v3 detector and its
validation-selected thresholds.

### Understand shadow mode

The bundled targets start in **shadow mode** by default. AEGIS reports what it
recommends but deliberately does not stop the request:

| Response field | Meaning in shadow mode |
| --- | --- |
| `recommended_action` | What the detector recommends: `allow`, `review`, or `block` |
| `action` | What the service actually enforces; this remains `allow` |
| `risk_score` | Detector score from `0.0` to `1.0` |
| `traffic_mode` | The active operating mode |

For example, `"recommended_action": "block"` with `"action": "allow"` is expected
in shadow mode. Record and evaluate the recommendation, but do not use the bundled
detectors for automatic production blocking without validating them on your own data
and policy.

AEGIS supports three traffic-mode behaviors:

| Mode | Effective `action` |
| --- | --- |
| `shadow` | Always `allow`; retain the detector result in `recommended_action` |
| `review` | Preserve `allow` and `review`; convert a recommended `block` to `review` |
| `enforce` | Apply the detector's `allow`, `review`, or `block` recommendation directly |

The GUI's Settings page can turn **Shadow mode** off for either bundled target. Off
means `enforce`: subsequent evaluations use the detector recommendation as the
effective action. The service also supports the intermediate `review` mode through a
deployment configuration or its authenticated runtime-control API.

Turning shadow mode off changes behavior; it does not validate the detector. Both
bundled targets remain research candidates and are not approved production blockers.
Before using `review` or `enforce` for real traffic:

1. Collect representative shadow-mode evidence and label the expected decisions.
2. Measure false positives, false negatives, subgroup behavior, and adversarial cases.
3. Train or calibrate a detector and thresholds for the intended model, data, and
   policy. Define review ownership, fail-closed behavior, monitoring, and rollback.
4. Set `"traffic_mode": "review"` for a staged rollout or `"traffic_mode": "enforce"`
   only after approval. Use a separately calibrated detector and configuration when
   the bundled research artifact is not valid for your traffic.
5. Give the validated deployment a distinct name and audit path, run its deployment
   preflight, and test it before serving traffic:

   ```bash
   aegis doctor --config YOUR-CUSTOM-CONFIG.json
   ```

For example, the relevant fields in a review-stage configuration are:

```json
{
  "name": "aegis-custom-review",
  "traffic_mode": "review"
}
```

This excerpt is not a complete configuration.

Finally, AEGIS returns a decision; it does not intercept the protected model by
itself. The calling application must check the effective `action`, send only `allow`
requests downstream, route `review` requests to the defined review process, and reject
`block` requests. The Python client's `guarded_call` method provides this gating only
when `required_traffic_mode` is explicitly set to the deployed mode.

## Prepare your computer

You need:

- Git
- Docker Engine or Docker Desktop with Docker Compose v2
- An NVIDIA GPU with a working driver and container GPU support
- Internet access for the initial image build and model download
- Python 3.10 or newer only for the optional local client and demo tools

The running container is forced into offline model mode after model preparation.

### Windows setup

1. Install the latest NVIDIA driver. Confirm that Windows can see the GPU:

   ```powershell
   nvidia-smi
   ```

2. Install [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/)
   using its WSL 2 backend. Docker documents the Windows GPU requirements in
   [GPU support in Docker Desktop](https://docs.docker.com/desktop/features/gpu/).

3. Start Docker Desktop and wait until its engine reports that it is running. Open a
   new PowerShell window and verify the command-line tools:

   ```powershell
   docker --version
   docker compose version
   ```

If PowerShell says that `docker` is not recognized, Docker Desktop is not installed,
has not finished starting, or the terminal has not picked up its updated `PATH`.
Restart Docker Desktop and open a new PowerShell window before continuing.

## Install and deploy

The LLaVA target is the smallest and is the easiest place to start.

1. Clone the project and enter its directory.

   ```bash
   git clone <YOUR-AEGIS-REPOSITORY-URL> aegis
   cd aegis
   ```

2. Create the private environment file. This file holds secrets that Docker passes
   to AEGIS; it must not be committed or shared.

   macOS or Linux:

   ```bash
   cp .env.example .env
   ```

   Windows PowerShell:

   ```powershell
   Copy-Item .env.example .env
   ```

   Open `.env` and replace both placeholder values with two different, long random
   secrets. The easiest option is the account-free
   [Bitwarden Password Generator](https://bitwarden.com/password-generator/). Select
   a random password at least 48 characters long using letters and numbers, then
   generate a separate value for each setting. Never reuse these values as account
   passwords. For the highest assurance, use the local method below instead of an
   online generator.

   If you prefer not to generate secrets on a website, PowerShell can generate a
   32-byte cryptographically random base64 value locally. Run this block twice:

   ```powershell
   $bytes = New-Object byte[] 32
   $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
   $rng.GetBytes($bytes)
   [Convert]::ToBase64String($bytes)
   $rng.Dispose()
   ```

   `HF_TOKEN` is only needed when the upstream model repository requires it.

   ```dotenv
   AEGIS_API_TOKEN=replace-with-a-long-random-secret
   AEGIS_FINGERPRINT_KEY=replace-with-a-different-long-random-secret
   HF_TOKEN=
   ```

3. Build AEGIS.

   ```bash
   docker compose --profile llava build aegis-llava
   ```

4. Download and verify the pinned LLaVA model files. This is the only step that
   downloads the model.

   ```bash
   docker compose --profile tools run --rm prepare-llava
   ```

5. Start the service.

   ```bash
   docker compose --profile llava up -d aegis-llava
   ```

6. Confirm that AEGIS is ready.

   On Windows PowerShell, use `curl.exe` anywhere the guide shows `curl`, because
   `curl` may be a PowerShell alias rather than the command-line program.

   ```bash
   curl http://127.0.0.1:8766/readyz
   ```

   A successful deployment returns JSON containing `"ok": true`. Model startup may
   take several minutes the first time.

7. Send the included image-and-text example. Replace `YOUR_API_TOKEN` with the value
   from `.env`.

   ```bash
   curl -X POST "http://127.0.0.1:8766/v1/guard" -H "Authorization: Bearer YOUR_API_TOKEN" -H "Content-Type: application/json" --data-binary "@configs/llava_service_warmup_request.example.json"
   ```

   In PowerShell, you can read the token from `.env` without printing it and send the
   request like this:

   ```powershell
   $token = (Get-Content .env |
     Where-Object { $_ -like "AEGIS_API_TOKEN=*" }) -replace "^AEGIS_API_TOKEN=", ""

   curl.exe -X POST "http://127.0.0.1:8766/v1/guard" `
     -H "Authorization: Bearer $token" `
     -H "Content-Type: application/json" `
     --data-binary "@configs/llava_service_warmup_request.example.json"
   ```

### Use Qwen instead

Qwen accepts text-only requests as well as image-and-text requests. Stop LLaVA first,
because both services use port `8766`, then run:

```bash
docker compose --profile llava stop aegis-llava
docker compose --profile qwen build aegis-qwen
docker compose --profile tools run --rm prepare-qwen
docker compose --profile qwen up -d aegis-qwen
curl http://127.0.0.1:8766/readyz
```

## Web Dashboard

The browser dashboard is an optional companion to the AEGIS HTTP service. It does not
replace or change any existing CLI command or API workflow. If you followed the Docker
deployment steps immediately above, the selected LLaVA or Qwen container is already
the required AEGIS service at `http://127.0.0.1:8766`. The dashboard adds separate
**Evaluate**, **History**, and **Settings** pages.

### First-time installation and launch

Complete these steps once per checkout after building and preparing a target in the
deployment guide. The selected container may be running or stopped. The Windows
launcher starts or reuses it, waits for the model to become ready, and then starts the
dashboard. Do not run `aegis serve` alongside the Docker service because both use port
`8766`. The host installation below provides the local `aegis gui` command only.

1. Install Python 3.10 or newer on the host computer.
2. From the project directory, create a virtual environment and install AEGIS.

   Windows PowerShell:

    ```powershell
    py -3 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -e .
    aegis --version
    ```

   If `py -3` reports that no suitable Python is installed, install Python 3.10 or
   newer, reopen PowerShell, and repeat these commands. If PowerShell activation is
   unavailable, install with
   `.\.venv\Scripts\python.exe -m pip install -e .`. You can then use
   `.\.venv\Scripts\aegis.exe` anywhere this guide uses `aegis`.

   macOS or Linux:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -e .
    aegis --version
    ```

3. Start the service and dashboard.

   On Windows PowerShell, use the one-command
   [`start-aegis.ps1`](start-aegis.ps1) launcher:

    ```powershell
    .\start-aegis.ps1
    ```

   This defaults to the prepared LLaVA target. To start a prepared Qwen target instead:

    ```powershell
    .\start-aegis.ps1 -Target qwen25vl3b
    ```

   The script checks Docker and the local installation, stops the alternate target,
   runs the selected `docker compose up -d` command, waits up to 15 minutes for the
   matching model to report ready, loads `AEGIS_API_TOKEN` from `.env` without printing
   it, and runs:

    ```text
    .\.venv\Scripts\aegis.exe gui --project-root <PROJECT> --open-browser
    ```

   The model must already have been built and downloaded by the deployment steps. Use
   `-ReadyTimeoutSeconds N` when the default 900-second startup timeout is insufficient.

   On macOS or Linux, load the token and start the GUI manually:

    ```bash
    export AEGIS_API_TOKEN="$(sed -n 's/^AEGIS_API_TOKEN=//p' .env)"
    aegis gui --open-browser
    ```

   The startup JSON printed in the terminal reports the exact dashboard URL, upstream
   service, project root, and history database path.

### Subsequent usage

Do not recreate the virtual environment or reinstall AEGIS. On Windows, run the same
launcher from the project directory:

```powershell
# LLaVA (default)
.\start-aegis.ps1

# OR Qwen
.\start-aegis.ps1 -Target qwen25vl3b
```

Do not run both commands. The launcher leaves the GUI attached to the terminal; press
`Ctrl+C` to stop the dashboard while leaving the selected Docker service running.

On macOS or Linux, or to perform the startup manually:

1. Check the existing service:

   ```bash
   curl http://127.0.0.1:8766/readyz
   ```

   When the response contains `"ok": true`, the container is already running; skip
   every Docker start command. If it was stopped, restart only the target that you
   were using:

   ```bash
   # LLaVA
   docker compose --profile llava up -d aegis-llava

   # OR Qwen
   docker compose --profile qwen up -d aegis-qwen
   ```

   Do not run both alternatives, and do not run `aegis serve` alongside Docker.
2. Activate the existing host environment.

   Windows PowerShell:

    ```powershell
    .\.venv\Scripts\Activate.ps1
    ```

   macOS or Linux:

    ```bash
    source .venv/bin/activate
    ```

3. In that same terminal, make the Docker service's API token available to the GUI
   process.

   Windows PowerShell:

    ```powershell
    $env:AEGIS_API_TOKEN = (Get-Content .env |
      Where-Object { $_ -like "AEGIS_API_TOKEN=*" }) -replace "^AEGIS_API_TOKEN=", ""
    ```

   macOS or Linux:

    ```bash
    export AEGIS_API_TOKEN="$(sed -n 's/^AEGIS_API_TOKEN=//p' .env)"
    ```

4. Start only the dashboard:

    ```bash
    aegis gui --open-browser
    ```

   If PowerShell activation is unavailable, use
   `.\.venv\Scripts\aegis.exe gui --open-browser` instead.

   The Docker service and its loaded model are reused; this command does not start a
   second model service.

### Use the dashboard

1. Confirm that the service indicator reports **Online**. If it reports **Offline**,
   verify that the AEGIS service is running at the configured URL and that
   `AEGIS_API_TOKEN` matches the service token.
2. Follow the input guidance shown above the form. The dashboard reads the active
   target's capabilities from the running service and disables unsupported
   combinations:

   | Active target | Supported dashboard input |
   | --- | --- |
   | LLaVA (`llava05b`) | Prompt text together with one image |
   | Qwen (`qwen25vl3b`) | Prompt text alone, or prompt text together with one image |

   The bundled targets do not support image-only evaluation. To assess an image, add
   prompt text such as `Assess this image.` GUI uploads accept one PNG, JPEG, or WebP
   image up to 8 MiB.
3. Select **Evaluate**. Analysis leads with the operational **Decision** (`Allow`,
   `Review`, or `Block`) and explains the risk relative to the detector thresholds.
   The binary classifier verdict and effective action remain visible as supporting
   details, together with reasons, traffic mode, model, modality, trace ID, and latency.
   A high score below the calibrated block threshold can therefore have classifier
   verdict `benign` and decision `Review`; the dashboard emphasizes `Review`.
4. Open **History** in the navigation to retrieve earlier GUI submissions. Search
   prompt text or request IDs, filter by effective action, classifier verdict,
   modality, status, or date, and open a row to view the complete prompt, image
   preview, parsed decision, and full JSON response.
5. To remove all GUI records, select **Clear** on the History page and confirm
   the warning. This deletes every stored raw prompt, uploaded image, result, and error
   from the GUI database and cannot be undone through AEGIS. A clear waits for an
   evaluation already in progress and includes that record in the deletion boundary.
   It does not delete the service's separate privacy-safe audit logs.
6. Open **Settings** to inspect the active target and its accepted inputs, select a
   prepared target, or change Shadow mode. Apply the change and confirm the warning:

   - Selecting another target stops the current Compose service, starts the selected
     service, waits for its readiness check, and then applies the chosen traffic mode.
     Only one bundled target uses port `8766` at a time.
   - A model swap can take several minutes. Evaluations may be unavailable while the
     new model loads. If startup fails, AEGIS attempts to restore the previous target
     and traffic mode and reports whether rollback succeeded.
   - Turning **Shadow mode** off selects `enforce`, so a malicious verdict can produce
     an effective `block`. The confirmation is intentional: bundled detectors remain
     research candidates.
   - A Settings-page mode change is runtime-only and is not written to deployment JSON.
     Restarting a model outside the Settings workflow restores the configured mode,
     which is `shadow` in both bundled container configurations. While the GUI remains
     open, it remembers each target's selected mode and reapplies it during a later
     hot-swap; restarting the GUI clears those in-memory preferences.

Target switching controls the fixed services in this checkout's
[`compose.yaml`](compose.yaml). Start `aegis gui` from the project directory, or pass
`--project-root PATH`. A target must have been built and prepared before the GUI can
start it. For example, prepare Qwen for later hot-swapping without starting it:

```bash
docker compose --profile qwen build aegis-qwen
docker compose --profile tools run --rm prepare-qwen
```

Each valid GUI submission is written before it is sent to AEGIS and then updated with
either its result or its upstream error, so service-side failures also remain available
for reference. History persists across GUI restarts in
`~/.aegis/gui-history.sqlite3` by default. Unlike AEGIS's privacy-safe audit log, this
SQLite database contains raw prompt text and uploaded image bytes. Protect it according
to your data-retention policy, do not place it in a shared directory, and stop the GUI
before manually moving or deleting it.

Press `Ctrl+C` in the dashboard terminal to stop only the GUI. The AEGIS service
continues running until it is stopped separately. To stop both components, use the
graceful shutdown workflow below.

### Stop AEGIS

On Windows, open another PowerShell window in the project directory and run:

```powershell
.\stop-aegis.ps1
```

The [`stop-aegis.ps1`](stop-aegis.ps1) launcher authenticates a loopback-only GUI
shutdown request with `AEGIS_API_TOKEN`, waits for an active evaluation or history
clear to finish, and closes the dashboard normally. It then runs Docker Compose's
graceful stop operation for both bundled model services, so it works regardless of
which target is active. Containers, downloaded models, audit data, and GUI history are
retained for the next launch.

The default GUI shutdown allowance is 180 seconds and the Docker stop grace period is
30 seconds. They can be changed when necessary. If the GUI was started manually on a
non-default port, pass that port as well:

```powershell
.\stop-aegis.ps1 -GuiPort 9000 -GuiTimeoutSeconds 300 -ContainerTimeoutSeconds 60
```

On macOS or Linux, stop the foreground GUI with `Ctrl+C`, then stop either possible
Compose target:

```bash
docker compose --profile llava --profile qwen stop aegis-llava aegis-qwen
```

### GUI options

`aegis gui` binds only to a loopback address. The upstream bearer token is read from
`AEGIS_API_TOKEN` by the GUI process and remains server-side; it is never sent to the
browser. Run `aegis gui --help` for the authoritative option list:

```text
--service-url URL       running AEGIS service (default http://127.0.0.1:8766)
--host HOST             loopback bind address (default 127.0.0.1)
--port PORT             dashboard port (default 8767)
--history-db PATH       local SQLite history path
--api-token-env NAME    environment variable holding the upstream token
--timeout-seconds N     upstream evaluation timeout
--project-root PATH     checkout containing compose.yaml for target switching
--target-switch-timeout-seconds N
                         maximum wait for a target to become ready
--open-browser          open the dashboard after startup
```

## Troubleshooting startup

The Compose services use `restart: unless-stopped`. If startup fails, Docker retries
automatically, so the same traceback may appear repeatedly. This does not necessarily
mean that several different things are failing.

First, capture only the newest logs:

```bash
docker compose --profile llava logs --since=2m --tail=120 aegis-llava
```

Use `aegis-qwen` instead when running Qwen. Stop a restart loop while investigating:

```bash
docker compose --profile llava stop aegis-llava
```

### `docker` is not recognized

Make sure Docker Desktop is installed and running, then open a new terminal and run:

```powershell
docker --version
docker compose version
```

On Windows, Docker Desktop must finish starting its Linux engine before Compose
commands work.

### `aegis` is not recognized

The host-side GUI command is installed in the project's virtual environment. From the
project directory, activate it before running `aegis`:

```powershell
.\.venv\Scripts\Activate.ps1
aegis --version
```

If activation is unavailable, invoke the installed executable directly:

```powershell
.\.venv\Scripts\aegis.exe gui --open-browser
```

If that file does not exist, repeat the first-time GUI installation step with
`.\.venv\Scripts\python.exe -m pip install -e .`.

### `/readyz` does not return `"ok": true`

The first model startup can take several minutes. Check container status and logs:

```bash
docker compose --profile llava ps -a
docker compose --profile llava logs --tail=120 aegis-llava
```

If the container is still running and the log shows model loading or warmup, wait and
try `/readyz` again. If it has exited, read the final traceback rather than repeatedly
calling the readiness endpoint.

### Resource preflight failure

AEGIS checks resources before loading a large model. A failure includes the actual
and required byte counts. Typical checks are:

- `total_physical_memory`: the RAM visible inside the container is below the target
  minimum. An 8 GB Windows computer can expose only about 7.6 GiB because some memory
  is hardware-reserved. Upgrading to 16 GB is the reliable fix.
- `available_physical_memory`: close browsers, development tools, games, and other
  memory-heavy applications, or restart the computer before starting AEGIS.
- `available_virtual_memory`: increase the Windows paging file and, when using the
  WSL 2 backend, the WSL swap allocation.
- `cuda_device_memory`: use a GPU with enough VRAM and confirm that `nvidia-smi`
  works both on the host and through Docker's GPU support.
- `model_cache_size`: rerun the appropriate `prepare-llava` or `prepare-qwen`
  download command and check that it finishes successfully.

#### Increase Windows virtual memory

The Windows paging file and WSL swap are related but separate settings.

1. Search for **View advanced system settings**.
2. Select **Performance > Settings > Advanced > Virtual memory > Change**.
3. Enable **Automatically manage paging file size for all drives**.
4. Restart Windows.

For Docker Desktop's WSL 2 environment, open or create `%UserProfile%\.wslconfig`:

```powershell
notepad "$env:USERPROFILE\.wslconfig"
```

For an 8 GB swap allocation, enter:

```ini
[wsl2]
swap=8GB
```

Save the file, completely quit Docker Desktop, and run:

```powershell
wsl --shutdown
```

Start Docker Desktop again. Microsoft documents this file and the restart requirement
in [Advanced settings configuration in WSL](https://learn.microsoft.com/windows/wsl/wsl-config).
Swap is slower than physical RAM, so this may allow a research run to start but is not
a substitute for sufficient installed memory.

### `target_profile_resource_budget_matches` failure

This check means a deployment configuration asks for fewer resources than the bundled
target profile's validated minimum. Normally, restore the value shown as
`profile_minimum` in the error rather than weakening the profile.

If a developer intentionally maintains an unsupported low-resource research fork,
the resource values in both the deployment JSON and `AEGIS/target_profiles.py` must
agree. Changing only the JSON produces this error. Such a fork must be revalidated and
must not be treated as a production-safe profile.

### Source or configuration changes do not appear in the container

Files under `AEGIS/` and `configs/` are copied into the image at build time. Rebuild
and recreate the service after editing them:

```bash
docker compose --profile llava stop aegis-llava
docker compose --profile llava build --no-cache aegis-llava
docker compose --profile llava up -d --force-recreate aegis-llava
docker compose --profile llava logs --since=2m --tail=120 aegis-llava
```

Do not rely on older unbounded log output after rebuilding; `--since` and `--tail`
make it clear which error belongs to the current container.

## Local Python installation

A local installation is useful for the lightweight demo and for importing the AEGIS
client in another Python application. The full GPU service is most reliably deployed
with Docker from the cloned project as shown above; the Python package by itself does
not include the container deployment assets or downloaded model files.

```bash
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
- macOS or Linux: `source .venv/bin/activate`

Then install and check AEGIS:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
aegis demo
```

The demo checks the installation with a deterministic test provider; it is not a
trained safety model.

## Usage examples

These examples assume AEGIS is running on `http://127.0.0.1:8766`. Replace
`YOUR_API_TOKEN` with the `AEGIS_API_TOKEN` value in `.env`.

### Check a text prompt

Text-only requests are supported by the Qwen target. LLaVA requires both text and an
image.

```bash
curl -X POST "http://127.0.0.1:8766/v1/guard" -H "Authorization: Bearer YOUR_API_TOKEN" -H "Content-Type: application/json" --data '{"request_id":"example-text-001","text":"Explain why the sky appears blue."}'
```

The response contains one entry in `decisions`. The most useful fields are:

- `recommended_action`: AEGIS's `allow`, `review`, or `block` recommendation.
- `action`: what the service actually enforces.
- `risk_score`: the detector score from `0.0` to `1.0`.
- `traffic_mode`: the active operating mode.

The bundled targets start in shadow mode, so `action` remains `allow` while
`recommended_action` shows what AEGIS would have done until an operator changes the
mode. Do not treat these research detectors as validated production blockers merely
because runtime enforcement is available.

### Check an image and prompt

Both targets accept one PNG, JPEG, or WebP image encoded as base64. The deployment
example file already contains a small PNG and can be sent directly:

```bash
curl -X POST "http://127.0.0.1:8766/v1/guard" -H "Authorization: Bearer YOUR_API_TOKEN" -H "Content-Type: application/json" --data-binary "@configs/llava_service_warmup_request.example.json"
```

Use `configs/service_warmup_request.example.json` instead when running Qwen.

### Use the Python client

Set the API token in the terminal where your Python application will run.

macOS or Linux:

```bash
export AEGIS_API_TOKEN='the-same-value-used-in-.env'
```

Windows PowerShell:

```powershell
$env:AEGIS_API_TOKEN = 'the-same-value-used-in-.env'
```

Then evaluate a Qwen text request:

```python
import os

from AEGIS import GuardrailClient

client = GuardrailClient(
    "http://127.0.0.1:8766",
    api_token=os.environ["AEGIS_API_TOKEN"],
    required_traffic_mode="shadow",
)

result = client.guard(
    {
        "request_id": "python-example-001",
        "text": "Explain why the sky appears blue.",
    }
)

print(result.to_dict())
print("Effective action:", result.action)
print("Recommendation:", result.decisions[0].recommended_action)
```

To send an image with either target, construct the payload like this:

```python
import base64
from pathlib import Path

image_bytes = Path("example.png").read_bytes()
result = client.guard(
    {
        "request_id": "python-image-001",
        "text": "Describe this image.",
        "images": [
            {
                "media_type": "image/png",
                "base64": base64.b64encode(image_bytes).decode("ascii"),
            }
        ],
    }
)

print(result.to_dict())
```

## CLI examples

Run `aegis --help` for the command list or `aegis COMMAND --help` for every option.
Operational commands print machine-readable JSON; help and version output are plain
text.

Check the installed version:

```bash
aegis --version
```

Run the dependency-light installation demo with the default prompt or your own text:

```bash
aegis demo
aegis demo "Explain why the sky appears blue." --request-id cli-example-001
```

List the supported targets and whether their bundled detector and example-request
files are present:

```bash
aegis targets
```

Check disk space and whether a target's pinned model files are already available:

```bash
aegis prepare --target llava05b --root .
aegis prepare --target qwen25vl3b --root .
```

Add `--download` to download and verify missing files. The Docker preparation
commands in the deployment steps are recommended; a local download requires the
optional dependencies installed with `python -m pip install -e ".[mllm]"`.

```bash
aegis prepare --target llava05b --root . --download
```

Run the deployment preflight for either target:

```bash
aegis doctor --config configs/aegis.llava.deployment.container.json
aegis doctor --config configs/aegis.deployment.container.json
```

`doctor` exits unsuccessfully and reports the failed checks when the required model,
GPU, memory, environment variables, or other deployment resources are missing.

Start the HTTP service directly with a prepared GPU environment:

```bash
aegis serve --config configs/aegis.llava.deployment.container.json
```

For a normal deployment, use Docker Compose instead. The `aegis-llava` and
`aegis-qwen` services invoke this command with the correct container paths,
dependencies, environment variables, and model volume.

## Operate or remove the deployment

View logs:

```bash
docker compose --profile llava logs -f aegis-llava
```

Stop LLaVA without deleting its downloaded model volume:

```bash
docker compose --profile llava stop aegis-llava
```

To stop Qwen, substitute `qwen` and `aegis-qwen`. To remove the containers and both
named volumes, including downloaded model files and audit records, run:

```bash
docker compose --profile llava --profile qwen down --volumes
```

Keep `.env` private. The API token protects the service, and the independent
fingerprint key protects prompt fingerprints written to the audit log.
