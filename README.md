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

### Understand shadow mode

The bundled targets are locked to **shadow mode**. AEGIS reports what it recommends
but deliberately does not stop the request:

| Response field | Meaning in shadow mode |
| --- | --- |
| `recommended_action` | What the detector recommends: `allow`, `review`, or `block` |
| `action` | What the service actually enforces; this remains `allow` |
| `risk_score` | Detector score from `0.0` to `1.0` |
| `traffic_mode` | The active operating mode; bundled targets report `shadow` |

For example, `"recommended_action": "block"` with `"action": "allow"` is expected
in shadow mode. Record and evaluate the recommendation, but do not use the bundled
detectors for automatic production blocking without validating them on your own data
and policy.

#### Can shadow mode be turned off?

Not on either bundled target. The `llava05b` and `qwen25vl3b` profiles are research
candidates, and `aegis serve` deliberately refuses to start a configuration that has
a built-in `target_profile` and a `traffic_mode` other than `shadow`. Simply changing
`"traffic_mode": "shadow"` to `"enforce"` in a bundled configuration produces this
startup error:

```text
Built-in AEGIS targets are research candidates and may only start in shadow mode.
```

AEGIS supports three traffic-mode behaviors for a separately validated custom
deployment:

| Mode | Effective `action` |
| --- | --- |
| `shadow` | Always `allow`; retain the detector result in `recommended_action` |
| `review` | Preserve `allow` and `review`; convert a recommended `block` to `review` |
| `enforce` | Apply the detector's `allow`, `review`, or `block` recommendation directly |

Moving beyond shadow mode is a deployment-validation process, not only a configuration
edit:

1. Collect representative shadow-mode evidence and label the expected decisions.
2. Measure false positives, false negatives, subgroup behavior, and adversarial cases.
3. Train or calibrate a detector and thresholds for the intended model, data, and
   policy. Define review ownership, fail-closed behavior, monitoring, and rollback.
4. Create a new deployment configuration for that validated artifact. A custom
   deployment is represented by `"target_profile": null`; keeping a bundled profile
   name retains the shadow-mode lock. Set `"traffic_mode": "review"` for a staged
   rollout or `"traffic_mode": "enforce"` only after approval.
5. Give the custom deployment a distinct name and audit path, run its deployment
   preflight, and test it before serving traffic:

   ```bash
   aegis doctor --config YOUR-CUSTOM-CONFIG.json
   ```

For example, the relevant fields in a custom review-stage configuration are:

```json
{
  "name": "aegis-custom-review",
  "target_profile": null,
  "traffic_mode": "review"
}
```

This excerpt is not a complete configuration, and setting `target_profile` to `null`
disables the bundled profile-compatibility checks. Do not copy a bundled detector into
this configuration merely to bypass its safety lock.

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

The bundled targets run in shadow mode, so `action` remains `allow` while
`recommended_action` shows what AEGIS would have done. Your application can record
or review that recommendation, but should not treat these research detectors as
validated production blockers.

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
