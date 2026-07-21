# AEGIS

AEGIS is a self-hosted guardrail for prompts sent to multimodal language models. It
runs before the protected model, converts text and images into hidden-state features,
scores them with a small detector, and returns an `allow`, `review`, or `block`
recommendation through an HTTP API.

AEGIS ships with two model-specific targets:

| Target | Input | Minimum host resources |
| --- | --- | --- |
| `llava05b` | One image plus text | 8 GB RAM, about 2.5 GiB GPU memory, 4 GB free disk |
| `qwen25vl3b` | Text, or one image plus text | 16 GB RAM, about 7 GiB GPU memory, 12 GB free disk |

Both targets are research candidates and start in **shadow mode**. AEGIS reports what
it recommends but does not automatically stop requests. Do not use the bundled
detectors for automatic production blocking without validating them for your own data
and policy.

## Requirements

- Git
- Docker Engine or Docker Desktop with Docker Compose v2
- An NVIDIA GPU with a working driver and NVIDIA Container Toolkit support
- Internet access for the initial image build and model download
- Python 3.10 or newer only if you want the optional local client and demo tools

The running container is forced into offline model mode after preparation.

## Install and deploy

The LLaVA target is the smallest and is the easiest place to start.

1. Clone the project and enter its directory.

   ```bash
   git clone <YOUR-AEGIS-REPOSITORY-URL> aegis
   cd aegis
   ```

2. Create the private environment file.

   macOS or Linux:

   ```bash
   cp .env.example .env
   ```

   Windows PowerShell:

   ```powershell
   Copy-Item .env.example .env
   ```

   Open `.env` and replace both placeholder values with two different, long random
   secrets. `HF_TOKEN` is only needed when the model repository requires it.

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

   On Windows PowerShell, use `curl.exe` anywhere the guide shows `curl`.

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
