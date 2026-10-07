# AEGIS

AEGIS is a self-hosted guardrail for prompts sent to multimodal language models. It
runs before the protected model, converts text and images into hidden-state features,
scores them with a dual-head detector pair, and returns an `allow`, `review`, or `block`
recommendation through an HTTP API.

```text
User text/image -> AEGIS -> recommendation -> your application -> protected model
```

AEGIS does not answer the prompt or automatically forward it to another model. Your
application sends the prompt to AEGIS first, reads the decision, and decides what to
do next.

This README and `aegis targets` describe the current bundled runtime and target
profiles. Historical benchmark and research material is retained under
[`deliverables/`](deliverables/README.md); it describes earlier experiments. The v7
research updates and frozen qualification evidence are outside this runtime release.
`/readyz` proves only coarse readiness and the active target;
the authenticated `/v1/status` response is the operational source of truth for the
detector identity actually loaded by a running service.

## Start here if you are new

### What is Docker?

[Docker](https://www.docker.com/) runs an application and its dependencies inside an
isolated package called a *container*. AEGIS uses Docker so that its Python packages,
model runtime, system tools, and startup command are installed consistently. You do
not need a local Python installation to run the full AEGIS service with Docker.

Docker Compose reads [`compose.yaml`](compose.yaml) and manages the AEGIS container,
downloaded-model volume, GPU access, network port, and audit volume for you. Model
preparation mounts the cache read-write; detector runtime services mount that verified
cache read-only. Packaged code, configurations, and detector artifacts remain
root-owned and read-only to the non-root runtime user.

Every service in `compose.yaml` is profile-gated. A bare `docker compose build`,
`up`, `logs`, or `stop` command selects no AEGIS service, so always include the
profile and service shown in this guide. For the LLaVA workflow:

- `docker compose --profile llava build aegis-llava` creates the AEGIS image.
- `docker compose --profile tools run --rm prepare-llava` performs the one-time model
  download and verification.
- `docker compose --profile llava up -d aegis-llava` starts AEGIS in the background.
- `docker compose --profile llava logs aegis-llava` shows service logs.
- `docker compose --profile llava stop aegis-llava` stops the service without deleting
  downloaded models.

The Qwen sections use the corresponding `qwen`, `aegis-qwen`, and `prepare-qwen`
names.

### Choose a target

AEGIS ships with two model-specific targets:

| Target | Accepted input | Deployment preflight minimums | Recommended for |
| --- | --- | --- | --- |
| `llava05b` | Non-empty text plus exactly one image | 8,000,000,000 bytes total RAM; 2 GiB available RAM; 4 GiB available virtual memory; 2.5 GiB free GPU memory; 4 GiB free disk; 1.5 GiB model cache | First-time setup |
| `qwen25vl3b` | Non-empty text plus exactly one image | 15 GiB total RAM; 6 GiB available RAM; 8 GiB available virtual memory; 7 GiB free GPU memory; 12 GiB free disk; 6 GiB model cache | Larger-model experiments |

The GPU-memory preflight measures currently free VRAM. Close other GPU workloads
before running `doctor`; total GPU capacity alone is not sufficient.

Start with `llava05b`. On Windows, a computer sold with 8 GB of RAM can expose less
than 8 GiB after hardware reservations. A 16 GB host is strongly recommended even
for LLaVA so that Windows, Docker, and model loading have enough working memory.

Local validation completed LLaVA v7 CUDA model loading, warmup, authenticated
status, dashboard inference, and graceful shutdown. Qwen validation stopped at
the RAM and free-VRAM preflight before model loading; its CUDA loading, warmup,
and live inference remain pending on a host meeting those requirements. Run the
preflight and API checks below on your own host before using either target.

Both targets require an NVIDIA GPU and remain research candidates. Validate their
decisions on traffic representative of your deployment before using them as a
production enforcement gate.

### Detector contracts

Both deployments use a v7 dual-head detector. One fused final-layer
`text_image_tokens` extraction is split into aligned text-token and image-token
views, and the two head recommendations are combined with OR precedence:
`block > review > allow`. Both built-in targets still require non-empty text and
exactly one image; unsupported modalities are rejected before model inference.

| Target / head | Detector artifact | Pooling | Dimension | Recommend allow below | Recommend review in range | Recommend block at or above |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `llava05b` v7 text | `models/aegis/llava05b_v7_dual_or/llava05b_text_head_v1.npz` | `text_tokens` | 896 | `0.9773003604375604` | `0.9773003604375604` to below `0.9973003604375604` | `0.9973003604375604` |
| `llava05b` v7 image | `models/aegis/llava05b_v7_dual_or/llava05b_image_head_v1.npz` | `image_tokens` | 896 | `0.41109073768976995` | `0.41109073768976995` to below `0.43109073768976996` | `0.43109073768976996` |
| `qwen25vl3b` v7 text | `models/aegis/qwen25vl3b_v7_dual_or/qwen25vl3b_text_head_v1.npz` | `text_tokens` | 2048 | `0.9775257227486033` | `0.9775257227486033` to below `0.9975257227486033` | `0.9975257227486033` |
| `qwen25vl3b` v7 image | `models/aegis/qwen25vl3b_v7_dual_or/qwen25vl3b_image_head_v1.npz` | `image_tokens` | 2048 | `0.1490249723273814` | `0.1490249723273814` to below `0.1690249723273814` | `0.1690249723273814` |

The reference SHA-256 digests of the bundled artifacts are:

- LLaVA v7 text head: `cd8502fc73ecaf1e6597d318d63a82fcdf7abae628c1a54e5390b82b2a999a61`
- LLaVA v7 image head: `84b12df0887f420318e2f3f530d0dc4391e8c1e1e5d1e15ff28f4f719c6b80b5`
- Qwen v7 text head: `8818a75eb0fbe9fb1a0acb9f2035cb8b9f48f083ea1ee7aee03e15ec549533c8`
- Qwen v7 image head: `e17eff8de97b45c22887139417d84bdd32ed067c9615c0d601f9af4467b18deb`

The corresponding runtime detector identities are:

| Target | Runtime detector identity |
| --- | --- |
| `llava05b` | `25cebcf1e91125f4823176111d88cddae7379a0f70a1d8cc8af5ff07e22f08f6` |
| `qwen25vl3b` | `960f53c6fa3e584bfd2ff16a6ec3cd2d54f592e640e32da55748c2b8facd3a8e` |

`doctor` enforces these detector digests and requires the provider's hidden layer,
pooling, feature dimension, model revision, tokenizer revision, and preprocessing
fingerprint to match the selected artifact. Manual file verification remains useful
when transferring a checkout. On Windows, use:

```powershell
Get-FileHash -Algorithm SHA256 models\aegis\llava05b_v7_dual_or\*.npz
Get-FileHash -Algorithm SHA256 models\aegis\qwen25vl3b_v7_dual_or\*.npz
```

On Linux, use:

```bash
sha256sum models/aegis/llava05b_v7_dual_or/*.npz
sha256sum models/aegis/qwen25vl3b_v7_dual_or/*.npz
```

These thresholds are selected by the shipped detector artifact and deployment
configuration. Do not edit them independently of a documented calibration and
validation process. Run `aegis targets` from the repository root to inspect the
installed artifact paths, model revisions, resource requirements, and current
caveats.

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
in shadow mode. Shadow mode is observational and must not be used as an enforcement
gate. Record and evaluate the recommendation, but do not use the bundled detectors
for automatic production blocking without validating them on your own data and
policy.

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
itself. A gating integration must first verify that the service is in the intended
`review` or `enforce` mode, send only effective `allow` requests downstream, route
`review` requests to the defined review process, and reject `block` requests. It must
also fail closed on authentication, transport, HTTP, protocol, timeout, and
`guardrail_error` failures. The Python client's `guarded_call` method enforces those
decision and service-failure checks when `required_traffic_mode` is explicitly set to
the deployed mode. The client and service reject unknown or ambiguous request fields,
and provider CSV staging preserves scalar-looking text such as `00123`, `NA`, and
`true` exactly. The ordinary `guard` API can carry `request_id` and `metadata` for
correlation, but neither is detector input. `guarded_call` therefore rejects both,
along with mutable local image paths, instead of passing unevaluated or replaceable
content to its callback.

## Prepare your computer

You need:

- Git
- Docker Engine or Docker Desktop with the Docker Compose CLI plugin
  (`docker compose`; Compose v2 or later)
- An NVIDIA GPU with a working driver and container GPU support
- Internet access for the initial image build and model download
- `curl` for the readiness and HTTP API examples (`curl.exe` on Windows)
- Python 3.10 or newer for the optional local client, demo tools, and dashboard;
  the optional host-side model-download workflow specifically uses Python 3.11

The running container is forced into offline model mode after model preparation.

The Python client and dashboard bypass environment proxies for loopback and clear-text
HTTP destinations. Remote HTTPS origins may use the operating system's proxy settings;
use only a trusted proxy and retain end-to-end TLS certificate validation.

### Platform support

The bundled GPU container has been built and validated only for x86-64 (`amd64`)
Windows/WSL 2 and Linux hosts. ARM64 and NVIDIA Jetson deployments require a
separately built and validated runtime.

- **Windows:** run Docker Desktop with its WSL 2 backend and NVIDIA GPU support, as
  described below.
- **Linux:** run Docker Engine with the Docker Compose CLI plugin and configure the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  for Docker. Confirm that `nvidia-smi` works on the host before continuing.
- **macOS:** the bundled detector services cannot run locally because both require an
  NVIDIA CUDA device exposed to the container. The dependency-light demo, Python
  client, and dashboard can run on macOS, but the client or dashboard must connect to
  an AEGIS service hosted on a supported Windows or Linux machine over HTTPS.

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
   curl.exe --version
   ```

If PowerShell says that `docker` is not recognized, Docker Desktop is not installed,
has not finished starting, or the terminal has not picked up its updated `PATH`.
Restart Docker Desktop and open a new PowerShell window before continuing.

### Linux setup

1. Install a supported NVIDIA driver and confirm that `nvidia-smi` works.
2. Install Docker Engine and the Docker Compose CLI plugin for your distribution.
3. Install and configure the NVIDIA Container Toolkit for Docker using the official
   guide linked above, then restart Docker as instructed there.
4. Confirm the required commands are available:

   ```bash
   nvidia-smi
   docker --version
   docker compose version
   curl --version
   ```

## Install and deploy

These local service steps apply to the supported Windows and Linux configurations
above. The LLaVA target is the smallest and is the easiest place to start.

1. Clone the project and enter its directory.

   ```bash
   git clone https://github.com/basilpesto2/AEGIS.git aegis
   cd aegis
   ```

2. Create the private environment file. This file holds secrets that Docker passes
   to AEGIS; it must not be committed or shared.

   Linux:

   ```bash
   cp .env.example .env
   chmod 600 .env
   ```

   Windows PowerShell:

   ```powershell
   Copy-Item .env.example .env
   ```

   Open `.env` and replace the API-token and fingerprint-key placeholders with two
   different random secrets. Also set `AEGIS_ADMIN_TOKEN` to a third independent
   random secret if you will use Settings-page target or traffic-mode changes, or use
   `start-aegis.ps1` to switch away from a different target that is already healthy.
   A first start and same-target reuse do not require the administration token. The
   easiest option is the account-free
   [Bitwarden Password Generator](https://bitwarden.com/password-generator/). Select
   a random password at least 48 characters long using letters and numbers, then
   generate a separate value for each setting. Never reuse these values as account
   passwords. For the highest assurance, use the local method below instead of an
   online generator. AEGIS rejects configured secrets shorter than 32 characters,
   common/example placeholders, repeated or character-dominated values, simple
   sequences, and reuse between the API token, admin token, and fingerprint key.

   If you prefer not to generate secrets on a website, PowerShell can generate a
   32-byte cryptographically random base64 value locally. Run this block once for each
   secret you configure:

   ```powershell
   $bytes = New-Object byte[] 32
   $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
   $rng.GetBytes($bytes)
   [Convert]::ToBase64String($bytes)
   $rng.Dispose()
   ```

   On Linux with OpenSSL installed, run this command once for each secret you
   configure:

   ```bash
   openssl rand -base64 32
   ```

   `HF_TOKEN` is only needed when the upstream model repository requires it.

   ```dotenv
   AEGIS_API_TOKEN=replace-with-a-long-random-secret
   AEGIS_ADMIN_TOKEN=
   AEGIS_FINGERPRINT_KEY=replace-with-a-different-long-random-secret
   HF_TOKEN=
   ```

   Confirm that Git ignores the secret file:

   ```bash
   git check-ignore .env
   ```

   Compose gives an already exported shell variable precedence over the value in
   `.env`. For the manual commands in this guide, use a clean terminal or unset stale
   `AEGIS_API_TOKEN`, `AEGIS_ADMIN_TOKEN`, `AEGIS_FINGERPRINT_KEY`, and `HF_TOKEN`
   variables before starting. The Windows launcher deliberately reloads the three
   AEGIS values from `.env`.

3. Build AEGIS.

   ```bash
   docker compose --profile llava build aegis-llava
   ```

4. Download and verify the pinned LLaVA model files. This is the only step that
   downloads the model.

   ```bash
   docker compose --profile tools run --rm prepare-llava
   ```

5. Run the deployment preflight inside the prepared container environment.

   ```bash
   docker compose --profile llava run --rm --no-deps aegis-llava doctor --config /app/configs/aegis.llava.deployment.container.json
   ```

   The command must finish with JSON containing `"ok": true`. It checks that the
   configured detector set loads; enforces its SHA-256, hidden layer, pooling, exact
   model/tokenizer revisions, preprocessing fingerprint, and complete model-content
   digest; and checks free GPU memory, RAM, virtual memory, disk, authentication,
   request limits, and the writable audit path without starting the HTTP server.

6. Start the service.

   ```bash
   docker compose --profile llava up -d aegis-llava
   ```

7. Confirm that AEGIS is ready.

   On Windows PowerShell, use `curl.exe` anywhere the guide shows `curl`, because
   `curl` may be a PowerShell alias rather than the command-line program.

   The automatic proxy bypass described earlier applies to the Python client and
   dashboard, not to the separate `curl` program. If your shell defines proxy
   variables, add `--noproxy 127.0.0.1` to each local `curl` or `curl.exe` command.

   ```bash
   curl http://127.0.0.1:8766/readyz
   ```

   A successful LLaVA deployment returns JSON containing `"ok": true` and
   `"target_profile": "llava05b"`. Model startup may take several minutes the first
   time, and the HTTP listener may not accept connections until initial worker startup
   finishes. If the connection is refused, wait and repeat the readiness request.

8. Send the included image-and-text example. Replace `YOUR_API_TOKEN` with the value
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

Qwen uses the same image-text request shape as LLaVA: every request needs non-empty prompt
text and exactly one image. Stop LLaVA first because both services use port `8766`,
then build, prepare, preflight, and start Qwen:

```bash
docker compose --profile llava stop aegis-llava
docker compose --profile qwen build aegis-qwen
docker compose --profile tools run --rm prepare-qwen
docker compose --profile qwen run --rm --no-deps aegis-qwen doctor --config /app/configs/aegis.deployment.container.json
docker compose --profile qwen up -d aegis-qwen
curl http://127.0.0.1:8766/readyz
```

Send `configs/service_warmup_request.example.json` rather than the LLaVA example when
testing the Qwen service. Its readiness response must contain `"ok": true` and
`"target_profile": "qwen25vl3b"`. If the listener is not ready, wait and repeat the
`curl` command until both fields match; do not send an evaluation before then.

## Web Dashboard

The browser dashboard is an optional companion to the AEGIS HTTP service. It does not
replace or change any existing CLI command or API workflow. If you followed the Docker
deployment steps immediately above, the selected LLaVA or Qwen container is already
the required AEGIS service at `http://127.0.0.1:8766`. The dashboard adds separate
**Evaluate**, **History**, and **Settings** pages.

### First-time installation and launch

Complete these steps once per checkout. For a local Windows or Linux deployment,
first build and prepare a target using the deployment guide. The selected container
may be running or stopped: the Windows launcher starts or reuses it, while the Linux
commands below start it explicitly. On macOS, use the remote-service instructions
below instead. Do not run `aegis serve` alongside a local Docker service because both
use port `8766`. Only the `aegis gui` command is needed from the host installation for
this dashboard workflow.

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
    .\start-aegis.ps1 -Target qwen25vl3b -ReadyTimeoutSeconds 1200
    ```

   If the other target is already healthy, the launcher requires the distinct
   `AEGIS_ADMIN_TOKEN` that was configured when that running service started. It
   authenticates and records the current traffic mode so that an unsuccessful switch
   can restore it exactly. Adding a token to `.env` does not enable administration
   in an already-running service that started without one. In that case, or when the
   administration token is intentionally unset, run `.\stop-aegis.ps1` before
   launching the other target. The launcher otherwise refuses the switch without
   stopping the healthy service.

   The script checks Docker and the local installation, validates the configured
   secrets, proves that any reported ready service is the expected Compose container
   bound at `127.0.0.1:8766`, and reuses it when it is already the requested target.
   With the administration token matching that running service, it can authenticate,
   record, and stop a ready alternate. It then runs the selected profile-qualified
   Compose `up -d --no-deps --build` command and waits for the matching model using
   the configured readiness timeout (1,200 seconds, or 20 minutes, by default), loads
   the API and configured administration tokens from `.env` without printing them,
   and runs:

    ```text
    .\.venv\Scripts\aegis.exe gui --project-root <PROJECT> --open-browser
    ```

   The model cache must already have been downloaded by the deployment steps. The
   launcher rebuilds the image when its inputs changed. The readiness timeout starts
   after Compose finishes its build/start command and covers container preflight and
   model startup. On a build, start, or readiness failure, it prints recent logs when
   available, stops the failed selection, and restarts the exact retained container
   that was previously ready, with its exact traffic mode, before returning an error.
   This avoids recreating the prior target from a newly built shared image after that
   build failed to start. If the retained container no longer exists, rollback fails
   explicitly rather than silently substituting an image or mode. Secrets loaded for
   the child processes are restored to their prior process-environment values when the
   launcher exits.

   If the selected service is healthy but reports a detector identity that does not
   match the current target profile, the launcher deliberately leaves that service
   untouched and exits. Stop the stale selection explicitly, then rerun the launcher:

    ```powershell
    .\stop-aegis.ps1
    .\start-aegis.ps1 -Target llava05b
    ```

   Substitute `qwen25vl3b` for the target when required. A newly started service is
   accepted only after its authenticated status reports the exact configured detector
   identity.

   On Linux, start the prepared service, wait for `/readyz` to return `"ok": true`,
   then load the token and start the GUI. This example uses LLaVA; substitute the Qwen
   profile and service names when that is the prepared target.

    ```bash
    docker compose --profile llava up -d --build aegis-llava
    curl http://127.0.0.1:8766/readyz
    ```

   Repeat the readiness command until it reports `"ok": true` for the selected
   target. Then load the tokens and start the GUI:

    ```bash
    export AEGIS_API_TOKEN="$(sed -n 's/^AEGIS_API_TOKEN=//p' .env)"
    export AEGIS_ADMIN_TOKEN="$(sed -n 's/^AEGIS_ADMIN_TOKEN=//p' .env)"
    aegis gui --open-browser
    ```

   On macOS, connect the dashboard to an operator-provided HTTPS deployment. Set the
   matching service token without committing it to the repository:

    ```bash
    export AEGIS_API_TOKEN='token-issued-for-the-remote-service'
    aegis gui --service-url https://aegis.example.com --open-browser
    ```

   Evaluation and local GUI history remain available. Local Compose target switching
   is unavailable for a remote service. Runtime traffic-mode control remains available
   only if the remote service advertises it and a distinct authorized administration
   token is supplied through `AEGIS_ADMIN_TOKEN`.

   The startup JSON printed in the terminal reports the per-launch bootstrap URL,
   clean dashboard origin, upstream service, project root, history database path, and
   private shutdown-state path. The fragment credential in the bootstrap URL is
   temporary and local; do not paste or share it.

### Subsequent usage

Do not recreate the virtual environment or reinstall AEGIS. On Windows, run the same
launcher from the project directory:

```powershell
# LLaVA (default)
.\start-aegis.ps1

# OR Qwen
.\start-aegis.ps1 -Target qwen25vl3b -ReadyTimeoutSeconds 1200
```

Do not run both commands. The launcher leaves the GUI attached to the terminal; press
`Ctrl+C` to stop the dashboard while leaving the selected Docker service running.
When the requested target differs from another healthy target, configure
`AEGIS_ADMIN_TOKEN` before using the launcher or run `.\stop-aegis.ps1` first.
Same-target reuse and a first start still work without the administration token.

On Linux, or to perform the local startup manually on Windows:

1. Check the existing service:

   ```bash
   curl http://127.0.0.1:8766/readyz
   ```

   A response containing `"ok": true` and the intended `target_profile` identifies
   only a reusable *candidate*. It does not expose the loaded detector identity. Do
   not skip the Docker start command unless you also complete the authenticated
   identity check in step 4. Both targets use port `8766`; a healthy response for the
   other profile is not the requested detector. Stop that profile before continuing.
   If the service is absent, stale, or not already identity-verified, start only the
   prepared target:

   ```bash
   # Stop the alternate target when it is the one currently reported by /readyz
   docker compose --profile qwen stop aegis-qwen       # before starting LLaVA
   docker compose --profile llava stop aegis-llava     # before starting Qwen

   # LLaVA
   docker compose --profile llava up -d --build aegis-llava

   # OR Qwen
   docker compose --profile qwen up -d --build aegis-qwen
   ```

   Do not run both alternatives, and do not run `aegis serve` alongside Docker.
2. Activate the existing host environment.

   Windows PowerShell:

    ```powershell
    .\.venv\Scripts\Activate.ps1
    ```

   Linux:

    ```bash
    source .venv/bin/activate
    ```

3. In that same terminal, make the Docker service's API token available to the GUI
   process.

    Windows PowerShell:

    ```powershell
    $env:AEGIS_API_TOKEN = (Get-Content .env |
      Where-Object { $_ -like "AEGIS_API_TOKEN=*" }) -replace "^AEGIS_API_TOKEN=", ""
    $env:AEGIS_ADMIN_TOKEN = (Get-Content .env |
      Where-Object { $_ -like "AEGIS_ADMIN_TOKEN=*" }) -replace "^AEGIS_ADMIN_TOKEN=", ""
    ```

    Linux:

    ```bash
    export AEGIS_API_TOKEN="$(sed -n 's/^AEGIS_API_TOKEN=//p' .env)"
    export AEGIS_ADMIN_TOKEN="$(sed -n 's/^AEGIS_ADMIN_TOKEN=//p' .env)"
    ```

4. Before reusing the service, query its authenticated status and require both the
   intended `target_profile` and the exact `detector_identity_sha256` listed in the
   identity table above. A mismatch is a stale or unintended runtime; stop it and run
   the matching Compose start command from step 1.

   Windows PowerShell:

    ```powershell
    Invoke-RestMethod http://127.0.0.1:8766/v1/status `
      -Headers @{ Authorization = "Bearer $env:AEGIS_API_TOKEN" } |
      Select-Object target_profile, detector_identity_sha256
    ```

   Linux:

    ```bash
    curl "http://127.0.0.1:8766/v1/status" \
      -H "Authorization: Bearer $AEGIS_API_TOKEN"
    ```

5. Start only the dashboard:

    ```bash
    aegis gui --open-browser
    ```

   If PowerShell activation is unavailable, use
   `.\.venv\Scripts\aegis.exe gui --open-browser` instead.

   The Docker service and its loaded model are reused; this command does not start a
   second model service.

On macOS, activate the existing environment, export the remote service's token, and
start only the dashboard:

```bash
source .venv/bin/activate
export AEGIS_API_TOKEN='token-issued-for-the-remote-service'
aegis gui --service-url https://aegis.example.com --open-browser
```

### Use the dashboard

1. Confirm that the service indicator reports **Online**. This proves readiness only:
   `/readyz` is unauthenticated, so an incorrect token can still show **Online** while
   evaluations and Settings fail. If it reports **Offline**, verify that the AEGIS
   service is running at the configured URL. If evaluation reports an authentication
   error or mode control is unavailable unexpectedly, verify that `AEGIS_API_TOKEN`
   matches the service evaluation token and that the optional, distinct
   `AEGIS_ADMIN_TOKEN` matches the service administration token.
2. Follow the input guidance shown above the form. The dashboard reads the active
   target's capabilities from the running service and disables unsupported
   combinations:

   | Active target | Supported dashboard input |
   | --- | --- |
   | LLaVA (`llava05b`) | Prompt text together with one image |
   | Qwen (`qwen25vl3b`) | Prompt text together with one image |

   The built-in targets support neither text-only nor image-only evaluation. Supply both
   fields; prompt text such as `Assess this image.` is sufficient when the image
   carries the main content. GUI uploads accept one PNG, JPEG, or WebP image up to
   8 MiB.
3. Select **Evaluate**. The prominent **Decision** banner displays the detector's
   recommendation (`Allow`, `Review`, or `Block`) and explains the risk relative to
   the detector thresholds; it is not the effective action. The binary classifier
   verdict and effective action remain visible separately, together with reasons,
   traffic mode, model, modality, trace ID, and latency. In shadow mode the banner can
   show `Block` while the effective action remains `Allow`. A high score below the
   calibrated block threshold can have classifier verdict `benign` and recommendation
   `Review`.
4. Open **History** in the navigation to retrieve earlier GUI submissions. Search
   prompt text or request IDs, filter by effective action, classifier verdict,
   modality, status, or date, and open a row to view the complete prompt, image
   preview, parsed decision, and full JSON response.
5. To remove all GUI records, select **Clear** on the History page and confirm the
   warning. A clear waits for an evaluation already in progress and includes that
   record in the deletion boundary. AEGIS enables SQLite `secure_delete`, deletes the
   rows, truncates the write-ahead log, and runs `VACUUM` to compact the database.
   Filesystem snapshots, backups, SSD wear-levelled blocks, or storage outside
   SQLite's control can still retain remnants; use storage-level sanitization when
   that threat is in scope. This operation does not delete the service's separate
   privacy-safe audit logs.
6. Open **Settings** to inspect the active target and its accepted inputs, select a
   prepared target, or change Shadow mode. Apply the change and confirm the warning:

   Settings target and traffic-mode changes require the same distinct
   `AEGIS_ADMIN_TOKEN` in both the running service and GUI process. When it is absent
   or does not match, inspection and evaluation remain available but mutation controls
   are unavailable.

   - Selecting another target stops the current Compose service, rebuilds and starts
     the selected service, waits for its readiness check, and then applies the chosen
     traffic mode. Only one bundled target uses port `8766` at a time.
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
`--project-root PATH`. A target's model cache must have been prepared before the GUI
can start it; the target switch builds the image. For example, prepare Qwen for later
hot-swapping without starting it:

```bash
docker compose --profile qwen build aegis-qwen
docker compose --profile tools run --rm prepare-qwen
docker compose --profile qwen run --rm --no-deps aegis-qwen doctor --config /app/configs/aegis.deployment.container.json
```

Each accepted GUI submission is written before it is sent to AEGIS and then updated
with either its result or its upstream error, so service-side failures also remain
available for reference. A capacity or initial storage failure is rejected before any
upstream request; a post-decision storage failure is reported explicitly and can leave
the record pending. On the next GUI start, any such pre-existing pending record is
marked as interrupted before retention limits are enforced. History persists across GUI restarts in
`~/.aegis/gui-history.sqlite3` by default. Unlike AEGIS's privacy-safe audit log, this
SQLite database contains raw prompt text and uploaded image bytes. Protect it according
to your data-retention policy and do not place it in a shared directory. By default,
AEGIS retains at most 1,000 records and 268,435,456 bytes (256 MiB) of logical history
payload, with a maximum record age of 30 days. On startup, before history reads, and
after history writes, it removes the oldest eligible records when any limit is
exceeded. Pending evaluations are preserved regardless of age until they receive a
terminal result. On restart, abandoned pending records are first marked interrupted
and then become eligible for the configured retention limits.
Adjust the limits with the GUI options below. Stop the GUI before manually moving or
deleting the database; use an appropriate storage-sanitization procedure when remnants
outside SQLite are in scope.

Press `Ctrl+C` in the dashboard terminal to stop only the GUI. The AEGIS service
continues running until it is stopped separately. To stop both components, use the
graceful shutdown workflow below.

### Stop AEGIS

On Windows, open another PowerShell window in the project directory and run:

```powershell
.\stop-aegis.ps1
```

The [`stop-aegis.ps1`](stop-aegis.ps1) launcher reads the GUI's independent,
per-launch shutdown credential from the private state file, sends it directly without
using a proxy, waits for active evaluation, history, or target-switch work to finish,
and closes the dashboard normally. It never sends the reusable AEGIS API token to a
listener merely because it occupies the dashboard port. It then runs Docker Compose's
graceful stop operation for both bundled model services, so it works regardless of
which target is active. Containers, downloaded models, audit data, and GUI history are
retained for the next launch.

The default GUI shutdown allowance is 180 seconds and the Docker stop grace period is
180 seconds. The container allowance exceeds the shipped 120-second inference bound
so an active request can finish, write its audit result, and tear down its worker before
Docker escalates to `SIGKILL`. They can be changed when necessary, but do not set the
container timeout below the configured inference timeout plus teardown margin. The
default state file is `~/.aegis/gui-state.json`. A manual GUI started with a custom
`--state-file` must pass that same file to the stop launcher. A custom `--history-db`
also changes the implicit state-file location to `gui-state.json` beside that history
database unless `--state-file` is supplied, so pass that resolved path as well:

```powershell
# Also correct when the GUI used --history-db C:\private\history.sqlite3
# without an explicit --state-file.
.\stop-aegis.ps1 -GuiStateFile C:\private\gui-state.json -GuiTimeoutSeconds 300 -ContainerTimeoutSeconds 180
```

On Linux with a local deployment, stop the foreground GUI with `Ctrl+C`, then stop
either possible Compose target:

```bash
docker compose --profile llava --profile qwen stop aegis-llava aegis-qwen
```

On macOS, `Ctrl+C` stops only the local dashboard. The remote AEGIS service must be
managed separately by its operator.

### GUI options

`aegis gui` binds only to a loopback address. It creates a random per-launch browser
session and opens a fragment-bearing bootstrap URL. The packaged application validates
the credential, stores it in origin-and-port-scoped `sessionStorage`, and immediately
removes it from the address bar. Static shell assets contain no history; every GUI API
request requires the session header. Keep the dashboard browser profile and host
account trusted. The upstream evaluation and optional administration bearer tokens are
read by the GUI process and remain server-side; they are never sent to the browser.
Run `aegis gui --help` for the authoritative option list:

```text
--service-url URL       running AEGIS service (default http://127.0.0.1:8766)
--host HOST             loopback bind address (default 127.0.0.1)
--port PORT             dashboard port (default 8767)
--max-http-connections N
                        maximum simultaneous GUI HTTP connections (default 32)
--request-read-timeout-seconds N
                        absolute request-header/body read deadline (default 15)
--history-db PATH       local SQLite history path
--state-file PATH       private per-launch GUI shutdown-state file
--history-max-records N maximum retained records (default 1000)
--history-max-age-days N
                        maximum retained age in days (default 30)
--history-max-bytes N   maximum logical history payload (default 268435456)
--api-token-env NAME    environment variable holding the upstream token
--admin-token-env NAME  environment variable holding the distinct administration token
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

For Qwen, use the matching profile and service instead:

```bash
docker compose --profile qwen logs --since=2m --tail=120 aegis-qwen
```

Stop the matching restart loop while investigating:

```bash
docker compose --profile llava stop aegis-llava

# OR Qwen
docker compose --profile qwen stop aegis-qwen
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

### `Unable to create process` from `.venv`

This usually means the Python installation used to create the virtual environment
was removed or moved. Install Python 3.10 or newer first, verify that the Windows
launcher can find it, and retain the broken environment as a backup until its
replacement passes the CLI checks.

Windows PowerShell, from the repository root:

```powershell
py -3 -c "import sys; assert sys.version_info >= (3, 10); print(sys.executable, sys.version)"
Move-Item -LiteralPath .\.venv -Destination .\.venv-python-old
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\aegis.exe --version
.\.venv\Scripts\aegis.exe targets
```

The first command must succeed before you move anything. Run the block only from the
AEGIS repository root, and use a different explicit backup name if
`.venv-python-old` already exists. After both checks succeed, the backup can be
removed manually when it is no longer needed.

### `/readyz` does not return `"ok": true`

The first model startup can take several minutes. Check container status and logs:

```bash
docker compose --profile llava ps -a
docker compose --profile llava logs --tail=120 aegis-llava

# OR Qwen
docker compose --profile qwen ps -a
docker compose --profile qwen logs --tail=120 aegis-qwen
```

If the container is still running and the log shows model loading or warmup, wait and
try `/readyz` again. If it has exited, read the final traceback rather than repeatedly
calling the readiness endpoint.

### Resource preflight failure

AEGIS checks resources before loading a large model. A failure includes the actual
and required byte counts. Typical checks are:

- `total_physical_memory`: the RAM visible inside the container is below the target
  minimum. LLaVA requires at least 8,000,000,000 container-visible bytes (about
  7.45 GiB), while Qwen requires 15 GiB. Hardware reservations and Docker or WSL
  limits can put an 8 GB Windows host below LLaVA's exact threshold; upgrading to
  16 GB is a reliable LLaVA fix. It is not sufficient for Qwen when WSL uses its
  default 50% host-memory limit; use a host with additional RAM for the supported
  Qwen service profile rather than allocating nearly all host memory to WSL.
- `available_physical_memory`: close browsers, development tools, games, and other
  memory-heavy applications, or restart the computer before starting AEGIS.
- `available_virtual_memory`: increase the Windows paging file and, when using the
  WSL 2 backend, the WSL swap allocation.
- `cuda_device_available_memory`: close other GPU workloads or use a GPU with enough
  free VRAM, and confirm that `nvidia-smi` works both on the host and through Docker's
  GPU support.
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
target profile's declared deployment minimum. Normally, restore the value shown as
`profile_minimum` in the error rather than weakening the profile.

If a developer intentionally maintains an unsupported low-resource research fork,
the resource values in both the deployment JSON and `AEGIS/target_profiles.py` must
agree. Changing only the JSON produces this error. Such a fork must be revalidated and
must not be treated as a production-safe profile.

### Source or configuration changes do not appear in the container

Files under `AEGIS/`, `configs/`, and `models/aegis/` are copied into the image at
build time. The Windows launcher and Settings target switch invoke Compose with
`--build`, so they rebuild when an image input changed. For a manual workflow, include
`--build` while recreating the matching service:

```bash
docker compose --profile llava stop aegis-llava
docker compose --profile llava up -d --build --force-recreate aegis-llava
docker compose --profile llava logs --since=2m --tail=120 aegis-llava
```

For Qwen, substitute `qwen` and `aegis-qwen` in all three commands. Run
`docker compose --profile llava build --no-cache aegis-llava` only when deliberately
diagnosing a stale Docker build cache; for Qwen, substitute the corresponding profile
and service. Routine source changes do not require `--no-cache`.

Do not rely on older unbounded log output after rebuilding; `--since` and `--tail`
make it clear which error belongs to the current container.

## Local Python installation

A local installation is useful for the lightweight demo and for importing the AEGIS
client in another Python application. The full GPU service is most reliably deployed
with Docker from the cloned project as shown above; the Python package by itself does
not include the container deployment assets or downloaded model files.

Create and activate the environment with the platform's Python launcher.

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

If `python3 -m venv` reports that venv support is unavailable, install the venv
package supplied by your operating-system distribution and repeat the command. On
Debian and Ubuntu, that package is commonly named `python3-venv`.

Then install and check AEGIS:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
aegis demo
```

The demo checks the installation with a deterministic test provider; it is not a
trained safety model. A successful default run exits with status 0 and reports a
`benign` verdict, an `allow` action, and a numeric `risk_score`. The command
exits unsuccessfully if the provider cannot produce a completed benign or malicious
decision.

## HTTP API usage

These examples assume AEGIS is running on `http://127.0.0.1:8766`. Replace
`YOUR_API_TOKEN` with the `AEGIS_API_TOKEN` value in `.env`. Use HTTPS for a remote
deployment; the shipped Compose file exposes the service only on host loopback. Do
not expose AEGIS's built-in HTTP server directly to an untrusted network. It limits
itself to 32 simultaneous connections and applies a 15-second absolute request-read
deadline by default, but a remote deployment still needs a hardened reverse proxy for
TLS, rate limits, request filtering, and deployment-level network controls.

The shipped profiles run one inference at a time. Overlapping guard requests are not
queued; they return HTTP 429 with `guardrail_busy`. Treat that response as a guardrail
failure, or apply bounded retry and backoff before sending the prompt downstream.

### Endpoints

| Method and path | Authentication | Purpose |
| --- | --- | --- |
| `GET /livez` | None | Confirm that the HTTP process is alive. |
| `GET /readyz` | None | Report only the minimum health contract: readiness, active target, traffic mode, input capabilities, and a coarse unavailable reason. During initial model loading the endpoint may not accept connections; an already-running service returns HTTP 503 if its worker becomes unavailable. |
| `GET /healthz` | None | Alias of `/readyz`. |
| `GET /v1/status` | Evaluation bearer token | Return the detailed provider, worker, audit-session, and deployment readiness report. |
| `GET /metrics` | Evaluation bearer token | Return in-memory request, action, error, and latency counters. |
| `POST /v1/guard` | Evaluation bearer token | Evaluate one request or a bounded request batch. |
| `PUT /v1/admin/traffic-mode` | Distinct administration bearer token | Change the runtime traffic mode without editing the deployment file. The endpoint is disabled when no administration token is configured. |

The unauthenticated health endpoints intentionally omit model revisions, filesystem
paths, hashes, worker process details, raw errors, and audit-session diagnostics. Use
the authenticated status endpoint when an operator needs that information:

```bash
curl "http://127.0.0.1:8766/v1/status" -H "Authorization: Bearer YOUR_API_TOKEN"
```

### Built-in request contract

Each built-in request must contain non-empty `text` and exactly one base64-encoded
image. The service rejects text-only, image-only, local-path, and multi-image
requests for both targets.

| Field | Requirement |
| --- | --- |
| `request_id` | Optional string, at most 256 characters; ordinary guard correlation only. `guarded_call` rejects it because it is not detector input. |
| `text` | Required non-empty string, at most 32,768 characters. |
| `images` | Required list containing exactly one image object. |
| `images[0].media_type` | `image/png`, `image/jpeg`, or `image/webp`; it must match the decoded file. |
| `images[0].base64` | Standard base64 without a data-URL prefix. |
| `metadata` | Optional JSON object carried with an ordinary guard request; it is not detector input and `guarded_call` rejects it. |

The table is the canonical request shape. For ordinary (non-batch) compatibility,
the service also accepts top-level `image_base64` with optional `image_media_type` in
place of `images`. The bundled services recognize local `image_path` and `image_paths`
fields but reject them; new integrations should always send `images`. Unknown or
conflicting top-level fields, unknown image-object fields, ambiguous aliases,
non-object image entries, and batch-envelope siblings are rejected before inference.

The API configuration permits at most 10 MiB of decoded image data, 20 million image
pixels, and a 12 MiB HTTP request body. Base64 expands the file, so the body limit may
be reached before the decoded-image limit. The GUI intentionally applies a smaller
8 MiB file limit. A batch uses `{"requests": [{...}, {...}]}` and is limited to four
items for LLaVA or eight for Qwen; every item must satisfy the same image-text
contract. The batch envelope must contain only `requests`; do not add sibling prompt
or model-control fields.

### Evaluate an image and prompt

The included files contain a small PNG and valid prompt. Use the file matching the
running target:

```bash
# LLaVA
curl -X POST "http://127.0.0.1:8766/v1/guard" -H "Authorization: Bearer YOUR_API_TOKEN" -H "Content-Type: application/json" --data-binary "@configs/llava_service_warmup_request.example.json"

# Qwen
curl -X POST "http://127.0.0.1:8766/v1/guard" -H "Authorization: Bearer YOUR_API_TOKEN" -H "Content-Type: application/json" --data-binary "@configs/service_warmup_request.example.json"
```

On Windows PowerShell use `curl.exe`. To load the token from `.env` without printing
it:

```powershell
$token = (Get-Content .env |
  Where-Object { $_ -like "AEGIS_API_TOKEN=*" }) -replace "^AEGIS_API_TOKEN=", ""

curl.exe -X POST "http://127.0.0.1:8766/v1/guard" `
  -H "Authorization: Bearer $token" `
  -H "Content-Type: application/json" `
  --data-binary "@configs/llava_service_warmup_request.example.json"
```

Use `configs/service_warmup_request.example.json` in the final line for Qwen.

### Interpret the response safely

The response contains one `decisions` entry per request. Important fields include:

- `verdict`: `benign` or `malicious` for a completed classifier evaluation;
  `guardrail_error` means evaluation failed.
- `recommended_action`: the detector's `allow`, `review`, or `block` recommendation.
- `action`: the effective action after applying the active traffic mode.
- `traffic_mode`: `shadow`, `review`, or `enforce`.
- `risk_score`, `review_threshold`, and `threshold`: the score and decision cutoffs.
- `trace_id`: the request identifier used for response correlation.

The bundled targets start in shadow mode, so `action` remains `allow` while
`recommended_action` shows what AEGIS would have done. An HTTP 200 response can still
contain `verdict: "guardrail_error"`; a gating caller must treat that as failure and
must not invoke the protected model. Do not treat runtime enforcement as evidence
that the research detectors are production-ready.

### Change traffic mode at runtime

Use the authenticated administration endpoint. This example selects the intermediate
`review` mode:

```bash
curl -X PUT "http://127.0.0.1:8766/v1/admin/traffic-mode" -H "Authorization: Bearer YOUR_ADMIN_TOKEN" -H "Content-Type: application/json" --data '{"traffic_mode":"review"}'
```

Windows PowerShell, after loading the distinct administration token from `.env`:

```powershell
$adminToken = (Get-Content .env |
  Where-Object { $_ -like "AEGIS_ADMIN_TOKEN=*" }) -replace "^AEGIS_ADMIN_TOKEN=", ""
$headers = @{ Authorization = "Bearer $adminToken" }
$body = @{ traffic_mode = "review" } | ConvertTo-Json -Compress
Invoke-RestMethod -Method Put `
  -Uri "http://127.0.0.1:8766/v1/admin/traffic-mode" `
  -Headers $headers `
  -ContentType "application/json" `
  -Body $body
```

Valid values are `shadow`, `review`, and `enforce`. This change is process-local and
is lost when the service restarts; the shipped configurations restart in `shadow`.
Confirm the active value through `/readyz` and every decision response before relying
on it. The administration token must differ from the API token and fingerprint key;
possession of the evaluation token alone cannot change traffic mode. Keep all secrets
server-side and do not give them to untrusted clients.

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

From the repository root, load the included request file matching the running target.
The file already contains the correctly encoded PNG.

```python
import json
import os
from pathlib import Path

from AEGIS import GuardrailClient

# LLaVA. For Qwen, use configs/service_warmup_request.example.json.
request_file = Path("configs/llava_service_warmup_request.example.json")
payload = json.loads(request_file.read_text(encoding="utf-8"))

client = GuardrailClient(
    "http://127.0.0.1:8766",
    api_token=os.environ["AEGIS_API_TOKEN"],
    required_traffic_mode="shadow",
    timeout_seconds=120.0,
)
result = client.guard(payload)

print(result.to_dict())
print("Effective action:", result.action)
print("Recommendation:", result.decisions[0].recommended_action)
```

This example observes shadow-mode results; it does not enforce them. For a gating
integration, configure and validate `review` or `enforce`, construct the client with
that exact `required_traffic_mode`, and use `guarded_call` only in controlled research
code. The client rejects a traffic-mode mismatch, review, block, guardrail error,
authentication error, protocol error, or transport failure before invoking the
callback. The client and service also enforce the strict request schema, and the CSV
provider staging path preserves text exactly instead of applying scalar or missing-
value coercion. `guarded_call` additionally rejects `request_id`, metadata, and
local-path images; submit image bytes in `images` or `image_base64` so the callback
receives the exact evaluated snapshot. These controls harden the integration boundary;
they do not make the research detectors production-ready.

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

A successful demo reports a completed `benign` or `malicious` decision with a
numeric `risk_score` and exits with status 0. It exits unsuccessfully if the
installation check produces a `guardrail_error` or another incomplete decision.

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

Without `--download`, `ok` reports whether the disk preflight passed; it does not mean
the model is present. The `ready` field is authoritative for cache availability. If
it is `false`, run the matching Docker preparation command or add `--download` after
installing the dependency below.

Add `--download` to download missing files and inspect the resulting target cache.
Both targets validate the exact pinned source revision and a complete
inference-runtime content digest. LLaVA uses source revision
`74dd0bf867a4cda7950c17663794267c60cf4b40` and runtime-content SHA-256
`c2cd35a65b8059c8add9e8901550c9e29d62d189cc7c2a5a1f6f715d7e05bb1c`.
Qwen uses source revision `66285546d2b821cf421d4f5eb2576359d3770cd3` and
runtime-content SHA-256
`45f7d1afd0ef8e09cb7a79456fd232014d2a482ca975d2008848c0efa77e4ce0`.
The Docker preparation commands in the deployment steps are recommended. A host-side
download needs `huggingface-hub`, but it does not need the full inference extras or a
host PyTorch installation. Use Python 3.11 for this constrained path because its
pinned pandas version matches the container's Python 3.11 environment. Run these
commands from an activated Python 3.11 virtual environment and confirm the interpreter
before installing:

```bash
python --version
python -m pip install --constraint requirements-runtime.constraints.txt -e . huggingface-hub
```

The version command must report Python 3.11.

Docker Compose reads `HF_TOKEN` from `.env`, but a host-side `aegis prepare`
process does not load that file. If the upstream repository requires authentication,
export the same value in the terminal without printing it.

Windows PowerShell:

```powershell
$env:HF_TOKEN = (Get-Content .env |
  Where-Object { $_ -like "HF_TOKEN=*" }) -replace "^HF_TOKEN=", ""
```

macOS or Linux:

```bash
export HF_TOKEN="$(sed -n 's/^HF_TOKEN=//p' .env)"
```

Skip this export when the repository is public and `HF_TOKEN` is empty.

```bash
# LLaVA
aegis prepare --target llava05b --root . --download

# OR Qwen
aegis prepare --target qwen25vl3b --root . --download
```

Run the shipped deployment preflight inside the matching Compose service, where the
container paths, model volume, environment variables, and GPU are available:

```bash
# LLaVA
docker compose --profile llava run --rm --no-deps aegis-llava doctor --config /app/configs/aegis.llava.deployment.container.json

# Qwen
docker compose --profile qwen run --rm --no-deps aegis-qwen doctor --config /app/configs/aegis.deployment.container.json
```

`doctor` exits unsuccessfully and reports the failed checks when the required model,
GPU, memory, environment variables, or other deployment resources are missing.

The checked-in `configs/*.container.json` files are container-only. Do not pass them
to a host-side `aegis doctor` or `aegis serve`: their `/app` model paths and `/var/lib`
runtime paths do not refer to the Docker volumes when interpreted by the host. The
host CLI can validate and serve an operator-created configuration containing genuine
host paths, but a working MLLM host deployment additionally requires the constrained
`mllm` dependencies, a CUDA-compatible PyTorch/torchvision installation, prepared
model caches, and all target resource requirements. The base `pip install -e .` used
for the client/demo/dashboard workflow does not install that inference stack:

```bash
aegis doctor --config YOUR-HOST-CONFIG.json
aegis serve --config YOUR-HOST-CONFIG.json
```

No ready-made host deployment configuration is shipped. For normal use, start
`aegis-llava` or `aegis-qwen` with Docker Compose as documented above.

## Operate or remove the deployment

View logs:

```bash
docker compose --profile llava logs -f aegis-llava

# OR Qwen
docker compose --profile qwen logs -f aegis-qwen
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

Keep `.env` private. The API token protects guard and metrics requests. The
administration token is optional for an evaluation-only first start and same-target
reuse, but it is required for runtime mode control, Settings mutations, and a
rollback-safe launcher switch away from another healthy target. The fingerprint key
protects prompt fingerprints written to the audit log. Every configured secret must
be an independent random value of at least 32 visible ASCII characters. The stable
per-target files `aegis-llava05b.jsonl` and `aegis-qwen25vl3b.jsonl` retain the
runtime session ID in every event; each target is bounded across restarts to the active
file plus five rotated backups of at most 100 MiB each.
