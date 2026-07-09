from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AEGIS.detector_artifact import load_detector_artifact
from AEGIS.guardrail import GuardrailPolicy
from AEGIS.provider_contract import load_provider
from AEGIS.service import evaluate_request_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve AEGIS decisions for live prompt requests using an embedding provider."
    )
    parser.add_argument("--provider", required=True, help="Provider spec: module:object.")
    parser.add_argument("--detector", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--block-threshold", type=float, default=None)
    parser.add_argument("--review-margin", type=float, default=None)
    parser.add_argument("--action-on-error", choices=["allow", "review", "block"], default="review")
    parser.add_argument("--require-matching-provenance", action="store_true")
    parser.add_argument("--hash-images", action="store_true")
    args = parser.parse_args()

    artifact = load_detector_artifact(args.detector)
    provider = load_provider(args.provider)
    policy = GuardrailPolicy(
        block_threshold=args.block_threshold,
        review_margin=args.review_margin,
        action_on_error=args.action_on_error,
        require_matching_provenance=args.require_matching_provenance,
        hash_images=args.hash_images,
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/healthz":
                self.send_error(404, "Use GET /healthz or POST /v1/guard")
                return
            self._send_json({"ok": True})

        def do_POST(self) -> None:
            if self.path != "/v1/guard":
                self.send_error(404, "Use POST /v1/guard")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                response = evaluate_request_payload(artifact, provider, payload, policy=policy)
                self._send_json(response)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)

        def log_message(self, format: str, *args) -> None:
            return

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer((args.host, args.port), Handler)
    print(f"Serving AEGIS provider guardrail at http://{args.host}:{args.port}/v1/guard")
    server.serve_forever()


if __name__ == "__main__":
    main()
