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
from AEGIS.service import evaluate_feature_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve AEGIS decisions for JSON feature-vector payloads."
    )
    parser.add_argument("--detector", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--block-threshold", type=float, default=None)
    parser.add_argument("--review-margin", type=float, default=None)
    args = parser.parse_args()

    artifact = load_detector_artifact(args.detector)
    policy = GuardrailPolicy(
        block_threshold=args.block_threshold,
        review_margin=args.review_margin,
    )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/v1/guard":
                self.send_error(404, "Use POST /v1/guard")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                response = evaluate_feature_payload(artifact, payload, policy=policy)
                body = json.dumps(response, sort_keys=True).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                body = json.dumps({"error": str(exc)}, sort_keys=True).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            return

    server = HTTPServer((args.host, args.port), Handler)
    print(f"Serving AEGIS feature guardrail at http://{args.host}:{args.port}/v1/guard")
    server.serve_forever()


if __name__ == "__main__":
    main()
