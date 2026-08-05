"""Command-line entry point for serving and inspecting Orphus."""

from __future__ import annotations

import argparse
import json

import uvicorn

from orphus.config import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(prog="orphus")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("show-config", help="print effective secret-safe configuration")
    serve = subcommands.add_parser("serve", help="start the API server")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    args = parser.parse_args()
    settings = load_settings()
    if args.command == "show-config":
        print(json.dumps(settings.describe(), indent=2, default=str))  # noqa: T201
        return
    uvicorn.run(
        "orphus.api.app:create_app",
        factory=True,
        host=args.host or settings.server.host,
        port=args.port or settings.server.port,
        workers=settings.server.api_workers,
    )


if __name__ == "__main__":
    main()
