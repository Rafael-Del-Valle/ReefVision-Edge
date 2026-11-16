from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from reefvision_edge import __version__
from reefvision_edge.app import create_app
from reefvision_edge.config import Settings, require_safe_bind


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reefvision-edge",
        description="Run the offline ReefVision-Edge field inference service.",
    )
    parser.add_argument("--host", help="Bind address (defaults to loopback).")
    parser.add_argument("--port", type=int, help="TCP port (defaults to 8765).")
    parser.add_argument("--data-dir", type=Path, help="Independent Edge data directory.")
    parser.add_argument("--username", help="Basic-auth username for non-loopback binding.")
    parser.add_argument("--password", help="Basic-auth password for non-loopback binding.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def settings_from_argv(argv: Sequence[str] | None = None) -> Settings:
    args = build_parser().parse_args(argv)
    base = Settings()
    settings = Settings(
        data_dir=args.data_dir if args.data_dir is not None else base.data_dir,
        bind_host=args.host if args.host is not None else base.bind_host,
        bind_port=args.port if args.port is not None else base.bind_port,
        auth_username=args.username if args.username is not None else base.auth_username,
        auth_password=args.password if args.password is not None else base.auth_password,
    )
    require_safe_bind(settings)
    return settings


def main(argv: Sequence[str] | None = None) -> None:
    settings = settings_from_argv(argv)
    uvicorn.run(
        create_app(settings),
        host=settings.bind_host,
        port=settings.bind_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
