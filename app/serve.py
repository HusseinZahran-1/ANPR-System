"""Manual launcher that shares listener settings with the camera setup page."""

import argparse
import os

import uvicorn

from app.api.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--require-models", action="store_true",
                        help="Load the versioned inference profile before binding")
    args = parser.parse_args()
    if args.host is not None:
        os.environ["HOST"] = args.host
    if args.port is not None:
        os.environ["PORT"] = str(args.port)
    settings = Settings.from_env()
    if args.require_models:
        from fastapi import HTTPException
        from app.api.routes import load_inference_profile

        try:
            load_inference_profile()
        except HTTPException as exc:
            raise SystemExit(exc.detail) from None
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, access_log=False)


if __name__ == "__main__":
    main()
