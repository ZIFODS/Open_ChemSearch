import argparse

import uvicorn

from chemsearch.app.main import app


def parse_args() -> argparse.Namespace:
    """Parse arguments from command line.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Host of running server.", default="127.0.0.1")
    parser.add_argument(
        "-p", "--port", help="Port of running server.", type=int, default=8000
    )

    args = parser.parse_args()

    return args


def main():
    """Entry point for script."""
    args = parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
