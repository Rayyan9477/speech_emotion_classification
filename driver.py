#!/usr/bin/env python3
"""
driver.py - Simple driver to launch the Streamlit UI or FastAPI server as used by tests.

Usage:
  python driver.py --app          # start Streamlit app
  python driver.py --api          # start FastAPI server (uvicorn)
"""

import argparse
import subprocess
import sys
import os


def run_app():
    cmd = [sys.executable, "-m", "streamlit", "run", "src/ui/streamlit_app.py", "--server.port", "8501", "--server.headless", "true"]
    return subprocess.Popen(cmd)


def run_api():
    cmd = [sys.executable, "-m", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8501"]
    return subprocess.Popen(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", action="store_true", help="Run Streamlit app")
    parser.add_argument("--api", action="store_true", help="Run FastAPI server")
    args = parser.parse_args()

    if args.app:
        proc = run_app()
        proc.wait()
    elif args.api:
        proc = run_api()
        proc.wait()
    else:
        parser.print_help()


if __name__ == "__main__":
    sys.exit(main() or 0)


