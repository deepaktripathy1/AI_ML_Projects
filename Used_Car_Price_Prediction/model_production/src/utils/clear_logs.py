#!/usr/bin/env python3
import argparse
from pathlib import Path


def clear_log_file(log_path: Path):
    if log_path.exists():
        with log_path.open("r+") as file:
            file.truncate(0)
        print(f"Cleared {log_path}")
    else:
        print(f"Log file not found : {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Clear log files.")
    parser.add_argument(
        "-p", "--project-root",
        type=Path,
        default=Path(__file__).parent.resolve(),
        help="Path to the project root directory"
    )
    args = parser.parse_args()

    logs_to_clear = [
        args.project_root / "logs" / "app.log",
        args.project_root / "prefect_automation" / "logs" / "app.log",
        args.project_root / "scripts" / "logs" / "app.log",
        args.project_root / "deployment" / "bentoml" / "logs" / "app.log"
    ]

    for log_file in logs_to_clear:
        clear_log_file(log_path=log_file)


if __name__ == "__main__":
    main()
