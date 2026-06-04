"""Run DeepResearch Bench RACE evaluation for one model."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_drb_evaluation(
    bench_dir: Path,
    model_name: str,
    raw_data_dir: Path,
    output_dir: Path,
    query_file: Path,
    process_trace_dir: Path | None = None,
    use_model_subdir: bool = True,
    max_workers: int = 10,
    limit: int | None = None,
    skip_cleaning: bool = False,
    cleaned_data_dir: Path | None = None,
    force: bool = False,
) -> None:
    race_output = (output_dir / "race" / model_name) if use_model_subdir else output_dir
    race_output.mkdir(parents=True, exist_ok=True)
    race_cmd = [
        "python",
        "-u",
        "deepresearch_bench_race.py",
        model_name,
        "--raw_data_dir",
        str(raw_data_dir),
        "--query_file",
        str(query_file),
        "--output_dir",
        str(race_output),
        "--max_workers",
        str(max_workers),
    ]
    if process_trace_dir is not None:
        race_cmd.extend(["--process_trace_dir", str(process_trace_dir)])
    if limit is not None:
        race_cmd.extend(["--limit", str(limit)])
    if skip_cleaning:
        race_cmd.append("--skip_cleaning")
    if cleaned_data_dir is not None:
        race_cmd.extend(["--cleaned_data_dir", str(cleaned_data_dir)])
    if force:
        race_cmd.append("--force")
    run_cmd(race_cmd, cwd=bench_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DRB RACE for one model output.")
    parser.add_argument("--bench_dir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--raw_data_dir", required=True)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--query_file", default="data/drb_test.jsonl")
    parser.add_argument("--process_trace_dir", default=None)
    parser.add_argument("--use_model_subdir", action="store_true")
    parser.add_argument("--max_workers", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip_cleaning", action="store_true")
    args = parser.parse_args()
    run_drb_evaluation(
        bench_dir=Path(args.bench_dir).resolve(),
        model_name=args.model_name,
        raw_data_dir=Path(args.raw_data_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        query_file=Path(args.query_file).resolve(),
        process_trace_dir=Path(args.process_trace_dir).resolve() if args.process_trace_dir else None,
        use_model_subdir=bool(args.use_model_subdir),
        max_workers=args.max_workers,
        limit=args.limit,
        skip_cleaning=args.skip_cleaning,
    )


if __name__ == "__main__":
    main()

