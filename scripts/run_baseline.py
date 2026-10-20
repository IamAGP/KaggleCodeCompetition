"""Run the predeclared two-candidate comparison locally."""

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from name_that_llm.baseline import BaselineExperiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/name-that-llm"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/baseline-v1"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.output_dir / "run.log"),
        ],
    )
    BaselineExperiment(args.data_dir, args.output_dir).run()


if __name__ == "__main__":
    main()
