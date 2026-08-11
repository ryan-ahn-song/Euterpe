from __future__ import annotations

import argparse
import json
from pathlib import Path

from hearing_assist.cape import save_examples
from hearing_assist.synthetic import generate_synthetic_examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate engineering-only CAPE data for pipeline verification"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--listeners", type=int, default=8)
    parser.add_argument("--segments-per-listener", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20_260_811)
    args = parser.parse_args()
    examples = generate_synthetic_examples(
        listener_count=args.listeners,
        music_segments_per_listener=args.segments_per_listener,
        seed=args.seed,
    )
    count = save_examples(args.output, examples)
    warning_path = args.output.with_suffix(args.output.suffix + ".NOTICE.json")
    warning_path.write_text(
        json.dumps(
            {
                "data_type": "synthetic software-verification data",
                "observations": count,
                "warning": "Not physiology, clinical data, or evidence of hearing benefit.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {count} observations: {args.output.resolve()}")
    print(f"saved provenance notice: {warning_path.resolve()}")


if __name__ == "__main__":
    main()
