"""Write D-Mercator edgelist (two names per line, no header) from edges.tsv."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("edges.tsv"),
        help="TSV with columns source, target (default: edges.tsv)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("edges.edge"),
        help="Output edgelist for Mercator (default: edges.edge)",
    )
    args = p.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Missing {args.input.resolve()}")

    n = 0
    with args.input.open(newline="", encoding="utf-8") as f_in, args.output.open(
        "w", encoding="ascii", newline="\n"
    ) as f_out:
        r = csv.DictReader(f_in, delimiter="\t")
        if "source" not in (r.fieldnames or []) or "target" not in (r.fieldnames or []):
            raise SystemExit(f"Expected tab-separated columns source, target; got {r.fieldnames!r}")
        for row in r:
            s = (row.get("source") or "").strip()
            t = (row.get("target") or "").strip()
            if not s or not t or s == t:
                continue
            f_out.write(f"{s} {t}\n")
            n += 1

    print(f"Wrote {n} edges to {args.output.resolve()}")


if __name__ == "__main__":
    main()
