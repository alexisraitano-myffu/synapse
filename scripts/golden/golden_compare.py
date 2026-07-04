"""
SYN-111 golden parity: Rust routing vs the frozen Python reference.

Runs the core's `golden_replay` example on the frozen corpus, normalizes its
output database with the SAME normalizer as the Python reference, and diffs.
Exit 0 = parity, 1 = divergence (first differing rows printed per table).

Usage:
    python -m scripts.golden.golden_compare [--core-repo PATH] [--keep-db]
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.golden.golden_lib import GOLDEN_DIR, normalize_db

CORE_REPO_DEFAULT = Path.home() / "Pro AR" / "synapse-core"
MODEL_DIR = Path.home() / ".synapse" / "models" / "paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-repo", default=str(CORE_REPO_DEFAULT))
    parser.add_argument("--keep-db", action="store_true")
    args = parser.parse_args()

    reference = json.loads((GOLDEN_DIR / "python_reference.json").read_text())
    today = reference["replayed_at_now"][:10]

    out_dir = Path(tempfile.mkdtemp(prefix="synapse-golden-rust-"))
    rust_db = out_dir / "rust_replay.db"

    print("build + run du runner Rust…")
    subprocess.run(
        ["cargo", "build", "--release", "--example", "golden_replay"],
        cwd=args.core_repo, check=True, capture_output=True,
    )
    binary = Path(args.core_repo) / "target" / "release" / "examples" / "golden_replay"
    run = subprocess.run(
        [str(binary), str(GOLDEN_DIR / "corpus.json"), str(rust_db),
         str(MODEL_DIR), today],
        capture_output=True, text=True,
    )
    if run.returncode != 0:
        print(run.stdout[-2000:])
        print(run.stderr[-2000:])
        return 1

    rust_state = normalize_db(rust_db)
    py_state = reference["final_state"]

    ok = True
    for table in py_state:
        a, b = py_state[table], rust_state.get(table)
        if json.dumps(a, sort_keys=True, ensure_ascii=False) == \
           json.dumps(b, sort_keys=True, ensure_ascii=False):
            continue
        ok = False
        print(f"\n✗ DIVERGENCE — table {table} (python {len(a)} vs rust {len(b or [])})")
        if isinstance(a, list) and isinstance(b, list):
            shown = 0
            for i in range(max(len(a), len(b))):
                ra = a[i] if i < len(a) else "<absent>"
                rb = b[i] if i < len(b) else "<absent>"
                if ra != rb:
                    print(f"  row {i}:")
                    if isinstance(ra, dict) and isinstance(rb, dict):
                        for k in sorted(set(ra) | set(rb)):
                            if ra.get(k) != rb.get(k):
                                print(f"    {k}: py={ra.get(k)!r} ≠ rust={rb.get(k)!r}")
                    else:
                        print(f"    py:   {ra}")
                        print(f"    rust: {rb}")
                    shown += 1
                    if shown >= 3:
                        print("  …")
                        break

    if ok:
        print(f"\n✓ PARITÉ — {sum(len(v) for v in py_state.values() if isinstance(v, list))} "
              f"lignes normalisées identiques sur {len(py_state)} tables")
    if args.keep_db or not ok:
        print(f"db rust conservée: {rust_db}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
