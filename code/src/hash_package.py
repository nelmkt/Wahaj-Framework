"""Hash v8 package files except this manifest and the unopened blinded key."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "tables/SHA256SUMS_v8.txt"
FOLDERS = ("figures_png", "figures_pdf", "tables", "code", "manuscript")
BLINDED_KEY = ROOT / "tables/imagery_sampling_key_v7.csv"


def entries():
    for folder in FOLDERS:
        for path in sorted((ROOT / folder).rglob("*")):
            if not path.is_file():
                continue
            if path != MANIFEST and path != BLINDED_KEY:
                yield f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT).as_posix()}"


if __name__ == "__main__":
    lines = list(entries())
    if MANIFEST.exists():
        raise SystemExit("Refusing to overwrite the v8 manifest; version a replacement")
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert MANIFEST.read_text(encoding="utf-8").splitlines() == list(entries())
    print(f"Hashed {len(lines)} package files.")
