"""Make the current complete table inventory without opening the blinded key."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "tables"
ORIGINAL = TABLES / "table_population_v8.csv"
LATEST_SNAPSHOT = TABLES / "table_population_v8_followup8.csv"
OUT = TABLES / "table_population_v8_followup10.csv"
ADDITIONS = ROOT / "manuscript/TABLE_POPULATION_ADDITIONS_V10.md"
KEY_NAME = "imagery_sampling_key_v7.csv"


def main():
    if OUT.exists() or ADDITIONS.exists():
        raise SystemExit("Refusing to overwrite versioned v10 inventory output")
    original = pd.read_csv(ORIGINAL)
    latest = pd.read_csv(LATEST_SNAPSHOT)
    fields = ["table", "population", "interpretation"]
    if list(original.columns) != fields or list(latest.columns) != fields:
        raise SystemExit("Inventory schema differs from the original")
    if original.table.duplicated().any() or latest.table.duplicated().any():
        raise SystemExit("Duplicate table in an older inventory")
    original_map = original.set_index("table")
    latest_map = latest.set_index("table")
    if not set(original.table).issubset(set(latest.table)):
        raise SystemExit("A baseline table is absent from the latest snapshot")
    if not latest_map.loc[original.table, ["population", "interpretation"]].reset_index(
            drop=True).equals(original_map.loc[original.table, ["population", "interpretation"]].reset_index(drop=True)):
        raise SystemExit("An original population or blinded-key marker was changed")
    key = original_map.loc[KEY_NAME]
    if (key.population != "unopened blinded key" or
            key.interpretation != "Filename inventoried only; contents not inspected"):
        raise SystemExit("Blinded-key handling differs from the original")
    filenames = {p.name for p in TABLES.glob("*.csv")} | {OUT.name}
    if filenames - set(latest.table) != {OUT.name}:
        raise SystemExit("A current table lacks existing population metadata")
    if set(latest.table) - filenames:
        raise SystemExit("A latest-snapshot table is missing from disk")
    rows = latest.copy()
    rows = pd.concat([rows, pd.DataFrame([{
        "table": OUT.name,
        "population": "all table CSV filenames in the v10 package",
        "interpretation": "Current inventory metadata only; blinded key not opened",
    }])], ignore_index=True).sort_values("table").reset_index(drop=True)
    if (set(rows.table) != filenames or len(rows) != len(filenames) or
            rows.table.duplicated().any() or
            rows.population.isna().any() or rows.interpretation.isna().any() or
            rows.population.str.strip().eq("").any() or
            rows.interpretation.str.strip().eq("").any()):
        raise SystemExit("The v10 inventory is not complete or has blank metadata")
    added = sorted(filenames - set(original.table))
    rows.to_csv(OUT, index=False)
    text = ["# Tables added after the original v8 inventory", "",
            f"The original inventory has {len(original)} entries. The v10 inventory has {len(rows)}: "
            f"{len(added)-1} previously saved follow-up CSVs plus its own new inventory row. "
            "The original three-column schema and the unopened blinded-key marker are preserved.", "",
            "| Table | Population | Interpretation |", "|---|---|---|"]
    for name in added:
        row = rows.loc[rows.table.eq(name)].iloc[0]
        text.append(f"| `{name}` | {row.population} | {row.interpretation} |")
    ADDITIONS.write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.name}: {len(rows)} entries for {len(filenames)} CSVs")
    print(f"Wrote {ADDITIONS.name}: {len(added)} additions over the original inventory")


if __name__ == "__main__":
    main()
