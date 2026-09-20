import json
from collections import Counter
from pathlib import Path
import pandas as pd

TRAIN_JSON = Path("fine_tune_train.json")      # adjust path if needed
TEST_JSON  = Path("fine_tune_test.json")       # adjust path if needed
OCM_JSON   = Path("ocm-slo-clean.json")        # adjust path if needed
OUT_FIELD = "ocm_top_categories"

def load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def count_categories(data):
    """
    Multi-label count:
    each image contributes +1 for every category in OUT_FIELD.
    """
    c = Counter()
    n_images = 0
    n_missing_field = 0
    n_empty = 0

    for item in data:
        if not isinstance(item, dict):
            continue
        n_images += 1
        cats = item.get(OUT_FIELD, None)
        if cats is None:
            n_missing_field += 1
            continue
        if not isinstance(cats, list) or len(cats) == 0:
            n_empty += 1
            continue
        for cat in cats:
            if isinstance(cat, str) and cat.strip():
                c[cat.strip()] += 1

    return c, n_images, n_missing_field, n_empty

def counter_to_df_include_zeros_and_pct_of_all(counter: Counter, all_categories: list[str]) -> tuple[pd.DataFrame, int]:
    rows = [{"category": cat, "count": int(counter.get(cat, 0))} for cat in all_categories]
    df = pd.DataFrame(rows)

    total_counts = int(df["count"].sum())
    df["pct_of_all_counts"] = df["count"].apply(lambda x: (100.0 * x / total_counts) if total_counts > 0 else 0.0)

    df = df.sort_values(["count", "category"], ascending=[False, True]).reset_index(drop=True)
    return df, total_counts

# ---- Load data + taxonomy ----
train = load_json(TRAIN_JSON)
test  = load_json(TEST_JSON)
ocm   = load_json(OCM_JSON)

all_top_categories = sorted(list(ocm.keys()))

# ---- Count ----
c_train, n_train, miss_train, empty_train = count_categories(train)
c_test,  n_test,  miss_test,  empty_test  = count_categories(test)
c_all = c_train + c_test

# ---- Build tables incl. zero-count + pct over ALL counts ----
df_train, total_train = counter_to_df_include_zeros_and_pct_of_all(c_train, all_top_categories)
df_test,  total_test  = counter_to_df_include_zeros_and_pct_of_all(c_test,  all_top_categories)
df_all,   total_all   = counter_to_df_include_zeros_and_pct_of_all(c_all,   all_top_categories)

# Optional: add split-level metadata for convenience
summary_df = pd.DataFrame([
    {"split": "TRAIN", "n_images": n_train, "missing_field": miss_train, "empty_list": empty_train,
     "total_assigned_category_counts": total_train},
    {"split": "TEST",  "n_images": n_test,  "missing_field": miss_test,  "empty_list": empty_test,
     "total_assigned_category_counts": total_test},
    {"split": "ALL",   "n_images": n_train + n_test, "missing_field": miss_train + miss_test,
     "empty_list": empty_train + empty_test, "total_assigned_category_counts": total_all},
])

# ---- Save XLSX ----
out_dir = Path("category_representation_outputs")
out_dir.mkdir(parents=True, exist_ok=True)
xlsx_path = out_dir / "category_representation.xlsx"

with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="summary", index=False)
    df_train.to_excel(writer, sheet_name="train", index=False)
    df_test.to_excel(writer, sheet_name="test", index=False)
    df_all.to_excel(writer, sheet_name="all", index=False)

print(f"✅ Saved Excel workbook to: {xlsx_path.resolve()}")
