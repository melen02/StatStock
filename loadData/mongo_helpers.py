import pandas as pd
from pymongo import UpdateOne


def is_missing(v):
    """True for None, NaN and NaT."""
    return v is None or (pd.api.types.is_scalar(v) and pd.isna(v))


def upsert_collection(col, df, unique_keys, name):
    if df.empty:
        print(f"No data to upsert into {name}")
        return

    # Without this index every upsert scans the whole collection.
    col.create_index([(key, 1) for key in unique_keys])

    operations = [
        UpdateOne({key: record[key] for key in unique_keys}, {"$set": record}, upsert=True)
        for record in df.to_dict("records")
    ]
    result = col.bulk_write(operations, ordered=False)
    print(f"{result.upserted_count} inserted, {result.matched_count} updated in {name}")


def merge_quarterly_rows(col, df, unique_keys, name, refresh_quarters=None):
    if df.empty:
        print(f"No data to merge into {name}")
        return

    count_inserted = 0
    count_updated = 0

    for record in df.to_dict("records"):
        query = {key: record[key] for key in unique_keys if key in record}
        existing = col.find_one(query)

        if not existing:
            col.insert_one(record)
            count_inserted += 1
            continue

        update_fields = {}

        for key, new_value in record.items():
            if key in unique_keys or is_missing(new_value):
                continue

            refresh = refresh_quarters and record.get("Quarter") in refresh_quarters
            if is_missing(existing.get(key)) or refresh:
                update_fields[key] = new_value

        if update_fields:
            col.update_one(query, {"$set": update_fields})
            count_updated += 1

    print(f"{count_inserted} inserted, {count_updated} updated in {name}")


def replace_collection(db, name, df):
    """Replace a derived-results collection with the contents of df."""
    records = df.copy()
    for c in records.select_dtypes(include="datetime").columns:
        records[c] = records[c].astype(str)
    records = records.astype(object).where(records.notna(), None).to_dict("records")

    db[name].drop()
    if records:
        db[name].insert_many(records)
    print(f"Saved {len(records)} rows to MongoDB:{name}")
