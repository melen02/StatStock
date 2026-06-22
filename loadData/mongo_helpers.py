import pandas as pd

def is_missing(v):
    return v is None or (isinstance(v, float) and pd.isna(v))


def upsert_collection(col, df, unique_keys, name):
    if df.empty:
        print(f"No data to upsert into {name}")
        return

    count_inserted = 0
    count_updated = 0

    for record in df.to_dict("records"):
        query = {key: record[key] for key in unique_keys if key in record}
        result = col.update_one(query, {"$set": record}, upsert=True)

        if result.matched_count > 0:
            count_updated += 1
        else:
            count_inserted += 1

    print(f"{count_inserted} inserted, {count_updated} updated in {name}")


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
            if key in unique_keys:
                continue

            old_value = existing.get(key)

            if is_missing(old_value) and not is_missing(new_value):
                update_fields[key] = new_value

            elif refresh_quarters and record.get("Quarter") in refresh_quarters:
                if not is_missing(new_value):
                    update_fields[key] = new_value

        if update_fields:
            col.update_one(query, {"$set": update_fields})
            count_updated += 1

    print(f"{count_inserted} inserted, {count_updated} updated in {name}")
    