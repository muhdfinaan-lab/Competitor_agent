import schedule
import time
from datetime import datetime
from collect_data import (
    search_competitor, discover_competitors, save_snapshot
)
import pandas as pd
import json

def load_catalog():
    with open("teamthai_catalog.json", "r") as f:
        return json.load(f)

def run_daily_collection():
    print(f"\n{'='*50}")
    print(f"Daily collection started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*50}")

    catalog = load_catalog()
    all_results = []

    brands_to_track = ["Dr.Wash", "Oliva", "Speed XL"]
    region = "Kerala, India"

    for brand in brands_to_track:
        if brand not in catalog:
            continue
        category = catalog[brand]["category"]
        print(f"Processing {brand}...")

        competitors = discover_competitors(brand, category, region)
        own_data = search_competitor(f"{brand} {category}", region)
        for row in own_data:
            row["is_teamthai"] = True
        all_results.extend(own_data)

        for comp in competitors:
            comp_data = search_competitor(f"{comp} {category}", region)
            for row in comp_data:
                row["is_teamthai"] = False
            all_results.extend(comp_data)

    df = pd.DataFrame(all_results)
    save_snapshot(df, filename_prefix="daily_teamthai_snapshot")
    print(f"Daily collection complete: {len(df)} records saved.\n")


schedule.every().day.at("08:00").do(run_daily_collection)

if __name__ == "__main__":
    print("Daily scheduler started. Waiting for scheduled time...")
    print("Press Ctrl+C to stop.\n")

    run_daily_collection()

    while True:
        schedule.run_pending()
        time.sleep(60)