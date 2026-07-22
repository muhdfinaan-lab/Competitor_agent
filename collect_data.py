from ddgs import DDGS
import pandas as pd
from datetime import datetime
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def search_competitor(product_name, region):
    """
    Searches DuckDuckGo for a competitor product in a specific region.
    Returns a list of search results (title, snippet, link).
    """
    query = f"{product_name} India {region} price"
    results = []

    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=5):
            results.append({
                "product": product_name,
                "region": region,
                "title": r["title"],
                "snippet": r["body"],
                "link": r["href"],
                "date_collected": datetime.now().strftime("%Y-%m-%d %H:%M")
            })

    return results


def save_snapshot(df, filename_prefix="competitor_data"):
    """
    Saves the collected data to a CSV file named with today's date.
    """
    if not os.path.exists("data"):
        os.makedirs("data")

    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"data/{filename_prefix}_{today}.csv"

    df.to_csv(filename, index=False)
    print(f"Saved snapshot to: {filename}")
    return filename


def discover_competitors(brand_name, category, region):
    """
    Uses web search + LLM to automatically find real competitor brands.
    """
    query = f"top {category} brands in {region} India competitors of {brand_name}"

    with DDGS() as ddgs:
        raw_results = [r["body"] for r in ddgs.text(query, max_results=5)]

    combined_text = "\n".join(raw_results)

    prompt = f"""Based on the following search results, list ONLY the real 
brand names (not {brand_name} itself) that are competitors in the 
{category} market in {region}, India. 

Search results:
{combined_text}

Return ONLY a comma-separated list of 2-4 brand names, nothing else. 
Example format: Rin, Wheel, Ghadi"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    competitors_text = response.choices[0].message.content.strip()
    competitors = [c.strip() for c in competitors_text.split(",")]
    return competitors


if __name__ == "__main__":
    teamthai_products = [
        {"brand": "Dr.Wash", "category": "laundry soap"},
        {"brand": "Oliva", "category": "toilet soap"},
        {"brand": "Speed XL", "category": "stain removal soap"},
    ]
    regions = ["Kerala", "Tamil Nadu"]

    all_results = []

    for product in teamthai_products:
        brand = product["brand"]
        category = product["category"]

        for region in regions:
            print(f"\n--- Processing {brand} ({category}) in {region} ---")

            competitors = discover_competitors(brand, category, region)
            print(f"Discovered competitors: {competitors}")

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
    print(f"\nTotal records collected: {len(df)}")
    save_snapshot(df, filename_prefix="teamthai_competitor_data")