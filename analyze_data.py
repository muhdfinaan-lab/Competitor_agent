import pandas as pd
import glob
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def load_latest_snapshot():
    files = glob.glob("data/teamthai_competitor_data_*.csv")
    if not files:
        raise FileNotFoundError("No data files found. Run collect_data.py first.")
    
    latest_file = max(files, key=os.path.getctime)
    print(f"Loading: {latest_file}")
    return pd.read_csv(latest_file)


def format_data_clearly(df):
    lines = []
    
    for _, row in df.iterrows():
        label = "[TEAM THAI'S OWN PRODUCT]" if row["is_teamthai"] else "[COMPETITOR PRODUCT]"
        lines.append(
            f"{label} Brand/Search term: {row['product']} | Region: {row['region']} | "
            f"Title: {row['title']} | Info: {row['snippet']}"
        )
    
    return "\n".join(lines)


def generate_insights_and_advice(df):
    data_summary = format_data_clearly(df)

    if len(data_summary) > 8000:
        data_summary = data_summary[:8000]

    teamthai_brands = ", ".join(df[df["is_teamthai"] == True]["product"].unique())

    prompt = f"""You are a senior FMCG market analyst working for Team Thai, 
an FMCG company selling laundry soaps, toilet soaps, and cleaning products 
IN INDIA ONLY.

CRITICAL CONTEXT: Team Thai's OWN brands being analyzed in this report are: 
{teamthai_brands}. These are NOT competitors — they are Team Thai's own 
products. Never refer to Team Thai's own brand as a competitor.

Each line of data below is explicitly labeled either 
"[TEAM THAI'S OWN PRODUCT]" or "[COMPETITOR PRODUCT]" — trust these labels 
exactly as given, do not reinterpret them.

IMPORTANT: Some search results may be irrelevant or about unrelated 
businesses that happen to share a similar name (e.g., unrelated laundromats, 
services, or companies in other countries). IGNORE any data that doesn't 
clearly relate to FMCG soap/detergent products in the Indian market.

DATA:
{data_summary}

Based on this data, provide a structured report with these exact sections:

1. KEY FINDINGS — 3-4 bullet points summarizing what the data shows about 
    Team Thai's OWN products vs COMPETITOR products (pricing signals, 
    review sentiment, market presence, region differences).

2. COMPETITIVE GAPS — where Team Thai's own products appear weaker than 
    competitor products based on this data.

3. ACTIONABLE ADVICE — 3-4 specific, practical recommendations for how 
    Team Thai can beat these competitors (pricing moves, marketing focus, 
    regional strategy, promotional timing). Be specific, not generic.

4. CONFIDENCE NOTE — briefly note that this is based on online search 
    snippets, not verified pricing data, and recommend which findings need 
    human verification before acting.

Keep the entire report concise and business-focused — this will be read 
by a manager, not a technical audience."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    df = load_latest_snapshot()
    print(f"Analyzing {len(df)} records...\n")
    
    report = generate_insights_and_advice(df)
    print("=" * 60)
    print("COMPETITIVE INSIGHTS REPORT")
    print("=" * 60)
    print(report)
    
    # Save the report too
    with open("data/latest_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("\n\nReport saved to: data/latest_report.txt")