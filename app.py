import streamlit as st
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
from ddgs import DDGS
from openai import OpenAI
from dotenv import load_dotenv
import json
from groq import Groq

def load_catalog():
    with open("teamthai_catalog.json", "r") as f:
        return json.load(f)

catalog = load_catalog()

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

sns.set_style("whitegrid")

def build_price_table(rows, teamthai_brand):
    import re

    def extract_prices(text):
        matches = re.findall(r'₹\s?(\d+(?:,\d{3})*(?:\.\d+)?)', str(text))
        prices = []
        for m in matches:
            try:
                val = float(m.replace(",", ""))
                if 5 <= val <= 5000:
                    prices.append(val)
            except ValueError:
                pass
        return prices

    table_rows = []
    for row in rows:
        title = row.get("title", "")
        snippet = row.get("snippet", "")
        prices = extract_prices(snippet) + extract_prices(title)
        brand_type = "Team Thai" if row.get("is_teamthai", False) else "Competitor"
        for p in prices:
            table_rows.append({
                "Brand": row.get("product", ""),
                "Type": brand_type,
                "Price Found (₹)": p,
                "Source": title[:60] + "..." if len(title) > 60 else title,
            })

    if not table_rows:
        return None

    price_table = pd.DataFrame(table_rows)
    price_table = price_table.sort_values(["Type", "Brand", "Price Found (₹)"])
    return price_table

def build_dashboard_charts(rows, teamthai_brand):
    df = pd.DataFrame(rows)

    # ensure expected columns exist to avoid KeyError
    for col, default in [("snippet", ""), ("title", ""), ("is_teamthai", False), ("product", "")]:
        if col not in df.columns:
            df[col] = default

    import re
    def extract_prices(text):
        matches = re.findall(r'₹\s?(\d+(?:,\d{3})*(?:\.\d+)?)', str(text))
        prices = []
        for m in matches:
            try:
                prices.append(float(m.replace(",", "")))
            except ValueError:
                pass
        return prices

    price_rows = []
    for _, row in df.iterrows():
        prices = extract_prices(row.get("snippet", "")) + extract_prices(row.get("title", ""))
        for p in prices:
            if 5 <= p <= 5000:
                price_rows.append({"product": row.get("product", ""), "price": p})

    price_df = pd.DataFrame(price_rows)

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    if not price_df.empty:
        avg_price = price_df.groupby("product")["price"].mean().reset_index()
        avg_price = avg_price.sort_values("price")
        colors = ["#1F3864" if p.lower().startswith(teamthai_brand.lower()) else "#B0B0B0"
                  for p in avg_price["product"]]
        sns.barplot(data=avg_price, x="price", y="product", palette=colors, ax=ax1)
        ax1.set_title("Estimated Average Price by Brand (₹)", fontsize=12, weight="bold")
        ax1.set_xlabel("Average price found in listings (₹)")
        ax1.set_ylabel("")
    else:
        ax1.text(0.5, 0.5, "No price data detected in this search.\nTry a different region or product.",
                  ha="center", va="center", fontsize=10, color="gray")
        ax1.set_title("Estimated Average Price by Brand (₹)", fontsize=12, weight="bold")
        ax1.axis("off")

    own_count = len(df[df["is_teamthai"] == True])
    competitor_count = len(df[df["is_teamthai"] == False])

    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.pie(
        [own_count, competitor_count],
        labels=[f"{teamthai_brand}", "Competitors (combined)"],
        autopct="%1.0f%%",
        colors=["#1F3864", "#D9D9D9"],
        startangle=90,
    )
    ax2.set_title("Share of Online Visibility", fontsize=12, weight="bold")

    positive_words = ["best", "quality", "trusted", "effective", "great",
                       "recommend", "excellent", "popular", "top", "good"]

    def positive_score(text):
        text = str(text).lower()
        return sum(text.count(w) for w in positive_words)

    df["positive_signal"] = df["snippet"].apply(positive_score)
    sentiment_by_brand = df.groupby("product")["positive_signal"].sum().reset_index()
    sentiment_by_brand = sentiment_by_brand.sort_values("positive_signal", ascending=False)

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    colors3 = ["#1F3864" if str(p).lower().startswith(teamthai_brand.lower()) else "#B0B0B0"
               for p in sentiment_by_brand["product"]]
    sns.barplot(data=sentiment_by_brand, x="positive_signal", y="product", palette=colors3, ax=ax3)
    ax3.set_title("Positive Language Signal by Brand", fontsize=12, weight="bold")
    ax3.set_xlabel("Positive keyword mentions (quality, trusted, best, etc.)")
    ax3.set_ylabel("")

    return fig1, fig2, fig3

def get_verdict(rows, teamthai_brand):
    df = pd.DataFrame(rows)
    own_count = len(df[df["is_teamthai"] == True])
    comp_count = len(df[df["is_teamthai"] == False])
    
    if own_count == 0 and comp_count == 0:
        return None, None
    
    own_share = own_count / (own_count + comp_count) * 100
    
    if own_share >= 40:
        return "success", f" {teamthai_brand} holds a strong {own_share:.0f}% share of online visibility in this comparison."
    elif own_share >= 20:
        return "warning", f" {teamthai_brand} holds a moderate {own_share:.0f}% share — room to grow visibility."
    else:
        return "error", f" {teamthai_brand} holds only {own_share:.0f}% share — competitors are dominating online presence here."

def build_dashboard_metrics(rows, teamthai_brand):
    df = pd.DataFrame(rows)
    
    own_count = len(df[df["is_teamthai"] == True])
    competitor_count = len(df[df["is_teamthai"] == False])
    num_competitors = df[df["is_teamthai"] == False]["product"].nunique()
    total_mentions = len(df)
    
    return {
        "own_count": own_count,
        "competitor_count": competitor_count,
        "num_competitors": num_competitors,
        "total_mentions": total_mentions,
    }

# ---------- Core Functions ----------

def search_competitor(product_name, region):
    query = f"{product_name} {region} price"
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append({
                    "product": product_name,
                    "region": region,
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "date_collected": datetime.now().strftime("%Y-%m-%d %H:%M")
                })
    except Exception:
        pass
    return results

def clean_llm_output(text):

    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    last_line = lines[-1] if lines else text
    if len(last_line) > 200 or "maybe" in last_line.lower() or "I think" in last_line:
        return ""
    return last_line

def discover_competitors(brand_name, specific_product, category, region):
    queries = [
        f"{specific_product} alternatives {region}",
        f"top {category} brands in {region} similar to {specific_product}",
        f"best selling {category} products {region} competitors of {brand_name}",
    ]

    raw_results = []
    try:
        with DDGS() as ddgs:
            for q in queries:
                raw_results.extend([r.get("body", "") for r in ddgs.text(q, max_results=5)])
    except Exception:
        pass

    combined_text = "\n".join(raw_results)

    prompt = f"""Based on the following search results, list ONLY real 
FMCG BRAND NAMES that make {category} PRODUCTS and are sold/available in 
{region}, NOT appliance/electronics brands.

CRITICAL: Do NOT include "{brand_name}" itself, or any close variation of 
its name, in the list.

Search results:
{combined_text}

Return ONLY a comma-separated list of 6-8 real FMCG competitor brand names, 
nothing else."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    raw_output = response.choices[0].message.content.strip()
    raw_output = clean_llm_output(raw_output)
    raw_list = [c.strip() for c in raw_output.split(",")] if raw_output else []

    def normalize(s):
        return "".join(ch for ch in (s or "").lower() if ch.isalnum())

    brand_norm = normalize(brand_name)
    BLOCKLIST = ["LG", "Samsung", "Bosch", "Whirlpool", "IFB", "Godrej Appliances",
                 "Bajaj Finance", "Voltas", "Haier", "Milma", "Amway India",
                 "Blinkit", "Zepto", "Swiggy Instamart", "Amazon", "Flipkart", "BigBasket"]
    blockset = set(normalize(x) for x in BLOCKLIST)

    filtered = []
    for c in raw_list:
        if not c:
            continue
        n = normalize(c)
        if n == brand_norm or n in blockset:
            continue
        filtered.append(c)

    return filtered[:8]


def format_data_clearly(rows, teamthai_brand):
    lines = []
    for row in rows:
        label = "[TEAM THAI'S OWN PRODUCT]" if row.get("is_teamthai", False) else "[COMPETITOR PRODUCT]"
        lines.append(f"{label} {row.get('product','')} | Region: {row.get('region','')} | "
                      f"{row.get('title','')} | {row.get('snippet','')}")
    return "\n".join(lines)


def generate_report(rows, teamthai_brand, category, known_facts="", product_count=0, region=""):
    data_summary = format_data_clearly(rows, teamthai_brand)[:8000]

    facts_block = ""
    if known_facts.strip():
        facts_block = f"""
=== VERIFIED GROUND TRUTH ABOUT {teamthai_brand} (DO NOT CONTRADICT) ===
{known_facts}

FACT: {teamthai_brand} currently has {product_count} distinct product 
variants already in its lineup. This is a WIDE/ESTABLISHED range, not a 
narrow one. NEVER describe {teamthai_brand}'s product range as "limited," 
"narrow," or suggest it "only" has one or two products — that directly 
contradicts this verified fact.
=== END VERIFIED GROUND TRUTH ===

RULE: Before writing ANY claim about what {teamthai_brand} does or doesn't 
offer, check it against the VERIFIED GROUND TRUTH above. This ground 
truth always overrides anything implied by search snippets.
"""

    prompt = f"""You are a senior FMCG market analyst working for Team Thai, 
analyzing the {region} market. Team Thai's OWN brand here is: {teamthai_brand} 
({category}).
{facts_block}
Each line below is labeled "[TEAM THAI'S OWN PRODUCT]" or 
"[COMPETITOR PRODUCT]" — trust these labels exactly.

IGNORE any data unrelated to {category} products in {region}.

DATA:
{data_summary}

IMPORTANT: Advice must help Team Thai compete against and outperform 
rivals — never suggest partnering with, merging with, or collaborating 
with a competing manufacturer. Also disregard generic DIY/tutorial content 
that isn't about the specific brands being compared — focus only on 
commercial products, pricing, and brand positioning.

Provide a structured report:
1. KEY FINDINGS (3-4 bullets)
2. COMPETITIVE GAPS (must be consistent with the verified ground truth — 
   do not claim a product is missing if ground truth says it exists)
3. ACTIONABLE ADVICE (specific, practical steps to beat competitors — 
   must not repeat something Team Thai already has, per ground truth)
4. CONFIDENCE NOTE (data limitations, what needs human verification)

Keep it concise and business-focused."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content

# ---------- Streamlit Interface ----------

st.set_page_config(
    page_title="Team Thai — Competitive Intelligence Agent",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    div.stButton > button:first-child {
        background-color: #1E2761;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        border: none;
    }
    div.stButton > button:first-child:hover {
        background-color: #3D5AFE;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background-color:#1E2761; padding:1.5rem 2rem; border-radius:10px; margin-bottom:1.5rem;">
    <h1 style="color:white; margin:0; font-size:1.8rem;"> Team Thai — Competitive Intelligence Agent</h1>
    <p style="color:#CADCFC; margin:0.3rem 0 0 0; font-size:0.95rem;">Real-time competitor insights from live online data — no physical surveys required.</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    product = st.selectbox("Team Thai Brand", options=list(catalog.keys()))

with col2:
    specific_product = st.selectbox(
        "Specific Product",
        options=catalog[product]["products"]
    )

with col3:
    category = catalog[product]["category"]
    st.text_input("Category", value=category, disabled=True)

with col4:
    region = st.text_input("Region", value="Kerala, India", help="Enter any region")
# --- KNOWN FACTS 
brand_description = catalog[product].get("description", "")
auto_facts = (
    f"Brand positioning: {brand_description} "
    f"Currently analyzing this specific product: {specific_product}. "
    f"Full existing range under {product}: " + "; ".join(catalog[product]["products"])
)

known_facts = st.text_area(
    "Known facts about this product (auto-filled from catalog, editable)",
    value=f"{auto_facts}. Do not recommend expanding into any of these existing categories.",
    height=120
)

if st.button(" Analyze Now", type="primary"):
    with st.spinner("Discovering competitors..."):
        competitors = discover_competitors(product, specific_product, category, region)
        st.success(f"Discovered competitors: {', '.join(competitors)}")

    with st.spinner("Collecting live data..."):
        all_rows = []
        own_data = search_competitor(f"{specific_product}", region)
        for r in own_data:
            r["is_teamthai"] = True
        all_rows.extend(own_data)

        for comp in competitors:
            comp_data = search_competitor(f"{comp} {category}", region)
            for r in comp_data:
                r["is_teamthai"] = False
            all_rows.extend(comp_data)

    st.info(f"Collected {len(all_rows)} live data points as of "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    st.markdown(f"""
    <div style="display:flex; gap:1rem; margin-bottom:1rem;">
        <span style="background:#F4F6FC; padding:0.4rem 0.9rem; border-radius:20px; font-size:0.85rem; color:#1E2761;"> {region}</span>
        <span style="background:#F4F6FC; padding:0.4rem 0.9rem; border-radius:20px; font-size:0.85rem; color:#1E2761;"> {product}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # ---------- DASHBOARD ----------
    st.markdown("###  Competitive Dashboard")

    fig1, fig2, fig3 = build_dashboard_charts(all_rows, product)

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.pyplot(fig1)
    with col_b:
        st.pyplot(fig2)

    st.pyplot(fig3)
    
    verdict_type, verdict_text = get_verdict(all_rows, product)
    if verdict_text:
        if verdict_type == "success":
            st.success(verdict_text)
        elif verdict_type == "warning":
            st.warning(verdict_text)
        else:
            st.error(verdict_text)
    
    st.markdown("###  Exact Prices Found (per listing)")
    price_table = build_price_table(all_rows, product)

    if price_table is not None:
        st.dataframe(price_table, use_container_width=True, hide_index=True)
    else:
        st.info("No specific price values were found in this search — try a more specific product name or different region.")

    st.markdown("---")
    # ---------- END DASHBOARD ----------

    with st.spinner("Generating insights and advice..."):
        product_count = len(catalog[product]["products"])
        report = generate_report(all_rows, product, category, known_facts, product_count, region)

    st.markdown("---")
    st.subheader(" Competitive Insights Report")
    st.markdown(report)
    st.download_button(
        label=" Download Report as Text",
        data=report,
        file_name=f"{product}_competitive_report_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain",
    )

    with st.expander("View raw collected data"):
        st.dataframe(pd.DataFrame(all_rows))
        
    with st.expander("How this agent works"):
        st.markdown("""
        1. **Discover** — finds real competitor brands for the selected product & region  
        2. **Collect** — pulls live pricing, reviews & trends from the open web  
        3. **Ground** — checks findings against Team Thai's real product catalog  
        4. **Analyze & Advise** — an LLM generates findings and specific competitive advice  
    
        *Built with Streamlit, Groq (Llama 3.1), and DuckDuckGo Search — running entirely on free-tier tools.*
        """)