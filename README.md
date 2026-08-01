#  Competitive Intelligence Agent

An AI agent that discovers competitors, pulls live market data, and generates actionable business advice — built in 48 hours for an interview challenge from **Team Thai**.

**The brief:** No physical surveys. Only public online data. Refreshed daily. Real, actionable advice — not just raw data.

## What it does

-  **Discovers** real competitor brands for any product, in any region — automatically
-  **Collects** live pricing, reviews, and market data from the open web
-  **Grounds** every insight in a verified product catalog, so the AI can't invent or contradict real facts
-  **Advises** — generates specific, actionable recommendations to beat competitors
-  **Refreshes daily** via an automated scheduler

## Tech Stack

Python · Streamlit · Groq (Llama 3.1) · Live web search · Pandas · Matplotlib/Seaborn — 100% free-tier.

## The real story

The demo was the easy part. Along the way, the AI got things wrong — confusing product categories, misidentifying its own brand as a competitor, contradicting verified facts. Every bug caught and fixed here taught more about building *reliable* AI than the working demo itself.

## Quick Start

```bash
pip install -r requirements.txt
# add your GROQ_API_KEY to a .env file
streamlit run app.py
```

## Demo

 Full walkthrough: [link to your LinkedIn post/video]

---

Built by [Muhammed Finan P C](https://linkedin.com/in/muhd-finan)
