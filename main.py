# =====================================================================
# 1. DEPENDENCIES, AUTHENTICATION & ENVIRONMENT SETUP
# =====================================================================

import os
import re
import time
import asyncio
import requests
import feedparser
import resend
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from google import genai
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# Authenticate Gemini & Setup LLM Client via System Environment Variables
gemini_key = os.getenv('GEMINI_API_KEY')
if not gemini_key:
    raise ValueError("GEMINI_API_KEY environment variable is missing!")

os.environ["GEMINI_API_KEY"] = gemini_key

gemini_llm = LLM(
    model="gemini/gemini-3.6-flash",
    api_key=gemini_key
)

# =====================================================================
# 2. EMAIL DISPATCH (RESEND API)
# =====================================================================

def send_email_briefing(briefing_text: str):
    """Dispatches the final executive briefing to configured recipients via Resend API."""
    resend.api_key = os.getenv('RESEND_API_KEY')
    if not resend.api_key:
        print("❌ ERROR: RESEND_API_KEY environment variable is missing. Skipping email dispatch.")
        return

    raw_receivers = os.getenv('RECEIVER_EMAIL') or ''
    recipient_list = [email.strip() for email in raw_receivers.split(',') if email.strip()]

    if not recipient_list:
        print("❌ ERROR: RECEIVER_EMAIL is empty. Skipping email dispatch.")
        return

    html_content = f"""
    <html>
      <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <h2 style="color: #1a365d;">Daily Executive Commodity Briefing</h2>
        <pre style="white-space: pre-wrap; font-family: inherit;">{briefing_text}</pre>
      </body>
    </html>
    """

    params = {
        "from": "Commodity Desk <onboarding@resend.dev>",
        "to": recipient_list,
        "subject": "📊 Daily Executive Commodity Briefing",
        "text": briefing_text,
        "html": html_content
    }

    try:
        print(f"\n[Email Dispatch] Sending briefing via Resend API...")
        response = resend.Emails.send(params)
        print(f" SUCCESS: Briefing dispatched! Resend ID: {response['id']}")
        print(f" Delivered to {len(recipient_list)} recipient(s): {', '.join(recipient_list)}")
    except Exception as e:
        print(f" ERROR sending via Resend: {str(e)}")

# =====================================================================
# 3. CUSTOM SCRAPING TOOLS
# =====================================================================

@tool("Yahoo Finance Direct Historical Multi-Commodity Scraper")
def fetch_multi_commodity_prices() -> str:
    """Scrapes daily close prices directly from Yahoo Finance backend API for Copper, Aluminum, Nickel, Zinc, Gold, Silver, and Brent Crude Oil to compute accurate WoW, MoM, and YoY trends."""

    commodities = {
        "Copper": ("HG=F", "lb", True),             # COMEX High Grade Copper ($/lb -> $/MT)
        "Aluminum": ("ALI=F", "MT", False),          # LME/COMEX Aluminum ($/MT)
        "Nickel": ("NIK=F", "MT", False),            # Nickel Futures ($/MT)
        "Zinc": ("ZNC=F", "MT", False),              # Zinc Futures ($/MT)
        "Gold": ("GC=F", "oz", False),               # COMEX Gold ($/oz)
        "Silver": ("SI=F", "oz", False),             # COMEX Silver ($/oz)
        "Brent Crude Oil": ("BZ=F", "bbl", False)    # Brent Crude Futures ($/bbl)
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    report = ["=== DIRECT FINANCIAL HISTORICAL BENCHMARKS ==="]

    for name, (ticker, unit, convert_to_mt) in commodities.items():
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=2y&interval=1d"

        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                report.append(f"\n• **{name}** ({ticker}): HTTP Error {response.status_code}")
                continue

            json_data = response.json()
            result = json_data.get("chart", {}).get("result", [])

            if not result:
                report.append(f"\n• **{name}** ({ticker}): No data returned from API.")
                continue

            timestamps = result[0].get("timestamp", [])
            indicators = result[0].get("indicators", {}).get("quote", [{}])[0]
            close_prices = indicators.get("close", [])

            historical_data = []
            for ts, price in zip(timestamps, close_prices):
                if price is not None:
                    date_str = time.strftime("%b %d, %Y", time.gmtime(ts))
                    historical_data.append({"date": date_str, "close": float(price)})

            if len(historical_data) < 2:
                report.append(f"\n• **{name}** ({ticker}): Insufficient price history.")
                continue

            historical_data.reverse()

            latest_date = historical_data[0]["date"]
            latest_price = historical_data[0]["close"]

            def get_change(offset_days):
                idx = min(offset_days, len(historical_data) - 1)
                past_price = historical_data[idx]["close"]
                pct = ((latest_price - past_price) / past_price) * 100
                arrow = "▲ +" if pct > 0 else ("▼ " if pct < 0 else "► ")
                return f"{arrow}{pct:.2f}%"

            wow = get_change(5)
            mom = get_change(21)
            yoy = get_change(252) if len(historical_data) >= 252 else get_change(len(historical_data) - 1)

            price_str = f"${latest_price:,.2f} / {unit}"
            if convert_to_mt:
                mt_price = latest_price * 2204.622
                price_str += f" (${mt_price:,.2f} / MT)"

            report.append(
                f"\n• **{name}** ({ticker} as of {latest_date})\n"
                f"  - **Closing Price:** {price_str}\n"
                f"  - **Trends:** WoW: {wow} | MoM: {mom} | YoY: {yoy}"
            )

        except Exception as e:
            report.append(f"\n• **{name}**: Error parsing API data: {str(e)}")

    return "\n".join(report)

@tool("Global Steel Benchmarks Direct Scraper")
def fetch_global_steel_benchmarks() -> str:
    """Scrapes US HRC (Yahoo API), China Steel Futures (LME FOB China Argus first row in CNY), and Central European HRC (LME NW Europe Argus first row in EUR) directly from LME web tables using real-time Forex APIs."""

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }

    report = ["=== GLOBAL STEEL BENCHMARKS ==="]

    usd_to_cny = 7.10
    eur_to_usd = 1.10

    try:
        fx_resp = requests.get("https://open.er-api.com/v6/latest/USD", timeout=5)
        if fx_resp.status_code == 200:
            fx_rates = fx_resp.json().get("rates", {})
            if "CNY" in fx_rates:
                usd_to_cny = float(fx_rates["CNY"])
            if "EUR" in fx_rates and float(fx_rates["EUR"]) > 0:
                eur_to_usd = 1.0 / float(fx_rates["EUR"])
    except Exception as e:
        print(f"[FX API Warning] Could not fetch live rates: {str(e)}")

    # 1. US HRC STEEL
    try:
        us_url = "https://query1.finance.yahoo.com/v8/finance/chart/HRC%3DF?range=2y&interval=1d"
        resp_us = requests.get(us_url, headers=headers, timeout=10)
        if resp_us.status_code == 200:
            result_us = resp_us.json().get("chart", {}).get("result", [])[0]
            timestamps = result_us.get("timestamp", [])
            closes = result_us.get("indicators", {}).get("quote", [{}])[0].get("close", [])

            data_us = [{"price": p} for t, p in zip(timestamps, closes) if p is not None]
            data_us.reverse()

            if len(data_us) >= 2:
                latest_price = data_us[0]["price"]

                def calc_pct(offset):
                    idx = min(offset, len(data_us) - 1)
                    past = data_us[idx]["price"]
                    pct = ((latest_price - past) / past) * 100
                    sym = "▲ +" if pct > 0 else ("▼ " if pct < 0 else "► ")
                    return f"{sym}{pct:.2f}%"

                wow = calc_pct(5)
                mom = calc_pct(21)
                yoy = calc_pct(252) if len(data_us) >= 252 else calc_pct(len(data_us) - 1)

                report.append(
                    f"• **US HRC Steel (CME HRC=F):**\n"
                    f"  - Price: ${latest_price:,.2f} / Short Ton\n"
                    f"  - Trends: WoW: {wow} | MoM: {mom} | YoY: {yoy}"
                )
    except Exception as e:
        report.append(f"• **US Steel Error:** {str(e)}")

    # 2. CHINA STEEL FUTURES
    try:
        cn_url = "https://www.lme.com/metals/ferrous/lme-steel-hrc-fob-china-argus"
        resp_cn = requests.get(cn_url, headers=headers, timeout=10)
        if resp_cn.status_code == 200:
            soup_cn = BeautifulSoup(resp_cn.content, "html.parser")
            tables_cn = soup_cn.find_all("table")
            if tables_cn:
                rows = tables_cn[0].find_all("tr")
                first_row_cn = None
                for row in rows:
                    cols = [c.text.strip() for c in row.find_all(["td", "th"])]
                    if len(cols) >= 2:
                        raw_price = cols[1].replace('$', '').replace(',', '').strip()
                        if raw_price.replace('.', '', 1).isdigit() and float(raw_price) > 0:
                            first_row_cn = (cols[0], float(raw_price))
                            break

                if first_row_cn:
                    contract_name, price_usd = first_row_cn
                    price_cny = price_usd * usd_to_cny
                    report.append(
                        f"• **China Steel Futures (LME FOB China Argus - Contract {contract_name}):** ¥{price_cny:,.2f} CNY / MT (${price_usd:,.2f} USD / MT @ FX {usd_to_cny:.4f})"
                    )
                else:
                    report.append("• **China Steel:** First row price could not be extracted.")
    except Exception as e:
        report.append(f"• **China Steel Error:** {str(e)}")

    # 3. CENTRAL EUROPEAN STEEL
    try:
        eu_url = "https://www.lme.com/metals/ferrous/lme-steel-hrc-nw-europe-argus"
        resp_eu = requests.get(eu_url, headers=headers, timeout=10)
        if resp_eu.status_code == 200:
            soup_eu = BeautifulSoup(resp_eu.content, "html.parser")
            tables_eu = soup_eu.find_all("table")
            if tables_eu:
                rows = tables_eu[0].find_all("tr")
                first_row_eu = None
                for row in rows:
                    cols = [c.text.strip() for c in row.find_all(["td", "th"])]
                    if len(cols) >= 2:
                        raw_price = cols[1].replace('$', '').replace(',', '').strip()
                        if raw_price.replace('.', '', 1).isdigit() and float(raw_price) > 0:
                            first_row_eu = (cols[0], float(raw_price))
                            break

                if first_row_eu:
                    contract_name, price_usd = first_row_eu
                    price_eur = price_usd / eur_to_usd
                    report.append(
                        f"• **Central European Steel (LME HRC NW Europe Argus - Contract {contract_name}):** €{price_eur:,.2f} EUR / MT (${price_usd:,.2f} USD / MT @ FX {eur_to_usd:.4f})"
                    )
                else:
                    report.append("• **Central European Steel:** First row price could not be extracted.")
    except Exception as e:
        report.append(f"• **Central European Steel Error:** {str(e)}")

    return "\n".join(report)

@tool("Multi-Source Commodity RSS Reader")
def fetch_commodity_rss_news() -> str:
    """Fetches energy, metals, and shipping headlines from RSS feeds, excluding items older than 14 days."""
    feeds = {
        "Mining.com (Metals & Steel)": "https://www.mining.com/feed/",
        "Oilprice.com (Brent & Energy)": "https://oilprice.com/rss/main",
        "FreightWaves (Sea Freight & Shipping)": "https://www.freightwaves.com/news/category/maritime/feed"
    }

    report = []
    cutoff_date = datetime.now() - timedelta(days=14)

    for source_name, url in feeds.items():
        try:
            parsed = feedparser.parse(url)
            report.append(f"\n=== {source_name} ===")
            entries = parsed.entries[:6] if parsed.entries else []

            valid_entries = 0
            for entry in entries:
                title = entry.get('title', 'No Title')
                link = entry.get('link', '')
                summary = entry.get('summary', '')[:200]

                pub_date_str = "Recent"
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    dt = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                    if dt < cutoff_date:
                        continue
                    pub_date_str = dt.strftime("%b %d, %Y")

                report.append(
                    f"• Headline: {title}\n"
                    f"  Date: {pub_date_str}\n"
                    f"  Source URL: {link}\n"
                    f"  Snippet: {summary}...\n"
                )
                valid_entries += 1
                if valid_entries >= 3:
                    break
        except Exception as e:
            report.append(f"Error fetching {source_name}: {str(e)}")

    return "\n".join(report)

@tool("Trading Economics Stream Reader")
def fetch_macro_stream() -> str:
    """Fetches real-time macroeconomic news from Trading Economics Stream, excluding items older than 14 days."""
    url = "https://tradingeconomics.com/ws/stream.ashx?start=0&size=30"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    shipping_keywords = ['freight', 'shipping', 'transportation', 'baltic dry', 'container', 'port', 'maritime', 'cargo']
    metals_keywords = ['lithium', 'copper', 'iron', 'steel', 'metal', 'aluminum', 'gold', 'silver', 'nickel', 'zinc', 'ore', 'mining']

    categorized_output = {"Shipping": [], "Metals": [], "Other Macro": []}
    cutoff_date = datetime.now() - timedelta(days=14)

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        for item in data:
            title = item.get('title', 'No Title')
            date_str = item.get('date', '')

            if "forecast" in title.lower():
                continue

            is_recent = True
            if date_str:
                try:
                    match = re.search(r'\d{4}-\d{2}-\d{2}', date_str)
                    if match:
                        item_date = datetime.strptime(match.group(0), "%Y-%m-%d")
                        if item_date < cutoff_date:
                            is_recent = False
                    elif str(datetime.now().year - 1) in date_str or str(datetime.now().year - 2) in date_str:
                        is_recent = False
                except Exception:
                    pass

            if not is_recent:
                continue

            news_item = (
                f"• Headline: {title}\n"
                f"  Date: {date_str if date_str else 'Recent'}\n"
                f"  Source: Trading Economics Stream\n"
                f"  Source URL: https://tradingeconomics.com/stream\n"
            )

            title_lower = title.lower()
            if any(k in title_lower for k in shipping_keywords):
                categorized_output["Shipping"].append(news_item)
            elif any(k in title_lower for k in metals_keywords):
                categorized_output["Metals"].append(news_item)
            else:
                categorized_output["Other Macro"].append(news_item)

        formatted_output = []
        for category, items in categorized_output.items():
            if items:
                formatted_output.append(f"\n=== {category} ===")
                formatted_output.extend(items)

        return "\n".join(formatted_output) if formatted_output else "No fresh stream items found."
    except Exception as e:
        return f"Error fetching stream: {str(e)}"

@tool("GMK Center Steel News Scraper")
def fetch_gmk_steel_news() -> str:
    """Scrapes breaking global steel, iron ore, scrap, and metallurgical coal news directly from GMK Center."""
    url = "https://gmk.center/en/news/"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        articles = soup.find_all(["article", "div"], class_=lambda x: x and ("post" in x or "news" in x or "item" in x), limit=8)
        if not articles:
            articles = soup.select(".news-list a, .posts-list a")[:8]

        results = []
        for article in articles:
            title = article.get_text(strip=True)
            link = article.get("href") if article.name == "a" else (article.find("a")["href"] if article.find("a") else url)

            if title and len(title) > 25:
                results.append(
                    f"• Headline: {title}\n"
                    f"  Source: GMK Center (Steel & Metallurgy)\n"
                    f"  Source URL: {link}\n"
                )

        return "\n".join(results[:6]) if results else "No specific GMK steel articles found."
    except Exception as e:
        return f"Error scraping GMK Center: {str(e)}"

# =====================================================================
# 4. CREWAI AGENTS & TASKS CONFIGURATION
# =====================================================================

news_researcher = Agent(
    role="Senior Commodity & Macro Market Researcher",
    goal="Gather strictly high-signal, RECENT raw material news, steel industry trends, and physical exchange benchmarks.",
    backstory=(
        "You are a data-driven commodity intelligence analyst. You leverage direct trade RSS feeds, "
        "GMK Center steel news, Trading Economics, and Yahoo Finance historical tables to build complete market datasets."
    ),
    tools=[
        fetch_commodity_rss_news,
        fetch_macro_stream,
        fetch_gmk_steel_news,
        fetch_multi_commodity_prices,
        fetch_global_steel_benchmarks
    ],
    llm=gemini_llm,
    verbose=True,
    memory=False
)

market_analyst = Agent(
    role="Lead Commodity Strategist",
    goal="Synthesize physical news and market metrics into an executive morning briefing providing exactly 3 bullet points per category.",
    backstory=(
        "You spent 15 years as a head trader in metals and energy. You demand price precision "
        "and active market catalysts combining macro data, physical supply fundamentals, and exchange benchmarks."
    ),
    llm=gemini_llm,
    verbose=True
)

research_task = Task(
    description=(
        "1. Execute 'Yahoo Finance Direct Historical Multi-Commodity Scraper' for Copper, Aluminum, Nickel, Zinc, Gold, Silver, and Brent Crude.\n"
        "2. Execute 'Global Steel Benchmarks Direct Scraper' for US HRC, China Steel Futures, and Central European LME HRC prices.\n"
        "3. Execute 'GMK Center Steel News Scraper' for breaking steel and iron ore updates.\n"
        "4. Execute 'Multi-Source Commodity RSS Reader' and 'Trading Economics Stream Reader' for physical news and macro events."
    ),
    expected_output="A consolidated research report containing real-time price benchmarks with WoW/MoM/YoY indicators and fresh physical news.",
    agent=news_researcher
)

analysis_task = Task(
    description=(
        "Review the gathered research data.\n"
        "Create an executive morning briefing organized strictly by category:\n\n"
        "CATEGORIES TO COVER:\n"
        "1. Copper\n"
        "2. Aluminum\n"
        "3. Steel & Iron Ore\n"
        "4. Base Metals (Nickel, Zinc)\n"
        "5. Precious Metals (Gold/Silver)\n"
        "6. Energy & Freight (Brent Crude / Baltic Dry)\n\n"
        "BENCHMARK REQUIREMENTS:\n"
        "- Display yesterday's closing price along with WoW, MoM, and YoY percentage changes at the top of EVERY category.\n"
        "STEEL BENCHMARK REQUIREMENTS:\n"
        "For Category 3 (Steel & Iron Ore), COPY THE SCRAPED NUMBERS DIRECTLY:\n"
        "  * **US HRC Steel (CME):** [Scraped price in USD/Short Ton] | WoW / MoM / YoY\n"
        "  * **Central European HRC (LME Argus):** [MUST USE EXACT FIRST-ROW CONTRACT NAME & EUR CONVERTED PRICE FROM SCRAPER]\n"
        "  * **China Steel Futures (LME FOB China Argus):** [MUST USE EXACT FIRST-ROW CONTRACT NAME & CNY CONVERTED PRICE FROM SCRAPER]\n"
        "CRITICAL BENCHMARK INSTRUCTION:\n"
        "- YOU MUST USE THE EXACT NUMBERS RETURNED BY 'Global Steel Benchmarks Direct Scraper'.\n"
        "- DO NOT USE HARDCODED MEMORY OR FALLBACKS LIKE €630/MT FOR EUROPEAN STEEL.\n"
        "RECENCY & BULLET REQUIREMENTS:\n"
        "- Discard any gathered news item older than 14 days.\n"
        "- Provide EXACTLY 3 distinct bullet points per category formatted as:\n"
        "  - **[Headline/Topic]** (Date | Source Name)\n"
        "    - **What happened:** (Key facts)\n"
        "    - **Market Impact:** [Bullish/Bearish] + directional driver\n"
        "    - **Source:** (Direct URL)"
    ),
    expected_output="An executive briefing grouped by metal category where EVERY category displays historical price benchmark headers followed by 3 cited news bullet points.",
    agent=market_analyst
)

# =====================================================================
# 5. EXECUTION & AUTOMATED EMAIL DISPATCH
# =====================================================================

commodity_crew = Crew(
    agents=[news_researcher, market_analyst],
    tasks=[research_task, analysis_task],
    process=Process.sequential
)

async def run_crew():
    print("--- Starting Agent Crew Workflow ---")
    result = await commodity_crew.kickoff_async()
    briefing_output = str(result)

    print("\n\n====================================")
    print("     FINAL EXECUTIVE BRIEFING")
    print("====================================")
    print(briefing_output)

    send_email_briefing(briefing_output)

if __name__ == "__main__":
    asyncio.run(run_crew())
