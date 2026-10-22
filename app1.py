"""Company Research Desk: NSE/BSE company research and share-price justification.

Every number comes from Yahoo Finance (via yfinance) or Google News. Nothing is
estimated or filled in. When a figure is missing it is shown as "n/a".
Valuation assumptions are inputs you set yourself and are labelled as such.
"""
from __future__ import annotations

import email.utils
import statistics
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

try:
    import plotly.graph_objects as go
except Exception:  # plotly is optional; charts fall back to st.line_chart
    go = None

CRORE = 1e7
NA = "n/a"

FIN_KEYWORDS = (
    "result", "earnings", "profit", "revenue", "q1", "q2", "q3", "q4", "dividend",
    "share price", "stock", "shares", "target price", "brokerage", "rating", "ebitda",
    "margin", "order win", "acquisition", "merger", "ipo", "fund raise", "debt",
    "stake", "buyback", "guidance", "sebi", "nse", "bse", "sensex", "nifty",
)


# ----------------------------------------------------------------------------
# Pure helpers (no network)
# ----------------------------------------------------------------------------
def resolve_symbol(raw: str, exchange: str) -> str:
    """Turn user input into a Yahoo symbol. Respects an explicit .NS / .BO suffix."""
    s = raw.strip().upper().replace(" ", "")
    if not s:
        return ""
    if s.endswith(".NS") or s.endswith(".BO"):
        return s
    if s.startswith("NSE:"):
        return s[4:] + ".NS"
    if s.startswith("BSE:"):
        return s[4:] + ".BO"
    return s + (".NS" if exchange == "NSE" else ".BO")


def fmt_num(x, digits: int = 2, prefix: str = "", suffix: str = "") -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return NA
    return f"{prefix}{x:,.{digits}f}{suffix}"


def price_on_or_before(close: pd.Series, when: pd.Timestamp):
    sub = close.loc[:when].dropna()
    if sub.empty:
        return None
    return float(sub.iloc[-1])


def period_returns(hist: pd.DataFrame) -> dict:
    """% change of the last close versus the close on or before N calendar months ago.

    Returns None for a window the downloaded history does not cover.
    """
    out = {"1 month": None, "3 months": None, "6 months": None, "1 year": None}
    if hist is None or hist.empty:
        return out
    close = hist["Close"].dropna()
    if close.empty:
        return out
    last_date = close.index[-1]
    last = float(close.iloc[-1])
    first_date = close.index[0]
    for label, months in (("1 month", 1), ("3 months", 3), ("6 months", 6), ("1 year", 12)):
        target = last_date - pd.DateOffset(months=months)
        if target < first_date - timedelta(days=5):
            continue  # history does not reach back far enough
        base = price_on_or_before(close, target)
        if base:
            out[label] = (last / base - 1) * 100
    return out


def range_52w(hist: pd.DataFrame):
    """52-week high/low from daily High/Low columns over the last 365 days."""
    if hist is None or hist.empty:
        return None, None, None, None
    last_date = hist.index[-1]
    win = hist.loc[hist.index > last_date - timedelta(days=365)]
    if win.empty:
        return None, None, None, None
    hi_row = win["High"].idxmax()
    lo_row = win["Low"].idxmin()
    return float(win["High"].max()), float(win["Low"].min()), hi_row, lo_row


def classify_news(title: str) -> str:
    t = title.lower()
    return "Financial" if any(k in t for k in FIN_KEYWORDS) else "General"


def dcf_value(fcf0, growth, terminal_growth, discount, years, net_debt, shares):
    """Plain two-stage DCF. All inputs in the same currency unit. Returns dict or None."""
    if not fcf0 or fcf0 <= 0 or not shares or shares <= 0:
        return None
    if discount <= terminal_growth:
        return None
    pv = 0.0
    fcf = fcf0
    for t in range(1, years + 1):
        fcf *= 1 + growth
        pv += fcf / (1 + discount) ** t
    tv = fcf * (1 + terminal_growth) / (discount - terminal_growth)
    pv_tv = tv / (1 + discount) ** years
    ev = pv + pv_tv
    equity = ev - (net_debt or 0.0)
    return {
        "ev": ev,
        "equity": equity,
        "per_share": equity / shares,
        "tv_share_of_ev": pv_tv / ev if ev else None,
    }


def median_or_none(vals):
    vals = [v for v in vals if v is not None and not pd.isna(v) and v > 0]
    return statistics.median(vals) if vals else None


# ----------------------------------------------------------------------------
# Data access (cached)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def load_history(symbol: str) -> pd.DataFrame:
    df = yf.Ticker(symbol).history(period="2y", interval="1d", auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_info(symbol: str) -> dict:
    try:
        return yf.Ticker(symbol).info or {}
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def load_statements(symbol: str):
    t = yf.Ticker(symbol)
    errors = {}

    def first_nonempty(name, *fns):
        for fn in fns:
            try:
                df = fn()
                if df is not None and not df.empty:
                    return df
            except Exception as e:  # keep the reason so the page can show it
                errors[name] = f"{type(e).__name__}: {e}"
        return pd.DataFrame()

    return {
        "income": first_nonempty("income", lambda: t.income_stmt, lambda: t.get_income_stmt(freq="yearly"), lambda: t.financials),
        "balance": first_nonempty("balance", lambda: t.balance_sheet, lambda: t.get_balance_sheet(freq="yearly")),
        "cashflow": first_nonempty("cashflow", lambda: t.cashflow, lambda: t.get_cashflow(freq="yearly")),
        "errors": errors,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def load_holders(symbol: str):
    t = yf.Ticker(symbol)
    out = {}
    for key, fn in (
        ("major", lambda: t.major_holders),
        ("institutions", lambda: t.institutional_holders),
    ):
        try:
            df = fn()
            out[key] = df if df is not None else pd.DataFrame()
        except Exception:
            out[key] = pd.DataFrame()
    return out


def _parse_iso(s: str):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def load_yahoo_news(symbol: str):
    items = []
    try:
        raw = yf.Ticker(symbol).news or []
    except Exception:
        raw = []
    for n in raw:
        c = n.get("content") if isinstance(n, dict) else None
        if c:  # current yfinance shape
            title = c.get("title")
            when = _parse_iso(c.get("pubDate") or "")
            link = (c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url")
            src = (c.get("provider") or {}).get("displayName")
        else:  # legacy shape
            title = n.get("title")
            ts = n.get("providerPublishTime")
            when = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
            link = n.get("link")
            src = n.get("publisher")
        if title and when:
            items.append({"title": title, "when": when, "link": link, "source": src or "Yahoo Finance"})
    return items


@st.cache_data(ttl=900, show_spinner=False)
def load_google_news(query: str):
    url = (
        "https://news.google.com/rss/search?q="
        + quote(query + " when:30d")
        + "&hl=en-IN&gl=IN&ceid=IN:en"
    )
    items = []
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception:
        return items
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        link = it.findtext("link")
        pub = it.findtext("pubDate")
        src = it.findtext("source")
        try:
            when = email.utils.parsedate_to_datetime(pub) if pub else None
        except Exception:
            when = None
        if title and when:
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            items.append({"title": title, "when": when, "link": link, "source": src or "Google News"})
    return items


# ----------------------------------------------------------------------------
# Statement helpers
# ----------------------------------------------------------------------------
def pick_row(df: pd.DataFrame, names):
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            return df.loc[n]
    return None


def five_year_table(income: pd.DataFrame, in_crore: bool):
    rows = {
        "Revenue": ["Total Revenue", "Operating Revenue"],
        "EBITDA": ["EBITDA", "Normalized EBITDA"],
        "EBIT": ["EBIT", "Operating Income"],
        "Net profit": ["Net Income", "Net Income Common Stockholders"],
    }
    if income is None or income.empty:
        return pd.DataFrame()
    cols = sorted(income.columns)[-5:]
    data = {}
    for label, names in rows.items():
        s = pick_row(income, names)
        if s is None:
            data[label] = [None] * len(cols)
        else:
            vals = [s.get(c) for c in cols]
            data[label] = [None if pd.isna(v) else (v / CRORE if in_crore else v) for v in vals]
    df = pd.DataFrame(data, index=[f"FY{c.year}" if c.month <= 6 else f"FY{c.year + 1}" for c in cols]).T
    df.columns = [f"{c.strftime('%b %Y')}" for c in cols]
    return df


def with_margins(tbl: pd.DataFrame) -> pd.DataFrame:
    if tbl.empty or "Revenue" not in tbl.index:
        return tbl
    out = tbl.copy()
    rev = tbl.loc["Revenue"]
    for lab, key in (("EBITDA margin %", "EBITDA"), ("EBIT margin %", "EBIT"), ("Net margin %", "Net profit")):
        if key in tbl.index:
            out.loc[lab] = (tbl.loc[key] / rev * 100).where(rev > 0)
    out.loc["Revenue growth %"] = (rev.pct_change() * 100)
    return out


def roce_latest(statements):
    inc, bal = statements["income"], statements["balance"]
    ebit = pick_row(inc, ["EBIT", "Operating Income"])
    ta = pick_row(bal, ["Total Assets"])
    cl = pick_row(bal, ["Current Liabilities", "Total Current Liabilities"])
    if ebit is None or ta is None or cl is None:
        return None
    common = [c for c in inc.columns if c in bal.columns]
    if not common:
        return None
    c = max(common)
    e, a, l = ebit.get(c), ta.get(c), cl.get(c)
    if any(pd.isna(v) for v in (e, a, l)) or (a - l) <= 0:
        return None
    return e / (a - l) * 100


def latest_fcf(cashflow: pd.DataFrame):
    if cashflow is None or cashflow.empty:
        return None, None
    col = max(cashflow.columns)
    fcf = pick_row(cashflow, ["Free Cash Flow"])
    if fcf is not None and not pd.isna(fcf.get(col)):
        return float(fcf.get(col)), col
    ocf = pick_row(cashflow, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
    capex = pick_row(cashflow, ["Capital Expenditure"])
    if ocf is not None and capex is not None and not pd.isna(ocf.get(col)) and not pd.isna(capex.get(col)):
        return float(ocf.get(col) + capex.get(col)), col  # capex is negative in Yahoo data
    return None, None


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
CSS = """
<style>
.block-container {padding-top: 4.5rem; max-width: 1200px;}
.hdr {display:flex; flex-wrap:wrap; align-items:baseline; gap:.8rem 1.2rem; margin-bottom:.2rem;}
.hdr h1 {margin:0; font-size:1.7rem;}
.hdr .px {font-size:1.7rem; font-weight:650; font-variant-numeric: tabular-nums;}
.sub {opacity:.7; font-size:.9rem; margin-bottom:.8rem;}
.rangebar {position:relative; height:10px; border-radius:6px; background:linear-gradient(90deg,#c0392b,#e0b84c,#2e8b57); margin:.4rem 0 .2rem;}
.rangebar .dot {position:absolute; top:-4px; width:6px; height:18px; border-radius:3px; background:#111; border:1px solid #fff;}
.rangelab {display:flex; justify-content:space-between; font-size:.85rem; font-variant-numeric: tabular-nums;}
.note {font-size:.82rem; opacity:.75;}
</style>
"""


def metric_row(items):
    cols = st.columns(len(items))
    for col, (label, val, delta) in zip(cols, items):
        col.metric(label, val, delta)


def price_chart(hist: pd.DataFrame, days: int, hi, lo):
    df = hist.loc[hist.index > hist.index[-1] - timedelta(days=days)]
    if go is None:
        st.line_chart(df["Close"])
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close", line=dict(width=2)))
    if hi:
        fig.add_hline(y=hi, line_dash="dot", annotation_text=f"52w high {hi:,.2f}", annotation_position="top left")
    if lo:
        fig.add_hline(y=lo, line_dash="dot", annotation_text=f"52w low {lo:,.2f}", annotation_position="bottom left")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=360, showlegend=False, yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)


def tab_overview(symbol, info, hist):
    last = float(hist["Close"].iloc[-1])
    prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else None
    hi, lo, hi_d, lo_d = range_52w(hist)
    rets = period_returns(hist)
    as_of = hist.index[-1].strftime("%d %b %Y")
    cur = info.get("currency") or ""

    metric_row([
        ("Last close", fmt_num(last, 2, "₹" if cur == "INR" else ""), fmt_num((last / prev - 1) * 100, 2, suffix="% 1d") if prev else None),
        ("52-week high", fmt_num(hi, 2), None),
        ("52-week low", fmt_num(lo, 2), None),
        ("Market cap", fmt_num(info.get("marketCap") / CRORE, 0, "₹", " cr") if info.get("marketCap") and cur == "INR" else NA, None),
    ])
    st.caption(f"Prices as of the last trading day in the data: {as_of}. Source: Yahoo Finance, unadjusted daily close.")

    if hi and lo and hi > lo:
        pos = (last - lo) / (hi - lo) * 100
        st.markdown(
            f"""<div class="rangebar"><div class="dot" style="left:calc({pos:.1f}% - 3px)"></div></div>
<div class="rangelab"><span>{lo:,.2f} (low{', ' + lo_d.strftime('%d %b %y') if lo_d is not None else ''})</span>
<span>{pos:.0f}% of the 52-week range</span>
<span>{hi:,.2f} (high{', ' + hi_d.strftime('%d %b %y') if hi_d is not None else ''})</span></div>""",
            unsafe_allow_html=True,
        )

    st.subheader("Price change")
    cols = st.columns(4)
    for col, (label, v) in zip(cols, rets.items()):
        col.metric(label, fmt_num(v, 1, suffix="%") if v is not None else NA)

    rng = st.radio("Chart range", ["1M", "3M", "6M", "1Y", "2Y"], horizontal=True, index=3)
    days = {"1M": 31, "3M": 92, "6M": 183, "1Y": 366, "2Y": 732}[rng]
    price_chart(hist, days, hi, lo)


def balance_ratios(statements):
    """ROE, debt/equity (%) and current ratio computed from the latest annual statements."""
    inc, bal = statements["income"], statements["balance"]
    out = {"roe": None, "de": None, "current": None}
    if bal is None or bal.empty:
        return out
    col = max(bal.columns)

    def val(series):
        if series is None:
            return None
        v = series.get(col)
        return None if v is None or pd.isna(v) else float(v)

    eq = val(pick_row(bal, ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]))
    debt = val(pick_row(bal, ["Total Debt"]))
    ca = val(pick_row(bal, ["Current Assets", "Total Current Assets"]))
    cl = val(pick_row(bal, ["Current Liabilities", "Total Current Liabilities"]))
    if eq and eq > 0:
        if debt is not None:
            out["de"] = debt / eq * 100
        ni = pick_row(inc, ["Net Income", "Net Income Common Stockholders"])
        if ni is not None and col in inc.columns and not pd.isna(ni.get(col)):
            out["roe"] = float(ni.get(col)) / eq * 100
    if ca is not None and cl:
        out["current"] = ca / cl
    return out


def render_ratios(info, statements):
    st.subheader("Key ratios")
    roce = roce_latest(statements)
    fb = balance_ratios(statements)

    def pct_info(x):
        return x * 100 if x is not None else None

    roe = pct_info(info.get("returnOnEquity"))
    de = info.get("debtToEquity")
    cur = info.get("currentRatio")
    metric_row([
        ("P/E (trailing)", fmt_num(info.get("trailingPE"), 1), None),
        ("P/B", fmt_num(info.get("priceToBook"), 2), None),
        ("EV/EBITDA", fmt_num(info.get("enterpriseToEbitda"), 1), None),
        ("Dividend yield", fmt_num(info.get("dividendYield"), 2, suffix="%") if info.get("dividendYield") is not None else NA, None),
    ])
    metric_row([
        ("ROE", fmt_num(roe if roe is not None else fb["roe"], 1, suffix="%"), None),
        ("ROCE (EBIT / capital employed)", fmt_num(roce, 1, suffix="%"), None),
        ("Debt / equity", fmt_num(de if de is not None else fb["de"], 1, suffix="%"), None),
        ("Current ratio", fmt_num(cur if cur is not None else fb["current"], 2), None),
    ])
    st.caption("Ratios come from Yahoo's summary. Where Yahoo has none, ROE, debt/equity and current ratio are computed from the latest annual statements. ROCE is always computed here from EBIT, total assets and current liabilities.")
    if not info or all(info.get(k) is None for k in ("trailingPE", "priceToBook", "returnOnEquity")):
        st.info("Yahoo's summary data (P/E, P/B, ROE) came back empty for this ticker. This usually means Yahoo is limiting the shared server. Wait a minute and press Research again.")


def tab_financials(symbol, info, statements):
    cur = info.get("financialCurrency") or info.get("currency")
    in_crore = cur == "INR"
    tbl = five_year_table(statements["income"], in_crore)
    if tbl.empty:
        msg = "Yahoo Finance returned no annual income statement for this ticker."
        errs = statements.get("errors") or {}
        if errs:
            msg += " Reason reported: " + "; ".join(f"{k}: {v}" for k, v in errs.items())
        else:
            msg += " No error was reported, so Yahoo may simply have no data, or is limiting this server. Wait a minute and try again."
        st.warning(msg)
        render_ratios(info, statements)
        return
    unit = "₹ crore" if in_crore else (cur or "reporting currency")
    st.subheader(f"Revenue and profit, last {tbl.shape[1]} fiscal years ({unit})")
    if tbl.shape[1] < 5:
        st.info(
            f"Yahoo Finance supplied only {tbl.shape[1]} annual periods for this company, so only those are shown. "
            "Missing years are not estimated. For the earlier years, check the company's annual reports or Screener.in."
        )
    st.dataframe(with_margins(tbl).style.format(lambda v: fmt_num(v, 1) if pd.notna(v) else NA), use_container_width=True)
    if go is not None:
        fig = go.Figure()
        for lab in ("Revenue", "EBITDA", "EBIT", "Net profit"):
            if lab in tbl.index:
                fig.add_trace(go.Bar(name=lab, x=list(tbl.columns), y=tbl.loc[lab].tolist()))
        fig.update_layout(barmode="group", height=340, margin=dict(l=10, r=10, t=10, b=10), yaxis_title=unit)
        st.plotly_chart(fig, use_container_width=True)

    render_ratios(info, statements)


def tab_valuation(symbol, info, hist, statements, exchange):
    last = float(hist["Close"].iloc[-1])
    cur = info.get("financialCurrency") or info.get("currency")
    st.markdown("Valuation inputs below are **your assumptions**, not data. Change them and see how the implied price moves.")

    results = {}

    # --- Multiples vs peers ---
    st.subheader("1. Multiples against peers")
    peers_raw = st.text_input(
        "Peer tickers (comma separated)",
        value="",
        placeholder="e.g. TCS, INFY, WIPRO. Choose peers yourself; none are assumed.",
        key="peers",
    )
    peers = [resolve_symbol(p, exchange) for p in peers_raw.split(",") if p.strip()]
    if not peers:
        st.info("Add peer tickers to compare P/E, P/B and EV/EBITDA and get a multiples-based price.")
    else:
        rows = []
        for p in peers:
            pi = load_info(p)
            rows.append({
                "Peer": p,
                "P/E": pi.get("trailingPE"),
                "P/B": pi.get("priceToBook"),
                "EV/EBITDA": pi.get("enterpriseToEbitda"),
            })
        pdf = pd.DataFrame(rows).set_index("Peer")
        med = {c: median_or_none(pdf[c].tolist()) for c in pdf.columns}
        own = {"P/E": info.get("trailingPE"), "P/B": info.get("priceToBook"), "EV/EBITDA": info.get("enterpriseToEbitda")}
        show = pdf.copy()
        show.loc["Peer median"] = [med[c] for c in pdf.columns]
        show.loc[f"{symbol} (this company)"] = [own[c] for c in pdf.columns]
        st.dataframe(show.style.format(lambda v: fmt_num(v, 1) if pd.notna(v) else NA), use_container_width=True)
        missing = [p for p, r in pdf.iterrows() if r.isna().all()]
        if missing:
            st.warning("No data returned for: " + ", ".join(missing))

        net_debt = None
        if info.get("totalDebt") is not None and info.get("totalCash") is not None:
            net_debt = info["totalDebt"] - info["totalCash"]
        shares = info.get("sharesOutstanding")
        if med["P/E"] and info.get("trailingEps"):
            results["Peer median P/E"] = med["P/E"] * info["trailingEps"]
        if med["P/B"] and info.get("bookValue"):
            results["Peer median P/B"] = med["P/B"] * info["bookValue"]
        if med["EV/EBITDA"] and info.get("ebitda") and shares and net_debt is not None:
            results["Peer median EV/EBITDA"] = (med["EV/EBITDA"] * info["ebitda"] - net_debt) / shares
        st.caption("Implied price = peer median multiple × this company's own EPS, book value per share, or EBITDA (less net debt). Only as good as the peer set you choose.")

    # --- DCF ---
    st.subheader("2. Simple DCF")
    fcf0, fcf_col = latest_fcf(statements["cashflow"])
    shares = info.get("sharesOutstanding")
    if fcf0 is None or shares is None:
        st.info("Free cash flow or share count is not available from Yahoo for this company, so the DCF is skipped.")
    elif fcf0 <= 0:
        st.info(f"Latest free cash flow is negative or zero ({fcf0 / CRORE:,.0f} cr), so a DCF would not be meaningful.")
    else:
        net_debt = None
        if info.get("totalDebt") is not None and info.get("totalCash") is not None:
            net_debt = info["totalDebt"] - info["totalCash"]
        st.caption(
            f"Starting free cash flow: {fcf0 / CRORE:,.0f} cr ({fcf_col.strftime('%b %Y')}). "
            f"Net debt: {fmt_num(net_debt / CRORE, 0, suffix=' cr') if net_debt is not None else NA}. Shares: {shares / 1e7:,.2f} cr."
        )
        c1, c2, c3, c4 = st.columns(4)
        g = c1.slider("FCF growth, years 1-5 (%)", 0.0, 30.0, 10.0, 0.5, key="g") / 100
        tg = c2.slider("Terminal growth (%)", 0.0, 7.0, 4.0, 0.25, key="tg") / 100
        r = c3.slider("Discount rate (%)", 6.0, 18.0, 12.0, 0.25, key="r") / 100
        yrs = c4.slider("Explicit years", 3, 10, 5, key="yrs")
        res = dcf_value(fcf0, g, tg, r, yrs, net_debt or 0.0, shares)
        if res is None:
            st.warning("Discount rate must be above terminal growth.")
        else:
            results["DCF (your assumptions)"] = res["per_share"]
            st.metric("DCF value per share", fmt_num(res["per_share"], 2, "₹" if cur == "INR" else ""))
            if res["tv_share_of_ev"] is not None:
                st.caption(f"{res['tv_share_of_ev'] * 100:.0f}% of this value comes from the terminal value, so it is sensitive to the terminal growth and discount rate.")
            grid_g = [max(0.0, g - 0.04), max(0.0, g - 0.02), g, g + 0.02, g + 0.04]
            grid_r = [r - 0.02, r - 0.01, r, r + 0.01, r + 0.02]
            sens = pd.DataFrame(
                {f"r={rr * 100:.1f}%": [
                    (dcf_value(fcf0, gg, tg, rr, yrs, net_debt or 0.0, shares) or {}).get("per_share") for gg in grid_g
                ] for rr in grid_r},
                index=[f"g={gg * 100:.1f}%" for gg in grid_g],
            )
            st.markdown("**Sensitivity** (value per share):")
            st.dataframe(sens.style.format(lambda v: fmt_num(v, 0) if pd.notna(v) else NA), use_container_width=True)
        st.caption("DCF is not suitable for banks and NBFCs, where debt is part of operations. Treat the output as a range of what the assumptions imply, not a forecast.")

    # --- Summary ---
    st.subheader("Does the current price hold up?")
    if not results:
        st.info("Add peers or enable the DCF to see implied prices next to the market price.")
        return
    summ = pd.DataFrame({"Method": list(results), "Implied price": list(results.values())})
    summ["Vs. last close"] = (summ["Implied price"] / last - 1) * 100
    st.dataframe(
        summ.style.format({"Implied price": lambda v: fmt_num(v, 2), "Vs. last close": lambda v: fmt_num(v, 1, suffix="%")}),
        hide_index=True,
        use_container_width=True,
    )
    lo_v, hi_v = min(results.values()), max(results.values())
    st.markdown(f"Last close **{last:,.2f}** against implied range **{lo_v:,.2f} to {hi_v:,.2f}** across {len(results)} method(s).")
    st.caption("This is a comparison tool, not investment advice. Each method rests on inputs you should check against the company's filings.")


def tab_shareholding(symbol, info, exchange):
    st.markdown(
        "Yahoo Finance does not split holdings into promoter, FII and DII for Indian companies. "
        "The official shareholding pattern is on the exchange site, linked below."
    )
    h = load_holders(symbol)
    if not h["major"].empty:
        st.subheader("Summary from Yahoo Finance")
        st.dataframe(h["major"], use_container_width=True)
    else:
        st.info("Yahoo Finance returned no holder summary for this ticker.")
    if not h["institutions"].empty:
        st.subheader("Top institutional holders (Yahoo Finance)")
        st.dataframe(h["institutions"], use_container_width=True, hide_index=True)
    base = symbol.split(".")[0]
    if exchange == "NSE":
        st.markdown(f"[NSE shareholding pattern for {base}](https://www.nseindia.com/get-quotes/equity?symbol={quote(base)})  (open the *Corporate Filings* section)")
    else:
        st.markdown(f"[BSE shareholding pattern for scrip code {base}](https://www.bseindia.com/stock-share-price/x/y/{quote(base)}/)")


def tab_news(symbol, info):
    name = info.get("longName") or info.get("shortName") or symbol.split(".")[0]
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    items = load_yahoo_news(symbol) + load_google_news(f'"{name}"') + load_google_news(
        f'"{name}" (results OR earnings OR profit OR revenue OR dividend OR "share price")'
    )
    seen, clean = set(), []
    for it in sorted(items, key=lambda x: x["when"], reverse=True):
        key = it["title"].lower()[:90]
        if it["when"] >= cutoff and key not in seen:
            seen.add(key)
            it["kind"] = classify_news(it["title"])
            clean.append(it)
    if not clean:
        st.info("No news found for the last 30 days. Searched Yahoo Finance and Google News for: " + name)
        return
    kinds = st.radio("Show", ["All", "Financial", "General"], horizontal=True)
    shown = [c for c in clean if kinds == "All" or c["kind"] == kinds]
    st.caption(
        f"{len(shown)} headline(s) from the last 30 days about “{name}”. "
        "Financial vs general is a keyword match on the headline, so a few may be mislabelled. Sources: Yahoo Finance, Google News."
    )
    for it in shown:
        local = it["when"].astimezone(timezone(timedelta(hours=5, minutes=30)))
        st.markdown(
            f"**[{it['title']}]({it['link']})**  \n"
            f"<span class='note'>{it['source']} · {local.strftime('%d %b %Y, %H:%M')} IST · {it['kind']}</span>",
            unsafe_allow_html=True,
        )


def main():
    st.set_page_config(page_title="Company Research Desk", page_icon="📈", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([4, 1.2, 1])
    raw = c1.text_input("Ticker", value="", placeholder="NSE symbol or BSE scrip code, e.g. RELIANCE or 500325", label_visibility="collapsed")
    exchange = c2.selectbox("Exchange", ["NSE", "BSE"], label_visibility="collapsed")
    go_btn = c3.button("Research", type="primary", use_container_width=True)

    if "symbol" not in st.session_state:
        st.session_state.symbol = ""
    if go_btn or (raw and raw != st.session_state.get("last_raw")):
        st.session_state.symbol = resolve_symbol(raw, exchange)
        st.session_state.last_raw = raw

    symbol = st.session_state.symbol
    if not symbol:
        st.markdown("### Company Research Desk")
        st.write("Type an NSE symbol (RELIANCE) or a BSE scrip code (500325), then press Research. You get price ranges, five years of revenue and profit, ratios, valuation against peers and a DCF, shareholding, and the last month of news.")
        return

    with st.spinner(f"Fetching {symbol} from Yahoo Finance…"):
        hist = load_history(symbol)
    if hist.empty:
        st.error(
            f"No price data came back for {symbol}. Check the symbol and exchange. "
            "If it is correct, Yahoo Finance may be rate-limiting or blocking this connection; wait a minute and retry."
        )
        return
    info = load_info(symbol)
    name = info.get("longName") or info.get("shortName") or symbol
    last = float(hist["Close"].iloc[-1])
    st.markdown(
        f"<div class='hdr'><h1>{name}</h1><span class='px'>{last:,.2f} {info.get('currency') or ''}</span></div>"
        f"<div class='sub'>{symbol} · {info.get('sector') or NA} · {info.get('industry') or NA}</div>",
        unsafe_allow_html=True,
    )

    statements = load_statements(symbol)
    t1, t2, t3, t4, t5 = st.tabs(["Price", "Financials", "Valuation", "Shareholding", "News"])
    with t1:
        tab_overview(symbol, info, hist)
    with t2:
        tab_financials(symbol, info, statements)
    with t3:
        tab_valuation(symbol, info, hist, statements, exchange)
    with t4:
        tab_shareholding(symbol, info, exchange)
    with t5:
        tab_news(symbol, info)


if __name__ == "__main__":
    main()
