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
    tbl = tbl.dropna(axis=1, how="all") if not tbl.empty else tbl
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
    shown = with_margins(tbl).apply(lambda col: col.map(lambda v: fmt_num(v, 1) if pd.notna(v) else NA))
    st.dataframe(shown, use_container_width=True)
    st.caption(
        "Figures are Yahoo Finance's standardised annual numbers. Revenue is usually net of excise and taxes, EBITDA is calculated by Yahoo, "
        "and net profit is Yahoo's net income. These can differ from a company's headline results, so confirm against its filings (see the Documents tab)."
    )
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


# ----------------------------------------------------------------------------
# Company search (name -> NSE / BSE ticker)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def search_companies(query: str):
    q = query.strip()
    rows = []
    try:
        rows = yf.Search(q, max_results=20, news_count=0).quotes or []
    except Exception:
        rows = []
    if not rows:
        try:
            r = requests.get(
                "https://query2.finance.yahoo.com/v1/finance/search",
                params={"q": q, "quotesCount": 20, "newsCount": 0},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            rows = r.json().get("quotes", [])
        except Exception:
            rows = []
    out, seen = [], set()
    for x in rows:
        sym = x.get("symbol", "") or ""
        if not (sym.endswith(".NS") or sym.endswith(".BO")) or sym in seen:
            continue
        seen.add(sym)
        out.append({
            "symbol": sym,
            "name": x.get("longname") or x.get("shortname") or sym,
            "exch": "NSE" if sym.endswith(".NS") else "BSE",
        })
    return out


def pick_symbol(q: str, pref: str):
    """Return (symbol, matches). symbol is set when the input is already an explicit ticker."""
    s = (q or "").strip()
    if not s:
        return None, []
    up = s.upper().replace(" ", "")
    if up.endswith((".NS", ".BO")) or up.startswith(("NSE:", "BSE:")):
        return resolve_symbol(s, pref), []
    if up.isdigit():  # BSE scrip code
        return up + ".BO", []
    matches = search_companies(s)
    matches = sorted(matches, key=lambda m: m["exch"] != pref)  # stable: preferred exchange first
    return None, matches


# ----------------------------------------------------------------------------
# Brokers / analysts
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_analyst(symbol: str):
    t = yf.Ticker(symbol)
    out = {}
    for key, fn in (
        ("targets", lambda: t.analyst_price_targets),
        ("recs", lambda: t.recommendations_summary),
        ("changes", lambda: t.upgrades_downgrades),
        ("eps", lambda: t.earnings_estimate),
        ("rev", lambda: t.revenue_estimate),
    ):
        try:
            out[key] = fn()
        except Exception:
            out[key] = None
    return out


def _clean_num(x):
    try:
        v = float(x)
        return None if pd.isna(v) else v
    except Exception:
        return None


def tab_brokers(symbol, info, hist):
    last = float(hist["Close"].iloc[-1])
    data = load_analyst(symbol)
    tg = data.get("targets") if isinstance(data.get("targets"), dict) else {}
    low = _clean_num(tg.get("low")) or _clean_num(info.get("targetLowPrice"))
    mean = _clean_num(tg.get("mean")) or _clean_num(info.get("targetMeanPrice"))
    med = _clean_num(tg.get("median")) or _clean_num(info.get("targetMedianPrice"))
    high = _clean_num(tg.get("high")) or _clean_num(info.get("targetHighPrice"))
    n = info.get("numberOfAnalystOpinions")
    key = info.get("recommendationKey")

    st.markdown("What brokerage analysts covered by Yahoo Finance expect for the share price.")
    st.info(
        "Horizon: analyst price targets are normally 12-month targets. Yahoo Finance does not publish a separate 2 to 3 month forecast, "
        "so the last 3 months of rating changes below are the closest view of how sentiment is moving in the short term."
    )

    have_any = any(v is not None for v in (low, mean, med, high))
    if have_any:
        st.subheader("Target price")
        up = (mean / last - 1) * 100 if mean else None
        metric_row([
            ("Lowest target", fmt_num(low, 2), None),
            ("Average target", fmt_num(mean, 2), fmt_num(up, 1, suffix="% vs last close") if up is not None else None),
            ("Median target", fmt_num(med, 2), None),
            ("Highest target", fmt_num(high, 2), None),
        ])
        st.caption(f"Last close {last:,.2f}. Analysts counted: {n if n else NA}. Consensus label from Yahoo: {key if key else NA}.")
    else:
        st.warning(
            "Yahoo Finance has no broker price targets for this stock. Coverage of Indian companies is patchy. "
            "For broker reports, check the brokerages directly or financial news sites."
        )

    recs = data.get("recs")
    if isinstance(recs, pd.DataFrame) and not recs.empty:
        st.subheader("Rating counts, now versus the last 3 months")
        show = recs.copy()
        if "period" in show.columns:
            names = {"0m": "Now", "-1m": "1 month ago", "-2m": "2 months ago", "-3m": "3 months ago"}
            show["period"] = show["period"].map(lambda p: names.get(p, p))
            show = show.set_index("period")
        show = show.rename(columns={"strongBuy": "Strong buy", "buy": "Buy", "hold": "Hold", "sell": "Sell", "strongSell": "Strong sell"})
        st.dataframe(show, use_container_width=True)

    ch = data.get("changes")
    if isinstance(ch, pd.DataFrame) and not ch.empty:
        d = ch.copy()
        try:
            d.index = pd.to_datetime(d.index).tz_localize(None)
        except Exception:
            pass
        try:
            d = d[d.index >= pd.Timestamp.now() - pd.Timedelta(days=90)]
        except Exception:
            pass
        if not d.empty:
            st.subheader("Rating changes in the last 90 days")
            d = d.sort_index(ascending=False)
            d.index = [i.strftime("%d %b %Y") if hasattr(i, "strftime") else i for i in d.index]
            d.index.name = "Date"
            st.dataframe(d, use_container_width=True)

    for key_, title in (("eps", "Earnings per share estimates"), ("rev", "Revenue estimates")):
        df = data.get(key_)
        if isinstance(df, pd.DataFrame) and not df.empty:
            st.subheader(title)
            st.dataframe(df, use_container_width=True)
    st.caption("Source: Yahoo Finance analyst data. Estimates are opinions of brokers, not facts, and can be wrong. Not investment advice.")


# ----------------------------------------------------------------------------
# Documents
# ----------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def load_bse_filings(code: str):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.bseindia.com/",
        "Origin": "https://www.bseindia.com",
        "Accept": "application/json, text/plain, */*",
    }
    to = datetime.now()
    frm = to - timedelta(days=365)
    rows, err = [], None
    for page in (1, 2, 3):
        try:
            r = requests.get(
                "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w",
                params={
                    "pageno": page, "strCat": "-1", "strPrevDate": frm.strftime("%Y%m%d"),
                    "strScrip": code, "strSearch": "P", "strToDate": to.strftime("%Y%m%d"),
                    "strType": "C", "subcategory": "-1",
                },
                headers=headers, timeout=15,
            )
            r.raise_for_status()
            tbl = r.json().get("Table") or []
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            break
        if not tbl:
            break
        rows += tbl
        if len(tbl) < 50:
            break
    return rows, err


def tab_documents(symbol, info):
    base = symbol.split(".")[0]
    name = info.get("longName") or info.get("shortName") or base
    st.markdown("Find the original documents from official sources: quarterly and annual results, annual reports, earnings call transcripts and investor presentations.")

    st.subheader("Official sources")
    links = [("Screener.in documents (annual reports, concall transcripts, presentations)", f"https://www.screener.in/company/{quote(base)}/consolidated/#documents")]
    if symbol.endswith(".NS"):
        links.append(("NSE company page (Corporate Filings section)", f"https://www.nseindia.com/get-quotes/equity?symbol={quote(base)}"))
    links.append(("BSE corporate announcements (search for the company)", "https://www.bseindia.com/corporates/ann.html"))
    site = info.get("website")
    if site:
        links.append(("Company website (look for Investors)", site))
    for label, url in links:
        st.markdown(f"- [{label}]({url})")

    st.subheader("Search for a specific document")
    kinds = [
        ("Annual report", "annual report pdf"),
        ("Quarterly results", "quarterly results press release pdf"),
        ("Earnings call transcript", "earnings call transcript"),
        ("Investor presentation", "investor presentation pdf"),
    ]
    cols = st.columns(len(kinds))
    for col, (label, term) in zip(cols, kinds):
        url = "https://www.google.com/search?q=" + quote(f"{name} {term}")
        col.link_button(label, url, use_container_width=True)
    st.caption("These open a web search in a new tab. Prefer results from the company's own website, NSE or BSE.")

    st.subheader("Latest filings from BSE")
    default_code = base if base.isdigit() else ""
    code = st.text_input(
        "BSE scrip code (6 digits, e.g. 500325 for Reliance)",
        value=default_code,
        key=f"bse_{symbol}",
        help="Find the code on the BSE company page or Screener.in. It is needed to list filings.",
    ).strip()
    if not code:
        st.info("Enter the company's BSE scrip code to list its filings from the last 12 months with direct PDF links.")
        return
    if not code.isdigit():
        st.warning("A BSE scrip code is a number, for example 500325.")
        return
    rows, err = load_bse_filings(code)
    if err and not rows:
        st.warning(
            f"Could not load filings from BSE ({err}). BSE sometimes blocks requests from shared servers. "
            "Use the official source links above instead."
        )
        return
    recs = []
    for r in rows:
        att = (r.get("ATTACHMENTNAME") or "").strip()
        recs.append({
            "Date": str(r.get("NEWS_DT") or r.get("DT_TM") or "")[:10],
            "Category": r.get("CATEGORYNAME") or "",
            "Subject": r.get("NEWSSUB") or r.get("HEADLINE") or "",
            "Document": f"https://www.bseindia.com/xml-data/corpfiling/AttachLive/{att}" if att else None,
        })
    if not recs:
        st.info("BSE returned no filings for this code in the last 12 months. Check that the code is right.")
        return
    df = pd.DataFrame(recs)
    pick = st.radio("Show", ["Results", "Transcripts and earnings calls", "Annual reports", "All filings"], horizontal=True, key=f"kind_{symbol}")
    text = (df["Category"] + " " + df["Subject"]).str.lower()
    if pick == "Results":
        df = df[text.str.contains("result|financial")]
    elif pick == "Transcripts and earnings calls":
        df = df[text.str.contains("transcript|earnings call|concall|conference call|analyst")]
    elif pick == "Annual reports":
        df = df[text.str.contains("annual report")]
    if df.empty:
        st.info("No filings of that type in the last 12 months. Try All filings.")
        return
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={"Document": st.column_config.LinkColumn("Document", display_text="Open PDF")},
    )
    st.caption("Source: BSE India corporate announcements, last 12 months. Links open the original PDF on bseindia.com, where you can download it.")



# ----------------------------------------------------------------------------
# Ask AI (answers grounded in the data already loaded for this company; uses Google Gemini)
# ----------------------------------------------------------------------------
SESSION_QUESTION_LIMIT = 10      # questions per visitor session
DEFAULT_DAILY_LIMIT = 200        # questions per day across all visitors (override with DAILY_LIMIT secret)
DEFAULT_MODEL = "gemini-2.5-flash"  # override with the GEMINI_MODEL secret if this name has been retired
MAX_QUESTION_CHARS = 600

AI_SYSTEM = (
    "You are a research assistant inside a stock research app for Indian listed companies. "
    "You are given a DATA block about one company, taken from Yahoo Finance and Google News. "
    "Rules: (1) For any figure about the company, use only numbers in the DATA block and quote them as given. "
    "(2) If the DATA block does not contain what is asked, say it is not available in the app; never guess or estimate a company figure. "
    "(3) You may explain finance concepts in general terms, and must make clear when you are giving general knowledge rather than company data. "
    "(4) Do not give buy, sell or hold advice or price predictions; present facts and trade-offs, and say the user should check filings. "
    "(5) Keep answers short and plain, under about 250 words. Amounts in rupees are in crore where the DATA says so."
)


def _get_secret(name: str):
    try:
        v = st.secrets[name]
        return str(v) if v is not None and str(v).strip() else None
    except Exception:
        return None


@st.cache_resource
def _usage_counter():
    return {"day": None, "count": 0}


def daily_budget_ok(limit: int) -> bool:
    c = _usage_counter()
    today = datetime.now(timezone.utc).date().isoformat()
    if c["day"] != today:
        c["day"], c["count"] = today, 0
    return c["count"] < limit


def daily_budget_use():
    _usage_counter()["count"] += 1


def build_company_context(symbol, info, hist, statements, max_chars: int = 9000) -> str:
    name = info.get("longName") or info.get("shortName") or symbol
    last = float(hist["Close"].iloc[-1])
    out = [f"Company: {name} ({symbol}). Sector: {info.get('sector') or NA}. Industry: {info.get('industry') or NA}."]
    out.append(f"Last close: {last:,.2f} {info.get('currency') or ''} on {hist.index[-1].strftime('%d %b %Y')}.")
    hi, lo, _, _ = range_52w(hist)
    out.append(f"52-week high: {fmt_num(hi)}. 52-week low: {fmt_num(lo)}.")
    rets = period_returns(hist)
    out.append("Price change: " + ", ".join(f"{k} {fmt_num(v, 1, suffix='%')}" for k, v in rets.items()) + ".")
    if info.get("marketCap"):
        out.append(f"Market cap: {info['marketCap'] / CRORE:,.0f} crore (if INR).")

    in_crore = (info.get("financialCurrency") or info.get("currency")) == "INR"
    try:
        tbl = five_year_table(statements["income"], in_crore)
        if not tbl.empty:
            tbl = tbl.dropna(axis=1, how="all")
            unit = "INR crore" if in_crore else (info.get("financialCurrency") or "reporting currency")
            out.append(f"Annual financials ({unit}), Yahoo Finance standardised figures:\n" + with_margins(tbl).round(1).to_string())
    except Exception:
        pass

    fb = {}
    try:
        fb = balance_ratios(statements)
    except Exception:
        pass
    ratio_bits = [
        f"P/E {fmt_num(info.get('trailingPE'), 1)}",
        f"P/B {fmt_num(info.get('priceToBook'), 2)}",
        f"EV/EBITDA {fmt_num(info.get('enterpriseToEbitda'), 1)}",
        f"ROE {fmt_num(info['returnOnEquity'] * 100 if info.get('returnOnEquity') is not None else fb.get('roe'), 1, suffix='%')}",
        f"Debt/equity {fmt_num(info.get('debtToEquity') if info.get('debtToEquity') is not None else fb.get('de'), 1, suffix='%')}",
        f"Current ratio {fmt_num(info.get('currentRatio') if info.get('currentRatio') is not None else fb.get('current'), 2)}",
    ]
    out.append("Ratios: " + ", ".join(ratio_bits) + ".")
    out.append(
        "Broker targets (usually 12-month): low {}, mean {}, high {}, analysts {}, consensus {}.".format(
            fmt_num(info.get("targetLowPrice")), fmt_num(info.get("targetMeanPrice")), fmt_num(info.get("targetHighPrice")),
            info.get("numberOfAnalystOpinions") or NA, info.get("recommendationKey") or NA,
        )
    )

    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        items = load_yahoo_news(symbol) + load_google_news(f'"{name}"')
        seen, heads = set(), []
        for it in sorted(items, key=lambda x: x["when"], reverse=True):
            k = it["title"].lower()[:90]
            if it["when"] >= cutoff and k not in seen:
                seen.add(k)
                heads.append(f"- {it['when'].strftime('%d %b %Y')} ({it['source']}): {it['title']}")
            if len(heads) >= 15:
                break
        if heads:
            out.append("News headlines, last 30 days:\n" + "\n".join(heads))
    except Exception:
        pass

    text = "\n".join(out)
    return text[:max_chars]


def tab_ask(symbol, info, hist, statements):
    name = info.get("longName") or info.get("shortName") or symbol
    st.markdown(f"Ask a question about **{name}**. The AI answers from the data in this app (prices, financials, ratios, broker targets and recent headlines).")
    try:
        from google import genai
        from google.genai import types
    except Exception:
        st.warning("The Google AI library is not installed on this app yet. Add `google-genai` to requirements.txt and redeploy.")
        return
    key = _get_secret("GEMINI_API_KEY")
    if not key:
        st.warning("This feature is switched off because no API key has been added to the app yet.")
        return
    model = _get_secret("GEMINI_MODEL") or DEFAULT_MODEL
    try:
        daily_limit = int(_get_secret("DAILY_LIMIT") or DEFAULT_DAILY_LIMIT)
    except ValueError:
        daily_limit = DEFAULT_DAILY_LIMIT

    hkey = f"chat_{symbol}"
    history = st.session_state.setdefault(hkey, [])
    used = st.session_state.get("ai_used", 0)

    for m in history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    with st.form(key=f"ask_form_{symbol}", clear_on_submit=True):
        q = st.text_area(
            "Your question",
            max_chars=MAX_QUESTION_CHARS,
            placeholder="e.g. How has revenue and margin trended over the years? What do the recent headlines say?",
            height=90,
        )
        sent = st.form_submit_button("Ask", type="primary")

    left = SESSION_QUESTION_LIMIT - used
    st.caption(f"{max(left, 0)} question(s) left in this visit. Answers use only this app's data and can be wrong; check the company's filings. Not investment advice. Do not type personal or sensitive information.")

    if not sent:
        return
    q = (q or "").strip()
    if not q:
        st.info("Type a question first.")
        return
    if used >= SESSION_QUESTION_LIMIT:
        st.warning("You have used all the questions for this visit. Reload the page later to ask more.")
        return
    if not daily_budget_ok(daily_limit):
        st.warning("The daily question limit for this app has been reached. Please try again tomorrow.")
        return

    context = build_company_context(symbol, info, hist, statements)
    contents = []
    for m in history[-6:]:
        role = "user" if m["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=q)]))
    with st.spinner("Reading the data…"):
        try:
            client = genai.Client(api_key=key)
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=AI_SYSTEM + "\n\nDATA:\n" + context,
                    max_output_tokens=2000,
                    temperature=0.2,
                ),
            )
            answer = (resp.text or "").strip()
        except Exception as e:
            st.error(f"The AI could not answer right now ({type(e).__name__}). This is often the free daily limit, a wrong model name or a bad key. Please try again later.")
            return
    if not answer:
        st.info("The AI returned no answer (it may have declined or hit a limit). Try rephrasing the question.")
        return
    daily_budget_use()
    st.session_state["ai_used"] = used + 1
    history.append({"role": "user", "content": q})
    history.append({"role": "assistant", "content": answer})
    st.rerun()


def main():
    st.set_page_config(page_title="Company Research Desk", page_icon="📈", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    c1, c2 = st.columns([5, 1.3])
    q = c1.text_input(
        "Company",
        value="",
        placeholder="Type a company name (e.g. reliance, tata motors) or a ticker, then press Enter",
        label_visibility="collapsed",
    )
    pref = c2.selectbox("Prefer exchange", ["NSE", "BSE"], label_visibility="collapsed")

    symbol, matches = pick_symbol(q, pref)
    if not q.strip():
        st.markdown("### Company Research Desk")
        st.write(
            "Type a company name or ticker and press Enter. The app finds the NSE or BSE ticker for you. "
            "You get price ranges, five years of revenue and profit, ratios, valuation, broker estimates, shareholding, official documents and the last month of news."
        )
        return
    if symbol is None:
        if matches:
            labels = [f"{m['name']} · {m['symbol']} ({m['exch']})" for m in matches]
            choice = st.selectbox("Matching companies", labels, key=f"match_{q.strip().lower()}_{pref}")
            symbol = matches[labels.index(choice)]["symbol"]
        else:
            symbol = resolve_symbol(q, pref)
            st.warning(f"No company matched “{q.strip()}”. Trying it as the ticker {symbol}.")
    exch = "NSE" if symbol.endswith(".NS") else "BSE"

    with st.spinner(f"Fetching {symbol} from Yahoo Finance…"):
        hist = load_history(symbol)
    if hist.empty:
        st.error(
            f"No price data came back for {symbol}. Check the name or ticker. "
            "If it is correct, Yahoo Finance may be limiting this server; wait a minute and retry."
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
    t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs(["Price", "Financials", "Valuation", "Brokers", "Shareholding", "Documents", "News", "Ask AI"])
    with t1:
        tab_overview(symbol, info, hist)
    with t2:
        tab_financials(symbol, info, statements)
    with t3:
        tab_valuation(symbol, info, hist, statements, exch)
    with t4:
        tab_brokers(symbol, info, hist)
    with t5:
        tab_shareholding(symbol, info, exch)
    with t6:
        tab_documents(symbol, info)
    with t7:
        tab_news(symbol, info)
    with t8:
        tab_ask(symbol, info, hist, statements)


if __name__ == "__main__":
    main()
