"""Fundamental snapshot: turn provider values (see the units contract in
``data/providers/base.py``) into labelled, plain-text rows. Pure functions, no I/O.

Honesty rules: a missing value is shown as "Not available", never as 0; implausible values are
dropped with a note; no comparison with sector averages is made because the data source does
not supply them. Fundamentals describe the past and are reported late, so they say nothing
reliable about the next few months.
"""

from __future__ import annotations

from dataclasses import dataclass

from stock_analyzer.analysis.benchmarks import major_currency
from stock_analyzer.data.types import Fundamentals

NOT_AVAILABLE = "Not available"
MAX_PLAUSIBLE_PE = 1000.0
BIG_GROWTH = 1.0  # +/-100% year on year: usually a tiny base, flag it


@dataclass(frozen=True)
class FundamentalRow:
    label: str
    value: str  # display text; NOT_AVAILABLE when unknown
    key: str  # glossary key for the tooltip
    note: str | None = None
    available: bool = True


@dataclass(frozen=True)
class FundamentalsSnapshot:
    ticker: str
    rows: tuple[FundamentalRow, ...]
    notes: tuple[str, ...]
    sector: str | None
    industry: str | None
    quote_type: str | None
    available: bool  # False when the provider returned nothing usable

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(r.label for r in self.rows if not r.available)


def _num(v: object) -> float | None:
    return float(v) if isinstance(v, int | float) and not isinstance(v, bool) else None


def format_money(x: float, currency: str | None) -> str:
    """Large amounts in words (5.06 trillion), so no one has to count zeros."""
    for scale, word in ((1e12, "trillion"), (1e9, "billion"), (1e6, "million")):
        if abs(x) >= scale:
            return f"{x / scale:,.2f} {word} {currency or ''}".strip()
    return f"{x:,.0f} {currency or ''}".strip()


def _row(label: str, value: str | None, key: str, note: str | None = None) -> FundamentalRow:
    if value is None:
        return FundamentalRow(label, NOT_AVAILABLE, key, note, available=False)
    return FundamentalRow(label, value, key, note)


def build_snapshot(f: Fundamentals) -> FundamentalsSnapshot:
    v = f.values or {}
    sector, industry, qtype = v.get("sector"), v.get("industry"), v.get("quote_type")
    cur = major_currency(v.get("currency"))
    fin_cur = major_currency(v.get("financial_currency")) or cur
    is_fund = isinstance(qtype, str) and qtype.upper() in {"ETF", "MUTUALFUND", "INDEX"}
    bank = isinstance(sector, str) and sector.lower().startswith("financial")
    notes: list[str] = list(f.warnings)

    cap = _num(v.get("market_cap"))
    pe, fpe, pb = _num(v.get("pe_ratio")), _num(v.get("forward_pe")), _num(v.get("pb_ratio"))
    eps = _num(v.get("eps_ttm"))
    rev, earn = _num(v.get("revenue_growth")), _num(v.get("earnings_growth"))
    de, dy = _num(v.get("debt_to_equity")), _num(v.get("dividend_yield"))

    def ratio(x: float | None) -> str | None:
        return f"{x:.1f}x" if x is not None and 0 < x <= MAX_PLAUSIBLE_PE else None

    pe_note = None
    if pe is not None and ratio(pe) is None:
        pe_note = "Not meaningful (zero, negative or extreme)."
    elif pe is None and eps is not None and eps <= 0:
        pe_note = "Earnings per share are not positive, so P/E is not meaningful."

    def growth(x: float | None, what: str) -> tuple[str | None, str | None]:
        if x is None:
            return None, None
        note = None
        if abs(x) > BIG_GROWTH:
            note = f"{what} changed by more than 100%: usually a small starting base."
        return f"{x:+.0%}", note

    rev_text, rev_note = growth(rev, "Revenue")
    earn_text, earn_note = growth(earn, "Earnings")
    eps_note = None
    if eps is not None and fin_cur and cur and fin_cur != cur:
        eps_note = f"Reported in {fin_cur}; the stock trades in {cur}. Not directly comparable."
    de_note = "Not meaningful for banks and insurers." if bank and de is not None else None
    if dy is None:
        dy_text = None
    elif dy == 0:
        dy_text = "None reported"
    else:
        dy_text = f"{dy:.2%}"

    rows = (
        _row("Market cap", format_money(cap, cur) if cap and cap > 0 else None, "market_cap"),
        _row("P/E (last 12 months)", ratio(pe), "pe", pe_note),
        _row("P/E (expected)", ratio(fpe), "pe"),
        _row("P/B", ratio(pb), "pb"),
        _row(
            "Earnings per share",
            f"{eps:,.2f} {fin_cur or ''}".strip() if eps is not None else None,
            "eps",
            eps_note,
        ),
        _row("Revenue growth", rev_text, "growth", rev_note),
        _row("Earnings growth", earn_text, "growth", earn_note),
        _row("Debt / equity", f"{de:.2f}x" if de is not None else None, "debt_to_equity", de_note),
        _row("Dividend yield", dy_text, "dividend_yield"),
    )
    usable = any(r.available for r in rows)
    if is_fund:
        notes.append(
            "This looks like a fund. P/E and P/B are averages across its holdings; growth and "
            "debt figures usually do not apply."
        )
    if usable:
        notes.append(
            "Growth figures compare the latest reported quarter with the same quarter a year "
            "earlier. Company figures are reported with a delay and are revised."
        )
        notes.append(
            "No comparison with sector averages is shown: this data source does not supply "
            "them, and we do not invent them."
        )
    else:
        notes.append("Fundamental data is not available for this ticker right now.")
    return FundamentalsSnapshot(f.ticker, rows, tuple(notes), sector, industry, qtype, usable)
