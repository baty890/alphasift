import pandas as pd

from alphasift.filter import apply_candidate_prefilter
from alphasift.models import CandidatePrefilterConfig


def _row(code, *, name="普通股份", market_cap=3_000_000_000, turnover=3, change=1, amount=1):
    return {
        "code": code,
        "name": name,
        "total_mv": market_cap,
        "turnover_rate": turnover,
        "change_pct": change,
        "amount": amount,
    }


def test_candidate_prefilter_includes_exact_bounds_and_excludes_risk_markets():
    rows = [
        _row("000001"),
        _row("000002", market_cap=30_000_000_000, turnover=15, change=6),
        _row("000003", name="ST测试"),
        _row("688001"),
        _row("689001"),
        _row("430001"),
        _row("830001"),
        _row("920001"),
        _row("000004", market_cap=2_999_999_999),
        _row("000005", turnover=2.99),
        _row("000006", turnover=15.01),
        _row("000007", change=0.99),
        _row("000008", change=6.01),
    ]

    screened = apply_candidate_prefilter(pd.DataFrame(rows), CandidatePrefilterConfig())

    assert screened["code"].tolist() == ["000001", "000002"]


def test_candidate_prefilter_caps_twenty_by_amount_then_code():
    rows = [
        _row(f"600{index:03d}", amount=100 - index // 2)
        for index in range(25)
    ]

    screened = apply_candidate_prefilter(pd.DataFrame(rows), CandidatePrefilterConfig())

    assert len(screened) == 20
    expected = sorted(
        (row for row in rows),
        key=lambda row: (-row["amount"], row["code"]),
    )[:20]
    assert screened["code"].tolist() == [row["code"] for row in expected]
