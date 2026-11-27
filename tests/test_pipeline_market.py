import pytest

from alphasift.config import Config
from alphasift.pipeline import screen


def test_screen_rejects_unknown_market():
    with pytest.raises(ValueError, match="Unsupported market"):
        screen("dual_low", market="xx", config=Config())


@pytest.mark.parametrize("market", ["us", "hk", "jp", "kr", "tw", "in"])
def test_screen_enforces_strategy_market_scope(market):
    # Supported markets pass the market gate, but cn-scoped strategies reject them.
    with pytest.raises(ValueError, match="does not support market"):
        screen("dual_low", market=market, config=Config())


def _hk_snapshot():
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "code": code,
                "name": name,
                "price": 10.0,
                "change_pct": 0.5,
                "amount": 5e8,
                "avg_amount_usd": avg_usd,
                "total_mv": 1e11,
                "circ_mv": 1e11,
                "pe_ratio": 8.0,
                "pb_ratio": 0.8,
                "industry": "Financial Services",
            }
            for code, name, avg_usd in [
                ("0005.HK", "HSBC", 6e7),
                ("0939.HK", "CCB", 4e7),
                ("1398.HK", "ICBC", 5e7),
                ("9999.HK", "Thin", 3e5),
            ]
        ]
    )


def test_non_cn_screen_skips_a_share_industry_provider(monkeypatch):
    calls = []

    def fake_enrich(df, **kwargs):
        calls.append(kwargs)
        return df, []

    monkeypatch.setattr(
        "alphasift.pipeline.fetch_snapshot_with_fallback", lambda *a, **k: _hk_snapshot()
    )
    monkeypatch.setattr("alphasift.pipeline.enrich_industry_concepts", fake_enrich)
    monkeypatch.setattr(
        "alphasift.pipeline.enrich_daily_features", lambda df, **k: (df, [])
    )

    result = screen(
        "global_value",
        market="hk",
        use_llm=False,
        industry_provider="akshare",
        config=Config(),
    )

    assert calls == []
    assert any("Industry provider akshare skipped" in item for item in result.degradation)
    codes = {pick.code for pick in result.picks}
    assert codes and codes <= {"0005.HK", "0939.HK", "1398.HK"}
    # global_value's USD 1M average traded value floor drops the thin listing.
    assert "9999.HK" not in codes
    assert result.after_filter_count == 3


def test_sina_rule_based_enriches_only_filtered_top_twenty_without_llm(monkeypatch):
    import pandas as pd

    from alphasift.pipeline import screen

    enrichment_calls = []
    context_calls = []
    frame = pd.DataFrame(
        [
            {
                "code": f"600{index:03d}",
                "name": f"候选{index}",
                "price": 10 + index,
                "change_pct": 1.0,
                "amount": 100 - index,
                "total_mv": 3_000_000_000,
                "turnover_rate": 3.0,
                "pe_ratio": 12,
                "pb_ratio": 1.5,
            }
            for index in range(23)
        ]
    )
    frame.attrs["snapshot_source"] = "sina"

    def fake_industry(df, **kwargs):
        enrichment_calls.append((df["code"].tolist(), kwargs))
        enriched = df.copy()
        enriched["industry"] = "测试行业"
        enriched["concepts"] = "测试题材"
        return enriched, []

    def fake_context(df, **kwargs):
        context_calls.append((df["code"].tolist(), kwargs))
        return [
            {"code": code, "fund_flow": f"{code} flow"}
            for code in df["code"].tolist()
        ], []

    monkeypatch.setattr(
        "alphasift.pipeline.fetch_snapshot_with_fallback",
        lambda *args, **kwargs: frame.copy(),
    )
    monkeypatch.setattr("alphasift.pipeline.enrich_industry_concepts", fake_industry)
    monkeypatch.setattr("alphasift.pipeline.collect_candidate_context", fake_context)
    monkeypatch.setattr(
        "alphasift.pipeline.rank_candidates_with_metadata",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM must not run")),
    )

    result = screen(
        "sina_rule_based",
        use_llm=True,
        post_analyzers=[],
        config=Config(risk_enabled=False, portfolio_diversity_enabled=False),
    )

    assert result.snapshot_count == 23
    assert result.after_filter_count == 20
    assert result.llm_ranked is False
    assert len(enrichment_calls[0][0]) == 20
    assert enrichment_calls[0][1]["provider"] == "eastmoney"
    assert len(context_calls[0][0]) == 20
    assert context_calls[0][1]["providers"] == ["fund_flow"]
    assert context_calls[0][1]["max_workers"] == 1
    assert all(pick.fund_flow.endswith(" flow") for pick in result.picks)
