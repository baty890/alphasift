from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import importlib.util
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from alphasift.config import Config
from alphasift.pack import DailyPack, FIELDS
from alphasift.relay import (
    STRATEGY_NAME,
    _limit_up,
    _streak_through_day,
    backtest_relay,
    screen_auction,
    screen_relay,
)
from alphasift.strategy import list_strategies


def _pack_for_screen(codes: list[str], two_boards: set[str]) -> DailyPack:
    dates = np.asarray(["2026-01-01", "2026-01-02"], dtype=str)
    matrices = {
        field: np.full((2, len(codes)), np.nan, dtype=np.float32)
        for field in FIELDS
    }
    for column, code in enumerate(codes):
        matrices["open"][:, column] = [10.0, 10.0]
        matrices["high"][:, column] = [11.2, 11.2]
        matrices["low"][:, column] = [9.9, 9.9]
        matrices["close"][:, column] = [10.0, 11.0 if code in two_boards else 10.0]
        matrices["pre_close"][:, column] = [10.0, 10.0]
        matrices["volume"][:, column] = [30_000_000.0, 30_000_000.0]
        matrices["amount"][:, column] = [300_000_000.0, 350_000_000.0]
    meta = pd.DataFrame(
        [{"code": code, "name": f"样本{code}", "circ_mv": 5_000_000_000.0} for code in codes]
    )
    return DailyPack(dates, np.asarray(codes, dtype=str), matrices, meta, ("test",))


def test_relay_screens_top_15_and_only_enriches_final_candidates(tmp_path: Path, monkeypatch):
    import alphasift.candidate_context as candidate_context
    import alphasift.relay as relay

    valid_codes = [f"{number:06d}" for number in range(1, 21)]
    boundary_codes = ["000022", "000023"]
    rejected_codes = [f"{number:06d}" for number in range(24, 30)]
    codes = [
        *valid_codes, *boundary_codes, *rejected_codes, "000021", "688001", "830001",
        "300001", "300002",
    ]
    two_boards = {*valid_codes[:8], "000023"}
    pack = _pack_for_screen(codes, two_boards)
    rows = []
    for number, code in enumerate(valid_codes, 1):
        ladder = 2 if code in two_boards else 1
        turnover = (8.0 + number % 17) if ladder == 2 else (5.0 + number % 14)
        rows.append(
            {
                "code": code,
                "name": f"样本{code}",
                "price": 11.0,
                "change_pct": 10.0,
                "amount": 300_000_000.0 + number * 1_000_000.0,
                "turnover_rate": turnover,
                "circ_mv": 5_000_000_000.0,
            }
        )
    rows.extend(
        [
            {"code": "000022", "name": "首板边界", "price": 11, "change_pct": 9.5, "amount": 250_000_000, "turnover_rate": 5.0, "circ_mv": 2_000_000_000},
            {"code": "000023", "name": "二板边界", "price": 11, "change_pct": 9.5, "amount": 250_000_000, "turnover_rate": 25.0, "circ_mv": 12_000_000_000},
            {"code": "000024", "name": "成交额不足", "price": 11, "change_pct": 10, "amount": 249_999_999, "turnover_rate": 10, "circ_mv": 5e9},
            {"code": "000025", "name": "市值过小", "price": 11, "change_pct": 10, "amount": 3e8, "turnover_rate": 10, "circ_mv": 1_999_999_999},
            {"code": "000026", "name": "市值过大", "price": 11, "change_pct": 10, "amount": 3e8, "turnover_rate": 10, "circ_mv": 12_000_000_001},
            {"code": "000027", "name": "换手过低", "price": 11, "change_pct": 10, "amount": 3e8, "turnover_rate": 4.99, "circ_mv": 5e9},
            {"code": "000028", "name": "换手过高", "price": 11, "change_pct": 10, "amount": 3e8, "turnover_rate": 18.01, "circ_mv": 5e9},
            {"code": "000029", "name": "涨幅不足", "price": 11, "change_pct": 9.49, "amount": 3e8, "turnover_rate": 10, "circ_mv": 5e9},
            {"code": "000021", "name": "ST样本", "price": 11, "change_pct": 10, "amount": 3e8, "turnover_rate": 10, "circ_mv": 5e9},
            {"code": "688001", "name": "科创样本", "price": 11, "change_pct": 10, "amount": 3e8, "turnover_rate": 10, "circ_mv": 5e9},
            {"code": "830001", "name": "北交所样本", "price": 11, "change_pct": 10, "amount": 3e8, "turnover_rate": 10, "circ_mv": 5e9},
            {"code": "300001", "name": "创业板涨停", "price": 12, "change_pct": 19.5, "amount": 350_000_000, "turnover_rate": 11.5, "circ_mv": 5e9},
            {"code": "300002", "name": "创业板普通大阳线", "price": 11, "change_pct": 10, "amount": 350_000_000, "turnover_rate": 11.5, "circ_mv": 5e9},
        ]
    )
    snapshot = pd.DataFrame(rows)
    snapshot.attrs.update(snapshot_source="sina", quote_date="2026-01-03", market="cn")
    calls: list[str] = []
    monkeypatch.setattr(candidate_context, "fetch_stock_fund_flow_summary", lambda code: calls.append(code) or "日期及资金流原始字段=2026-01-02,100000000,0.2")

    def enrich(frame, **kwargs):
        assert kwargs["candidate_codes"] == frame["code"].tolist()
        enriched = frame.copy()
        enriched["industry"] = enriched["code"].map(lambda code: f"行业{code}")
        enriched["concepts"] = "['题材甲', '共同主题']"
        return enriched, ["test mapping"]

    monkeypatch.setattr(relay, "enrich_industry_concepts", enrich)
    config = replace(Config.from_env(), data_dir=tmp_path, industry_provider_cache_dir=tmp_path / "cache")
    result = screen_relay(config=config, snapshot_df=snapshot, daily_pack=pack)

    assert result.snapshot_source == "sina"
    assert not result.llm_ranked
    assert result.after_filter_count == 23
    assert len(result.picks) == 15
    assert len(calls) == 15
    assert set(calls) == {pick.code for pick in result.picks}
    assert all(not pick.code.startswith(("688", "4", "8")) for pick in result.picks)
    chinext_pick = next(pick for pick in result.picks if pick.code == "300001")
    assert chinext_pick.board == "创业板"
    assert chinext_pick.consecutive_limit_ups == 1
    assert all(pick.board == "主板" for pick in result.picks if not pick.code.startswith(("300", "301")))
    assert "300002" not in {pick.code for pick in result.picks}
    assert all(not pick.name.startswith("ST") for pick in result.picks)
    assert all(pick.consecutive_limit_ups == 2 for pick in result.picks[:9])
    assert all(pick.consecutive_limit_ups == 1 for pick in result.picks[9:])
    assert all(pick.sector_assist and pick.same_topic_limitup_count == 15 for pick in result.picks)
    assert result.picks[0].concepts == "题材甲|共同主题"
    assert result.picks[0].main_net_inflow == 100_000_000.0
    pool = __import__("json").loads((tmp_path / "relay" / "candidate_pool.json").read_text(encoding="utf-8"))
    assert len(pool["stocks"]) == 15
    assert pool["is_final"] is False
    assert pool["pool_status"] == "provisional_intraday"
    assert pool["stocks"][0]["auction_reference"] == 35_000_000.0


def test_board_specific_limit_up_thresholds():
    assert _limit_up(10.95, 10.0, "600001")
    assert not _limit_up(11.0, 10.0, "300001")
    assert _limit_up(11.95, 10.0, "300001")

    main_board_pack = _pack_for_screen(["600001"], {"600001"})
    chinext_pack = _pack_for_screen(["300001"], {"300001"})
    assert _streak_through_day(main_board_pack, 1, 0) == 1
    assert _streak_through_day(chinext_pack, 1, 0) == 0


def test_relay_explain_uses_simplified_chinese_headers_and_board_label():
    from alphasift.cli import _format_relay_screen_explain
    from alphasift.models import Pick, ScreenResult

    pick = Pick(
        rank=1,
        code="300001",
        name="创业板样本",
        final_score=100,
        screen_score=100,
        board="创业板",
        consecutive_limit_ups=1,
        circ_mv=5_000_000_000,
        turnover_rate=11.5,
        main_net_inflow=120_000_000,
        concepts="算力|人工智能",
        auction_reference=35_000_000,
        same_topic_limitup_count=3,
    )
    result = ScreenResult(
        strategy=STRATEGY_NAME,
        market="cn",
        after_filter_count=1,
        picks=[pick],
    )

    report = _format_relay_screen_explain(result)
    assert "代码\t名称\t板块\t连板数\t流通市值(亿)\t换手率\t主力净流入(亿)\t所属题材\t竞价参考线(亿)\t题材助攻数" in report
    assert "300001\t创业板样本\t创业板\t1\t50.00\t11.50%\t1.200\t算力/人工智能\t0.35\t3" in report


def _pack_for_backtest() -> DailyPack:
    dates = np.asarray(pd.bdate_range("2026-01-01", periods=65).strftime("%Y-%m-%d"), dtype=str)
    codes = np.asarray(["000001", "600001"], dtype=str)
    matrices = {field: np.full((65, 2), np.nan, dtype=np.float32) for field in FIELDS}
    for column in range(2):
        matrices["open"][:, column] = 10.0
        matrices["high"][:, column] = 10.2
        matrices["low"][:, column] = 9.8
        matrices["close"][:, column] = 10.0
        matrices["pre_close"][:, column] = 10.0
        matrices["volume"][:, column] = 50_000_000.0
        matrices["amount"][:, column] = 500_000_000.0
        matrices["close"][60, column] = 11.0
    matrices["open"][61, :] = 11.33
    matrices["close"][61, 0] = 11.5
    matrices["pre_close"][61, 0] = 11.0
    matrices["close"][61, 1] = 12.1
    matrices["pre_close"][61, 1] = 11.0
    matrices["close"][62, 1] = 12.3
    matrices["pre_close"][62, 1] = 12.1
    meta = pd.DataFrame(
        [{"code": code, "name": f"样本{code}", "circ_mv": 5_000_000_000.0} for code in codes]
    )
    return DailyPack(dates, codes, matrices, meta, ("test",))


def test_relay_backtest_simulates_one_day_exit_and_limit_up_hold():
    result = backtest_relay(_pack_for_backtest())
    assert result["actual_signal_days"] == 60
    assert result["trades"] == 2
    by_code = {trade["code"]: trade for trade in result["trade_rows"]}
    assert by_code["000001"]["exit_index"] == 61
    assert by_code["600001"]["exit_index"] == 62
    assert result["tiers"]["1to2"]["trades"] == 2
    assert result["cumulative_return_pct"] > 0


def test_auction_filters_open_gap_and_amount_then_ranks_top_three():
    pool = {
        "strategy": STRATEGY_NAME,
        "trade_date": "2026-01-02",
        "is_final": True,
        "stocks": [
            {"code": f"{code:06d}", "name": f"样本{code}", "prior_day_amount": 1000.0}
            for code in range(1, 6)
        ],
    }
    quotes = {
        "000001": {"opening": 10.3, "previous_close": 10.0, "auction_amount": 100.0},
        "000002": {"opening": 10.5, "previous_close": 10.0, "auction_amount": 80.0},
        "000003": {"opening": 10.8, "previous_close": 10.0, "auction_amount": 200.0},
        "000004": {"opening": 10.1, "previous_close": 10.0, "auction_amount": 200.0},
        "000005": {"opening": 10.25, "previous_close": 10.0, "auction_amount": 100.0},
    }
    calls: list[str] = []
    sleeps: list[float] = []
    picks = screen_auction(
        pool,
        quote_fetcher=lambda code: calls.append(code) or quotes[code],
        sleep=sleeps.append,
        jitter=lambda _low, _high: 0.2,
        now=datetime(2026, 1, 5, 9, 25, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    assert calls == [f"{number:06d}" for number in range(1, 6)]
    assert sleeps == [0.7] * 4
    assert [item["code"] for item in picks] == ["000005", "000001", "000002"]


def test_auction_rejects_regular_session_and_sina_quote_parser():
    import pytest

    from alphasift.relay import _parse_sina_auction_quote

    with pytest.raises(RuntimeError, match="finalized"):
        screen_auction(
            {"strategy": STRATEGY_NAME, "trade_date": "2026-01-02", "is_final": False, "stocks": []},
            now=datetime(2026, 1, 5, 9, 25, tzinfo=ZoneInfo("Asia/Shanghai")),
        )
    with pytest.raises(RuntimeError, match="9:25"):
        screen_auction(
            {"strategy": STRATEGY_NAME, "trade_date": "2026-01-02", "is_final": True, "stocks": []},
            now=datetime(2026, 1, 5, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
        )
    parsed = _parse_sina_auction_quote(
        'var hq_str_sz000001="平安银行,10.30,10.00,10.30,10.30,10.30,0,0,100,80000000";'
    )
    assert parsed == {"opening": 10.3, "previous_close": 10.0, "auction_amount": 80_000_000.0}


def test_relay_strategy_is_discoverable_and_daily_workflow_is_separate():
    repo = Path(__file__).resolve().parents[1]
    names = {strategy.name for strategy in list_strategies()}
    assert STRATEGY_NAME in names
    workflow = (repo / ".github" / "workflows" / "relay-screen.yml").read_text(encoding="utf-8")
    assert 'cron: "45 10 * * 1-5"' in workflow
    assert f"alphasift screen {STRATEGY_NAME} --no-llm" in workflow
    spec = importlib.util.spec_from_file_location("relay_track_record", repo / "scripts" / "track_record.py")
    assert spec is not None and spec.loader is not None
    track_record = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = track_record
    spec.loader.exec_module(track_record)
    assert STRATEGY_NAME not in track_record.strategies_for("cn")
