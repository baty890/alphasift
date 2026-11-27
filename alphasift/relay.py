"""First-board and second-board relay screening, backtesting, and auction checks."""

from __future__ import annotations

import ast
import json
import math
import random
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from alphasift.config import Config
from alphasift.industry import enrich_industry_concepts
from alphasift.jsonutil import atomic_write_strict_json
from alphasift.models import Pick, ScreenResult
from alphasift.normalize import normalize_code, safe_float
from alphasift.pack import DailyPack, load_pack, pack_dir
from alphasift.timeutil import now_iso

STRATEGY_NAME = "relay_1to2_2to3"
MAX_CANDIDATES = 15
MIN_AMOUNT = 250_000_000.0
LIMIT_UP_PCT = 9.5
FIRST_BOARD_MIN_CIRC_MV = 2_000_000_000.0
FIRST_BOARD_MAX_CIRC_MV = 15_000_000_000.0
FIRST_BOARD_MIN_TURNOVER = 3.5
FIRST_BOARD_MAX_TURNOVER = 16.0
FIRST_BOARD_MIN_NET_INFLOW = 30_000_000.0
_CN_TZ = ZoneInfo("Asia/Shanghai")
_AUCTION_URL = "https://hq.sinajs.cn/list={symbol}"


@dataclass(frozen=True)
class RelayTrade:
    code: str
    name: str
    ladder: int
    signal_date: str
    entry_index: int
    exit_index: int
    entry_price: float
    exit_price: float
    return_pct: float


def candidate_pool_path(config: Config) -> Path:
    return config.data_dir / "relay" / "candidate_pool.json"


def _code_is_excluded(code: str, name: str) -> bool:
    cleaned = normalize_code(code)
    normalized_name = str(name).strip().upper()
    return (
        cleaned.startswith(("688", "4", "8", "92"))
        or bool(re.match(r"^(?:\*?ST|退)", normalized_name))
    )


def _is_main_board(code: str) -> bool:
    return code.startswith(("000", "001", "002", "003", "600", "601", "603", "605"))


def _row_value(row: dict[str, Any], names: tuple[str, ...]) -> tuple[str, object] | None:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() and str(value).strip().lower() != "nan":
            return name, value
    return None


def _parse_seal_minutes(value: object) -> int | None:
    match = re.search(r"(?<!\d)(\d{1,2}):(\d{2})(?::\d{2})?(?!\d)", str(value).strip())
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    if hour > 23 or minute > 59:
        return None
    return hour * 60 + minute


def _first_board_seal_fields(row: dict[str, Any]) -> tuple[bool, int | None]:
    time_field = _row_value(row, ("last_seal_time", "seal_time", "最后封板时间", "封板时间"))
    seal_minutes = _parse_seal_minutes(time_field[1]) if time_field else None
    if seal_minutes is not None and seal_minutes > 14 * 60 + 30:
        return False, seal_minutes

    broken_field = _row_value(row, ("broken_board_count", "炸板次数"))
    if broken_field:
        broken_count = safe_float(broken_field[1])
        if broken_count is not None and broken_count > 2:
            return False, seal_minutes

    amount_field = _row_value(
        row,
        ("seal_amount_yi", "封单金额(亿)", "封单金额（亿）", "seal_amount", "封单金额", "封单额"),
    )
    if amount_field:
        seal_amount = safe_float(amount_field[1])
        if seal_amount is not None:
            amount_is_yi = (
                "亿" in amount_field[0]
                or amount_field[0] == "seal_amount_yi"
                or abs(seal_amount) < 10_000
            )
            seal_amount_yi = seal_amount if amount_is_yi else seal_amount / 100_000_000.0
            if seal_amount_yi < 0.2:
                return False, seal_minutes
    return True, seal_minutes


def _first_board_score(
    *,
    topic_count: int,
    turnover: float,
    net_inflow: float,
    seal_minutes: int | None,
    has_seal_time: bool,
) -> tuple[float, dict[str, float]]:
    topic_score = 35.0 if topic_count >= 5 else 20.0 if topic_count >= 3 else 5.0
    turnover_score = 25.0 if 5.0 <= turnover <= 13.0 else 12.0
    flow_score = 25.0 if net_inflow >= 80_000_000.0 else 12.0
    seal_score = 10.0
    if has_seal_time and seal_minutes is not None:
        seal_score = 15.0 if seal_minutes < 10 * 60 else 10.0 if seal_minutes < 11 * 60 + 30 else 5.0
    scores = {
        "topic_heat": topic_score,
        "turnover_golden_range": turnover_score,
        "fund_strength": flow_score,
        "seal_quality": seal_score,
    }
    return sum(scores.values()), scores


def _pack_code_index(pack: DailyPack) -> dict[str, int]:
    return {str(code): index for index, code in enumerate(pack.codes)}


def _limit_up(close: float, pre_close: float) -> bool:
    return (
        math.isfinite(close)
        and math.isfinite(pre_close)
        and pre_close > 0
        and (close / pre_close - 1.0) * 100.0 >= LIMIT_UP_PCT
    )


def _bar_change(pack: DailyPack, day_index: int, code_index: int) -> float | None:
    close = safe_float(pack.matrices["close"][day_index, code_index])
    pre_close = safe_float(pack.matrices["pre_close"][day_index, code_index])
    if close is None or pre_close is None or pre_close <= 0:
        if day_index <= 0:
            return None
        pre_close = safe_float(pack.matrices["close"][day_index - 1, code_index])
    if close is None or pre_close is None or pre_close <= 0:
        return None
    return (close / pre_close - 1.0) * 100.0


def _streak_through_day(pack: DailyPack, day_index: int, code_index: int) -> int:
    streak = 0
    for index in range(day_index, -1, -1):
        change = _bar_change(pack, index, code_index)
        if change is None or change < LIMIT_UP_PCT:
            break
        streak += 1
    return streak


def _last_pack_index_before(pack: DailyPack, trade_date: str) -> int:
    dates = np.asarray(pack.dates, dtype=str)
    return int(np.searchsorted(dates, trade_date, side="left")) - 1


def _latest_pack_index_at_or_before(pack: DailyPack, trade_date: str) -> int:
    dates = np.asarray(pack.dates, dtype=str)
    return int(np.searchsorted(dates, trade_date, side="right")) - 1


def _circulating_market_cap(pack: DailyPack, code_index: int, bar_index: int) -> float | None:
    meta = pack.meta
    if "circ_mv" not in meta.columns or "code" not in meta.columns:
        return None
    code = str(pack.codes[code_index])
    row = meta.loc[meta["code"].astype(str) == code]
    if row.empty:
        return None
    latest_cap = safe_float(row.iloc[0].get("circ_mv"))
    latest_close = safe_float(pack.matrices["close"][-1, code_index])
    historical_close = safe_float(pack.matrices["close"][bar_index, code_index])
    if not latest_cap or not latest_close or not historical_close or latest_close <= 0:
        return None
    shares = latest_cap / latest_close
    return shares * historical_close


def _turnover_estimate(pack: DailyPack, code_index: int, bar_index: int) -> float | None:
    market_cap = _circulating_market_cap(pack, code_index, bar_index)
    close = safe_float(pack.matrices["close"][bar_index, code_index])
    volume = safe_float(pack.matrices["volume"][bar_index, code_index])
    if not market_cap or not close or close <= 0 or volume is None:
        return None
    estimated_shares = market_cap / close
    return volume / estimated_shares * 100.0 if estimated_shares > 0 else None


def _turnover_health(turnover: float, ladder: int) -> float:
    low, high = (5.0, 18.0) if ladder == 1 else (8.0, 25.0)
    midpoint = (low + high) / 2.0
    return max(0.0, 1.0 - abs(turnover - midpoint) / ((high - low) / 2.0))


def _parse_main_net_inflow(summary: str) -> float | None:
    match = re.search(r"日期及资金流原始字段=([^|]+)", summary)
    if not match:
        return None
    fields = match.group(1).split(",")
    return safe_float(fields[1]) if len(fields) > 1 else None


def _topic_values(value: object) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except ValueError:
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = None
        if isinstance(parsed, (list, tuple)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [
        topic.strip().strip("[]'\" ")
        for topic in re.split(r"[|,，;；/]+", text)
        if topic.strip().strip("[]'\" ")
    ]


def _topic_text(value: object) -> str:
    return "|".join(_topic_values(value))


def _main_flow_score(net_inflow: float | None, circ_mv: float, amount: float) -> float:
    if net_inflow is not None and circ_mv > 0:
        return net_inflow / circ_mv
    return amount / circ_mv if circ_mv > 0 else 0.0


def screen_relay(
    *,
    config: Config,
    snapshot_df: pd.DataFrame | None = None,
    daily_pack: DailyPack | None = None,
    max_output: int | None = None,
    use_llm: bool = False,
) -> ScreenResult:
    """Run the relay screen; only finalists receive Eastmoney enrichment."""
    from alphasift.candidate_context import fetch_stock_fund_flow_summary
    from alphasift.snapshot import fetch_snapshot_with_fallback

    pack = daily_pack or load_pack(pack_dir(config))
    if snapshot_df is None:
        snapshot_df = fetch_snapshot_with_fallback(
            ["sina"],
            required_columns=[
                "code", "name", "price", "change_pct", "amount", "turnover_rate", "circ_mv",
            ],
            market="cn",
            source_health_path=config.source_health_path,
        )
    if snapshot_df.empty:
        raise RuntimeError("relay_1to2_2to3 requires a non-empty live Sina snapshot")
    source = str(snapshot_df.attrs.get("snapshot_source", ""))
    if source.lower() != "sina":
        raise RuntimeError(f"relay_1to2_2to3 requires Sina snapshot; received {source or 'unknown'}")
    required = {"code", "name", "price", "change_pct", "amount", "turnover_rate", "circ_mv"}
    missing = sorted(required - set(snapshot_df.columns))
    if missing:
        raise RuntimeError(f"Sina snapshot is missing required fields: {', '.join(missing)}")

    quote_date = str(
        snapshot_df.attrs.get("quote_date")
        or datetime.now(_CN_TZ).date().isoformat()
    )[:10]
    previous_index = _last_pack_index_before(pack, quote_date)
    ref_index = _latest_pack_index_at_or_before(pack, quote_date)
    if previous_index < 0:
        raise RuntimeError(f"DailyPack has no session before snapshot date {quote_date}")
    if ref_index < 0:
        raise RuntimeError(f"DailyPack has no completed session on or before {quote_date}")
    screen_clock = datetime.now(_CN_TZ)
    is_final = screen_clock.hour >= 15 and pack.trade_date == quote_date

    pack_codes = _pack_code_index(pack)
    meta_by_code = (
        pack.meta.assign(code=pack.meta["code"].astype(str)).set_index("code")
        if "code" in pack.meta.columns
        else pd.DataFrame()
    )
    candidates: list[dict[str, Any]] = []
    for row in snapshot_df.to_dict(orient="records"):
        code = normalize_code(row.get("code"))
        name = str(row.get("name") or "")
        if not code or _code_is_excluded(code, name):
            continue
        code_index = pack_codes.get(code)
        if code_index is None:
            continue
        change_pct = safe_float(row.get("change_pct"))
        if change_pct is None or change_pct < LIMIT_UP_PCT:
            continue
        previous_streak = _streak_through_day(pack, previous_index, code_index)
        ladder = previous_streak + 1
        if ladder not in (1, 2):
            continue
        circ_mv = safe_float(row.get("circ_mv"))
        turnover = safe_float(row.get("turnover_rate"))
        amount = safe_float(row.get("amount"))
        price = safe_float(row.get("price"))
        if ladder == 1 and not _is_main_board(code):
            continue
        market_cap_min, market_cap_max = (
            (FIRST_BOARD_MIN_CIRC_MV, FIRST_BOARD_MAX_CIRC_MV)
            if ladder == 1
            else (2_000_000_000.0, 12_000_000_000.0)
        )
        if (
            circ_mv is None or not market_cap_min <= circ_mv <= market_cap_max
            or turnover is None or amount is None or price is None or amount < MIN_AMOUNT
        ):
            continue
        turnover_min, turnover_max = (
            (FIRST_BOARD_MIN_TURNOVER, FIRST_BOARD_MAX_TURNOVER)
            if ladder == 1
            else (8.0, 25.0)
        )
        if not turnover_min <= turnover <= turnover_max:
            continue
        seal_passes, seal_minutes = _first_board_seal_fields(row) if ladder == 1 else (True, None)
        if not seal_passes:
            continue
        prior_amount = safe_float(pack.matrices["amount"][ref_index, code_index])
        health = _turnover_health(turnover, ladder)
        candidates.append(
            {
                "code": code,
                "name": name,
                "ladder": ladder,
                "price": price,
                "change_pct": change_pct,
                "circ_mv": circ_mv,
                "turnover_rate": turnover,
                "amount": amount,
                "prior_day_amount": prior_amount or 0.0,
                "auction_reference": (prior_amount or 0.0) * 0.1,
                "health": health,
                "pre_score": (ladder, health, amount / circ_mv, code),
                "main_net_inflow": None,
                "industry": str(meta_by_code.loc[code].get("industry", "") if code in meta_by_code.index else ""),
                "concepts": "",
                "fund_flow": "",
                "seal_minutes": seal_minutes,
                "has_seal_time": _row_value(
                    row, ("last_seal_time", "seal_time", "最后封板时间", "封板时间")
                ) is not None,
                "same_topic_limitup_count": 0,
                "sector_assist": False,
            }
        )
    candidates.sort(key=lambda item: (-item["pre_score"][0], -item["pre_score"][1], -item["pre_score"][2], item["code"]))
    flow_errors: list[str] = []
    for item in (candidate for candidate in candidates if candidate["ladder"] == 1):
        try:
            summary = fetch_stock_fund_flow_summary(item["code"])
        except Exception as exc:
            summary = ""
            flow_errors.append(f"{item['code']} 主力资金: {exc}")
        item["fund_flow"] = summary
        item["main_net_inflow"] = _parse_main_net_inflow(summary)
        if not summary or item["main_net_inflow"] is None:
            flow_errors.append(f"{item['code']} 主力资金: 接口未返回可解析的净流入值")
    candidates = [
        item for item in candidates
        if item["ladder"] != 1
        or (
            item["main_net_inflow"] is not None
            and item["main_net_inflow"] >= FIRST_BOARD_MIN_NET_INFLOW
        )
    ]

    second_board_finalists = [
        item for item in candidates if item["ladder"] == 2
    ][: min(MAX_CANDIDATES, max_output or MAX_CANDIDATES)]
    finalists = [
        *[item for item in candidates if item["ladder"] == 1],
        *second_board_finalists,
    ]
    for item in second_board_finalists:
        try:
            summary = fetch_stock_fund_flow_summary(item["code"])
        except Exception as exc:
            summary = ""
            flow_errors.append(f"{item['code']} 主力资金: {exc}")
        item["fund_flow"] = summary
        item["main_net_inflow"] = _parse_main_net_inflow(summary)
        if not summary or item["main_net_inflow"] is None:
            flow_errors.append(f"{item['code']} 主力资金: 接口未返回可解析的净流入值")

    if finalists:
        enriched, notes = enrich_industry_concepts(
            pd.DataFrame(finalists),
            provider="eastmoney",
            candidate_codes=[item["code"] for item in finalists],
            provider_cache_dir=config.industry_provider_cache_dir,
            provider_cache_ttl_hours=config.industry_provider_cache_ttl_hours,
        )
        for item in finalists:
            mapped = enriched.loc[enriched["code"].astype(str) == item["code"]]
            if not mapped.empty:
                item["industry"] = str(mapped.iloc[0].get("industry") or "")
                item["concepts"] = _topic_text(mapped.iloc[0].get("concepts"))
    else:
        notes = []

    for item in finalists:
        topics = set(_topic_values(item["industry"])) | set(_topic_values(item["concepts"]))
        if topics:
            same_topic_count = sum(
                bool(
                    topics
                    & (
                        set(_topic_values(other["industry"]))
                        | set(_topic_values(other["concepts"]))
                    )
                )
                for other in finalists
            )
            item["same_topic_limitup_count"] = same_topic_count
            item["sector_assist"] = same_topic_count > 1

    for item in finalists:
        if item["ladder"] == 1:
            item["selected_score"], item["selection_factors"] = _first_board_score(
                topic_count=item["same_topic_limitup_count"],
                turnover=item["turnover_rate"],
                net_inflow=item["main_net_inflow"],
                seal_minutes=item["seal_minutes"],
                has_seal_time=item["has_seal_time"],
            )

    finalists.sort(
        key=lambda item: (
            -item["ladder"],
            -item["selected_score"] if item["ladder"] == 1 else -item["health"],
            0.0 if item["ladder"] == 1 else -_main_flow_score(
                item["main_net_inflow"], item["circ_mv"], item["amount"]
            ),
            item["code"],
        )
    )
    finalists = finalists[: min(MAX_CANDIDATES, max_output or MAX_CANDIDATES)]
    picks: list[Pick] = []
    for rank, item in enumerate(finalists, 1):
        flow_ratio = _main_flow_score(item["main_net_inflow"], item["circ_mv"], item["amount"])
        score = (
            item["selected_score"]
            if item["ladder"] == 1
            else item["ladder"] * 40.0 + item["health"] * 35.0
            + min(25.0, max(0.0, flow_ratio * 100.0))
        )
        ranking_reason = (
            f"精选得分={score:.0f}/100；题材热度={item['selection_factors']['topic_heat']:.0f}；"
            f"换手区间={item['selection_factors']['turnover_golden_range']:.0f}；"
            f"资金强度={item['selection_factors']['fund_strength']:.0f}；"
            f"封板质量={item['selection_factors']['seal_quality']:.0f}"
            if item["ladder"] == 1
            else f"{item['ladder']}连板优先；换手健康度={item['health']:.2f}；资金强度按流入/流通市值排序"
        )
        picks.append(
            Pick(
                rank=rank,
                code=item["code"],
                name=item["name"],
                final_score=score,
                screen_score=score,
                ranking_reason=ranking_reason,
                price=item["price"],
                change_pct=item["change_pct"],
                amount=item["amount"],
                fund_flow=item["fund_flow"],
                circ_mv=item["circ_mv"],
                turnover_rate=item["turnover_rate"],
                main_net_inflow=item["main_net_inflow"],
                consecutive_limit_ups=item["ladder"],
                same_topic_limitup_count=item["same_topic_limitup_count"],
                sector_assist=item["sector_assist"],
                auction_reference=item["auction_reference"],
                prior_day_amount=item["prior_day_amount"],
                industry=item["industry"],
                concepts=item["concepts"],
                factor_scores=(
                    item["selection_factors"]
                    if item["ladder"] == 1
                    else {"turnover_health": item["health"], "flow_intensity": flow_ratio}
                ),
            )
        )

    degradation = [
        "初筛使用本地 DailyPack + Sina 快照；资金流与题材评分查询东方财富，不调用 LLM。",
        "一进二先按真实主力净流入硬筛；二进三仍按原候选预排序后查询资金流。",
        f"题材热度计数范围仅为成功映射题材的候选集合，不代表全市场涨停家数。",
        *notes,
        *flow_errors,
    ]
    if not is_final:
        degradation.append("盘后候选池尚未确认：仅供盘中观察，竞价命令不会接受此候选池。")
    result = ScreenResult(
        strategy=STRATEGY_NAME,
        market="cn",
        strategy_version="1.0",
        strategy_category="framework",
        snapshot_count=len(snapshot_df),
        after_filter_count=len(candidates),
        picks=picks,
        run_id=f"relay-{datetime.now(_CN_TZ):%Y%m%d%H%M%S}",
        track="rules",
        llm_ranked=False,
        degradation=degradation,
        snapshot_source="sina",
        source_errors=flow_errors,
        risk_enabled=False,
        portfolio_diversity_enabled=False,
    )
    pool = {
        "strategy": STRATEGY_NAME,
        "created_at": now_iso(),
        "trade_date": quote_date,
        "snapshot_source": "sina",
        "is_final": is_final,
        "pool_status": "confirmed_after_close" if is_final else "provisional_intraday",
        "assist_count_scope": "selected_candidates",
        "stocks": [
            {
                "code": pick.code,
                "name": pick.name,
                "consecutive_limit_ups": pick.consecutive_limit_ups,
                "circ_mv": pick.circ_mv,
                "turnover_rate": pick.turnover_rate,
                "amount": pick.amount,
                "prior_day_amount": pick.prior_day_amount,
                "auction_reference": pick.auction_reference,
                "main_net_inflow": pick.main_net_inflow,
                "selection_score": pick.final_score if pick.consecutive_limit_ups == 1 else None,
                "industry": pick.industry,
                "concepts": pick.concepts,
                "same_topic_limitup_count": pick.same_topic_limitup_count,
                "sector_assist": pick.sector_assist,
            }
            for pick in picks
        ],
    }
    atomic_write_strict_json(candidate_pool_path(config), pool)
    return result


def _backtest_candidates(pack: DailyPack, signal_index: int) -> list[tuple[int, int, float]]:
    meta = pack.meta
    if "code" not in meta.columns:
        return []
    meta_rows = {str(row["code"]): row for row in meta.to_dict(orient="records")}
    candidates: list[tuple[int, int, float]] = []
    for code_index, code_value in enumerate(pack.codes):
        code = str(code_value)
        row = meta_rows.get(code)
        if not row or _code_is_excluded(code, str(row.get("name") or "")):
            continue
        streak = _streak_through_day(pack, signal_index, code_index)
        if streak not in (1, 2):
            continue
        circ_mv = safe_float(row.get("circ_mv"))
        latest_close = safe_float(pack.matrices["close"][-1, code_index])
        signal_close = safe_float(pack.matrices["close"][signal_index, code_index])
        volume = safe_float(pack.matrices["volume"][signal_index, code_index])
        amount = safe_float(pack.matrices["amount"][signal_index, code_index])
        if not circ_mv or not latest_close or not signal_close or volume is None or amount is None:
            continue
        shares = circ_mv / latest_close
        cap_estimate = shares * signal_close
        turnover = volume / shares * 100.0 if shares > 0 else None
        if turnover is None or not 2e9 <= cap_estimate <= 12e9 or amount < MIN_AMOUNT:
            continue
        low, high = (5.0, 18.0) if streak == 1 else (8.0, 25.0)
        if not low <= turnover <= high:
            continue
        candidates.append((code_index, streak, _turnover_health(turnover, streak) + amount / cap_estimate))
    candidates.sort(key=lambda item: (-item[1], -item[2], str(pack.codes[item[0]])))
    return candidates[:MAX_CANDIDATES]


def backtest_relay(pack: DailyPack, *, signal_days: int = 60) -> dict[str, Any]:
    """Backtest the last N settled signal days using only the local daily pack."""
    if signal_days < 1:
        raise ValueError("signal_days must be positive")
    dates_count = len(pack.dates)
    if dates_count < 4:
        raise ValueError("DailyPack must contain at least four trading sessions")
    first_signal = max(1, dates_count - signal_days - 2)
    last_signal = dates_count - 3
    trades: list[RelayTrade] = []
    for t in range(first_signal, last_signal + 1):
        signal_date = str(pack.dates[t])
        for code_index, ladder, _score in _backtest_candidates(pack, t):
            prev_close = safe_float(pack.matrices["close"][t, code_index])
            open_price = safe_float(pack.matrices["open"][t + 1, code_index])
            day1_close = safe_float(pack.matrices["close"][t + 1, code_index])
            day2_close = safe_float(pack.matrices["close"][t + 2, code_index])
            if not prev_close or not open_price or not day1_close or not day2_close:
                continue
            gap_pct = (open_price / prev_close - 1.0) * 100.0
            if not 2.0 <= gap_pct <= 6.0:
                continue
            day1_pre_close = safe_float(pack.matrices["pre_close"][t + 1, code_index]) or prev_close
            holds_to_day2 = _limit_up(day1_close, day1_pre_close)
            exit_index = t + 2 if holds_to_day2 else t + 1
            exit_price = day2_close if holds_to_day2 else day1_close
            trade_return = (exit_price / open_price - 1.0) * 100.0
            code = str(pack.codes[code_index])
            meta_row = pack.meta.loc[pack.meta["code"].astype(str) == code]
            name = str(meta_row.iloc[0].get("name") or code) if not meta_row.empty else code
            trades.append(
                RelayTrade(
                    code=code,
                    name=name,
                    ladder=ladder,
                    signal_date=signal_date,
                    entry_index=t + 1,
                    exit_index=exit_index,
                    entry_price=open_price,
                    exit_price=exit_price,
                    return_pct=trade_return,
                )
            )

    code_indexes = _pack_code_index(pack)
    daily_returns: dict[int, list[float]] = {}
    for day_index in sorted({day for trade in trades for day in range(trade.entry_index, trade.exit_index + 1)}):
        for trade in trades:
            if not trade.entry_index <= day_index <= trade.exit_index:
                continue
            code_index = code_indexes[trade.code]
            close = safe_float(pack.matrices["close"][day_index, code_index])
            base = (
                trade.entry_price
                if day_index == trade.entry_index
                else safe_float(pack.matrices["close"][day_index - 1, code_index])
            )
            if close is not None and base and base > 0:
                daily_returns.setdefault(day_index, []).append(close / base - 1.0)
    equity, peak, max_drawdown = 1.0, 1.0, 0.0
    for day_index in sorted(daily_returns):
        returns = daily_returns[day_index]
        equity *= 1.0 + float(np.mean(returns))
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)

    def summary(group: list[RelayTrade]) -> dict[str, float | int | None]:
        if not group:
            return {"trades": 0, "win_rate_pct": None, "average_return_pct": None}
        returns = [trade.return_pct for trade in group]
        return {
            "trades": len(group),
            "win_rate_pct": sum(value > 0 for value in returns) / len(returns) * 100.0,
            "average_return_pct": float(np.mean(returns)),
        }

    overall = summary(trades)
    return {
        "strategy": STRATEGY_NAME,
        "pack_trade_date": pack.trade_date,
        "requested_signal_days": signal_days,
        "actual_signal_days": max(0, last_signal - first_signal + 1),
        "first_signal_date": str(pack.dates[first_signal]),
        "last_signal_date": str(pack.dates[last_signal]),
        "trades": overall["trades"],
        "win_rate_pct": overall["win_rate_pct"],
        "average_return_pct": overall["average_return_pct"],
        "cumulative_return_pct": (equity - 1.0) * 100.0,
        "max_drawdown_pct": max_drawdown * 100.0,
        "tiers": {
            "1to2": summary([trade for trade in trades if trade.ladder == 1]),
            "2to3": summary([trade for trade in trades if trade.ladder == 2]),
        },
        "assumptions": [
            "历史流通市值按最新 circ_mv / 最新收盘价估算流通股数，再乘历史收盘价；换手率据此估算。",
            "历史资金流不可用，候选排序以成交额/估算流通市值作为资金强度代理。",
            "当前名称用于排除 ST，历史 ST 状态不可从 DailyPack 还原。",
            "胜率和单笔收益按交易统计；累计收益/回撤按每日持仓等权、不计滑点费用模拟。",
        ],
        "trade_rows": [trade.__dict__ for trade in trades],
    }


def load_candidate_pool(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Relay candidate pool not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    stocks = payload.get("stocks")
    if (
        payload.get("strategy") != STRATEGY_NAME
        or not isinstance(stocks, list)
        or not payload.get("trade_date")
        or payload.get("is_final") is not True
    ):
        raise ValueError(f"Invalid or unfinalized relay candidate pool: {source}")
    if len(stocks) > MAX_CANDIDATES:
        raise ValueError(f"Relay candidate pool exceeds {MAX_CANDIDATES} stocks")
    return payload


def _parse_sina_auction_quote(text: str) -> dict[str, float]:
    match = re.search(r'="([^"]*)"', text)
    if not match:
        raise ValueError("Sina returned an empty auction quote")
    fields = match.group(1).split(",")
    if len(fields) < 10:
        raise ValueError("Sina auction quote is missing opening/amount fields")
    opening = safe_float(fields[1])
    previous_close = safe_float(fields[2])
    amount = safe_float(fields[9])
    if not opening or not previous_close or amount is None or previous_close <= 0:
        raise ValueError("Sina auction quote has invalid opening, previous close, or amount")
    return {"opening": opening, "previous_close": previous_close, "auction_amount": amount}


def _sina_symbol(code: str) -> str:
    normalized = normalize_code(code)
    if not normalized or _code_is_excluded(normalized, ""):
        raise ValueError(f"Unsupported relay stock code: {code}")
    return ("sh" if normalized.startswith("6") else "sz") + normalized


def fetch_auction_quote(code: str, *, timeout: float = 10.0) -> dict[str, float]:
    symbol = _sina_symbol(code)
    response = requests.get(
        _AUCTION_URL.format(symbol=symbol),
        headers={"Referer": "https://finance.sina.com.cn/"},
        timeout=timeout,
    )
    response.raise_for_status()
    response.encoding = "gbk"
    return _parse_sina_auction_quote(response.text)


def screen_auction(
    pool: dict[str, Any],
    *,
    quote_fetcher=fetch_auction_quote,
    sleep=time.sleep,
    jitter=random.uniform,
    require_auction_window: bool = True,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Filter candidates using 9:25 Sina auction snapshots, serially paced."""
    if pool.get("is_final") is not True:
        raise RuntimeError("Auction screening requires a finalized after-close relay candidate pool")
    clock = now or datetime.now(_CN_TZ)
    if require_auction_window and not (
        clock.hour == 9 and 25 <= clock.minute < 30
    ):
        raise RuntimeError("alphasift auction 仅可在北京时间 9:25–9:29 运行，避免误用盘中行情")
    try:
        pool_date = date.fromisoformat(str(pool.get("trade_date", ""))[:10])
    except ValueError as exc:
        raise ValueError("Relay candidate pool is missing a valid trade_date") from exc
    pool_age = (clock.date() - pool_date).days
    if pool_age < 1 or pool_age > 7:
        raise RuntimeError("Relay candidate pool must come from the prior session and be no more than 7 calendar days old")
    stocks = pool.get("stocks") or []
    quotes: list[dict[str, Any]] = []
    for index, stock in enumerate(stocks):
        if index:
            sleep(0.5 + jitter(0.0, 0.5))
        code = normalize_code(stock.get("code"))
        prior_amount = safe_float(stock.get("prior_day_amount"))
        if not code or not prior_amount or prior_amount <= 0:
            raise ValueError(f"Candidate {stock.get('code', '(missing)')} has no valid prior-day amount")
        quote = quote_fetcher(code)
        gap_pct = (quote["opening"] / quote["previous_close"] - 1.0) * 100.0
        ratio = quote["auction_amount"] / prior_amount * 100.0
        if 2.0 <= gap_pct <= 6.0 and ratio >= 8.0:
            quotes.append(
                {
                    **stock,
                    "opening": quote["opening"],
                    "open_gap_pct": gap_pct,
                    "auction_amount": quote["auction_amount"],
                    "auction_amount_pct": ratio,
                    "intensity": ratio / max(gap_pct, 0.01),
                }
            )
    quotes.sort(key=lambda item: (-item["intensity"], -item["auction_amount_pct"], item["code"]))
    return quotes[:3]
