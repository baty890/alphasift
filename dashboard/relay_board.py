"""Read-only Streamlit view of the latest relay candidate pool."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


_DISPLAY_COLUMNS = [
    "代码",
    "名称",
    "板块",
    "精选得分",
    "流通市值(亿)",
    "换手率(%)",
    "主力净流入(亿)",
    "最后封板时间",
    "炸板次数",
    "封单金额(亿)",
    "所属题材",
]


def _candidate_table(stocks: list[dict[str, Any]]) -> pd.DataFrame:
    if not stocks:
        return pd.DataFrame(columns=_DISPLAY_COLUMNS)
    frame = pd.DataFrame(stocks)
    display = pd.DataFrame(index=frame.index)
    def series(field: str) -> pd.Series:
        return frame[field] if field in frame.columns else pd.Series(index=frame.index, dtype=object)

    display["代码"] = frame.get("code", "")
    display["名称"] = frame.get("name", "")
    display["板块"] = frame.get("board", "")
    display["精选得分"] = series("selection_score")
    display["流通市值(亿)"] = pd.to_numeric(series("circ_mv"), errors="coerce") / 1e8
    display["换手率(%)"] = series("turnover_rate")
    display["主力净流入(亿)"] = pd.to_numeric(series("main_net_inflow"), errors="coerce") / 1e8
    display["最后封板时间"] = series("last_seal_time")
    display["炸板次数"] = series("broken_board_count")
    display["封单金额(亿)"] = series("seal_amount_yi")
    concepts = series("concepts").fillna("").astype(str)
    industry = series("industry").fillna("").astype(str)
    display["所属题材"] = concepts.where(concepts.str.strip().ne(""), industry)
    return display[_DISPLAY_COLUMNS]


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="AlphaSift 连板备选池", layout="wide")
    st.title("AlphaSift 连板备选池")
    data_dir = Path(os.environ.get("ALPHASIFT_DATA_DIR", "data"))
    pool_path = data_dir / "relay" / "candidate_pool.json"
    if not pool_path.is_file():
        st.info(f"尚无候选池文件：{pool_path}")
        return
    try:
        payload = json.loads(pool_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"读取候选池失败：{exc}")
        return

    st.caption(
        f"交易日：{payload.get('trade_date', '--')} | "
        f"状态：{payload.get('pool_status', '--')} | "
        f"更新时间：{payload.get('created_at', '--')}"
    )
    stocks = payload.get("stocks", [])
    if not isinstance(stocks, list):
        st.error("候选池 stocks 字段格式错误")
        return
    first_board, second_board = st.tabs(["一进二备选池", "二进三备选池"])
    with first_board:
        rows = [stock for stock in stocks if stock.get("consecutive_limit_ups") == 1]
        st.dataframe(_candidate_table(rows), use_container_width=True, hide_index=True)
    with second_board:
        rows = [stock for stock in stocks if stock.get("consecutive_limit_ups") == 2]
        st.dataframe(_candidate_table(rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
