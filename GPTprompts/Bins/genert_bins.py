#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用 LLM 基于经验分布生成 15 个分箱边界（bins）。
严格按照 gpt_query.py 的简洁双阶段生成模式。

示例：
python /opt/DM/OCT/CLIP_Code/CLIP-EBC/GPTprompts/Bins/genert_bins.py --excel_path /path/to/file.xlsx --metric_name diabp
"""

import os
import os.path as osp
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser


# -----------------------------
# Pydantic 输出结构：仅包含 bins 边界
# -----------------------------
class BinRanges(BaseModel):
    BINS: List[List[int]] = Field(
        description="A list of [low, high] integer pairs representing the bins. The number of bins should be appropriate for the data, ideally between 5 and 15. Bins must be contiguous, non-overlapping, ordered, and cover the entire integer range from the minimum to the maximum value observed in the data."
    )


bin_parser = PydanticOutputParser(pydantic_object=BinRanges)

# API 配置
API_SECRET_KEY = "bf7c2cc6-c975-4cc8-9b80-05639cd673cf"
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

os.environ["OPENAI_API_KEY"] = API_SECRET_KEY
# 关键：对 openai>=1.x/ langchain_openai，需要使用 OPENAI_BASE_URL，或者直接在 ChatOpenAI 里传 base_url
os.environ["OPENAI_BASE_URL"] = BASE_URL

# 显式传 api_key / base_url，并加上超时，避免网络卡死
model = ChatOpenAI(model="kimi-k2-250905", api_key=API_SECRET_KEY, base_url=BASE_URL, timeout=60)


def _format_value(v: float) -> str:
    """友好显示数值：90.0 -> 90，保留必要精度"""
    try:
        vf = float(v)
        if vf.is_integer():
            return str(int(round(vf)))
        return f"{vf:.6g}"
    except Exception:
        return str(v)


def _read_distribution(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取 Excel/CSV 的分布数据。
    期望至少包含：
      - 数值 / value
      - 数量 / count  （若缺失，则尝试 百分比 / percent / 百分比）
    返回：
      - values: 升序唯一数值 (float)
      - weights: 对应权重 (float)
    """
    if not osp.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = osp.splitext(file_path)[1].lower()
    print("[DEBUG] Reading file...", flush=True)
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="gbk")
    print("[DEBUG] File read. Processing columns...", flush=True)

    cols = {str(c).strip() for c in df.columns}
    cand_value = [c for c in ["数值", "value", "Value"] if c in cols]
    cand_count = [c for c in ["数量", "count", "Count"] if c in cols]
    cand_pct = [c for c in ["百分比", "percent", "Percent", "比例"] if c in cols]

    if not cand_value:
        raise ValueError(f"未找到 '数值/value' 列，当前列名：{list(cols)}")
    value_col = cand_value[0]

    print("[DEBUG] Columns processed. Coercing to numeric...", flush=True)
    values = pd.to_numeric(df[value_col], errors="coerce")

    weights = None
    if cand_count:
        count_col = cand_count[0]
        weights = pd.to_numeric(df[count_col], errors="coerce")

    if weights is None or not np.isfinite(weights).any() or float(weights.fillna(0).sum()) <= 0:
        if not cand_pct:
            raise ValueError("未找到有效的 '数量' 或 '百分比' 列以构建分布。")
        pct_col = cand_pct[0]
        pct_raw = df[pct_col].astype(str).str.strip().str.replace("%", "", regex=False)
        weights = pd.to_numeric(pct_raw, errors="coerce") / 100.0

    # 清洗
    print("[DEBUG] Numeric coercion done. Cleaning data...", flush=True)
    values = values.to_numpy(dtype=float)
    weights = pd.Series(weights).fillna(0).to_numpy(dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]

    if values.size == 0:
        raise ValueError("有效的数据行为 0，请检查输入文件内容。")

    # 同值聚合
    print("[DEBUG] Data cleaned. Aggregating unique values...", flush=True)
    uniq_vals, inv = np.unique(values, return_inverse=True)
    agg_weights = np.zeros_like(uniq_vals, dtype=float)
    np.add.at(agg_weights, inv, weights)

    # 升序
    order = np.argsort(uniq_vals)
    return uniq_vals[order], agg_weights[order]


def _summarize_distribution(values: np.ndarray, weights: np.ndarray, topk_heavy: int = 30) -> str:
    """
    将分布压缩成简要摘要，便于放入 LLM 提示。
    """
    total = float(weights.sum())
    if total <= 1:
        return "Distribution is empty or has insufficient data."

    uniq_n = int(values.size)
    vmin, vmax = float(values[0]), float(values[-1])

    # Top-K heavy values
    order_desc = np.argsort(-weights)
    topk = min(topk_heavy, uniq_n)
    heavy = []
    for i in range(topk):
        idx = order_desc[i]
        share = float(weights[idx] / total) if total > 0 else 0.0
        heavy.append({"value": float(values[idx]), "count": float(weights[idx]), "share": share})

    # --- NEW: Dynamically suggest number of bins based on data size ---
    # Use Sturges' formula (k = 1 + log2(N)) as a starting point.
    # Clamp the suggestion to a reasonable range to guide the LLM.
    num_bins_sturges = int(np.ceil(1 + np.log2(total)))
    num_bins_suggestion = np.clip(num_bins_sturges, 5, 15) # Suggest between 5 and 15 bins

    # Generate integer quantile boundaries based on the dynamic suggestion
    cum_weights = np.cumsum(weights)
    cum_share = cum_weights / total
    
    quantiles = np.linspace(0, 1, num_bins_suggestion + 1)
    
    indices = np.searchsorted(cum_share, quantiles, side='left')
    indices = np.clip(indices, 0, len(values) - 1)
    quantile_values_from_data = values[indices]
    
    quantile_values_from_data[-1] = values[-1]
    unique_quantile_values = sorted(list(set(quantile_values_from_data)))
    
    quantile_summary = ", ".join([_format_value(v) for v in unique_quantile_values])
    # --- END NEW ---

    # 简洁文本
    lines = [
        f"min_value={_format_value(vmin)}, max_value={_format_value(vmax)}, unique_values={uniq_n}, total_weight={total:.6g}",
        f"Data-driven suggestion for number of bins is ~{num_bins_suggestion}. The final number should be between 5 and 15.",
        "heavy_values (top by weight): " + ", ".join(
            [f"{_format_value(h['value'])}:{h['count']:.6g} ({h['share']:.4f})" for h in heavy]
        ),
        f"Suggested integer boundaries (for ~{num_bins_suggestion} bins): {quantile_summary}"
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="使用 LLM 基于经验分布生成分箱边界（bins）。")
    parser.add_argument("--excel_path", type=str, default="/opt/DM/OCT/CLIP_Code/CLIP-EBC/GPTprompts/Bins/hba1c.csv", help="Excel/CSV 文件路径")
    parser.add_argument("--metric_name", type=str, default="hba1c", help="指标名称")
    # 修正：去掉错误的 dest，保持参数名就是 output_path
    parser.add_argument("--output_path", type=str, default="/opt/DM/OCT/CLIP_Code/CLIP-EBC/configs/generated_bins_llm.json", help="输出 JSON 保存路径")
    parser.add_argument("--topk_heavy", type=int, default=30, help="传递给 LLM 的重单值个数")
    args = parser.parse_args()

    print(f"[DEBUG] excel_path={args.excel_path}, metric_name={args.metric_name}", flush=True)
    print(f"[DEBUG] OPENAI_BASE_URL={os.getenv('OPENAI_BASE_URL')}", flush=True)

    # 准备输出路径
    repo_root = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
    default_out = osp.join(repo_root, "configs", "generated_bins_llm.json")
    output_path = args.output_path or default_out
    os.makedirs(osp.dirname(output_path), exist_ok=True)

    print("[DEBUG] 开始读取并汇总分布...", flush=True)
    # 读取分布并摘要
    values, weights = _read_distribution(args.excel_path)
    dist_text = _summarize_distribution(values, weights, topk_heavy=args.topk_heavy)
    print("[DEBUG] 分布汇总完成。", flush=True)

    # 阶段一：候选生成Prompt
    generator_prompt_text = """
Role briefing (internal collaboration; do not output the process):
- Senior biostatistician: design contiguous equal-mass bins with sound statistical rationale.
- Data distribution analyst: align boundaries to high-density singletons for interpretability while preserving near-equal mass.
- Domain expert of {metric_name}: ensure ranges are plausible and continuous over the observed domain.
- ML engineer: ensure exact coverage, no overlaps, and a sensible number of bins for downstream training.

Input empirical distribution (summary):
{dist_summary}

Goal:
Propose a set of contiguous numeric bins for the metric '{metric_name}'. The number of bins should be chosen thoughtfully based on the data's complexity and range, ideally between 5 and 15. The goal is to create bins that are roughly equal-mass.

Hard constraints:
- The number of bins must be between 5 and 15.
- Bin boundaries MUST be integers.
- Bins must be contiguous, non-overlapping, and fully cover the integer range from `min_value` to `max_value`.
- Each bin must be represented as a `[low, high]` pair. The `low` of the first bin must be `min_value`. The `high` of the last bin must be `max_value`. For contiguous bins, the `low` of bin `i` must be `high` of bin `i-1` + 1.
- Each bin must contain at least one observed value from the data.

Strong guidance:
- The `Suggested integer boundaries` are your primary reference. They are derived from data quantiles to create bins with roughly equal numbers of observations (equal-mass). Start with these.
- For better interpretability, you can slightly adjust these boundaries. For example, if a suggested boundary is 98 and there is a `heavy_value` at 100, prefer 100 as the boundary.
- You can merge adjacent suggested intervals if they are too small or if it creates a more logical grouping. The final number of bins must be within the allowed range.
- The primary goal is a statistically sound, equal-mass distribution. The secondary goal is interpretability by aligning with dominant values.

Output format (JSON only, no explanation):
- A single object with key "BINS".
- "BINS" is a list of `[low, high]` integer arrays.
{format_instructions}
""".strip()

    generator_prompt = ChatPromptTemplate.from_template(generator_prompt_text).partial(
        format_instructions=bin_parser.get_format_instructions()
    )

    # 阶段二：验证/修正Prompt
    validator_prompt_text = """
You are the "bin layout validator and corrector". Your task is to validate and, if necessary, correct the candidate JSON to ensure it strictly adheres to all constraints.

Input:
- Empirical distribution summary:
{dist_summary}
- Candidate JSON:
```json
{candidate_json}
```

Strict Constraints to Enforce:
1.  **Structure**: The output must be a JSON object `{{"BINS": [...]}}` where `BINS` is a list of `[low, high]` pairs.
2.  **Integer Boundaries**: All `low` and `high` values must be integers.
3.  **Coverage**: The `low` of the first bin must be the `min_value` from the summary. The `high` of the last bin must be the `max_value`.
4.  **Contiguity**: Bins must be perfectly contiguous without gaps or overlaps. For any two adjacent bins `[..., high_prev]` and `[low_next, ...]`, it must be that `low_next == high_prev + 1`.
5.  **Bin Count**: The total number of bins MUST be strictly between 5 and 15, inclusive.
6.  **Monotonicity**: For every bin `[low, high]`, `low <= high`. For the list of bins, the values must be strictly increasing.

Review the candidate JSON against these constraints. If it is perfect, return it as is. If there are any violations (e.g., float values, gaps, overlaps, wrong start/end), produce a minimally corrected version that satisfies all rules.

Output only the final, valid JSON.
{format_instructions}
""".strip()

    validator_prompt = ChatPromptTemplate.from_template(validator_prompt_text).partial(
        format_instructions=bin_parser.get_format_instructions()
    )

    print("开始调用 LLM 生成候选分箱边界...", flush=True)
    try:
        # 生成候选
        candidate_json = (generator_prompt | model | StrOutputParser()).invoke({
            "metric_name": args.metric_name,
            "dist_summary": dist_text
        })
        print("候选生成完成，开始验证和格式化...", flush=True)
    except Exception as e:
        print(f"[ERROR] 候选生成阶段失败：{repr(e)}", flush=True)
        raise

    try:
        # 验证并解析成结构化结果
        final_obj = (validator_prompt | model | bin_parser).invoke({
            "dist_summary": dist_text,
            "candidate_json": candidate_json
        })
        print("[DEBUG] 验证与解析完成。", flush=True)
    except Exception as e:
        print(f"[ERROR] 验证阶段失败：{repr(e)}", flush=True)
        raise

    result_bins = final_obj.BINS

    # 打印并保存结果
    print(f"\n=== LLM 生成的 {len(result_bins)} 分箱（最终） ===")
    for i, rng in enumerate(result_bins):
        print(f"[{i:02d}] {_format_value(rng[0])} - {_format_value(rng[1])}")

    # 保存 JSON
    out_obj = {"BINS": result_bins}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {output_path}")
    print("任务完成。")


if __name__ == "__main__":
    main()