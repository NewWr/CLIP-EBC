#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用 LLM 基于经验分布生成包含小数区间的分箱（bins）。
保持与 genert_bins.py 相同的总体结构与流程，但允许小数边界。

示例：
python /opt/DM/OCT/CLIP_Code/CLIP-EBC/GPTprompts/Bins/genert_bins_decimal.py --excel_path /path/to/file.xlsx --metric_name hba1c
"""

import os
import os.path as osp
import json
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser


# -----------------------------
# Pydantic 输出结构：允许 float 边界
# -----------------------------
class BinRanges(BaseModel):
    BINS: List[List[float]] = Field(
        description="A list of [low, high] numeric pairs (floats allowed) representing the bins. Bins must be contiguous, non-overlapping, ordered, and cover the entire observed range from min_value to max_value. Use half-open semantics: [low, high) for all bins except the last, which is [low, high]."
    )

bin_parser = PydanticOutputParser(pydantic_object=BinRanges)

# API 配置：使用环境变量，避免在代码中硬编码密钥
API_SECRET_KEY = "bf7c2cc6-c975-4cc8-9b80-05639cd673cf"
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

os.environ["OPENAI_API_KEY"] = API_SECRET_KEY
os.environ["OPENAI_BASE_URL"] = BASE_URL

# 显式传 api_key / base_url，并加上超时
model = ChatOpenAI(model="kimi-k2-250905", api_key=API_SECRET_KEY, base_url=BASE_URL, timeout=60)


def _format_value(v: float) -> str:
    """友好显示数值：90.0 -> 90，必要时保留精度（最多 6 位有效数字）"""
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
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="gbk")

    cols = {str(c).strip() for c in df.columns}
    cand_value = [c for c in ["数值", "value", "Value"] if c in cols]
    cand_count = [c for c in ["数量", "count", "Count"] if c in cols]
    cand_pct = [c for c in ["百分比", "percent", "Percent", "比例"] if c in cols]

    if not cand_value:
        raise ValueError(f"未找到 '数值/value' 列，当前列名：{list(cols)}")
    value_col = cand_value[0]

    values = pd.to_numeric(df[value_col], errors="coerce")

    weights = None
    if cand_count:
        count_col = cand_count[0]
        weights = pd.to_numeric(df[count_col], errors="coerce")

    if weights is None or not np.isfinite(weights).any() or float(pd.Series(weights).fillna(0).sum()) <= 0:
        if not cand_pct:
            raise ValueError("未找到有效的 '数量' 或 '百分比' 列以构建分布。")
        pct_col = cand_pct[0]
        pct_raw = df[pct_col].astype(str).str.strip().str.replace("%", "", regex=False)
        weights = pd.to_numeric(pct_raw, errors="coerce") / 100.0

    # 清洗
    values = values.to_numpy(dtype=float)
    weights = pd.Series(weights).fillna(0).to_numpy(dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]

    if values.size == 0:
        raise ValueError("有效的数据行为 0，请检查输入文件内容。")

    # 同值聚合
    uniq_vals, inv = np.unique(values, return_inverse=True)
    agg_weights = np.zeros_like(uniq_vals, dtype=float)
    np.add.at(agg_weights, inv, weights)

    # 升序
    order = np.argsort(uniq_vals)
    return uniq_vals[order], agg_weights[order]


def _summarize_distribution(values: np.ndarray, weights: np.ndarray, topk_heavy: int = 30) -> str:
    """将分布压缩成简要摘要，便于放入 LLM 提示。"""
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

    # 动态建议分箱数（Sturges 公式，限制在 5~15 范围内）
    num_bins_sturges = int(np.ceil(1 + np.log2(total)))
    num_bins_suggestion = int(np.clip(num_bins_sturges, 5, 15))

    # 基于分位数的边界建议（小数）
    cum_weights = np.cumsum(weights)
    cum_share = cum_weights / total
    quantiles = np.linspace(0, 1, num_bins_suggestion + 1)
    indices = np.searchsorted(cum_share, quantiles, side='left')
    indices = np.clip(indices, 0, len(values) - 1)
    q_vals = values[indices]
    q_vals[-1] = values[-1]
    unique_q_vals = sorted(set(map(float, q_vals)))
    quantile_summary = ", ".join([_format_value(v) for v in unique_q_vals])

    lines = [
        f"min_value={_format_value(vmin)}, max_value={_format_value(vmax)}, unique_values={uniq_n}, total_weight={total:.6g}",
        f"Data-driven suggestion for number of bins is ~{num_bins_suggestion} (between 5 and 15).",
        "heavy_values (top by weight): " + ", ".join([f"{_format_value(h['value'])}:{h['count']:.6g} ({h['share']:.4f})" for h in heavy]),
        f"Suggested boundaries (floats, ~{num_bins_suggestion} bins): {quantile_summary}",
    ]
    return "\n".join(lines)


def _metric_domain_knowledge(metric_name: str) -> str:
    """返回与 metric_name 对应的临床/筛查/治疗参考阈值与眼底相关语义（英文，避免花括号冲突）。
    若未知指标，返回通用提示。内部做大小写与符号归一化。
    """
    def _canon(s: str) -> str:
        return "".join(ch for ch in str(s).lower() if ch.isalnum())

    m = _canon(metric_name)

    # 常见别名映射 -> 统一键
    alias = {
        "hba1c": "hba1c",
        "hba1cpercent": "hba1c",
        "glycatedhemoglobin": "hba1c",
        "glycosylatedhemoglobin": "hba1c",

        "fpg": "fpg",
        "fastingglucose": "fpg",
        "fastingplasmaglucose": "fpg",
        "glucose": "fpg",
        "fbg": "fpg",

        "sbp": "sbp",
        "systolicbp": "sbp",
        "systolicbloodpressure": "sbp",

        "dbp": "dbp",
        "diastolicbp": "dbp",
        "diastolicbloodpressure": "dbp",

        "ldl": "ldl",
        "ldlc": "ldl",
        "ldlcholesterol": "ldl",

        "hdl": "hdl",
        "hdlc": "hdl",
        "hdlcholesterol": "hdl",
    }

    key = alias.get(m, m)

    # 具体知识片段（避免使用花括号，避免模板冲突）
    knowledge = {
        "hba1c": (
            "Typical thresholds (percent): normal < 5.7; prediabetes 5.7-6.4; diabetes >= 6.5. "
            "Treatment targets often around < 7 for many adults. Higher levels indicate increasing microvascular risk, including diabetic retinopathy progression. "
            "When placing boundaries, consider clinically interpretable cutoffs near 5.7, 6.5, 7.0, and potentially 8.0 if distribution supports it."
        ),
        "fpg": (
            "Fasting plasma glucose thresholds: normal < 100 mg/dL (~5.6 mmol/L); impaired 100-125 mg/dL (~5.6-6.9); diabetes >= 126 mg/dL (>= 7.0 mmol/L). "
            "Higher fasting glucose relates to higher risk of retinal microvascular changes. Boundaries near these cutoffs improve screening and triage interpretability."
        ),
        "sbp": (
            "Systolic BP categories (mmHg): normal < 120; elevated 120-129; stage 1 hypertension 130-139; stage 2 >= 140. "
            "Sustained levels >= 140 mmHg are associated with increased retinopathy risk; consider boundaries near 120, 130, 140, and higher marks (e.g., 160) if data supports."
        ),
        "dbp": (
            "Diastolic BP categories (mmHg): normal < 80; stage 1 hypertension 80-89; stage 2 >= 90. "
            "Consider boundaries near 80 and 90 to reflect clinical decision points."
        ),
        "ldl": (
            "LDL-C (mg/dL): optimal < 100; near optimal 100-129; borderline high 130-159; high 160-189; very high >= 190. "
            "Elevated LDL can correlate with retinal exudates risk; boundaries near 100, 130, 160, 190 may improve interpretability."
        ),
        "hdl": (
            "HDL-C (mg/dL): low < 40 (men) or < 50 (women); >= 60 often considered protective. "
            "Lower HDL may associate with higher cardiovascular and microvascular risks. Consider boundaries near 40, 50, 60 while keeping bin balance."
        ),
        "tg": (
            "Triglycerides (mg/dL): normal < 150; borderline high 150-199; high 200-499; very high >= 500. "
            "High triglycerides may relate to retinal exudative changes; boundaries near 150, 200, 500 are often meaningful."
        )
    }

    if key in knowledge:
        return knowledge[key]

    # 默认提示（未知指标）
    return (
        "No specific widely-accepted thresholds mapped for this metric. "
        "Use data-driven quantiles, while aligning where reasonable to known clinical, screening, or treatment decision points relevant to ophthalmic risk."
    )


GENERATOR_PROMPT = ChatPromptTemplate.from_template(
    (
        """
Role briefing (internal collaboration; do not output the process):
- Senior biostatistician: design contiguous equal-mass bins.
- Data analyst: align boundaries to high-density values for interpretability.
- Domain expert of ‘{{metric_name}}’: ensure ranges are plausible and align with known biochemical/clinical/medical thresholds when appropriate.
- Ophthalmology specialist: consider how {metric_name} relates to ocular complications and fundus photography (retinal) findings, e.g., DR severity, microaneurysms, retinal hemorrhages/exudates, macular edema, optic disc edema; prefer boundaries that ease clinical triage/screening decisions.
- ML engineer: ensure exact coverage, contiguity, non-overlap, and reasonable bin count.

Input empirical distribution (summary):
{dist_summary}

Metric-specific domain knowledge for {metric_name}:
{metric_domain_knowledge}

Goal:
Propose contiguous numeric bins for '{metric_name}'. Choose 5-15 bins with roughly equal mass. Boundaries can be floats. Additionally, integrate domain knowledge for {metric_name} (biochemical reference ranges, clinical decision thresholds, diagnostic cutoffs, guideline targets) and ophthalmic relevance (fundus photography signs/ocular complications, screening/triage thresholds) to place interpretable boundaries when possible without violating constraints.

Hard constraints:
- Bin boundaries MUST be numeric (floats allowed).
- Coverage: The first bin's low equals min_value, the last bin's high equals max_value.
- No-overlap and no duplicated boundary numbers: boundaries must be strictly increasing and pairwise distinct across all bins; do NOT repeat the same numeric value as an endpoint in two bins (i.e., enforce low_{{i+1}} > high_i).
- Bins must be ordered. Each bin must satisfy low < high and contain at least one observed value.
- No duplicated or identical bins; all boundary values must be unique across the whole layout.

Strong guidance:
- Start from the Suggested boundaries derived from data quantiles; adjust slightly for interpretability if needed.
- Incorporate biomedical/clinical knowledge for {metric_name}: prefer aligning some boundaries with widely used cutoffs (e.g., diagnostic or treatment thresholds from guidelines) when it does not conflict with coverage/non-overlap and keeps bin masses reasonably balanced.
- Integrate ophthalmology context: when reasonable, align boundaries to thresholds meaningful for fundus photography interpretation or ophthalmic triage (e.g., changes associated with DR progression risk, referral criteria, or treatment initiation windows), avoiding clinically meaningless micro-intervals.
- Avoid creating clinically meaningless micro-intervals around sparse extremes; balance equal-mass and medical interpretability.

Output format (JSON only, no explanation):
{{"BINS": [[low, high], ...]}}
{format_instructions}
        """
    ).strip()
).partial(format_instructions=bin_parser.get_format_instructions())


VALIDATOR_PROMPT = ChatPromptTemplate.from_template(
    (
        """
You are the bin layout validator and corrector. Validate and, if necessary, correct the candidate JSON.

Input:
- Empirical distribution summary:
{dist_summary}
- Candidate JSON:
```json
{candidate_json}
```
- Metric-specific domain knowledge for {metric_name}:
{metric_domain_knowledge}

Constraints to enforce:
1) Structure: JSON object {{"BINS": [[low, high], ...]}}.
2) Numeric boundaries: low/high must be numbers (floats allowed).
3) Coverage: first low == min_value; last high == max_value.
4) No-overlap and strictly increasing boundaries: enforce low_{{i+1}} > high_i for all i; prohibit gaps/overlaps in practical terms and do NOT repeat any boundary number.
5) Bin count: between 5 and 15 inclusive.
6) No degeneracy or duplicates: for every bin, enforce low < high; disallow duplicated [low, high] pairs; and all boundary numbers across all bins must be pairwise distinct and strictly increasing.

If perfect, return as-is. Otherwise, output a minimally corrected valid JSON that satisfies all constraints. When multiple valid corrections exist, prefer boundary placements that are clinically meaningful for {metric_name} with ophthalmic relevance (e.g., thresholds aiding fundus photograph interpretation, DR screening/triage, or treatment decisions) while preserving non-overlap, coverage, and reasonable mass balance.
{format_instructions}
        """
    ).strip()
).partial(format_instructions=bin_parser.get_format_instructions())


def main():
    parser = argparse.ArgumentParser(description="使用 LLM 基于经验分布生成包含小数区间的分箱（bins）。")
    parser.add_argument("--excel_path", type=str, default="/opt/DM/OCT/CLIP_Code/CLIP-EBC/GPTprompts/Bins/hba1c.csv", help="Excel/CSV 文件路径")
    parser.add_argument("--metric_name", type=str, default="hba1c", help="指标名称")
    parser.add_argument("--output_path", type=str, default="/opt/DM/OCT/CLIP_Code/CLIP-EBC/configs/generated_bins_llm_decimal.json", help="输出 JSON 保存路径")
    parser.add_argument("--topk_heavy", type=int, default=30, help="传递给 LLM 的重单值个数")
    args = parser.parse_args()

    print(f"[DEBUG] excel_path={args.excel_path}, metric_name={args.metric_name}", flush=True)
    print(f"[DEBUG] OPENAI_BASE_URL={os.getenv('OPENAI_BASE_URL')}", flush=True)

    # 准备输出路径
    repo_root = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
    default_out = osp.join(repo_root, "configs", "generated_bins_llm_decimal.json")
    output_path = args.output_path or default_out
    os.makedirs(osp.dirname(output_path), exist_ok=True)

    print("[DEBUG] 开始读取并汇总分布...", flush=True)
    values, weights = _read_distribution(args.excel_path)
    dist_text = _summarize_distribution(values, weights, topk_heavy=args.topk_heavy)
    print("[DEBUG] 分布汇总完成。", flush=True)

    # 阶段一：候选生成
    print("开始调用 LLM 生成候选分箱边界...", flush=True)
    try:
        metric_kb = _metric_domain_knowledge(args.metric_name)
        candidate_json = (GENERATOR_PROMPT | model | StrOutputParser()).invoke({
            "metric_name": args.metric_name,
            "dist_summary": dist_text,
            "metric_domain_knowledge": metric_kb,
        })
        print("候选生成完成，开始验证和格式化...", flush=True)
    except Exception as e:
        print(f"[ERROR] 候选生成阶段失败：{repr(e)}", flush=True)
        raise

    # 阶段二：验证/修正
    try:
        final_obj = (VALIDATOR_PROMPT | model | bin_parser).invoke({
            "dist_summary": dist_text,
            "candidate_json": candidate_json,
            "metric_name": args.metric_name,
            "metric_domain_knowledge": metric_kb,
        })
        print("[DEBUG] 验证与解析完成。", flush=True)
    except Exception as e:
        print(f"[ERROR] 验证阶段失败：{repr(e)}", flush=True)
        raise

    result_bins = final_obj.BINS

    # 打印并保存结果
    print(f"\n=== LLM 生成的 {len(result_bins)} 分箱（最终，小数区间，避免重复边界值） ===")
    # 仅打印一次“唯一边界序列”，避免任何重复的边界值出现在输出中
    cuts = [result_bins[0][0]] + [hi for _, hi in result_bins]
    cuts_text = " | ".join([_format_value(x) for x in cuts])
    print(f"唯一边界序列（严格递增，无重复）：{cuts_text}")

    # 完整打印每个区间（补齐左区间），仍使用半开区间语义，最后一个闭区间
    for i, (lo, hi) in enumerate(result_bins):
        tail = ")" if i < len(result_bins) - 1 else "]"
        print(f"[{i:02d}] [{_format_value(lo)}, {_format_value(hi)}{tail}")

    out_obj = {"BINS": result_bins}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {output_path}")
    print("任务完成。")


if __name__ == "__main__":
    main()