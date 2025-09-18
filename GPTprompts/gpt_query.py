import json
import os
import os.path as osp
from sys import api_version
from typing import Dict, List
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
import argparse

# -----------------------------
# 新的输出结构：为每个 hba1c 分箱生成多条短语
# -----------------------------
# 为hba1c分箱生成短语的输出结构
class BinPhraseParser(BaseModel):
    BIN_PHRASES: Dict[str, List[str]] = Field(
        description="Mapping from 'index: range' (e.g., '0: 45-59') to 3–5 short noun phrases (<= 6 words), strictly describing visible signs in color fundus photographs only, with clear inter-bin differentiation and global uniqueness across all bins."
    )

bin_parser = PydanticOutputParser(pydantic_object=BinPhraseParser)

# API_SECRET_KEY = "bb0dad61-34ce-4961-b027-d1196ef826db"
# BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

# API_SECRET_KEY = "AIzaSyDgVb07uNhcjpf8alYV5AyF3V6YtRS4zOU"
# BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

API_SECRET_KEY = "bf7c2cc6-c975-4cc8-9b80-05639cd673cf"
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


os.environ["OPENAI_API_KEY"] = API_SECRET_KEY
os.environ["OPENAI_API_BASE"] = BASE_URL

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# model = ChatOpenAI(model="deepseek-r1-250528")
model = ChatOpenAI(model="kimi-k2-250905")

# 读取15分箱的hba1c区间配置
config_path = osp.join(osp.dirname(osp.dirname(__file__)), "configs", "reduction_8.json")
with open(config_path, "r", encoding="utf-8") as f:
    reduction_cfg = json.load(f)

# === 新增：从配置中可选 metric，并通过命令行超参数选择 ===
available_metrics = sorted(list(reduction_cfg["15"].keys()))
parser = argparse.ArgumentParser(description="Generate fundus phrases for metric bins")
parser.add_argument(
    "--metric_name", "--metric", dest="metric_name",
    choices=available_metrics, default="hba1c",
    help=f"Metric name to generate bin phrases for. Available: {', '.join(available_metrics)}"
)
args = parser.parse_args()
metric_name = args.metric_name

bins = reduction_cfg["15"][metric_name]["bins"]["fine"]

def _format_range(lo, hi):
    """格式化分箱范围显示"""
    if lo == hi:
        return f"{lo}"
    hi_str = f"{hi}" if hi != "inf" else "inf"
    return f"{lo}-{hi_str}"

bin_labels = [f"{i}: {_format_range(lo, hi)}" for i, (lo, hi) in enumerate(bins)]
bin_list_text = "\n".join(bin_labels)

# === 指标特定的眼底反应轴指导函数（提前定义并赋值） ===
def _metric_visual_guidance(metric_name: str) -> str:
    m = metric_name.lower()
    if m == "hba1c":
        return (
            "- microaneurysm frequency and prominence\n"
            "- dot blot hemorrhage spread\n"
            "- hard exudate clusters and brightness\n"
            "- cotton wool spots presence\n"
            "- venous beading and tortuosity\n"
            "- vessel density or capillary rarefaction\n"
            "- neovascularization at disc or elsewhere\n"
            "- macular or foveal reflex attenuation"
        )
    if m in ("sysbp", "diabp"):
        return (
            "- arteriolar narrowing and attenuation\n"
            "- arteriolar light reflex (copper wiring)\n"
            "- arteriovenous crossing changes\n"
            "- venular tortuosity or beading\n"
            "- flame or blot hemorrhages\n"
            "- cotton wool spots\n"
            "- vessel density or capillary rarefaction\n"
            "- disc edema or crowding (severe)"
        )
    return (
        "- generalized vessel caliber and tortuosity\n"
        "- background fundus hue and contrast\n"
        "- macular and foveal reflex appearance\n"
        "- exudate distribution and brightness\n"
        "- hemorrhage extent and pattern\n"
        "- vessel density and capillary rarefaction\n"
        "- neovascularization presence and location"
    )

metric_axes_guidance = _metric_visual_guidance(metric_name)

# 阶段一：多角色协作的候选生成Prompt
generator_prompt_text = """
Role briefing (internal collaboration; do not output the process):
- Retinal fundus imaging clinician: ensure phrases reflect directly observable signs in color fundus photographs.
- Image analysis specialist: ensure all descriptors are strictly visual and present-time; no physiology speculation or prognosis.
- Medical terminology curator: unify style; concise English noun phrases; no diagnoses and no numbers.
- ML training engineer: enforce inter-bin separability via intensity gradation while staying visual-only.

Background:
We have {metric_name} bins as follows (index: range):
{bin_list}

Metric-specific fundus response cues to prioritize (use widely where plausible):
{metric_axes_guidance}

Goal:
For each bin, produce 3–5 English noun phrases (<= 6 words each) that strictly describe visible, present-time signs in color fundus photographs that are characteristic for that bin. Phrases must differentiate adjacent bins via intensity or feature-axis changes while maintaining a smooth visual gradation.

Style and scope boundaries:
- Use lowercase English; noun phrases only; avoid verbs with tense; no punctuation (hyphen allowed); no numbers.
- No disease/diagnosis/therapy names; no measurement units; do not mention metric names or lab terms (e.g., hba1c, glucose, glycemia, sugar, blood sugar, systolic, diastolic, blood pressure, bp, mmhg, percent).
- Absolutely avoid risk/prognosis/physiology terms (e.g., risk, burden, stress, perfusion, ischemia, ischemic, edema, leakage, autoregulation, hemodynamic, microcirculation, pressure). Replace them with visual-only descriptors if needed.

Global uniqueness requirement:
- Do not reuse any phrase across bins. Each phrase must be globally unique across all bins and also unique within its own bin.

Suggested visual feature axes (choose 3–5 to structure the gradation):
- arteriolar caliber, arteriolar light reflex, generalized vessel attenuation
- venular caliber, venular tortuosity
- vessel wall sheen, arterial light reflex prominence
- arteriovenous crossing changes
- background fundus hue/contrast
- macular or foveal reflex appearance
- optic disc margin clarity, peripapillary vessel crowding

Workflow (perform internally; do not output steps):
1) Plan 3–5 visual axes with clear intensity ladder (e.g., slight → mild → moderate → marked), prioritizing the metric-specific cues above when plausible.
2) For each bin, output 3–5 compact phrases (<= 6 words), covering the chosen axes and reflecting that bin’s plausible visible appearance in color fundus photos.
3) Differentiate adjacent bins and maintain global uniqueness: if duplication or near-synonym occurs, adjust intensity wording or switch axis.
4) Normalize: lowercase, noun phrases, no numbers, no diagnoses, visual-only wording.

Output requirement (output JSON only; no explanation):
- JSON keys: exactly "index: range" (e.g., "0: 45-59"), ordered by ascending index.
- JSON values: an array of 3–5 phrases per bin (lowercase English, <= 6 words, noun phrases, visual-only, no diagnoses, no numbers, no banned terms), all globally unique across bins.
""".strip()

generator_prompt = ChatPromptTemplate.from_template(generator_prompt_text)

# 阶段二：验证/修正/格式化Prompt
validator_prompt_text = """
You are the "terminology validator and formatter". Rigorously validate and minimally fix the candidate JSON so it satisfies ALL of the following:

Input:
- Bins (index: range):
{bin_list}
- Candidate JSON:
```json
{candidate_json}
```

Metric-specific axes to prefer (do not output this section):
{metric_axes_guidance}

Constraints to satisfy:
1) Keys must match the bins exactly and preserve names (e.g., "0: 45-59", "105-inf"), ordered by ascending index.
2) Each bin has 3–5 phrases; each phrase <= 6 English words; all lowercase; noun phrases; no numbers; no punctuation except hyphen.
3) Visual-only language for color fundus photographs: do NOT include risk, prognosis, or physiology/measurement terms. Prohibited terms (replace if present): risk, burden, stress, perfusion, ischemia, ischemic, edema, leakage, autoregulation, hemodynamic, microcirculation, pressure, hypotension, hypertension, diastolic, systolic, blood pressure, bp, mmhg, hba1c, glucose, glycemia, sugar, blood sugar, percent.
4) Uniqueness:
   - Within-bin: no duplicate phrases inside the same bin.
   - Global: no duplicate phrases across any two bins. If duplicates or near-synonyms are detected, replace minimally while keeping a smooth visual gradation and axis consistency.
5) Unified terminology and style suitable for model training; concise, present-time, visual-only descriptors; no disease/diagnosis/therapy names.

Output only the final JSON, with no explanations or extra text.
{format_instructions}
""".strip()

# 修复：正确通过 partial 传入 format_instructions
validator_prompt = ChatPromptTemplate.from_template(validator_prompt_text).partial(
    format_instructions=bin_parser.get_format_instructions()
)

# 执行双阶段生成：候选生成 -> 验证修正 -> 解析保存
print(f"开始生成{metric_name}分箱的描述短语...")

# 生成候选
candidate_json = (generator_prompt | model | StrOutputParser()).invoke({
    "bin_list": bin_list_text,
    "metric_name": metric_name,
    "metric_axes_guidance": metric_axes_guidance,
})
print("候选生成完成，开始验证和格式化...")

# 验证并解析成结构化结果
final_obj = (validator_prompt | model | bin_parser).invoke({
    "bin_list": bin_list_text,
    "candidate_json": candidate_json,
    "metric_axes_guidance": metric_axes_guidance,
})
result = final_obj.BIN_PHRASES

# === 新增：全局去重与最小替换修复 ===
def _compute_conflicts(bin_phrases: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Return phrases to replace for each bin due to duplicates (within-bin or cross-bin)."""
    conflicts = {}
    phrase_to_bins = {}
    for bin_key, phrases in bin_phrases.items():
        seen = set()
        for p in phrases:
            # within-bin duplicate
            if p in seen:
                conflicts.setdefault(bin_key, set()).add(p)
            seen.add(p)
            phrase_to_bins.setdefault(p, []).append(bin_key)
    # cross-bin duplicates: keep the lowest-index owner, replace others
    for p, keys in phrase_to_bins.items():
        if len(keys) > 1:
            keeper = min(keys, key=lambda k: int(k.split(":")[0].strip()))
            for k in keys:
                if k != keeper:
                    conflicts.setdefault(k, set()).add(p)
    return {k: sorted(list(v)) for k, v in conflicts.items()}

_uniqueness_fix_prompt_text = """
You are the "global uniqueness enforcer and formatter" for visual-only color fundus descriptors.

Input:
- Bins (index: range):
{bin_list}
- Current JSON:
```json
{current_json}
```
- Conflicts (phrases that MUST be replaced; only modify these and keep all other phrases unchanged):
```json
{conflicts}
```
- Metric-specific axes to prefer (do not output this section):
{metric_axes_guidance}

Tasks:
1) For each bin, ensure 3–5 phrases; each phrase <= 6 words; all lowercase; noun phrases; no numbers; no punctuation except hyphen.
2) Visual-only language for color fundus photographs; no diagnoses; no banned terms: risk, burden, stress, perfusion, ischemia, ischemic, edema, leakage, autoregulation, hemodynamic, microcirculation, pressure, hypotension, hypertension, diastolic, systolic, blood pressure, bp, mmhg, hba1c, glucose, glycemia, sugar, blood sugar, percent.
3) Replace only the listed conflicting phrases with new, globally unique phrases that maintain plausible visual gradation and the chosen visual axes for that bin. Do not alter any other phrases.
4) Preserve all keys exactly and order by ascending index.

Output only the final JSON, with no explanations.
{format_instructions}
""".strip()

_uniqueness_fix_prompt = ChatPromptTemplate.from_template(_uniqueness_fix_prompt_text).partial(
    format_instructions=bin_parser.get_format_instructions()
)

max_iters = 3
for i in range(max_iters):
    conflicts = _compute_conflicts(result)
    if not conflicts:
        break
    fix_obj = (_uniqueness_fix_prompt | model | bin_parser).invoke({
        "bin_list": bin_list_text,
        "current_json": json.dumps(result, ensure_ascii=False),
        "conflicts": json.dumps(conflicts, ensure_ascii=False),
        "metric_axes_guidance": metric_axes_guidance,
    })
    result = fix_obj.BIN_PHRASES

# 打印并保存结果
print(f"\n=== 生成的{metric_name}分箱描述短语 ===")
print(json.dumps(result, ensure_ascii=False, indent=2))

output_path = osp.join(osp.dirname(osp.dirname(__file__)), "configs", f"{metric_name}_bin_phrases_15.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"\n结果已保存到: {output_path}")
print("任务完成！")


def _metric_visual_guidance(metric_name: str) -> str:
    m = metric_name.lower()
    if m == "hba1c":
        return (
            "- microaneurysm frequency and prominence\n"
            "- dot blot hemorrhage spread\n"
            "- hard exudate clusters and brightness\n"
            "- cotton wool spots presence\n"
            "- venous beading and tortuosity\n"
            "- vessel density or capillary rarefaction\n"
            "- neovascularization at disc or elsewhere\n"
            "- macular or foveal reflex attenuation"
        )
    if m in ("sysbp", "diabp"):
        return (
            "- arteriolar narrowing and attenuation\n"
            "- arteriolar light reflex (copper wiring)\n"
            "- arteriovenous crossing changes\n"
            "- venular tortuosity or beading\n"
            "- flame or blot hemorrhages\n"
            "- cotton wool spots\n"
            "- vessel density or capillary rarefaction\n"
            "- disc edema or crowding (severe)"
        )
    return (
        "- generalized vessel caliber and tortuosity\n"
        "- background fundus hue and contrast\n"
        "- macular and foveal reflex appearance\n"
        "- exudate distribution and brightness\n"
        "- hemorrhage extent and pattern\n"
        "- vessel density and capillary rarefaction\n"
        "- neovascularization presence and location"
    )

metric_axes_guidance = _metric_visual_guidance(metric_name)
