from typing import Union, Tuple, List


num_to_word = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine", 
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen", "18": "eighteen", "19": "nineteen", 
    "20": "twenty", "21": "twenty-one", "22": "twenty-two", "23": "twenty-three", "24": "twenty-four", "25": "twenty-five", "26": "twenty-six", "27": "twenty-seven", "28": "twenty-eight", "29": "twenty-nine",
    "30": "thirty", "31": "thirty-one", "32": "thirty-two", "33": "thirty-three", "34": "thirty-four", "35": "thirty-five", "36": "thirty-six", "37": "thirty-seven", "38": "thirty-eight", "39": "thirty-nine",
    "40": "forty", "41": "forty-one", "42": "forty-two", "43": "forty-three", "44": "forty-four", "45": "forty-five", "46": "forty-six", "47": "forty-seven", "48": "forty-eight", "49": "forty-nine",
    "50": "fifty", "51": "fifty-one", "52": "fifty-two", "53": "fifty-three", "54": "fifty-four", "55": "fifty-five", "56": "fifty-six", "57": "fifty-seven", "58": "fifty-eight", "59": "fifty-nine",
    "60": "sixty", "61": "sixty-one", "62": "sixty-two", "63": "sixty-three", "64": "sixty-four", "65": "sixty-five", "66": "sixty-six", "67": "sixty-seven", "68": "sixty-eight", "69": "sixty-nine",
    "70": "seventy", "71": "seventy-one", "72": "seventy-two", "73": "seventy-three", "74": "seventy-four", "75": "seventy-five", "76": "seventy-six", "77": "seventy-seven", "78": "seventy-eight", "79": "seventy-nine",
    "80": "eighty", "81": "eighty-one", "82": "eighty-two", "83": "eighty-three", "84": "eighty-four", "85": "eighty-five", "86": "eighty-six", "87": "eighty-seven", "88": "eighty-eight", "89": "eighty-nine",
    "90": "ninety", "91": "ninety-one", "92": "ninety-two", "93": "ninety-three", "94": "ninety-four", "95": "ninety-five", "96": "ninety-six", "97": "ninety-seven", "98": "ninety-eight", "99": "ninety-nine",
    "100": "one hundred", "200": "two hundred", "300": "three hundred", "400": "four hundred", "500": "five hundred", "600": "six hundred", "700": "seven hundred", "800": "eight hundred", "900": "nine hundred",
    "1000": "one thousand"
}


def num2word(num: Union[int, str]) -> str:
    """
    Convert the input number to the corresponding English word. For example, 1 -> "one", 2 -> "two", etc.
    """
    num = str(int(num))
    return num_to_word.get(num, num)


def format_count(count: Union[float, Tuple[float, float]], prompt_type: str = "word") -> str:
    if count == 0:
        return "There is no person." if prompt_type == "word" else "There is 0 person."
    elif count == 1:
        return "There is one person." if prompt_type == "word" else "There is 1 person."
    elif isinstance(count, (int, float)):
        return f"There are {num2word(int(count))} people." if prompt_type == "word" else f"There are {int(count)} people."
    elif count[1] == float("inf"):
        return f"There are more than {num2word(int(count[0]))} people." if prompt_type == "word" else f"There are more than {int(count[0])} people."
    else:  # count is a tuple of finite numbers
        left, right = int(count[0]), int(count[1])
        left, right, _ = num2word(left), num2word(right) if prompt_type == "word" else left, right
        return f"There are between {left} and {right} people."


def format_blood_indicator(
    indicator_ranges: List[Tuple[float, float]], 
    indicator_name: str,
    unit: str = "",
    prompt_type: str = "word"
) -> List[str]:
    """
    Generate short, single-sentence English descriptions for indicator ranges.
    Updated to:
    - Ground prior knowledge in the context of fundus/retinal imaging.
    - Respect `prompt_type="word"` by expressing numbers in words for all ranges.
    """
    import os, json, math, random

    # Load and cache the 15-bin DBP ranges from reduction_8.json
    if not hasattr(format_blood_indicator, "_dbp_bins_15"):
        cfg_path = "/opt/DM/OCT/CLIP_Code/CLIP-EBC/configs/reduction_8.json"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        raw_bins = cfg.get("15", {}).get("diabp", {}).get("bins", {}).get("fine", [])
        dbp_bins_15 = []
        for lo, hi in raw_bins:
            lo_f = float(lo)
            hi_f = float("inf") if (isinstance(hi, str) and hi == "inf") else float(hi)
            dbp_bins_15.append((lo_f, hi_f))
        format_blood_indicator._dbp_bins_15 = tuple(dbp_bins_15)

    dbp_bins_15 = format_blood_indicator._dbp_bins_15

    # 两组名称池（保留原有），句式模板更新为"眼底/视网膜"语境下的短句，不再是纯粹的"心血管/高血压"表述。
    pro_name_syns = [
        "DBP",
        "diastolic BP",
        "diastolic pressure",
        "diastolic blood pressure",
        "arterial diastolic pressure",
        "diastolic arterial pressure",
        "arterial pressure in diastole",
        "pressure during diastole",
        "pressure during cardiac relaxation",
        "diastolic measure",
        "diastolic index",
        "diastolic parameter",
        "DBP level",
    ]
    plain_name_syns = [
        "DBP",
        "diastolic BP",
        "diastolic pressure",
        "lower BP reading",
        "bottom number",
        "diastolic value",
        "diastolic number",
    ]

    # 视网膜/眼底语境的固定句式模板（不包含括号，固定开头）
    template = "A found image of {name} {rng} suggests {label}."

    # 15 个 bin 的视网膜相关短标签（精简、无括号）
    bin_label_syns = [
        # 0: 45–59
        ["reduced retinal perfusion", "low microvascular tone", "retinal hypoperfusion pattern"],
        # 1: 60–69
        ["low-normal retinal perfusion", "lower-normal vascular tone", "retinal circulation at low-normal"],
        # 2: 70–73
        ["normal retinal microvasculature", "typical retinal appearance", "physiologic vessel caliber"],
        # 3: 74–77
        ["upper-normal arteriolar tone", "mild trend to arteriolar narrowing", "borderline vascular tension"],
        # 4: 78–79
        ["borderline retinal stress", "subtle arteriolar narrowing", "borderline vascular load"],
        # 5: 80–80
        ["borderline retinal stress", "subtle arteriolar narrowing", "borderline vascular load"],
        # 6: 81–82
        ["early retinopathy signs", "arteriolar narrowing", "arteriovenous nicking"],
        # 7: 83–84
        ["early retinopathy signs", "arteriolar narrowing", "increased vascular tortuosity"],
        # 8: 85–85
        ["early retinopathy signs", "arteriolar narrowing", "arteriovenous nicking"],
        # 9: 86–87
        ["early retinopathy signs", "arteriolar narrowing", "increased vascular tortuosity"],
        # 10: 88–89
        ["early retinopathy signs", "arteriolar narrowing", "arteriovenous nicking"],
        # 11: 90–90
        ["moderate retinopathy risk", "retinal hemorrhages", "hard exudates"],
        # 12: 91–94
        ["moderate retinopathy risk", "cotton wool spots", "retinal hemorrhages"],
        # 13: 95–104
        ["moderate retinopathy risk", "venous beading", "clustered exudates"],
        # 14: 105–inf
        ["severe retinopathy risk", "optic disc edema", "widespread hemorrhages"],
    ]

    def _is_diastolic(name: str) -> bool:
        n = (name or "").lower()
        return ("dbp" in n) or ("diabp" in n) or ("diastolic" in n) or ("舒张" in n)

    # === 新增：数字转英文单词的区间格式化，支持 word / 数字 两种模式 ===
    def _num_word(v: float) -> str:
        return num2word(int(v))

    def _fmt_range_words(lo: float, hi: float, unit_str: str) -> str:
        if math.isinf(hi):
            return f"at least {_num_word(lo)} {unit_str}".strip()
        if int(lo) == int(hi):
            return f"{_num_word(lo)} {unit_str}".strip()
        return f"{_num_word(lo)} to {_num_word(hi)} {unit_str}".strip()

    def _fmt_range_digits(lo: float, hi: float, unit_str: str) -> str:
        if math.isinf(hi):
            return f"≥{int(lo)} {unit_str}".strip()
        if int(lo) == int(hi):
            return f"{int(lo)} {unit_str}".strip()
        return f"{int(lo)}–{int(hi)} {unit_str}".strip()

    def _match_bin(lo: float, hi: float):
        for idx, (blo, bhi) in enumerate(dbp_bins_15):
            if int(blo) == int(lo) and ((math.isinf(bhi) and math.isinf(hi)) or int(bhi) == int(hi)):
                return idx
        return None

    # 根据 prompt_type 决定是否使用"英文单词"模式
    use_plain = isinstance(prompt_type, str) and prompt_type.lower().startswith("plain")
    use_word = isinstance(prompt_type, str) and prompt_type.lower().startswith("word")
    name_pool = plain_name_syns if use_plain else pro_name_syns

    is_dbp = _is_diastolic(indicator_name)
    unit_str = unit or ("mmHg" if is_dbp else "")

    outputs = []
    for i, r in enumerate(indicator_ranges or []):
        lo_raw, hi_raw = r
        lo = float(lo_raw)
        hi = float("inf") if (isinstance(hi_raw, str) and str(hi_raw).lower() == "inf") else float(hi_raw)

        # 使用单词 or 数字格式化区间
        rng_phrase = (_fmt_range_words if use_word else _fmt_range_digits)(lo, hi, unit_str)

        bin_idx = _match_bin(lo, hi) if is_dbp else None

        # 统一名称与模板：固定句式，不再随机模板；DBP 统一为专业名称
        name_syn = "diastolic blood pressure" if is_dbp else indicator_name
        tpl = template

        if is_dbp and bin_idx is not None and 0 <= bin_idx < len(bin_label_syns):
            # 对应 bin 的标签（无括号，眼底彩照相关）
            label = random.choice(bin_label_syns[bin_idx])
            sent = tpl.format(name=name_syn, rng=rng_phrase, label=label)
        else:
            # 非 DBP 指标：同样使用固定句式与中性视网膜标签
            label = "a typical retinal appearance"
            sent = tpl.format(name=name_syn, rng=rng_phrase, label=label)

        outputs.append(sent)

    return outputs