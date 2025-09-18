from typing import Union, Tuple, List, Optional


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
    "100": "one hundred", "104": "one hundred and four", "105": "one hundred and five", "106": "one hundred and six",   
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
    prompt_type: str = "word",
    phrase_config_path: Optional[str] = None,
    template: Optional[str] = None,
    name_syns_mode: str = "pro",     # ['pro', 'plain', 'random']
    pick_mode: str = "random",       # ['random', 'first']
    seed: Optional[int] = None
) -> List[str]:
    """
    根据分箱范围与可配置的短语JSON，生成每个bin对应的一句式提示语。
    - pro_name_syns / plain_name_syns / bin_label_syns 均从 phrase_config_path 所指的 JSON 中加载
    - template 从参数注入，默认 "A fundus image of {name} {rng} suggests {label}."
    - prompt_type: 'word'/'number' 控制数字用词/用数
    - name_syns_mode: 使用专业名/普通名/随机
    - pick_mode: 随机选取短语或固定取第一条
    """
    import os, json, math, random, re

    if seed is not None:
        random.seed(seed)

    # 默认配置路径（可被参数覆盖）
    if phrase_config_path is None:
        phrase_config_path = "/opt/DM/OCT/CLIP_Code/CLIP-EBC/configs/hba1c_bin_phrases_15.json"

    assert os.path.exists(phrase_config_path), f"Phrase config not found: {phrase_config_path}"
    with open(phrase_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 读取名称同义词
    pro_name_syns = cfg.get("pro_name_syns", [])
    plain_name_syns = cfg.get("plain_name_syns", [])

    # 解析 bin 短语映射：支持形如 "0: 78-110" / "15: 90-inf"
    idx_to_phrases = {}
    range_to_phrases = {}  # ((lo, hi), phrases)
    range_key_pattern = re.compile(r"^\s*(\d+)\s*:\s*([-\d\.]+)\s*-\s*([-\d\.]+|inf)\s*$")

    for k, v in cfg.items():
        if not isinstance(k, str) or not isinstance(v, list):
            continue
        m = range_key_pattern.match(k)
        if not m:
            continue
        idx = int(m.group(1))
        lo = float(m.group(2))
        hi_raw = m.group(3).lower()
        hi = float("inf") if hi_raw == "inf" else float(hi_raw)
        idx_to_phrases[idx] = v
        range_to_phrases[(int(lo), int(hi) if math.isfinite(hi) else float("inf"))] = v

    # 模板处理
    template = (template or "A fundus image of {name} {rng} suggests {label}.").strip()

    # 数字转词
    def _num_word(v: float) -> str:
        return num2word(int(v))

    # 区间格式化（词/数）
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

    use_word = (isinstance(prompt_type, str) and prompt_type.lower().startswith("word"))
    fmt_range = _fmt_range_words if use_word else _fmt_range_digits

    # 名称选择
    def _pick_name() -> str:
        if name_syns_mode == "random":
            pool = (pro_name_syns or []) + (plain_name_syns or [])
            return random.choice(pool) if pool else indicator_name
        elif name_syns_mode == "plain":
            return random.choice(plain_name_syns) if (pick_mode == "random" and plain_name_syns) else (plain_name_syns[0] if plain_name_syns else indicator_name)
        else:  # "pro" or fallback
            return random.choice(pro_name_syns) if (pick_mode == "random" and pro_name_syns) else (pro_name_syns[0] if pro_name_syns else indicator_name)

    outputs = []
    unit_str = unit or ""

    for i, r in enumerate(indicator_ranges or []):
        lo_raw, hi_raw = r
        lo = float(lo_raw)
        hi = float("inf") if (isinstance(hi_raw, str) and str(hi_raw).lower() == "inf") else float(hi_raw)

        rng_phrase = fmt_range(lo, hi, unit_str)

        # 先按区间匹配（更稳健），失败则按索引匹配
        phrases = range_to_phrases.get(
            (int(lo), int(hi) if math.isfinite(hi) else float("inf"))
        )
        if phrases is None:
            phrases = idx_to_phrases.get(i, None)

        if not phrases:
            # 找不到则给出中性描述
            label = "a typical retinal appearance"
        else:
            label = random.choice(phrases) if pick_mode == "random" else phrases[0]

        name_syn = _pick_name()
        sent = template.format(name=name_syn, rng=rng_phrase, label=label)
        outputs.append(sent)

    return outputs