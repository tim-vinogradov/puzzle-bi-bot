"""Встроенный разборщик запросов на русском языке.

Работает без интернета и без ключей: покрывает все формулировки из кейса и их
естественные вариации («отклонение факта от плана», «долю каждого региона»,
«перестрой третий график», «открой дашборд «Продажи Q3»» и т. д.).
Если фраза не распознана уверенно, её получает языковая модель (режим hybrid).
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Контекст диалога
# ---------------------------------------------------------------------------


@dataclass
class NLUContext:
    has_dashboard: bool = False
    dataset: str | None = None
    charts: list[dict] = field(default_factory=list)
    calcs: list[dict] = field(default_factory=list)
    switches: dict = field(default_factory=dict)
    table_values: dict[str, list[str]] = field(default_factory=dict)
    catalog: dict[str, list[str]] = field(default_factory=dict)
    saved_names: list[str] = field(default_factory=list)
    current_name: str | None = None
    last_chart: int | None = None
    columns: list[str] = field(default_factory=list)


@dataclass
class ParseResult:
    actions: list[dict]
    confident: bool = True


# ---------------------------------------------------------------------------
# Нормализация и мелкие помощники
# ---------------------------------------------------------------------------

QUOTE_RE = re.compile(r"[«\"“„']([^»\"”“']{1,120})[»\"”“']")


def norm(text: str) -> str:
    text = (text or "").lower().replace("ё", "е")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def quoted(text: str) -> list[str]:
    return [q.strip() for q in QUOTE_RE.findall(text or "") if q.strip()]


NUM_WORDS = {
    "один": 1, "одна": 1, "одного": 1, "два": 2, "две": 2, "двух": 2, "три": 3, "трех": 3, "четыре": 4, "четырех": 4,
    "пять": 5, "пяти": 5, "шесть": 6, "шести": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10, "десяти": 10,
    "двенадцать": 12, "двенадцати": 12, "пятнадцать": 15, "двадцать": 20, "двадцати": 20,
}

ORDINALS = {
    1: r"перв\w*|1-?(?:й|ый|ой|го|му)?", 2: r"втор\w*|2-?(?:й|ой|ый|го|му)?", 3: r"трет\w*|3-?(?:й|ий|го|му)?",
    4: r"четверт\w*|4-?(?:й|ый|го|му)?", 5: r"пят(?:ый|ого|ому|ом)|5-?(?:й|ый|го)?", 6: r"шест(?:ой|ого|ому)|6-?(?:й|ой|го)?",
}

MONTH_STEMS = ["январ", "феврал", "март", "апрел", "ма[йяе]", "июн", "июл", "август", "сентябр", "октябр", "ноябр", "декабр"]


def to_int(token: str) -> int | None:
    token = token.strip()
    if token.isdigit():
        return int(token)
    return NUM_WORDS.get(token)


# ---------------------------------------------------------------------------
# Периоды
# ---------------------------------------------------------------------------


def parse_period(t: str) -> dict | None:
    """«за последний квартал» → {'kind': 'last_quarter'} и т. п. Даты вычисляет бот."""
    if re.search(r"за вс[ею] (?:время|историю|период)|весь период|за все годы|все данные|вся история", t):
        return {"kind": "all"}
    if re.search(r"с начала (?:этого |текущего )?года|\bytd\b|текущ\w* год|этот год|этом году|текущем году", t):
        return {"kind": "ytd"}

    # диапазон месяцев «с марта 2025 по июнь 2026»
    m = re.search(r"с (" + "|".join(MONTH_STEMS) + r")\w*\s*(\d{4})?\s*(?:года?)?\s*(?:по|до) (" + "|".join(MONTH_STEMS) + r")\w*\s*(\d{4})?", t)
    if m:
        m1 = _month_index(m.group(1))
        m2 = _month_index(m.group(3))
        return {"kind": "range", "start_month": m1, "start_year": int(m.group(2)) if m.group(2) else None,
                "end_month": m2, "end_year": int(m.group(4)) if m.group(4) else None}
    m = re.search(r"(?:с|от) (\d{4})\s*(?:г\.?|года)?\s*(?:по|до|-) (\d{4})", t) or re.search(r"\b(20\d{2})\s*-\s*(20\d{2})\b", t)
    if m:
        return {"kind": "range", "start": f"{m.group(1)}-01-01", "end": f"{m.group(2)}-12-31"}
    m = re.search(r"\bс (20\d{2})\s*(?:г\.?|года)?", t)
    if m:
        return {"kind": "range", "start": f"{m.group(1)}-01-01", "end": None}

    m = re.search(r"последн\w* (\d+|[а-я]+) (месяц\w*|мес\.?|квартал\w*|год\w*|лет|недел\w*)", t)
    if m:
        n = to_int(m.group(1))
        unit = m.group(2)
        if n:
            if unit.startswith("мес"):
                return {"kind": "last_n_months", "n": n}
            if unit.startswith("квартал"):
                return {"kind": "last_n_months", "n": 3 * n}
            if unit.startswith(("год", "лет")):
                return {"kind": "last_n_months", "n": 12 * n}
    if re.search(r"(?:последн\w*|прошл\w*|предыдущ\w*) квартал|за квартал\b", t):
        return {"kind": "last_quarter"}
    if re.search(r"(?:последн\w*|прошл\w*|предыдущ\w*) месяц|за месяц\b", t):
        return {"kind": "last_month"}
    if re.search(r"(?:прошл\w*|предыдущ\w*) год", t):
        return {"kind": "prev_year"}
    if re.search(r"последн\w* (?:год|12 месяц\w*|двенадцать месяц\w*)|за год\b|годовой|за 12 месяц|скользящ\w* год", t):
        return {"kind": "last_year"}
    m = re.search(r"(перв\w*|втор\w*|1|2)[- ]?(?:е|ое)?\s+полугоди\w*\s*(\d{4})?", t)
    if m:
        first = m.group(1).startswith(("перв", "1"))
        year = m.group(2)
        if year:
            return {"kind": "range", "start": f"{year}-{'01' if first else '07'}-01", "end": f"{year}-{'06-30' if first else '12-31'}"}
        return {"kind": "range", "start_month": 1 if first else 7, "start_year": None, "end_month": 6 if first else 12, "end_year": None}
    if re.search(r"полгода|полугоди", t):
        return {"kind": "last_n_months", "n": 6}

    m = re.search(r"\b([1-4])\s*-?\s*(?:й|го|ый|ом)?\s*(?:квартал\w*|кв\.?)\s*(\d{4})?", t) or \
        re.search(r"\bq([1-4])\s*(\d{4})?", t)
    if m:
        return {"kind": "quarter", "quarter": int(m.group(1)), "year": int(m.group(2)) if m.group(2) else None}
    for word, q in (("перв", 1), ("втор", 2), ("трет", 3), ("четверт", 4)):
        m = re.search(word + r"\w* квартал\w*\s*(\d{4})?", t)
        if m:
            return {"kind": "quarter", "quarter": q, "year": int(m.group(1)) if m.group(1) else None}

    m = re.search(r"\b(?:за|в|во|на)\s+(" + "|".join(MONTH_STEMS) + r")\w*\s*(\d{4})?", t)
    if m:
        return {"kind": "month", "month": _month_index(m.group(1)), "year": int(m.group(2)) if m.group(2) else None}
    m = re.search(r"\b(" + "|".join(MONTH_STEMS) + r")\w*\s+(20\d{2})", t)
    if m:
        return {"kind": "month", "month": _month_index(m.group(1)), "year": int(m.group(2))}
    m = re.search(r"\b(?:только|оставь)\s+(" + "|".join(MONTH_STEMS) + r")\w*", t)
    if m:
        return {"kind": "month", "month": _month_index(m.group(1)), "year": None}

    m = re.search(r"\b(?:за|в|на)\s+(20\d{2})\s*(?:г\.?|год\w*)?", t) or re.search(r"\b(20\d{2})\s*год", t)
    if m:
        return {"kind": "year", "year": int(m.group(1))}
    return None


def _month_index(stem: str) -> int:
    for i, pattern in enumerate(MONTH_STEMS):
        if re.fullmatch(pattern + r"\w*", stem) or re.match(pattern, stem):
            return i + 1
    return 1


# ---------------------------------------------------------------------------
# Разрезы, метрики, сущности
# ---------------------------------------------------------------------------

DIM_PATTERNS = [
    ("outlet_type", r"тип\w* точ\w*|формат\w* (?:точ|магаз)\w*"),
    ("outlet", r"торгов\w* точ\w*|\bточк(?:ам|ами|ах|и|у|е|а)?\b|\bточек\b|магазин\w*(?! у дома)"),
    ("region", r"регион\w*|округ\w*"),
    ("category", r"категори\w*"),
    ("product", r"продукт\w*|товар\w*(?! для дома)|\bsku\b|ассортимент\w*|позици\w*"),
    ("client", r"клиент\w*|покупател\w*|контрагент\w*"),
    ("city", r"город\w*"),
    ("segment", r"сегмент\w*"),
    ("reason", r"причин\w*"),
]

DIM_WORD_GEN = {
    "region": r"регион", "outlet": r"точ[еки]|магазин", "product": r"продукт|товар", "category": r"категори",
    "client": r"клиент", "city": r"город", "segment": r"сегмент", "reason": r"причин",
}


def find_dims(t: str) -> list[str]:
    found: list[tuple[int, str]] = []
    for dim, pattern in DIM_PATTERNS:
        m = re.search(pattern, t)
        if m and dim not in [d for _, d in found]:
            if dim == "outlet" and re.search(r"тип\w* точ", t) and not re.search(r"торгов\w* точ|по точкам", t):
                continue
            if dim == "client" and re.search(r"(?:нов\w*|лояльн\w*) клиент", t):
                continue
            found.append((m.start(), dim))
    return [d for _, d in sorted(found)]


def find_metric(t: str) -> str | None:
    if re.search(r"(?:количеств\w*|числ\w*|кол-в\w*) сдел|\bсделк|\bсделок", t):
        return "deals"
    if re.search(r"скидк|скидок", t):
        return "discount"
    if re.search(r"в штуках|\bшт\b|штук|количеств\w*|кол-в\w*|объем\w* в натур", t):
        return "qty"
    if re.search(r"в рублях|выручк|объем|сумм\w* продаж|оборот", t):
        return "amount"
    return None


REGION_PATTERNS = {
    "Центральный": r"центральн",
    "Северо-Западный": r"северо[- ]?запад|\bсзфо\b",
    "Южный": r"\bюжн\w*|\bюг(?:а|е|у|ом)?\b",
    "Северо-Кавказский": r"северо[- ]?кавказ|\bкавказ\w*|\bскфо\b",
    "Приволжский": r"приволж\w*|поволж\w*|\bпфо\b",
    "Уральский": r"уральск\w*|\bурал(?:а|е|у|ом)?\b",
    "Сибирский": r"сибирск\w*|\bсибир(?:ь|и|ью)\b",
    "Дальневосточный": r"дальневосточн\w*|дальн\w* восток\w*|\bдфо\b",
}

CATEGORY_PATTERNS = {
    "Напитки": r"напит",
    "Молочная продукция": r"молочн|молочк",
    "Бакалея": r"бакале",
    "Снеки и кондитерские изделия": r"снек|кондитер",
    "Бытовая химия": r"бытов\w* хими|\bхими[яию]",
    "Товары для дома": r"товар\w* для дома",
}

REASON_PATTERNS = {
    "Лояльность клиента": r"лояльн",
    "Объёмная скидка": r"объемн\w* скидк|за объем",
    "Промо-акция": r"промо",
    "Сезонная распродажа": r"сезонн|распродаж",
    "Истекающий срок годности": r"срок\w* годност|истекающ",
    "Компенсация за задержку поставки": r"компенсац|задержк",
    "Согласование руководителем": r"согласован|руководител",
    "Новый клиент": r"нов\w* клиент",
}

SEGMENT_PATTERNS = {
    "Сеть": r"\bсет(?:ь|и|ям|ей|евы\w*)\b",
    "HoReCa": r"horeca|хорек",
    "Независимый ритейл": r"независим",
    "Дистрибьютор": r"дистриб",
    "Онлайн-ритейл": r"онлайн|интернет",
}

TYPE_PATTERNS = {
    "Магазин у дома": r"магазин\w* у дома",
    "Супермаркет": r"супермаркет",
    "Оптовая база": r"оптов",
    "Гипермаркет": r"гипермаркет",
}


def _stem_word(word: str) -> str:
    word = word.lower().replace("ё", "е")
    if len(word) <= 3:
        return word
    return re.sub(r"(ами|ями|ого|его|ому|ему|ыми|ими|ах|ях|ов|ев|ей|ой|ый|ий|ая|яя|ое|ее|ые|ие|ам|ям|ом|ем|ы|и|а|я|о|е|у|ю|ь)$", "", word) or word


def product_patterns(products: list[str]) -> dict[str, str]:
    patterns = {}
    for name in products:
        words = [w for w in re.findall(r"[а-яёa-z]+", name.lower().replace("ё", "е")) if len(w) > 1]
        if not words:
            continue
        stems = [_stem_word(w) for w in words[:2]]
        if len(stems) == 1:
            patterns[name] = r"\b" + re.escape(stems[0]) + r"\w*"
        else:
            patterns[name] = r"\b" + re.escape(stems[0]) + r"\w*\s+" + re.escape(stems[1]) + r"\w*"
    return patterns


def city_patterns(cities: list[str]) -> dict[str, str]:
    patterns = {}
    for city in cities:
        c = city.lower().replace("ё", "е")
        if c == "санкт-петербург":
            patterns[city] = r"санкт-петербург\w*|петербург\w*|\bспб\b|питер\w*"
        elif c == "москва":
            patterns[city] = r"\bмоскв\w*|\bмск\b"
        elif c == "нижний новгород":
            patterns[city] = r"нижн\w* новгород\w*"
        else:
            base = c[:-1] if len(c) > 5 and c[-1] in "аяьйео" else c
            patterns[city] = r"\b" + re.escape(base) + r"\w*"
    return patterns


def match_values(t: str, patterns: dict[str, str]) -> list[str]:
    return [value for value, pattern in patterns.items() if re.search(pattern, t)]


def clean_client_name(name: str) -> str:
    name = norm(name)
    name = re.sub(r"\b(ооо|оао|зао|пао|ао|ип)\b", " ", name)
    name = re.sub(r"[«»\"'.,]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def match_clients(fragment: str, clients: list[str]) -> list[str]:
    frag = clean_client_name(fragment)
    if not frag or len(frag) < 3:
        return []
    cleaned = {c: clean_client_name(c) for c in clients}
    exact = [c for c, cc in cleaned.items() if cc == frag]
    if exact:
        return exact
    frag_stems = [_stem_word(w) for w in frag.split() if len(w) > 1]
    contains = [c for c, cc in cleaned.items() if all(s in cc for s in frag_stems)]
    if contains:
        return contains
    close = difflib.get_close_matches(frag, list(cleaned.values()), n=5, cutoff=0.72)
    return [c for c, cc in cleaned.items() if cc in close]


def extract_client_fragment(text: str, t: str) -> str | None:
    q = quoted(text)
    m = re.search(r"(?:по|для|у|от|с|со|о|об|про)\s+клиент\w*\s+(.+?)(?:\s+за\s|\s+в\s+20|\s*,|\s+какие|\s+что|\s+почему|\s+с\s+|\?|$)", t)
    if m:
        frag = m.group(1).strip(" .?!")
        for item in q:
            if norm(item) in frag or frag in norm(item):
                return item
        return frag
    m = re.search(r"клиент\w*\s+((?:ооо|ао|ип|пао|зао)\s+.+?)(?:\s+за\s|,|$)", t)
    if m:
        return m.group(1)
    for item in q:
        if re.search(r"\b(ооо|ао|ип|пао)\b", norm(item)):
            return item
    return None


def entities(text: str, t: str, ctx: NLUContext) -> dict[str, list[str]]:
    """Все упомянутые в тексте значения разрезов: регионы, категории, продукты, города…"""
    cat = ctx.catalog or {}
    client_frag = extract_client_fragment(text, t)
    t_wo_client = t.replace(norm(client_frag), " ") if client_frag else t
    out: dict[str, list[str]] = {}
    regions = match_values(t_wo_client, REGION_PATTERNS)
    if regions:
        out["region"] = regions
    cats = match_values(t_wo_client, CATEGORY_PATTERNS)
    if cats:
        out["category"] = cats
    if cat.get("product"):
        prods = match_values(t_wo_client, product_patterns(cat["product"]))
        if prods:
            out["product"] = prods
    reasons = match_values(t_wo_client, REASON_PATTERNS)
    if reasons and re.search(r"скидк|причин", t):
        out["reason"] = reasons
    segments = match_values(t_wo_client, SEGMENT_PATTERNS)
    if segments and re.search(r"сегмент|клиент|сет", t):
        out["segment"] = segments
    types = match_values(t_wo_client, TYPE_PATTERNS)
    if types:
        out["outlet_type"] = types
    if cat.get("city"):
        cities = [c for c in match_values(t_wo_client, city_patterns(cat["city"])) if not _city_is_region_word(c, t_wo_client)]
        if cities:
            out["city"] = cities
    for q in quoted(text):
        qn = norm(q)
        outlets = [o for o in cat.get("outlet", []) if qn and qn in norm(o)]
        if outlets and len(qn) > 2:
            out.setdefault("outlet", []).extend(outlets)
    if client_frag:
        clients = match_clients(client_frag, cat.get("client", []))
        out["client"] = clients if clients else ["?" + client_frag]
    else:
        named = client_by_name(t, cat.get("client", []))
        if named:
            out["client"] = named
            # «Сибирскому Продукту» — это клиент, а не регион и не разрез «продукты»
            if out.get("region") and any("сибир" in norm(n) or "урал" in norm(n) or "централ" in norm(n) for n in named):
                out.pop("region", None)
    return out


def client_by_name(t: str, clients: list[str]) -> list[str]:
    """Клиент, названный без слова «клиент»: два слова названия подряд в любом падеже."""
    hits = []
    for client in clients:
        words = [w for w in clean_client_name(client).split() if len(w) > 2 and not re.fullmatch(r"[а-я]\.[а-я]\.?", w)]
        if len(words) < 2:
            continue
        stems = [re.escape(_stem_word(w)[:max(4, len(_stem_word(w)))]) for w in words[:2]]
        if re.search(r"\b" + stems[0] + r"\w*\s+" + stems[1] + r"\w*", t):
            hits.append(client)
    return hits


def _city_is_region_word(city: str, t: str) -> bool:
    # «Центральный» и т. п. не путаем с городами
    return False


# ---------------------------------------------------------------------------
# Вычисления
# ---------------------------------------------------------------------------

MEASURE_WORDS = [
    (r"план\w* в штуках|план\w* шт|план\w* по количеств\w*", "plan_qty"),
    (r"факт\w* в штуках|количеств\w*|штук\w*|\bшт\b|кол-в\w*", "qty"),
    (r"скидк\w*|скидок", "discount"),
    (r"сделк\w*|сделок", "deals"),
    (r"\bплан\w*", "plan"),
    (r"факт\w*|выручк\w*|продаж\w*|объем\w*|оборот\w*", "fact"),
]

CALC_TRIGGER = re.compile(
    r"колонк|столбец|столбц|вычислени|расчет|рассчит|посчитай|вычисли|процент\w* выполнен|% выполнен|выполнени\w* план|"
    r"отклонени|динамик|прирост|изменени\w* к|дол[юияей]\b|долей|накопительн|нарастающ|ранг|ранжир|"
    r"средн\w*|медиан|минимальн|максимальн|формул|разниц|\bmin\b|\bmax\b|м/м|г/г|mom|yoy|минус|делить|умножить|плюс")


def _measure_in(fragment: str, default: str = "fact") -> str:
    for pattern, col in MEASURE_WORDS:
        if re.search(pattern, fragment):
            return col
    return default


def _dim_after(t: str, words: str) -> str | None:
    """Разрез после слов «по …», «каждого …», «внутри …»."""
    m = re.search(words + r"\s+(\w+)(?:\s+(\w+))?", t)
    if not m:
        return None
    fragment = " ".join(x for x in m.groups() if x)
    for dim, pattern in DIM_WORD_GEN.items():
        if re.search(pattern, fragment):
            return dim
    return None


def parse_formula(t: str, calcs: list[dict]) -> str | None:
    """«факт минус план, делить на план, умножить на 100» → «((fact - plan) / plan) * 100»."""
    m = re.search(r"(?:формул\w*|посчитай|рассчитай|вычисли|колонк\w*|столбец)\s*[:\-]?\s*(.+)$", t)
    body = m.group(1) if m else t
    body = re.sub(r"^(?:как|по формуле|формулой|которая|равную|равна|=)\s*", "", body.strip())
    body = body.replace("×", "*").replace("÷", "/").replace("−", "-")
    name_map: list[tuple[str, str]] = []
    for calc in calcs:
        label = norm(calc["label"]).split(",")[0]
        first = label.split()[0] if label else ""
        if first and len(first) > 3:
            name_map.append((re.escape(_stem_word(first)) + r"\w*", calc["id"]))
    tokens_map = [
        (r"план\w* в штуках|план\w* шт", " plan_qty "),
        (r"факт\w* в штуках|количеств\w*|кол-в\w*", " qty "),
        (r"фактическ\w* \w+|факт\w*|выручк\w*", " fact "),
        (r"план\w*", " plan "),
        (r"скидк\w*|скидок", " discount "),
        (r"сделк\w*|сделок", " deals "),
    ] + [(p, f" {cid} ") for p, cid in name_map] + [
        (r"умножить на|умноженн\w* на|умножь на|помножить на|\bх\b|\bx\b", " * "),
        (r"(?:раз)?делить на|(?:раз|по)?деленн\w* на|подели\w* на|раздели\w* на|делим на", " / "),
        (r"плюс|прибавить|сложить с", " + "),
        (r"минус|вычесть|отнять", " - "),
        (r"в процентах|процентах", " * 100 "),
        (r"\bсто\b", " 100 "),
    ]
    segments = [s for s in re.split(r",|;|\bзатем\b|\bпотом\b|\bи потом\b", body) if s.strip()]
    expr = ""
    for seg in segments:
        s = " " + seg + " "
        for pattern, repl in tokens_map:
            s = re.sub(pattern, repl, s)
        s = re.sub(r"(\d),(\d)", r"\1.\2", s)
        s = re.sub(r"[^a-z_0-9.+\-*/() ]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"\s*[+\-*/]\s*$", "", s).strip()
        if not s:
            continue
        if expr and s[0] in "+-*/":
            expr = f"({expr}) {s}"
        elif expr:
            expr = f"{expr} {s}"
        else:
            expr = s
    if not expr or not re.search(r"[a-z]", expr) or not re.search(r"[+\-*/]", expr):
        return None
    from ..calc import CalcError, parse_expr

    try:
        parse_expr(expr)
    except CalcError:
        return None
    return expr


def parse_calcs(t: str, ctx: NLUContext) -> list[dict] | dict | None:
    """Возвращает список описаний вычислений или dict с уточнением."""
    metric = (ctx.switches or {}).get("metric") or "amount"
    fact_col, plan_col = ("qty", "plan_qty") if metric == "qty" else ("fact", "plan")
    calcs: list[dict] = []
    label_m = re.search(r"(?:назови|назвать|под названием|с названием|озаглавь)\w*\s+(?:ее|его|колонку)?\s*[«\"“]?([^»\"”]+)", t)
    custom_label = label_m.group(1).strip() if label_m else None

    explicit_formula = re.search(r"минус|плюс|делить|умножить|деленн|умноженн|[a-zа-я]\s*[-+*/]\s*[a-zа-я0-9(]|формул", t)
    if explicit_formula and not re.search(r"отклонени\w* (?:факта )?от плана\s*$", t):
        expr = parse_formula(t, ctx.calcs)
        if expr:
            unit = "%" if re.search(r"\*\s*100\s*\)?\s*$", expr) else ("₽" if re.search(r"fact|plan|discount", expr) and "/" not in expr else "")
            from ..calc import humanize_expr

            label = custom_label or ("Расчёт: " + humanize_expr(expr, ctx.calcs))
            calcs.append({"kind": "expr", "expr": expr, "label": label[:80] + (", %" if unit == "%" and not label.endswith("%") and custom_label else ""), "unit": unit})
            return calcs

    if re.search(r"отклонени|разниц|недовыполнени|перевыполнени", t):
        both = re.search(r"абсолютн\w* и (?:в )?процент|и в процентах|рубл\w* и процент|в рублях и в %", t)
        pct = re.search(r"процент|%|относительн", t)
        if both or not pct:
            calcs.append({"kind": "expr", "expr": f"{fact_col} - {plan_col}", "unit": "шт" if metric == "qty" else "₽",
                          "label": custom_label or ("Отклонение от плана, шт" if metric == "qty" else "Отклонение от плана, ₽")})
        if both or pct:
            calcs.append({"kind": "expr", "expr": f"({fact_col} - {plan_col}) / {plan_col} * 100", "unit": "%",
                          "label": "Отклонение от плана, %"})
        return calcs
    if re.search(r"(?:процент|%|доля)\w* выполнени|выполнени\w* план|исполнени\w* план", t):
        calcs.append({"kind": "expr", "expr": f"{fact_col} / {plan_col} * 100", "unit": "%",
                      "label": custom_label or "Выполнение плана, %"})
        return calcs
    if re.search(r"динамик|прирост|изменени\w* к|рост\w* к|м/м|г/г|\bmom\b|\byoy\b|к предыдущ|к прошл|год к году|месяц к месяцу", t):
        if re.search(r"год к году|г/г|yoy|к прошл\w* году|к аналогичн|год назад|к предыдущ\w* году", t):
            lag = "year"
        elif re.search(r"месяц|м/м|mom", t):
            lag = "month"
        else:
            lag = "prev"
        mode = "abs" if re.search(r"абсолютн|в рублях|в штуках|разниц", t) else "pct"
        measure = _measure_in(re.sub(r"динамик\w*|прирост\w*", "", t), fact_col)
        calcs.append({"kind": "change", "measure": measure, "lag": lag, "mode": mode, **({"label": custom_label} if custom_label else {})})
        return calcs
    if re.search(r"\bдол[юияей]\b|долей|удельн\w* вес|вклад", t):
        by = _dim_after(t, r"(?:кажд\w*|дол[юияей]|долей|вклад)") or next((d for d in find_dims(t) if d != "period"), None)
        if not by and ctx.switches.get("dim"):
            return {"clarify": "share_dim"}
        measure = _measure_in(re.sub(r"дол[юияей]|долей", "", t), fact_col)
        calcs.append({"kind": "share", "measure": measure, "by": by, **({"label": custom_label} if custom_label else {})})
        return calcs
    if re.search(r"накопительн|нарастающ|кумулятив|с начала периода", t):
        measure = _measure_in(re.sub(r"накопительн\w*|нарастающ\w*|итог\w*", "", t), fact_col)
        calcs.append({"kind": "cumsum", "measure": measure})
        if measure == fact_col and re.search(r"план", t) and not re.search(r"только факт", t):
            calcs.append({"kind": "cumsum", "measure": plan_col})
        return calcs
    if re.search(r"ранг|ранжир|место\b|рейтинг", t):
        by = _dim_after(t, r"(?:ранг\w*|ранжир\w*|рейтинг\w*|место)") or next((d for d in find_dims(t)), None) or ctx.switches.get("dim")
        measure = _measure_in(re.sub(r"ранг\w*|ранжир\w*", "", t), fact_col)
        desc = not re.search(r"по возрастани|от меньш|с конца", t)
        calcs.append({"kind": "rank", "measure": measure, "by": by, "desc": desc})
        return calcs
    stat = None
    if re.search(r"медиан", t):
        stat = "median"
    elif re.search(r"средн", t):
        stat = "mean"
    elif re.search(r"минимальн|\bmin\b|минимум", t):
        stat = "min"
    elif re.search(r"максимальн|\bmax\b|максимум", t):
        stat = "max"
    if stat:
        if re.search(r"средн\w* чек", t):
            calcs.append({"kind": "expr", "expr": "fact / deals", "unit": "₽", "label": "Средний чек, ₽"})
            return calcs
        by = _dim_after(t, r"(?:по|внутри|для кажд\w*|в кажд\w*|в разрезе)") or ctx.switches.get("dim")
        measure = _measure_in(re.sub(r"средн\w*|медиан\w*|минимальн\w*|максимальн\w*", "", t), fact_col)
        calcs.append({"kind": "group_stat", "stat": stat, "measure": measure, "by": by})
        return calcs
    expr = parse_formula(t, ctx.calcs)
    if expr:
        from ..calc import humanize_expr

        calcs.append({"kind": "expr", "expr": expr, "unit": "%" if expr.rstrip(") ").endswith("100") else "",
                      "label": custom_label or ("Расчёт: " + humanize_expr(expr, ctx.calcs))})
        return calcs
    return None


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------


def chart_index(t: str) -> int | str | None:
    for n, pattern in ORDINALS.items():
        if re.search(r"(?:\b(?:" + pattern + r")\s+(?:график\w*|диаграмм\w*|гистограмм\w*|визуализаци\w*))", t):
            return n
    m = re.search(r"график\w*\s*(?:№|номер)?\s*(\d)\b", t)
    if m:
        return int(m.group(1))
    if re.search(r"все графики|всех графиков|каждый график", t):
        return "all"
    if re.search(r"последн\w* график|этот график|текущ\w* график|его\b", t):
        return "last"
    return None


def sort_target(t: str, ctx: NLUContext) -> str | None:
    fragment = t.split("сортир", 1)[1] if "сортир" in t else t
    fragment = fragment.split(" по ", 1)[1] if " по " in fragment else fragment
    calcs = ctx.calcs

    def calc_like(*needles: str) -> str | None:
        for calc in reversed(calcs):
            lbl = norm(calc["label"])
            if any(n in lbl for n in needles):
                return calc["id"]
        return None

    if re.search(r"процент\w* выполнени|выполнени\w*|% выполн", fragment):
        return calc_like("выполнение") or "exec_pct"
    if re.search(r"отклонени", fragment):
        if re.search(r"процент|%", fragment):
            return calc_like("отклонение от плана, %") or "deviation_pct"
        return calc_like("отклонение от плана, ₽", "отклонение") or "deviation"
    if re.search(r"динамик|прирост", fragment):
        return calc_like("динамика")
    if re.search(r"дол[еияю]", fragment):
        return calc_like("доля")
    if re.search(r"ранг", fragment):
        return calc_like("ранг")
    if re.search(r"скидк", fragment):
        return "discount"
    if re.search(r"количеств|штук", fragment):
        return "qty"
    if re.search(r"план", fragment):
        return "plan"
    if re.search(r"факт|объем|продаж|выручк", fragment):
        return "fact"
    for calc in calcs:
        first = _stem_word(norm(calc["label"]).split()[0])
        if first and first in fragment:
            return calc["id"]
    return None


def chart_type_from(t: str) -> dict:
    if re.search(r"тепло\w* карт|heatmap|хитмап", t):
        return {"type": "exec_line", "variant": "heatmap"}
    if re.search(r"кругов|пирог|пай|donut|бублик|кольцев", t):
        return {"type": "structure"}
    if re.search(r"точечн|scatter|рассеян", t):
        return {"type": "scatter_planfact"}
    if re.search(r"линейн|линией|линии|линия|тренд", t):
        return {"type": "exec_line", "variant": None}
    if re.search(r"горизонтальн|гистограмм|рейтинг", t):
        return {"type": "top"}
    if re.search(r"столбчат|столбик|столбц", t):
        return {"type": "planfact_period"}
    return {}


def parse_chart_changes(t: str, ctx: NLUContext) -> dict:
    changes: dict[str, Any] = {}
    if "сортир" in t or re.search(r"упорядоч|расположи", t):
        target = sort_target(t, ctx)
        if target:
            changes["sort_by"] = target
        changes["sort_dir"] = "asc" if re.search(r"возрастани|от меньш|снизу|худш|отстающ|сначала мал", t) else "desc"
    m = re.search(r"\bтоп[- ]?(\d+|[а-я]+)", t) or re.search(r"(?:первы\w*|лучши\w*|худши\w*) (\d+|[а-я]+)", t)
    if m:
        n = to_int(m.group(1))
        if n:
            changes["top_n"] = n
            if re.search(r"худши\w*|отстающ", t):
                changes["sort_dir"] = "asc"
    if re.search(r"с учетом|учти|учитывая|на основе|по новой|используя|на новую|через", t):
        target = None
        if re.search(r"нов\w* колонк|нов\w* вычислен|последн\w* (колонк|вычислен)|добавленн", t):
            target = ctx.calcs[-1]["id"] if ctx.calcs else None
        else:
            target = sort_target("сортир по " + t.split("учет", 1)[-1], ctx)
        if target:
            changes["value"] = target
        elif ctx.calcs:
            changes["use_last_calc"] = True
    dims = [d for d in find_dims(t) if d != "period"]
    if dims and re.search(r"\bпо\s+(?:" + "|".join(p for _, p in DIM_PATTERNS) + ")", t):
        changes["dim"] = dims[0]
    if re.search(r"по кварталам|поквартальн", t):
        changes["grain"] = "quarter"
    elif re.search(r"по месяцам|помесячн", t):
        changes["grain"] = "month"
    elif re.search(r"по годам", t):
        changes["grain"] = "year"
    if re.search(r"в процентах", t) and "value" not in changes:
        changes["view"] = "pct"
    elif re.search(r"абсолютн", t):
        changes["view"] = "abs"
    changes.update(chart_type_from(t))
    return changes


# ---------------------------------------------------------------------------
# Главный разбор
# ---------------------------------------------------------------------------

COMMAND_VERBS = r"(?:перестрой|покажи|сохрани|отсортируй|добавь|посчитай|сделай|переключи|отфильтруй|оставь|открой|удали|убери|скачай|выгрузи|переименуй|построй|рассчитай)"


def split_compound(t: str) -> list[str]:
    parts = re.split(r";\s*|\s+(?:а )?(?:потом|затем|после этого)\s+|\.\s+(?=[а-я])", t)
    result: list[str] = []
    for part in parts:
        sub = re.split(r",?\s+и\s+(?=" + COMMAND_VERBS + r"\b)", part)
        result.extend(s.strip(" ,.") for s in sub if s.strip(" ,."))
    return result or [t]


def parse(text: str, ctx: NLUContext) -> ParseResult:
    t = norm(text)
    if not t:
        return ParseResult([], False)
    parts = split_compound(t)
    if len(parts) > 1:
        actions: list[dict] = []
        confident = True
        for part in parts:
            # кавычки ищем в исходном тексте соответствующего куска
            res = parse_single(_orig_fragment(text, part), part, ctx)
            actions.extend(res.actions)
            confident = confident and res.confident
            # следующий кусок видит результат предыдущего (например, новую колонку)
            _advance_context(ctx, res.actions)
        return ParseResult(actions, confident and bool(actions))
    return parse_single(text, t, ctx)


def _orig_fragment(text: str, part: str) -> str:
    q = quoted(text)
    return part if not q else part + " " + " ".join(f"«{x}»" for x in q if norm(x) in part)


def _advance_context(ctx: NLUContext, actions: list[dict]) -> None:
    for action in actions:
        if action["type"] == "add_calc":
            for calc in action["calcs"]:
                ctx.calcs = ctx.calcs + [{"id": f"__pending_{len(ctx.calcs)}", "label": calc.get("label") or _calc_label_hint(calc), **calc}]
        if action["type"] == "build":
            ctx.has_dashboard = True


def _calc_label_hint(calc: dict) -> str:
    return {"change": "Динамика", "share": "Доля", "cumsum": "Накопительный итог", "rank": "Ранг",
            "group_stat": "Среднее"}.get(calc["kind"], "Расчёт")


THANKS = {"спасибо", "спс", "благодарю", "thanks", "thank", "you", "отлично", "супер", "класс", "круто", "понятно", "ясно",
          "ок", "окей", "ok", "хорошо", "большое", "огромное", "отличная", "работа", "здорово", "молодец", "принято", "да"}

FOLLOW_STOP = {"а", "и", "ну", "теперь", "за", "в", "во", "на", "тот", "та", "же", "то", "самое", "покажи", "давай", "если", "как",
               "насчет", "период", "это", "тоже", "так", "что", "с", "до", "по", "тогда", "еще", "ещё", "сделай", "мне", "нужно", "надо"}
PERIOD_WORD = re.compile(r"^(?:последн|прошл|предыдущ|текущ|этот|этом|начал|все|всё|время|истори|месяц|мес|квартал|кв|год|лет|полгод|полугод"
                         r"|янв|фев|мар|апр|ма[йяе]|июн|июл|авг|сен|окт|ноя|дек|q\d|\d+|перв|втор|трет|четверт|один|два|две|три|четыр|пят|шест|двенадцат|ytd)")


def is_period_only(t: str) -> bool:
    words = re.findall(r"[а-яa-z0-9]+", t)
    return bool(words) and all(w in FOLLOW_STOP or PERIOD_WORD.match(w) for w in words)


QUESTION_START = re.compile(r"^(?:а\s+|ну\s+|и\s+)?(?:какой|какая|какие|какое|каков\w*|кто|где|сколько|чей|чья|у кого|в каком|на каком|насколько|на сколько)\b")
BEST_RE = r"лучш|лидер|больше всего|максимальн|самы[йея] (?:больш|крупн|успешн|сильн|прибыльн|продаваем)|сильне|перв\w* мест|топ[- ]?1\b|рекорд"
WORST_RE = r"худш|отста|слабе|меньше всего|минимальн|самы[йея] (?:слаб|плох|мал)|аутсайдер|провал|не выполн|недовыполн|последн\w* мест|проседа"


def parse_question(text: str, t: str, ctx: NLUContext, period: dict | None, ents: dict[str, list[str]]) -> dict | None:
    """Вопросы, на которые нужен ответ числом или именем: «какой регион лучший?», «сколько продали за сентябрь?»."""
    ents = {k: v for k, v in ents.items() if not (k == "client" and v and v[0].startswith("?"))}
    dims = [d for d in find_dims(t) if d != "period"]
    base = {"type": "answer", "period": period, "query": text.strip()}
    if re.search(r"скидк|скидок|причин", t):
        base["measure"] = "discount"
    elif re.search(r"штук|количеств|единиц|\bшт\b", t):
        base["measure"] = "qty"
    elif re.search(r"выполнени|план", t) and not re.search(r"сколько", t):
        base["measure"] = "exec"
    first_ent = next(iter(ents.items()), None)

    if first_ent and first_ent[0] == "client" and len(first_ent[1]) > 1:
        return {"type": "clarify", "question": "Нашёл несколько клиентов с таким названием. Кого показать?",
                "options": [{"label": c, "action": {**base, "kind": "entity", "field": "client", "value": c}} for c in first_ent[1][:6]]}
    # «как дела у Урала?», «что с напитками?»
    if first_ent and re.search(r"^(?:а\s+|ну\s+)?(?:как\s+(?:дела|там|обсто\w+|себя|идут|у нас)|что\s+(?:с|со|у|по)\b|как\s+)", t) \
            and not re.search(BEST_RE + "|" + WORST_RE + r"|сколько", t):
        return {**base, "kind": "entity", "field": first_ent[0], "value": first_ent[1][0]}

    # «доля промо-акций в скидках», «какой процент продаж даёт Москва»
    if first_ent and re.search(r"\bдол[яиюе]\b|какой процент|сколько процентов|какую часть|удельн\w* вес|вклад", t) \
            and not re.search(r"кажд\w*|посчитай|добавь|колонк", t):
        return {**base, "kind": "share", "field": first_ent[0], "value": first_ent[1][0]}

    question = QUESTION_START.search(t) or t.endswith("?")
    if not question:
        return None
    if re.search(r"^(?:а\s+)?сколько\b|какова?\w* (?:выручк|сумм|объем)|какой (?:факт|план|объем|оборот)", t):
        lagging = bool(re.search(r"не выполн|отста|ниже плана|недовыполн|провал", t))
        if lagging:
            what = "client" if re.search(r"клиент", t) else "product" if re.search(r"продукт|товар", t) else \
                "region" if re.search(r"регион", t) else "category" if re.search(r"категори", t) else "outlet"
            return {**base, "kind": "count", "what": what, "lagging": True}
        if re.search(r"клиент|покупател", t):
            kind, what = "count", "client"
        elif re.search(r"точ\w*|магазин", t):
            kind, what = "count", "outlet"
        elif re.search(r"продукт|товар|позици|sku", t):
            kind, what = "count", "product"
        elif re.search(r"сделок|сделк|заказ", t):
            kind, what = "total", "deals"
        elif re.search(r"скидк|скидок", t):
            kind, what = "total", "discount"
        elif re.search(r"штук|единиц|количеств|\bшт\b", t):
            kind, what = "total", "qty"
        elif re.search(r"\bплан", t) and not re.search(r"факт|продал|заработ", t):
            kind, what = "total", "plan"
        else:
            kind, what = "total", "fact"
        action = {**base, "kind": kind, "what": what}
        if first_ent:
            action.update(field=first_ent[0], value=first_ent[1][0])
        return action
    grow = re.search(r"раст\w*|рост|прирост|быстрее|ускор", t)
    fall = re.search(r"пада\w*|снижа\w*|сокраща\w*|пада|пр[оа]сел|упал", t)
    if grow or fall:
        return {**base, "kind": "bottom" if fall and not grow else "top", "measure": "growth", "dim": dims[0] if dims else None}
    if re.search(r"^(?:а\s+)?кому\b", t) and "client" not in dims:
        dims = ["client"] + dims
    best = re.search(BEST_RE, t)
    worst = re.search(WORST_RE, t)
    if best or worst:
        action = {**base, "kind": "bottom" if worst and not best else "top", "dim": dims[0] if dims else None}
        m = re.search(r"\bтоп[- ]?(\d+)|(\d+)\s+(?:лучш|худш)", t)
        if m:
            action["n"] = int(m.group(1) or m.group(2))
        if re.search(r"выручк|продаж|объем|оборот|факт", t) and not re.search(r"выполнени|план", t):
            action["measure"] = action.get("measure") if action.get("measure") in ("discount", "qty") else "fact"
        if first_ent and first_ent[0] != action.get("dim"):
            action.update(field=first_ent[0], value=first_ent[1][0])
        return action
    return None


def unknown_entity_filter(t: str, ctx: NLUContext) -> dict | None:
    """«оставь только сыр» — ищем слово по всем справочникам, если прямого совпадения нет."""
    m = re.search(r"(?:только|оставь|отфильтруй(?: по)?|выбери)\s+(.+)$", t)
    if not m:
        return None
    fragment = re.sub(r"\b(?:только|регион\w*|категори\w*|продукт\w*|товар\w*|точк\w*|город\w*|клиент\w*)\b", " ", m.group(1))
    words = [w for w in re.findall(r"[а-яa-z0-9-]+", fragment) if len(w) > 2 and w not in FOLLOW_STOP]
    if not words:
        return None
    stems = [_stem_word(w) for w in words]
    cat = ctx.catalog or {}
    for fld in ("product", "category", "region", "city", "outlet_type", "reason", "segment", "client", "outlet"):
        if ctx.columns and fld not in ctx.columns and not (fld in ("city", "outlet_type") and "outlet" in ctx.columns):
            continue
        hits = [v for v in cat.get(fld, []) if all(s in norm(v) for s in stems)]
        if hits and len(hits) <= 40:
            return {"type": "set_filter", "field": fld, "values": hits, "mode": "add" if re.search(r"добавь", t) else "only"}
    return {"type": "reply", "text": f"Не нашёл «{' '.join(words)}» ни среди продуктов, ни среди категорий, регионов, городов и клиентов. "
                                     f"Категории: {', '.join(cat.get('category', []))}. Откройте фильтр над графиками — в нём работает поиск."}


def parse_single(text: str, t: str, ctx: NLUContext) -> ParseResult:  # noqa: C901 — это и есть «словарь» бота
    q = quoted(text)

    # --- справка ---
    if re.search(r"^(?:помощь|help|справка|\?|что ты умеешь|что умеешь|как (?:тобой )?пользоваться|примеры|что можно спросить)", t):
        return ParseResult([{"type": "help"}])
    if re.fullmatch(r"(?:привет|здравствуй\w*|добрый (?:день|вечер)|доброе утро|hi|hello)[!. ]*", t):
        return ParseResult([{"type": "help", "greeting": True}])
    if re.search(r"(?:какие|что за|каки\w+)\s+(?:\w+\s+)?(?:дашборд\w*|отчет\w*|график\w*|вопрос\w*|анализ\w*|визуализац\w*)\s+(?:ты\s+)?(?:можешь|умеешь|можно|бывают|строишь|собираешь|делаешь)"
                 r"|что (?:ты )?(?:можешь|умеешь)|что можно (?:спросить|сделать|посмотреть)|какие у тебя (?:возможности|функции)|с чего начать|как (?:с тобой|тобой) работать", t):
        return ParseResult([{"type": "help"}])
    if re.search(r"(?:какие|что за|какими|что у тебя за)\s+(?:\w+\s+)?(?:данн\w*|таблиц\w*|баз\w*)|что (?:есть|лежит|хранится) в (?:базе|данных)|откуда (?:ты берешь )?данные"
                 r"|за какой период (?:есть |у тебя )?данные|опиши данные|что в базе|какая база", t) and not re.search(r"выгрузи|скачай|покажи данные", t):
        return ParseResult([{"type": "about_data"}])
    words = re.findall(r"[а-яa-z]+", t)
    if words and len(words) <= 5 and all(w in THANKS for w in words):
        return ParseResult([{"type": "reply", "text": "Пожалуйста! Продолжим? Например: «Сохрани дашборд», «Добавь колонку с процентом выполнения плана» или «Кто отстаёт?»."}])

    # --- экспорт ---
    if re.search(r"скача|выгрузи|экспорт|сохрани\w* (?:в|как) (?:файл|html|csv|excel|эксель|xlsx|картинк|png|pdf|пдф)|отправь файлом|дай файл|\bв (?:pdf|пдф)\b", t):
        fmt = None
        if re.search(r"html|хтмл|веб|страниц", t):
            fmt = "html"
        elif re.search(r"excel|эксел|xlsx|xls", t):
            fmt = "xlsx"
        elif re.search(r"csv|цсв", t):
            fmt = "csv"
        elif re.search(r"картинк|изображени|png|скриншот|jpe?g|снимок", t):
            fmt = "png"
        elif re.search(r"pdf|пдф", t):
            fmt = "pdf"
        elif re.search(r"таблиц|данны", t):
            fmt = "xlsx"
        elif re.search(r"дашборд|отчет", t):
            fmt = "html"
        if fmt:
            return ParseResult([{"type": "export", "format": fmt}])

    # --- ссылка ---
    if re.search(r"поделит|поделись|ссылк\w* (?:на|для)|опубликуй|дай ссылку|share", t):
        return ParseResult([{"type": "share"}])

    # --- управление сохранёнными дашбордами ---
    if re.search(r"(?:мои|все|список|перечень|сохраненн\w*)\s+(?:\w+\s+)?(?:дашборд|отчет)|дашборды\s*$|какие (?:есть|у меня) дашборд", t) \
            and not re.search(r"найди|поиск|ищи", t):
        sort = "name" if re.search(r"по (?:названи|алфавит|имени)", t) else ("created" if re.search(r"по дате созд", t) else "updated")
        return ParseResult([{"type": "list", "sort": sort}])
    m = re.search(r"(?:найди|поищи|ищи|поиск)\s+(?:мне\s+)?(?:дашборд\w*|отчет\w*)?\s*(?:про|по|о|об|с|где|на тему)?\s*(.*)$", t)
    if m and re.search(r"дашборд|отчет", t):
        query = (q[0] if q else m.group(1)).strip(" ?.!")
        return ParseResult([{"type": "list", "query": query}])
    if (re.search(r"восстанови\w*", t) and re.search(r"дашборд|отчет|удален", t)) or \
            re.fullmatch(r"(?:восстанови\w*|верни (?:его|обратно|удаленн\w*)|отмени удаление)[!. ]*", t):
        return ParseResult([{"type": "restore", "name": q[0] if q else None}])
    if re.search(r"переименуй|переименовать|смени\w* название|измени\w* название|назови дашборд", t):
        if len(q) >= 2:
            return ParseResult([{"type": "rename", "name": q[0], "new_name": q[-1]}])
        m = re.search(r"\s(?:в|на)\s+[«\"“]?(.+?)[»\"”]?\s*$", text.strip())
        new_name = q[-1] if q else (m.group(1).strip() if m else None)
        if new_name:
            return ParseResult([{"type": "rename", "name": None, "new_name": new_name}])
        return ParseResult([{"type": "clarify", "question": "Как назвать дашборд?", "options": [],
                             "pending": {"type": "rename", "slot": "new_name"}}])
    if re.search(r"(?:удали|удалить|сотри)\s+(?:\w+\s+)?(?:дашборд|отчет)", t):
        m = re.search(r"(?:дашборд|отчет)\w*\s+(.+)$", t)
        name = q[0] if q else (m.group(1).strip(" .") if m else None)
        return ParseResult([{"type": "delete", "name": name}])
    if re.search(r"(?:сделай|создай|сохрани)\s+(?:\w+\s+)?копи|скопируй|дублируй|продублируй|дубликат", t):
        new_name = None
        m = re.search(r"(?:назови|под названием|как)\s+[«\"“]?([^»\"”]+)", text)
        if m:
            new_name = m.group(1).strip()
        return ParseResult([{"type": "copy", "name": q[0] if q and not new_name else None, "new_name": new_name}])
    if re.search(r"^(?:открой|загрузи|верни|покажи|вернись к)\s+(?:мне\s+)?(?:сохраненн\w*\s+)?(?:дашборд|отчет)", t) and (q or re.search(r"^(?:открой|загрузи|вернись)", t)):
        m = re.search(r"(?:дашборд|отчет)\w*\s+(.+)$", t)
        name = q[0] if q else (m.group(1).strip(" .") if m else None)
        if name and not re.search(r"^(?:по|про|за)\s", name or ""):
            return ParseResult([{"type": "open", "name": name}])
        if name:
            return ParseResult([{"type": "list", "query": re.sub(r"^(?:по|про)\s", "", name)}])
    if re.search(r"^открой\s", t) and q:
        return ParseResult([{"type": "open", "name": q[0]}])
    m = re.fullmatch(r"(?:открой|загрузи)\s+(?!таблиц|данн|график|список)(.{2,80}?)[!. ]*", t)
    if m and not re.search(r"продаж|скидк|план|\bза\s|\bпо\s", t):
        return ParseResult([{"type": "open", "name": m.group(1)}])
    if re.search(r"\bсохрани|\bсохранить|запомни дашборд", t):
        name = q[0] if q else None
        if not name:
            m = re.search(r"(?:как|под названием|под именем|с названием|назови)\s+(?!нов)(.+)$", text.strip(), re.IGNORECASE)
            if m:
                name = m.group(1).strip(" .«»\"")
        as_new = bool(re.search(r"как нов\w*|новым|отдельн\w*|как копи\w*|в новый", t))
        force = bool(re.search(r"перезапиш|замени", t))
        return ParseResult([{"type": "save", "name": name, "as_new": as_new, "force": force}])

    # --- история и отмена ---
    m = re.search(r"(?:откати|верни|восстанови)\w*\s+(?:к|на|до)?\s*верси\w*\s*(?:№\s*)?(\d+)", t)
    if m:
        return ParseResult([{"type": "revert", "version": int(m.group(1))}])
    if re.search(r"истори\w* (?:версий|изменений|правок)|какие (?:есть )?версии|что (?:уже )?(?:добавлено|менялось)", t) or re.fullmatch(r"(?:покажи\s+)?истори\w*[!?. ]*", t):
        return ParseResult([{"type": "history"}])
    if re.search(r"обнови данные|перезагрузи данные|освежи|актуализируй|подтяни свеж", t):
        return ParseResult([{"type": "refresh"}])
    if re.search(r"(?:убери|удали|отмени|верни без)\s+(?:последн\w*\s+|эт[уоиа]\w*\s+|нов\w*\s+|е[её]\s+)?(?:колонк|столбец|столбц|вычислени|расчет)", t):
        ref = q[0] if q else ("last" if re.search(r"последн|\bэт[уоиа]|\bнов", t) or not re.search(r"(?:колонк|столбец|вычислени|расчет)\w*\s+\S", t) else
                              re.sub(r".*(?:колонк|столбец|столбц|вычислени|расчет)\w*\s+(?:с\s+)?", "", t))
        return ParseResult([{"type": "remove_calc", "ref": ref}])
    if re.search(r"^(?:отмени|откати|undo|верни как было|назад$|шаг назад)", t):
        return ParseResult([{"type": "undo"}])

    if re.search(r"\bsql\b|скл\b|какой запрос|запрос к баз|как (?:ты )?(?:достал|получил|посчитал) данн", t) and not re.search(r"напиши|выполни", t):
        return ParseResult([{"type": "show_sql"}])
    if re.search(r"прибыл|марж|себестоим|рентабельн|наценк|валов\w* доход", t):
        return ParseResult([{"type": "reply", "text": "В данных кейса нет себестоимости и затрат, поэтому прибыль и маржу посчитать не из чего. "
                                                     "Есть план, факт, количество, скидки и число сделок — например: «Посчитай долю скидок от продаж» "
                                                     "или формула словами: «факт минус скидка»."}])
    if re.search(r"(?:поменяй|смени|измени)\s+(?:\w+\s+)?цвет", t):
        return ParseResult([{"type": "reply", "text": "Цвета закреплены за сущностями (регион «Центральный» везде одного цвета) и подобраны так, "
                                                     "чтобы их различали и люди с нарушением цветового зрения. Можно поменять вид графика: "
                                                     "«сделай четвёртый график тепловой картой» или «сделай первый график круговым»."}])
    if re.search(r"(?:побольше|крупнее|больше размер|на весь экран|на всю ширину|разверни|увеличь)", t) and re.search(r"график|диаграмм", t):
        return ParseResult([{"type": "ui_info", "topic": "enlarge"}])

    # --- вопросы про интерфейс (фильтры/переключатели уже есть) ---
    if re.search(r"(?:добавь|сделай|нужен|включи)\s+(?:\w+\s+)?фильтр", t) and not re.search(r"только|оставь", t):
        field = next((d for d in find_dims(t) if d != "period"), None)
        topic = "cascade" if re.search(r"каскад|зависим|регион\w*\s*(?:->|→|-)\s*точ", t) else "filter"
        return ParseResult([{"type": "ui_info", "topic": topic, "text": field or ""}])
    if re.search(r"(?:добавь|сделай|нужен|включи)\s+(?:\w+\s+)?переключател", t):
        if re.search(r"план\s*/\s*факт|план[- ]факт|план и факт|план или факт", t):
            return ParseResult([{"type": "ui_info", "topic": "series"}])
        if re.search(r"период|месяц|квартал|год", t):
            return ParseResult([{"type": "ui_info", "topic": "grain"}])
        return ParseResult([{"type": "ui_info", "topic": "switch"}])
    if re.search(r"итогов\w* цифр|kpi|кпи|ключев\w* (?:показател|метрик)|карточк", t) and not re.search(r"собери|построй", t):
        return ParseResult([{"type": "ui_info", "topic": "kpi"}])
    if re.search(r"каскадн", t):
        return ParseResult([{"type": "ui_info", "topic": "cascade"}])

    # --- добавление / изменение вычислений ---
    if re.search(r"(?:измени|поменяй|исправь|замени)\s+(?:\w+\s+)?формул", t):
        m = re.search(r"формул\w*\s+(.+?)\s+на\s+(.+)$", t)
        ref = q[0] if q else (m.group(1) if m else "last")
        new_part = m.group(2) if m else t
        new_ctx = NLUContext(**{**ctx.__dict__})
        calcs = parse_calcs("посчитай " + new_part, new_ctx)
        if isinstance(calcs, list) and calcs:
            return ParseResult([{"type": "modify_calc", "ref": ref, "calc": calcs[0]}])
    is_calc_verb = re.search(r"^(?:добавь|посчитай|рассчитай|вычисли|сделай|создай|выведи|прибавь|дай|нужн\w*|хочу|а\s+|ранжируй|отранжируй|проранжируй)", t) \
        or re.search(r"\b(?:посчитай|рассчитай|вычисли)\b", t)
    not_calc = re.search(r"(?:добавь|сделай)\s+(?:\w+\s+)?(?:график|фильтр|переключател|разбивк|копи|дашборд)", t)
    if not is_calc_verb and ctx.has_dashboard and re.search(r"минус|делить на|умножить на|плюс", t) and parse_formula(t, ctx.calcs):
        is_calc_verb = True
    if is_calc_verb and CALC_TRIGGER.search(t) and not not_calc:
        parsed = parse_calcs(t, ctx)
        if isinstance(parsed, dict) and parsed.get("clarify") == "share_dim":
            return ParseResult([{"type": "clarify", "question": "Долю в каком разрезе посчитать?",
                                 "options": [{"label": lbl, "action": {"type": "add_calc", "calcs": [{"kind": "share", "measure": "fact", "by": d}]}}
                                             for d, lbl in (("region", "Регионов"), ("outlet", "Точек"), ("category", "Категорий"), ("product", "Продуктов"))]}])
        if isinstance(parsed, list) and parsed:
            actions: list[dict] = [{"type": "add_calc", "calcs": parsed}]
            # «…и перестрой график» в той же фразе
            if re.search(r"перестро|обнови график|покажи на график", t):
                idx = chart_index(t)
                actions.append({"type": "rebuild_chart", "chart": idx, "changes": {"use_last_calc": True}})
            return ParseResult(actions)

    # --- графики ---
    if re.search(r"(?:добавь|построй еще|добавить)\s+(?:\w+\s+)?(?:график|диаграмм|гистограмм|визуализац)", t):
        changes = parse_chart_changes(t, ctx)
        if "type" not in changes:
            changes["type"] = "exec_line" if re.search(r"выполнени|динамик", t) else "top"
        dims = [d for d in find_dims(t) if d != "period"]
        if dims and "dim" not in changes:
            changes["dim"] = dims[0]
        return ParseResult([{"type": "add_chart", "changes": changes}])
    if ctx.calcs and re.search(r"(?:покажи|выведи|добавь|отобрази|нарисуй|вынеси|построй|перенеси)\s+(?:\w+\s+)?(?:его|ее|её|это|их|эту колонку|этот показатель|нов\w* колонк\w*|это вычисление)\b", t) \
            and re.search(r"график|диаграмм|визуал|наглядно", t):
        return ParseResult([{"type": "rebuild_chart", "chart": chart_index(t), "changes": {"use_last_calc": True}}])
    if re.search(r"(?:убери|удали|скрой|спрячь)\s+(?:\w+\s+)?(?:график|диаграмм|гистограмм)", t):
        return ParseResult([{"type": "remove_chart", "chart": chart_index(t) or "last"}])
    idx = chart_index(t)
    rebuild_verb = re.search(r"перестро|отсортируй|упорядоч|пересобер|переделай|поменяй|измени|преврати|сделай|покажи|выведи|замени|обнови|переведи|разверни|отобрази", t)
    if (idx is not None and rebuild_verb) or re.search(r"перестро\w*\s+(?:\w+\s+)?(?:график|диаграмм)", t):
        changes = parse_chart_changes(t, ctx)
        if changes or re.search(r"перестро", t):
            if not changes and ctx.calcs:
                changes = {"use_last_calc": True}
            return ParseResult([{"type": "rebuild_chart", "chart": idx, "changes": changes}])
    if ctx.has_dashboard and re.search(r"отсортируй|упорядоч", t) and parse_period(t) is None:
        changes = parse_chart_changes(t, ctx)
        target_top = next((i + 1 for i, c in enumerate(ctx.charts) if c.get("type") in ("top", "bar_dim", "planfact_dim")), None)
        if target_top:
            return ParseResult([{"type": "rebuild_chart", "chart": target_top, "changes": changes}])
        return ParseResult([{"type": "show_table", "sort_by": changes.get("sort_by"), "sort_dir": changes.get("sort_dir", "desc"),
                             "top_n": changes.get("top_n")}])

    # --- таблица ---
    if re.search(r"таблиц|выгрузи данные|покажи данные|сырые данные|исходные данные|выгрузку", t) and not re.search(r"собери|дашборд", t):
        group_by = [d for d in find_dims(t) if re.search(r"\bпо\s+\w*" , t)] if re.search(r"(?:сгруппир|агрегир|свод|итоги)\w*|таблиц\w* по", t) else []
        changes = parse_chart_changes(t, ctx) if "сортир" in t else {}
        return ParseResult([{"type": "show_table", "group_by": group_by, "sort_by": changes.get("sort_by"),
                             "sort_dir": changes.get("sort_dir", "desc"), "top_n": changes.get("top_n")}])

    if re.search(r"^(?:удали|сотри|очисти|измени|исправь|вставь|запиши|обнули|перезапиши|drop|delete|update|insert)\b", t) \
            and re.search(r"продаж|данн|таблиц|запис|клиент|план|баз|строк|скидк", t):
        return ParseResult([{"type": "reply", "text": "Я работаю с базой только на чтение: могу выгрузить, посчитать и показать данные, "
                                                     "но не изменяю и не удаляю их. Могу убрать колонку-вычисление, фильтр или дашборд."}])

    period = parse_period(t)
    ents = entities(text, t, ctx)
    wants_new = re.search(r"собери|построй|сделай дашборд|создай дашборд|новый дашборд|дашборд по|дашборд про|отчет по|сравни(?! с прошл)|шаблон|стандартн", t)

    # --- продолжение разговора о текущем дашборде ---
    if ctx.has_dashboard and not wants_new:
        if period and is_period_only(t):  # «а за прошлый год?»
            return ParseResult([{"type": "set_period", "period": period}])
        m = re.fullmatch(r"(?:а|и|ну)?\s*(?:теперь\s+|если\s+)?(?:отсортируй\s+)?по\s+(выполнени\w*(?: плана)?|отклонени\w*|выручк\w*|продаж\w*|факт\w*|объем\w*|скидк\w*|количеств\w*)\s*\??", t)
        target_top = next((i + 1 for i, c in enumerate(ctx.charts) if c.get("type") in ("top", "bar_dim", "planfact_dim")), None)
        if m and target_top:
            word = m.group(1)
            sort_by = ("exec_pct" if word.startswith("выполн") else "deviation" if word.startswith("отклон")
                       else "discount" if word.startswith("скидк") else "qty" if word.startswith("количеств") else "fact")
            return ParseResult([{"type": "rebuild_chart", "chart": target_top, "changes": {"sort_by": sort_by, "sort_dir": "desc"}}])
        if re.search(r"сравни\w* с прошл\w* год|к прошлому году|год к году|против прошлого года|чем в прошлом году", t) and not re.search(r"добавь|посчитай|колонк", t):
            ents_ok = {k: v for k, v in ents.items() if not (k == "client" and v[0].startswith("?"))}
            first = next(iter(ents_ok.items()), None)
            action = {"type": "answer", "kind": "yoy", "query": text.strip()}
            if first:
                action.update(field=first[0], value=first[1][0])
            return ParseResult([action])
        fields_ok = not ctx.columns or all(f in ctx.columns or (f in ("city", "outlet_type") and "outlet" in ctx.columns) for f in ents)
        words = re.findall(r"[а-яa-z0-9]+", t)
        if ents and fields_ok and len(words) <= 7 and re.match(r"^(?:а|и|ну)?\s*(?:теперь\s+|если\s+|давай\s+)?(?:в|во|по|для|у|про)\s", t) \
                and not re.search(r"продаж|план|дашборд|структур|сравни|топ|динамик|график", t) and not QUESTION_START.search(t):
            return ParseResult([{"type": "set_filter", "field": f, "values": v, "mode": "only"} for f, v in ents.items()
                                if not (f == "client" and v[0].startswith("?"))])
    question = parse_question(text, t, ctx, period, ents)
    if question:
        return ParseResult([question])

    # --- фильтры на текущем дашборде ---
    if ctx.has_dashboard and not wants_new:
        if re.search(r"сбрось|сбросить|убери\w* (?:все )?фильтр|очисти\w* фильтр|без фильтр|сними\w* (?:все )?фильтр|(?:верни|покажи|по)\s+(?:вс[её]|всем)\b|все (?:регионы|точки|продукты|категории|клиенты|города|типы|причины|месяцы)", t):
            field = None
            for dim, word in (("region", "регион"), ("outlet", "точк|город|тип"), ("product", "продукт|товар"), ("category", "категори"), ("client", "клиент"), ("reason", "причин"), ("period", "период|месяц")):
                if re.search(r"(?:все|фильтр\w* по|фильтр\w*)\s+(?:\w+\s+)?(?:" + word + ")", t):
                    field = dim
            return ParseResult([{"type": "set_filter", "field": field, "values": [], "mode": "clear" if field else "clear_all"}])
        filter_verbs = re.search(r"только|оставь|отфильтруй|фильтр|выбери|исключи|кроме|без\s|убери|добавь|покажи|по\s|для\s", t)
        strong_filter = re.search(r"только|оставь|отфильтруй|фильтр|исключи|кроме", t)
        fields_ok = not ctx.columns or all(f in ctx.columns for f in ents)
        if ents and filter_verbs and fields_ok and (period is None or (strong_filter and not re.search(r"продаж|план|дашборд|скидк", t))):
            actions = []
            mode = "exclude" if re.search(r"исключи|кроме|без\s|убери", t) else ("add" if re.search(r"добавь|плюс|также|тоже", t) else "only")
            for field, values in ents.items():
                if field == "client" and values and values[0].startswith("?"):
                    continue
                if field == "city":
                    field_out, values_out = "city", values
                else:
                    field_out, values_out = field, values
                actions.append({"type": "set_filter", "field": field_out, "values": values_out, "mode": mode})
            if period and period.get("kind") in ("month", "quarter"):
                actions.append({"type": "set_filter", "field": "period", "values": [], "mode": "only", "period": period})
            if actions:
                return ParseResult(actions)
        if period and period.get("kind") in ("month", "quarter", "year") and re.search(r"только|оставь|отфильтруй|покажи", t) \
                and not re.search(r"продаж|план|дашборд|скидк|структур", t):
            return ParseResult([{"type": "set_filter", "field": "period", "values": [], "mode": "only", "period": period}])
        m = re.fullmatch(r"(?:покажи|оставь|верни)\s+(?:только\s+)?(план|факт|план и факт)[!. ]*", t)
        if m:
            return ParseResult([{"type": "set_switch", "series": {"план": "plan", "факт": "fact"}.get(m.group(1), "both")}])
        if not ents and period is None and strong_filter and not re.search(r"исключи|кроме", t):
            guess = unknown_entity_filter(t, ctx)
            if guess:
                return ParseResult([guess])

    # --- переключатели на текущем дашборде ---
    if ctx.has_dashboard and not wants_new and period is None:
        switch: dict[str, Any] = {}
        if re.search(r"переключ|разрез|покажи по|сделай по|в разрезе|\bпо\s+(?:" + "|".join(p for _, p in DIM_PATTERNS) + ")", t):
            dims = [d for d in find_dims(t) if d != "period"]
            if dims:
                switch["dim"] = dims[-1] if re.search(r"с\s+\w+\s+на\s", t) else dims[0]
        m = re.search(r"метрик\w*\s+(\w+)|покажи\s+(?:в\s+)?(скидк\w*|количеств\w*|штук\w*|сделк\w*|рубл\w*|объем\w*)|переключи\w*\s+на\s+(скидк\w*|количеств\w*|штук\w*|сделк\w*|объем\w*|рубл\w*)", t)
        if m or re.search(r"в штуках|в рублях", t):
            metric = find_metric(t)
            if metric:
                switch["metric"] = metric
        if re.search(r"разбивк\w* по кварт|по кварталам|поквартальн|квартальн", t):
            switch["grain"] = "quarter"
        elif re.search(r"по месяцам|помесячн|разбивк\w* по месяц", t):
            switch["grain"] = "month"
        elif re.search(r"по годам|разбивк\w* по год", t):
            switch["grain"] = "year"
        if re.search(r"в процентах|процентный вид|вид\w* процент", t):
            switch["view"] = "pct"
        elif re.search(r"абсолютн", t):
            switch["view"] = "abs"
        elif re.search(r"(?:переключи|покажи|вид)\w*\s+(?:на\s+)?динамик", t):
            switch["view"] = "dyn"
        if re.search(r"только план\b", t):
            switch["series"] = "plan"
        elif re.search(r"только факт\b", t):
            switch["series"] = "fact"
        elif re.search(r"план и факт|план-факт|оба ряда", t) and re.search(r"покажи|верни|переключ", t):
            switch["series"] = "both"
        if switch:
            return ParseResult([{"type": "set_switch", **switch}])
        m = re.search(r"\bтоп[- ]?(\d+|[а-я]+)", t)
        if m and to_int(m.group(1)):
            target_top = next((i + 1 for i, c in enumerate(ctx.charts) if c.get("type") in ("top", "bar_dim", "planfact_dim")), None)
            if target_top:
                changes = parse_chart_changes(t, ctx)
                dims = [d for d in find_dims(t) if d != "period"]
                if dims and "dim" not in changes:
                    changes["dim"] = dims[0]
                if "sort_by" not in changes:
                    if re.search(r"по (?:выручк|продаж|объем|факт|оборот)", t):
                        changes.update(sort_by="fact", sort_dir="desc")
                    elif re.search(r"по выполнени", t):
                        changes.update(sort_by="exec_pct", sort_dir="desc")
                    elif re.search(r"по отклонени", t):
                        changes.update(sort_by="deviation", sort_dir="desc")
                return ParseResult([{"type": "rebuild_chart", "chart": target_top, "changes": changes}])

    # --- построение ---
    subject = re.search(r"продаж|план|факт|выручк|дашборд|отчет|сводк|структур|динамик|выполнени|скидк|сравни|отстающ|"
                        r"клиент|регион|точк|продукт|категори|топ|рейтинг|объем|оборот|сделк|аналитик|визуализ|график|покажи|построй|собери", t)
    if subject or period:
        action = build_action(text, t, ctx, period, ents)
        confident = bool(re.search(r"продаж|план|факт|выручк|дашборд|скидк|клиент|структур|сравни|отстающ|топ|выполнени|динамик|объем|оборот|сделк", t))
        return ParseResult([action], confident)
    if ctx.has_dashboard:
        dims = [d for d in find_dims(t) if d != "period"]
        if dims:
            return ParseResult([{"type": "set_switch", "dim": dims[0]}], False)
    return ParseResult([], False)


def build_action(text: str, t: str, ctx: NLUContext, period: dict | None, ents: dict[str, list[str]]) -> dict:
    client_frag = extract_client_fragment(text, t)
    t_dims = t.replace(norm(client_frag), " ") if client_frag else t
    for name in (ents.get("client") or []):
        words = [w for w in clean_client_name(name).split() if len(w) > 2]
        if len(words) >= 2:
            t_dims = re.sub(r"\b" + re.escape(_stem_word(words[0])) + r"\w*\s+" + re.escape(_stem_word(words[1])) + r"\w*", " ", t_dims)
    dims = [d for d in find_dims(t_dims) if d != "period"]
    metric = find_metric(t)
    lagging = bool(re.search(r"отстающ|аутсайдер|худш|провал|не выполн|недовыполн|слаб", t))
    top_m = re.search(r"\bтоп[- ]?(\d+|[а-я]+)?", t)
    top_n = to_int(top_m.group(1)) if top_m and top_m.group(1) else None

    dataset = "sales"
    client_filter = ents.get("client")
    if re.search(r"скидк|скидок", t) and (re.search(r"почему|причин|какие скидки|скидки дела|скидок|по скидкам|про скидки|анализ скидок|скидки по|скидки за|^покажи скидки|дашборд\w* (?:по|про) скидк", t)
                                          or client_filter) and not re.search(r"план|выполнени", t):
        dataset = "discounts"
    elif ("client" in dims and not re.search(r"план|выполнени", t)) or (client_filter and not re.search(r"скидк", t)):
        dataset = "clients"

    if re.search(r"дашборд|сводк|собери|отчет|шаблон|стандартн|обзор|картин\w* по", t):
        intent = "dashboard"
    elif re.search(r"сравни|отстающ|аутсайдер|выдели|лидер", t):
        intent = "compare"
    elif re.search(r"структур|состав|доли|долю|распределени", t):
        intent = "structure"
    elif re.search(r"\bтоп\b|топ[- ]?\d|рейтинг|лучш\w* \d|худш\w* \d", t):
        intent = "top"
    elif re.search(r"выполнени\w* плана по (?:месяц|квартал|период)|динамик\w* выполнени", t):
        intent = "exec"
    elif re.search(r"план[- ]?факт|план и факт|план/факт|выполнени", t) and dims:
        intent = "planfact_dim"
    elif re.search(r"динамик|по месяцам|тренд|по кварталам|как менял", t):
        intent = "trend"
    elif not dims:
        intent = "dashboard"
    else:
        intent = "planfact_dim"
    if dataset != "sales" and intent not in ("dashboard",):
        intent = "dashboard" if dataset == "discounts" else ("chart" if intent in ("top", "planfact_dim") else "dashboard")

    structure_dim = None
    if intent in ("structure", "dashboard"):
        m = re.search(r"структур\w*\s+(?:\w+\s+)?по\s+(\w+)|дол\w+\s+(\w+)", t)
        if m:
            fragment = m.group(1) or m.group(2) or ""
            for dim, pattern in DIM_WORD_GEN.items():
                if re.search(pattern, fragment):
                    structure_dim = dim
        if intent == "structure" and not structure_dim and dims:
            structure_dim = dims[0]
    top_dim = None
    m = re.search(r"\bтоп[- ]?\d*\s+(\w+)", t)
    if m:
        for dim, pattern in DIM_WORD_GEN.items():
            if re.search(pattern, m.group(1)):
                top_dim = dim
    switch_dims = [top_dim] if top_dim else []
    switch_dims += [d for d in dims if d not in switch_dims and d != structure_dim]
    if not switch_dims and structure_dim:
        switch_dims = [structure_dim]
    if intent == "compare" and not switch_dims:
        switch_dims = ["outlet"] if re.search(r"точк|магазин", t) else ["region"]

    grain = None
    if re.search(r"по кварталам|поквартальн", t):
        grain = "quarter"
    elif re.search(r"по годам", t):
        grain = "year"
    elif re.search(r"по месяцам|помесячн", t):
        grain = "month"

    filters = {k: v for k, v in ents.items() if k != "client"}
    if dataset == "discounts" and metric == "discount":
        metric = None
    client = None
    if client_filter:
        client = client_filter[0][1:] if client_filter[0].startswith("?") else None
        if not client_filter[0].startswith("?"):
            filters["client"] = client_filter
    sort_by = "exec_pct" if dataset == "sales" and re.search(r"выполнени|план", t) and intent in ("compare", "top") and not lagging else None
    return {"type": "build", "dataset": dataset, "intent": intent, "period": period, "dims": switch_dims, "sort_by": sort_by,
            "metric": metric, "grain": grain, "top_n": top_n, "lagging": lagging, "structure_dim": structure_dim,
            "filters": filters, "client": client, "query": text.strip()}
