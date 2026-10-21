import argparse
import json
import re
import time

from articles import article_key, get_article, load_articles
from config import (
    REWRITE_REASONING_EFFORT,
    REWRITTEN_PATH,
    SPECIAL_PART_FIRST_ARTICLE,
)
from llm import ask_model

NOTE_PART = "примечание"

PART_PATTERN = re.compile(r"^(\d+)\.\s+")
REPEALED_POINT_PATTERN = re.compile(r"^[а-я](?:\.\d+)?\)\s*утратил", re.I)
POINT_LINE_PATTERN = re.compile(r"(?m)^([а-я](?:\.\d+)?)\)\s*")
ORDINALS = {
    "перв": "1", "втор": "2", "трет": "3", "четв": "4", "пят": "5",
    "шест": "6", "седьм": "7", "восьм": "8",
}
REFERENCE_PATTERN = re.compile(
    r"(?:част\w*\s+(перв|втор|трет|четв|пят|шест|седьм|восьм)\w*"
    r"(?:\s*(?:,|или|и)\s*\w+)*\s+)?"
    r"стат(?:ь\w*|ей)\s+"
    r"(\d+(?:\.\d+)*(?:\s*(?:,|-|–|и|или)\s*\d+(?:\.\d+)*)*)"
)
LEFTOVER_REFERENCE_PATTERN = re.compile(
    r"настоящ\w+ (?:статьи|статьей|кодекса|глав\w*)|част\w+ (?:перв|втор"
    r"|трет|четв|пят|шест|седьм)\w*|стат(?:ь\w*|ей)\s+\d",
    re.I,
)


def split_parts(article: dict) -> list[dict]:
    lines = article["text"].splitlines()
    parts: list[dict] = []
    current: dict | None = None

    for line in lines:
        if line.startswith("Примечани"):
            current = {"part": NOTE_PART, "lines": [line]}
            parts.append(current)
            continue

        match = PART_PATTERN.match(line)
        if match and (current is None or current["part"] != NOTE_PART):
            current = {"part": match.group(1), "lines": [line]}
            parts.append(current)
            continue

        if current is None:
            current = {"part": "", "lines": []}
            parts.append(current)
        current["lines"].append(line)

    result = []
    for part in parts:
        lines = [
            line for line in part["lines"]
            if not REPEALED_POINT_PATTERN.match(line)
        ]
        text = "\n".join(lines).strip()
        body = PART_PATTERN.sub("", text, count=1)
        body = re.sub(r"^Примечани\w*\.\s*", "", body).lower()
        if not text or body.startswith(("утратил", "утратила")):
            continue

        disposition, punishment = split_punishment(text)
        result.append({
            "id": f"{article['number']}_{part['part'] or '0'}",
            "article_number": article["number"],
            "title": article["title"],
            "chapter": article["chapter"],
            "part": part["part"],
            "text": text,
            "disposition": disposition,
            "punishment": punishment,
        })
    return result


def split_points(text: str) -> tuple[str, list[tuple[str, str]]]:
    matches = list(POINT_LINE_PATTERN.finditer(text))
    if not matches:
        return text.strip(), []
    lead = text[:matches[0].start()].strip()
    points = []
    for match, following in zip(matches, [*matches[1:], None]):
        end = following.start() if following else len(text)
        body = text[match.end():end].strip().rstrip(" ;,.-–—").strip()
        points.append((match.group(1), body))
    return lead, points


def point_letters(text: str) -> list[str]:
    return [letter for letter, _ in split_points(text)[1]]


def keeps_points(part: dict, text: str) -> bool:
    letters = point_letters(part["disposition"])
    return not letters or letters == point_letters(text)


def split_punishment(text: str) -> tuple[str, str]:
    match = re.search(r"(?m)^наказыва\w*\s.*", text, flags=re.S)
    if not match:
        return text, ""
    disposition = text[:match.start()].strip().rstrip("-").strip()
    return disposition, match.group(0).strip()


def referenced_articles(
    text: str, own_number: str
) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    for match in REFERENCE_PATTERN.finditer(text):
        group = match.group(0)
        numbers_text = match.group(2)
        part_numbers = [
            digit
            for stem, digit in ORDINALS.items()
            if re.search(rf"\b{stem}\w*", group.split("стат")[0])
        ]

        numbers: list[str] = []
        for start, end in re.findall(
            r"(\d+(?:\.\d+)*)\s*[-–]\s*(\d+(?:\.\d+)*)", numbers_text
        ):
            if "." not in start + end and 0 < int(end) - int(start) <= 10:
                numbers += [str(n) for n in range(int(start), int(end) + 1)]
        numbers += re.findall(r"\d+(?:\.\d+)*", numbers_text)

        for number in dict.fromkeys(numbers):
            if number == own_number or not get_article(number):
                continue
            for part in (part_numbers if len(numbers) == 1 else []) or [""]:
                if (number, part) not in refs:
                    refs.append((number, part))
    return refs


def reference_text(number: str, part: str) -> str:
    article = get_article(number)
    article_parts = split_parts(article)
    chosen = [p for p in article_parts if p["part"] == part] if part else []
    if not chosen:
        chosen = [p for p in article_parts if p["part"] in ("1", "")][:1]
    label = f", часть {part}" if part and chosen and chosen[0]["part"] == part else ""
    body = chosen[0]["disposition"] if chosen else ""
    return f"Статья {number}. {article['title']}{label}\n{body}"


INTERNAL_REFERENCE_PATTERN = re.compile(
    r"настоящ\w+ (?:статьи|статьей|кодекса|глав\w*)"
    r"|^(?:\d+\.\s*)?(?:То же|Те же|Деяни\w+,? предусмотр|Действи\w+,? предусмотр)",
    re.I | re.M,
)


def strip_sanction(text: str) -> str:
    text = re.split(r"[,\s]*\bнаказыва\w*", text, maxsplit=1)[0]
    return text.strip().rstrip(" ,;-–—").strip()


def has_references(part: dict) -> bool:
    return bool(
        INTERNAL_REFERENCE_PATTERN.search(part["disposition"])
        or REFERENCE_PATTERN.search(part["disposition"])
    )


def plain_disposition(part: dict) -> str:
    text = PART_PATTERN.sub("", part["disposition"], count=1)
    text = text.strip().rstrip(" ,;-–—").strip()
    return text + ("" if text.endswith(".") else ".")


def part_label(part: dict) -> str:
    if part["part"] == NOTE_PART:
        return "примечание"
    label = (
        f"часть {part['part']}" if part["part"]
        else "статья без деления на части"
    )
    if part.get("point"):
        label += f", пункт «{part['point']}»"
    return label


REWRITER_PROMPT = """
Ты — юрист-кодификатор, редактор текстов Уголовного кодекса РФ.
Твоя работа — превращать отдельную часть статьи в самостоятельный,
понятный без контекста текст для справочной базы.

Тебе дают:
- номер и название статьи;
- полный текст статьи (для понимания внутренних ссылок);
- справку по статьям, на которые есть ссылки;
- часть, которую нужно переписать.

Как переписывать:
1. Раскрывай внутренние ссылки. «Те же деяния», «То же деяние»,
   «Деяния, предусмотренные частью первой настоящей статьи» замени
   кратким, но полным описанием этих деяний из соответствующих частей.
2. Раскрывай ссылки на другие статьи. Вместо «статьей 159.3 настоящего
   Кодекса» пиши суть нормы словами: «мошенничество с использованием
   электронных средств платежа». Номера других статей не упоминай.
   Бери суть только из справки. Если ссылка ведёт на конкретную часть
   другой статьи, найди эту часть в справке. Если точно раскрыть
   ссылку нельзя — назови преступление по названию той статьи.
3. Сохраняй все признаки состава и все квалифицирующие пункты.
   Каждый пункт — с новой строки и с той же буквой, что в оригинале
   («а) ...», «е.1) ...»). Буквы не переименовывай, даже если в
   оригинале есть пропуски. Ничего не выбрасывай и не обобщай.
4. Не добавляй ничего от себя: никаких толкований, примеров и выводов.
5. Не пиши наказание и санкцию — только описание деяния.
6. Не используй слова «настоящей статьи», «частью первой»,
   «статьей N», «настоящего Кодекса».
7. Если часть — примечание, перескажи определения так, чтобы было
   ясно, к каким преступлениям они относятся (по названию статьи
   или главы), без номеров. Вместо «в статьях настоящей главы» пиши
   «в преступлениях главы «<название главы>»».
8. Пиши обычным текстом на русском, без markdown, без вступлений
   вроде «Вот текст». Только сам переписанный текст.

Пример.
Статья 111. Умышленное причинение тяжкого вреда здоровью, часть 3:
«Деяния, предусмотренные частями первой или второй настоящей статьи,
если они совершены: а) группой лиц, группой лиц по предварительному
сговору или организованной группой; б) в отношении двух или более лиц»

Ответ:
Умышленное причинение тяжкого вреда здоровью (опасного для жизни либо
повлекшего потерю зрения, речи, слуха, органа или утрату органом его
функций, прерывание беременности, психическое расстройство, наркоманию
или токсикоманию, неизгладимое обезображивание лица, стойкую утрату
трудоспособности не менее чем на одну треть), в том числе совершённое
при отягчающих обстоятельствах (в связи со служебной деятельностью
потерпевшего, в отношении малолетнего или беспомощного, с особой
жестокостью, общеопасным способом, по найму, из хулиганских побуждений,
по мотивам ненависти или вражды, в целях использования органов
потерпевшего, с применением оружия), если оно совершено:
а) группой лиц, группой лиц по предварительному сговору или
организованной группой;
б) в отношении двух или более лиц.
"""


def build_rewrite_prompt(part: dict) -> str:
    article = get_article(part["article_number"])
    references = referenced_articles(part["text"], part["article_number"])
    references_text = "\n\n".join(
        reference_text(number, ref_part) for number, ref_part in references
    ) or "нет"

    return f"""
СТАТЬЯ: Статья {part['article_number']}. {part['title']}
ГЛАВА: {part['chapter']}

ПОЛНЫЙ ТЕКСТ СТАТЬИ:
{article['text']}

СПРАВКА ПО СТАТЬЯМ, НА КОТОРЫЕ ЕСТЬ ССЫЛКИ:
{references_text}

ПЕРЕПИШИ — {part_label(part)}:
{part['disposition']}
"""


def rewrite_part(part: dict, retries: int = 1) -> str:
    prompt = build_rewrite_prompt(part)
    text = ""
    for _ in range(retries + 1):
        text = ask_model(
            REWRITER_PROMPT,
            prompt,
            temperature=0.2,
            reasoning_effort=REWRITE_REASONING_EFFORT,
        )
        text = strip_sanction(text)
        if (
            text
            and not LEFTOVER_REFERENCE_PATTERN.search(text)
            and keeps_points(part, text)
        ):
            return text + ("" if text.endswith(".") else ".")
        letters = ", ".join(point_letters(part["disposition"]))
        prompt += (
            "\n\nВ прошлом ответе остались ссылки на части или статьи "
            "либо потерялись пункты. Раскрой ссылки словами"
            + (f" и сохрани пункты {letters} — каждый с новой строки "
               "с исходной буквой" if letters else "")
            + "."
        )
    return text or part["disposition"]


def all_parts() -> list[dict]:
    parts = []
    for article in load_articles().values():
        if article_key(article["number"])[0] < SPECIAL_PART_FIRST_ARTICLE:
            continue
        if "утратил" in article["title"].lower():
            continue
        parts += split_parts(article)
    return parts


def load_rewritten() -> dict[str, dict]:
    if not REWRITTEN_PATH.exists():
        return {}
    with REWRITTEN_PATH.open(encoding="utf-8") as file:
        items = [json.loads(line) for line in file if line.strip()]
    return {item["id"]: item for item in items}


def rewrite_all(limit: int | None = None) -> None:
    parts = all_parts()
    done = load_rewritten()
    todo = [part for part in parts if part["id"] not in done]
    if limit:
        todo = todo[:limit]

    print(f"Всего частей: {len(parts)}, готово: {len(done)}, "
          f"в работе: {len(todo)}")

    with REWRITTEN_PATH.open("a", encoding="utf-8") as file:
        for index, part in enumerate(todo, start=1):
            started = time.time()
            rewritten = rewrite_part(part)
            file.write(json.dumps(
                {**part, "rewritten": rewritten}, ensure_ascii=False
            ) + "\n")
            file.flush()
            print(
                f"[{index}/{len(todo)}] ст. {part['article_number']}, "
                f"{part_label(part)} — {time.time() - started:.1f} с"
            )


def fix_rewritten(max_growth: float = 1.5) -> None:
    items = load_rewritten()
    refetched = restored = 0

    for item in items.values():
        rewritten = strip_sanction(item["rewritten"])
        if not rewritten:
            rewritten = plain_disposition(item)
        rewritten += "" if rewritten.endswith(".") else "."
        item["rewritten"] = rewritten
        if (
            not has_references(item)
            and len(rewritten) > max_growth * len(item["disposition"])
        ):
            item["rewritten"] = plain_disposition(item)
            restored += 1
        elif (
            LEFTOVER_REFERENCE_PATTERN.search(rewritten)
            or not keeps_points(item, rewritten)
        ):
            item["rewritten"] = rewrite_part(item, retries=3)
            refetched += 1
            if LEFTOVER_REFERENCE_PATTERN.search(item["rewritten"]):
                status = "ссылки остались"
            elif not keeps_points(item, item["rewritten"]):
                status = "пункты не совпали"
            else:
                status = "исправлено"
            print(f"ст. {item['article_number']}, {part_label(item)} — {status}")

    tmp_path = REWRITTEN_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        for item in items.values():
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    tmp_path.replace(REWRITTEN_PATH)

    print(f"Переписано заново: {refetched}, "
          f"возвращён оригинал: {restored}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fix", action="store_true",
                        help="исправить уже переписанные части")
    args = parser.parse_args()
    if args.fix:
        fix_rewritten()
    else:
        rewrite_all(args.limit)
