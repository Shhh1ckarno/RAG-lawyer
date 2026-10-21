import re
from functools import lru_cache

from config import SOURCE_PATH

ARTICLE_PATTERN = re.compile(
    r"(?m)^[ \t]*Статья\s+(\d+(?:\.\d+)*)\.?[ \t]*(.*)$"
)
CHAPTER_PATTERN = re.compile(r"(?m)^Глава\s+\d+(?:\.\d+)*\.\s*(.+)$")
HEADER_PATTERN = re.compile(
    r"(?m)^(?:Раздел|Глава|ОСОБЕННАЯ ЧАСТЬ|ОБЩАЯ ЧАСТЬ).*$"
)


def article_key(number: str) -> tuple[int, ...]:
    return tuple(int(part) for part in number.split("."))


@lru_cache(maxsize=1)
def load_articles() -> dict[str, dict]:
    text = SOURCE_PATH.read_text(encoding="utf-8-sig")

    chapters = [
        (match.start(), match.group(1).strip().capitalize())
        for match in CHAPTER_PATTERN.finditer(text)
    ]
    matches = list(ARTICLE_PATTERN.finditer(text))
    articles = {}

    for index, match in enumerate(matches):
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )
        body = text[match.end():end]
        body = HEADER_PATTERN.sub("", body)
        body = "\n".join(
            line.strip() for line in body.splitlines() if line.strip()
        )

        chapter = ""
        for position, name in chapters:
            if position > match.start():
                break
            chapter = name

        number = match.group(1)
        articles[number] = {
            "number": number,
            "title": match.group(2).strip(),
            "chapter": chapter,
            "text": body,
        }

    return articles


def get_article(number: str) -> dict | None:
    return load_articles().get(number.strip().rstrip("."))


def format_article(article: dict, max_chars: int | None = None) -> str:
    text = article["text"]
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + " ..."
    return f"Статья {article['number']}. {article['title']}\n{text}"
