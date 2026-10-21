import re
from itertools import groupby
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from articles import format_article, get_article
from config import JUDGE_REASONING_EFFORT, MIN_CONFIDENCE
from llm import ask_structured
from parts import part_label
from rag import get_article_parts, search_parts

MAX_CANDIDATES = 5
PARTS_PER_QUERY = 10


class LegalSummary(BaseModel):
    legal_essence: str = Field(
        description="Суть деяния, изложенная юридическим языком УК РФ"
    )
    qualifying_signs: list[str] = Field(
        description="Юридически значимые признаки: объект, деяние, "
        "последствия, способ, форма вины, квалифицирующие признаки"
    )
    search_queries: list[str] = Field(
        description="3-5 формулировок в стиле названий и диспозиций "
        "статей УК РФ для поиска подходящих норм"
    )


class SelectedArticle(BaseModel):
    article_number: str = Field(description="Номер статьи, например '158'")
    part: str = Field(description="Только номер части цифрой, например '2', "
                      "или пустая строка, если статья без частей")
    points: list[str] = Field(description="Только буквы применимых пунктов "
                              "части из текста статьи, например ['а', 'в']; "
                              "пустой список, если у части нет пунктов")
    confidence: float = Field(description="Уверенность от 0 до 1")
    reasoning: str = Field(description="Почему статья подходит")


class LawyerDecision(BaseModel):
    articles: list[SelectedArticle]
    missing_facts: list[str] = Field(
        description="Каких фактов не хватает для точной квалификации"
    )


class ArticleVerdict(BaseModel):
    article_number: str
    part: str
    description: str = Field(description="Описание статьи и состава")
    punishment: str = Field(
        description="Наказание строго по санкции приведённой статьи"
    )
    mitigating_circumstances: list[str] = Field(
        description="Возможные смягчающие обстоятельства (ст. 61 УК РФ), "
        "применимые к ситуации"
    )


class JudgeDecision(BaseModel):
    verdicts: list[ArticleVerdict]
    conclusion: str = Field(description="Итоговое резюме судьи")


class State(TypedDict, total=False):
    situation: str
    summary: LegalSummary
    candidates: list[dict]
    selected: list[SelectedArticle]
    missing_facts: list[str]
    decision: JudgeDecision


ANALYST_PROMPT = """
Ты — аналитик-криминалист. Извлеки из описания ситуации самую суть
и переведи её на язык уголовного права РФ.

Правила:
1. Опиши деяние терминами УК РФ: «тайное хищение чужого имущества»,
   «умышленное причинение тяжкого вреда здоровью», «с применением
   насилия, опасного для жизни» и т.п.
2. Выдели объект посягательства, действие, последствия, способ,
   форму вины и квалифицирующие признаки (группа лиц, оружие,
   незаконное проникновение, размер ущерба и т.д.).
3. Не додумывай факты, которых нет в описании. Если признак
   неясен — так и укажи.
4. Поисковые формулировки пиши так, как звучат названия и
   диспозиции статей УК РФ, без номеров статей.
"""

LAWYER_PROMPT = f"""
Ты — опытный юрист по уголовному праву РФ. Тебе даны исходное
описание ситуации, его юридическая выжимка от аналитика и
статьи-кандидаты УК РФ, найденные по сходству.

Задача: выбрать статью или статьи, по которым следует
квалифицировать деяние, и для каждой указать уверенность от 0 до 1.

Правила:
1. Выбирай только из приведённых кандидатов.
2. Сопоставляй признаки состава из текста статьи с фактами.
3. Учитывай конкуренцию норм: общая и специальная норма,
   основной и квалифицированный состав — выбирай наиболее точную.
4. Если деяние образует совокупность преступлений — укажи
   несколько статей.
5. Уверенность ниже {MIN_CONFIDENCE} ставь, если состав
   подтверждается лишь частично.
6. Не выдумывай части и пункты, которых нет в тексте статьи.
7. Часть указывай только цифрой, пункты — только буквами из текста
   статьи, без пояснений: часть "2", пункты ["а", "г"].
8. Факты бери из исходного описания. Выжимка аналитика — подсказка:
   если она расходится с исходным описанием или в ней не хватает
   фактов, верь исходному описанию.
"""

JUDGE_PROMPT = """
Ты — судья по уголовным делам РФ. Тебе даны исходное описание
ситуации, статьи УК РФ, отобранные юристом, и текст статьи 61 УК РФ
о смягчающих обстоятельствах.

Для каждой статьи вынеси решение:
- номер статьи, часть и пункт;
- описание статьи: какой состав и почему он применим к ситуации;
- наказание — строго по санкции из приведённого текста
  (соответствующей части), ничего не выдумывай;
- возможные смягчающие обстоятельства из ст. 61 УК РФ, которые
  реально могут следовать из ситуации или которые стоит проверить.

Если статья, отобранная юристом, по твоему мнению не подходит
к исходной ситуации — не включай её и объясни это в резюме.
В резюме отметь, что решение предварительное и не заменяет
рассмотрение дела судом.
"""


def analyst(state: State) -> State:
    summary = ask_structured(
        ANALYST_PROMPT,
        f"Описание ситуации:\n{state['situation']}",
        LegalSummary,
    )
    return {"summary": summary}


def compare(state: State) -> State:
    summary = state["summary"]
    queries = [summary.legal_essence, *summary.search_queries]
    found = search_parts(queries, n_results=PARTS_PER_QUERY)

    candidates: dict[str, dict] = {}
    for item in found:
        number = item["article_number"]
        if number not in candidates:
            if len(candidates) >= MAX_CANDIDATES:
                continue
            candidates[number] = {
                "number": number,
                "title": item["title"],
                "distance": item["distance"],
                "matched_parts": [],
                "matched_points": [],
            }
        candidate = candidates[number]
        if item["part"] not in candidate["matched_parts"]:
            candidate["matched_parts"].append(item["part"])
        candidate["matched_points"].append([item["part"], item["point"]])
    return {"candidates": list(candidates.values())}


FOUND_MARK = " [найдено поиском]"


def format_parts(
    article_number: str,
    title: str,
    with_punishment: bool = False,
    matched: list[list[str]] | None = None,
) -> str:
    matched_set = {tuple(item) for item in matched or []}
    lines = [f"Статья {article_number}. {title}"]
    records = get_article_parts(article_number)
    for part, group in groupby(records, key=lambda record: record["part"]):
        group = list(group)
        first = group[0]
        mark = FOUND_MARK if (part, "") in matched_set else ""
        label = part_label({"part": part}).capitalize()
        lines.append(f"— {label}{mark}: {first['lead']}")
        for record in group:
            if record["point"]:
                mark = FOUND_MARK if (part, record["point"]) in matched_set else ""
                lines.append(f"    {record['point']}) {record['point_text']}{mark}")
        if with_punishment and first["punishment"]:
            lines.append(f"  Санкция: {first['punishment']}")
    return "\n".join(lines)


def normalize_part(value: str) -> str:
    match = re.search(r"\d+", value)
    return match.group(0) if match else ""


POINT_VALUE_PATTERN = re.compile(
    r"^\W*(?:п(?:ункт)?\.?\s*)?«?([а-я](?:\.\d+)?)(?:\)|»|$)"
)


def normalize_points(values: list[str], allowed: set[str]) -> list[str]:
    points = []
    for value in values:
        match = POINT_VALUE_PATTERN.match(value.strip().lower())
        if match and match.group(1) in allowed and match.group(1) not in points:
            points.append(match.group(1))
    return points


def qualification(article: str, part: str, points: list[str]) -> str:
    text = f"ст. {article}"
    if part:
        text += f" ч. {part}"
    if points:
        prefix = "пп." if len(points) > 1 else "п."
        text += f" {prefix} " + ", ".join(f"«{point}»" for point in points)
    return text


def normalize_article_number(value: str) -> str:
    match = re.search(r"\d+(?:\.\d+)*", value)
    return match.group(0) if match else value.strip()


def lawyer(state: State) -> State:
    summary = state["summary"]
    candidates_text = "\n\n".join(
        format_parts(
            article["number"],
            article["title"],
            matched=article["matched_points"],
        )
        for article in state["candidates"]
    )
    user_prompt = f"""
ИСХОДНОЕ ОПИСАНИЕ СИТУАЦИИ:
{state['situation']}

ЮРИДИЧЕСКАЯ ВЫЖИМКА АНАЛИТИКА:
{summary.legal_essence}

ПРИЗНАКИ ПО ВЕРСИИ АНАЛИТИКА:
{chr(10).join(f"- {sign}" for sign in summary.qualifying_signs)}

СТАТЬИ-КАНДИДАТЫ:
{candidates_text}
"""
    decision = ask_structured(LAWYER_PROMPT, user_prompt, LawyerDecision)

    candidate_numbers = {article["number"] for article in state["candidates"]}
    for article in decision.articles:
        article.article_number = normalize_article_number(article.article_number)
        if article.article_number not in candidate_numbers:
            continue
        article.part = normalize_part(article.part)
        allowed = {
            record["point"]
            for record in get_article_parts(article.article_number)
            if record["part"] == article.part and record["point"]
        }
        article.points = normalize_points(article.points, allowed)
    selected = [
        article
        for article in decision.articles
        if article.confidence >= MIN_CONFIDENCE
        and article.article_number in candidate_numbers
    ]
    selected.sort(key=lambda article: article.confidence, reverse=True)
    return {"selected": selected, "missing_facts": decision.missing_facts}


def judge(state: State) -> State:
    titles = {article["number"]: article["title"] for article in state["candidates"]}
    articles_text = "\n\n".join(
        f"[Юрист: {qualification(item.article_number, item.part, item.points)}, "
        f"уверенность {item.confidence:.2f}]\n"
        + format_parts(
            item.article_number,
            titles[item.article_number],
            with_punishment=True,
        )
        for item in state["selected"]
    )
    mitigating = get_article("61")
    user_prompt = f"""
ИСХОДНОЕ ОПИСАНИЕ СИТУАЦИИ:
{state['situation']}

СТАТЬИ, ОТОБРАННЫЕ ЮРИСТОМ:
{articles_text}

{format_article(mitigating) if mitigating else ''}
"""
    decision = ask_structured(
        JUDGE_PROMPT,
        user_prompt,
        JudgeDecision,
        reasoning_effort=JUDGE_REASONING_EFFORT,
    )
    for verdict in decision.verdicts:
        verdict.article_number = normalize_article_number(verdict.article_number)
    return {"decision": decision}


def route_after_lawyer(state: State) -> str:
    return "judge" if state.get("selected") else END


def build_graph():
    graph = StateGraph(State)
    graph.add_node("analyst", analyst)
    graph.add_node("compare", compare)
    graph.add_node("lawyer", lawyer)
    graph.add_node("judge", judge)

    graph.add_edge(START, "analyst")
    graph.add_edge("analyst", "compare")
    graph.add_edge("compare", "lawyer")
    graph.add_conditional_edges("lawyer", route_after_lawyer, ["judge", END])
    graph.add_edge("judge", END)
    return graph.compile()
