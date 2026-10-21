import argparse
import contextlib
import io
import json
import re
import time
from pathlib import Path

from config import BASE_DIR

DATASET_PATH = BASE_DIR / "uk_eval_dataset.json"
RESULTS_PATH = BASE_DIR / "eval_results.jsonl"

DEFAULT_IDS = [
    "theft_004", "murder_002", "traffic_002",
    "robbery_003", "grievous_harm_003", "bribery_002",
    "armed_robbery_003", "drugs_005", "complex_003",
]

POINT_PATTERN = re.compile(r"^\W*(?:п(?:ункт)?\.?\s*)?«?([а-я](?:\.\d+)?)(?:\)|»|$)")


def norm_part(value: str) -> str:
    match = re.search(r"\d+", value or "")
    return match.group(0) if match else ""


def norm_points(values: list[str]) -> set[str]:
    return {
        match.group(1)
        for value in values
        if (match := POINT_PATTERN.match(value.strip().lower()))
    }


def is_trap(trap: dict, selected: dict, expected_pairs: set) -> bool:
    article, part = trap["article"], norm_part(trap["part"])
    chosen = selected.get(article)
    if not chosen or trap.get("ref"):
        return False
    expected_numbers = {number for number, _ in expected_pairs}
    if not part:
        return article not in expected_numbers
    return (
        (article, part) not in expected_pairs
        and norm_part(chosen["part"]) == part
    )


def mitigating_points(texts: list[str]) -> set[str]:
    found = set()
    for text in texts:
        found |= set(re.findall(r"п(?:ункт)?\.?\s*«([а-я](?:\.\d)?)»", text))
    return found


def run_case(app, case: dict) -> dict:
    started = time.time()
    try:
        state = app.invoke({"situation": case["situation"]})
        error = None
    except Exception as exc:
        state, error = {}, f"{type(exc).__name__}: {exc}"

    summary = state.get("summary")
    decision = state.get("decision")
    return {
        "id": case["id"],
        "seconds": round(time.time() - started, 1),
        "error": error,
        "summary": summary.model_dump() if summary else None,
        "candidates": [
            {"number": c["number"], "matched_parts": c["matched_parts"]}
            for c in state.get("candidates", [])
        ],
        "selected": [item.model_dump() for item in state.get("selected", [])],
        "missing_facts": state.get("missing_facts", []),
        "decision": decision.model_dump() if decision else None,
    }


def score(case: dict, result: dict) -> dict:
    expected = list({
        (a["article"], norm_part(a["part"])): a
        for a in case["expected_articles"]
    }.values())
    expected_numbers = {a["article"] for a in expected}
    allowed = expected_numbers | {
        a["article"] for a in case["acceptable_alternatives"]
    }
    expected_pairs = {(a["article"], norm_part(a["part"])) for a in expected}

    candidates = {c["number"] for c in result["candidates"]}
    selected = {s["article_number"]: s for s in result["selected"]}
    verdicts = (result["decision"] or {}).get("verdicts", [])

    metrics: dict = {"error": bool(result["error"])}

    if not expected:
        metrics["no_crime_ok"] = not selected
        return metrics

    metrics["retrieval"] = len(expected_numbers & candidates) / len(expected_numbers)
    metrics["art_recall"] = len(expected_numbers & set(selected)) / len(expected_numbers)
    metrics["art_prec"] = (
        len(set(selected) & allowed) / len(selected) if selected else 0.0
    )
    metrics["trap"] = any(
        is_trap(trap, selected, expected_pairs)
        for trap in case["wrong_but_plausible"]
    )

    parts_ok, points_hit, points_total = [], 0, 0
    for item in expected:
        chosen = selected.get(item["article"])
        if not chosen:
            continue
        parts_ok.append(norm_part(chosen["part"]) == norm_part(item["part"]))
        want = norm_points(item["points"])
        if want:
            got = norm_points(chosen.get("points", []))
            points_hit += len(want & got)
            points_total += len(want)
    metrics["part"] = sum(parts_ok) / len(parts_ok) if parts_ok else None
    metrics["points"] = points_hit / points_total if points_total else None

    verdict_numbers = {re.sub(r"[^\d.]", "", v["article_number"]) for v in verdicts}
    metrics["judge_art"] = (
        len(expected_numbers & verdict_numbers) / len(expected_numbers)
        if result["decision"] else 0.0
    )

    found_61 = mitigating_points([
        text for v in verdicts for text in v["mitigating_circumstances"]
    ])
    want_61 = {m["point"] for m in case["mitigating"]}
    metrics["mitig"] = (
        len(want_61 & found_61) / len(want_61)
        if want_61 and result["decision"] else None
    )
    metrics["bad_61a"] = (
        "а" in found_61
        and case.get("crime_gravity") in ("тяжкое", "особо тяжкое")
    )
    return metrics


COLUMNS = ["retrieval", "art_recall", "art_prec", "part", "points",
           "trap", "judge_art", "mitig", "bad_61a"]


def fmt(value) -> str:
    if value is None:
        return "  —  "
    if isinstance(value, bool):
        return " да  " if value else " нет "
    return f"{value:5.2f}"


def mean(values: list) -> float | None:
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def report(cases: dict[str, dict], results: list[dict]) -> None:
    rows = []
    for result in results:
        case = cases[result["id"]]
        rows.append((case, result, score(case, result)))

    header = f"{'id':<20}{'ур.':<7}{'сек':>6}  " + " ".join(
        f"{c[:9]:>9}" for c in COLUMNS
    )
    print(header)
    print("-" * len(header))
    for case, result, metrics in rows:
        if "no_crime_ok" in metrics:
            line = f"состава нет: {'верно' if metrics['no_crime_ok'] else 'ошибка'}"
        else:
            line = " ".join(f"{fmt(metrics.get(c)):>9}" for c in COLUMNS)
        flag = " ОШИБКА" if metrics["error"] else ""
        print(f"{case['id']:<20}{case['difficulty']:<7}"
              f"{result['seconds']:>6.0f}  {line}{flag}")

    print()
    print("Средние по уровням (доля; для trap/bad_61a — доля случаев «да»):")
    for level in ("easy", "medium", "hard", "все"):
        group = [
            (r, m) for c, r, m in rows
            if (level == "все" or c["difficulty"] == level)
            and "no_crime_ok" not in m
        ]
        if not group:
            continue
        values = " ".join(
            f"{fmt(mean([m.get(c) for _, m in group])):>9}" for c in COLUMNS
        )
        seconds = mean([r["seconds"] for r, _ in group])
        print(f"{level:<20}{'':<7}{seconds:>6.0f}  {values}")

    print()
    print("Выбор юриста против ожидаемого:")
    from graph import qualification
    for case, result, _ in rows:
        want = "; ".join(
            qualification(a["article"], norm_part(a["part"]), a["points"])
            for a in case["expected_articles"]
        ) or "состава нет"
        got = "; ".join(
            qualification(s["article_number"], norm_part(s["part"]), s.get("points", []))
            + f" ({s['confidence']:.2f})"
            for s in result["selected"]
        ) or "ничего"
        cands = ", ".join(c["number"] for c in result["candidates"])
        print(f"  {case['id']}: ожидалось {want}\n"
              f"      юрист: {got}\n"
              f"      кандидаты: {cands}")
        if result["error"]:
            print(f"      ошибка: {result['error']}")


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_report(path: Path, cases: dict[str, dict], results: list[dict],
                 total: int) -> None:
    buffer = io.StringIO()
    seconds = mean([r["seconds"] for r in results])
    buffer.write(f"Готово {len(results)}/{total}, среднее время "
                 f"{seconds or 0:.0f} с\n\n")
    with contextlib.redirect_stdout(buffer):
        report(cases, results)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(buffer.getvalue(), encoding="utf-8")
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--per-level", type=int)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--rest", action="store_true",
                        help="все случаи, кроме DEFAULT_IDS")
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    parser.add_argument("--report-file", type=Path,
                        help="перезаписывать отчёт после каждого случая")
    args = parser.parse_args()
    results_path = args.out

    dataset = json.loads(Path(DATASET_PATH).read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in dataset}

    if args.ids:
        ids = args.ids
    elif args.rest:
        ids = [case["id"] for case in dataset if case["id"] not in DEFAULT_IDS]
    elif args.per_level:
        ids = [
            case["id"]
            for level in ("easy", "medium", "hard")
            for case in [c for c in dataset if c["difficulty"] == level][:args.per_level]
        ]
    else:
        ids = DEFAULT_IDS

    if args.fresh and results_path.exists():
        results_path.unlink()

    done = {r["id"]: r for r in load_results(results_path)}

    if not args.report:
        from graph import build_graph
        app = build_graph()
        todo = [i for i in ids if i not in done or done[i]["error"]]
        for index, case_id in enumerate(todo, start=1):
            print(f"[{index}/{len(todo)}] {case_id} ...", flush=True)
            result = run_case(app, cases[case_id])
            done[case_id] = result
            with results_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"    готово за {result['seconds']:.0f} с"
                  + (f", ошибка: {result['error']}" if result["error"] else ""),
                  flush=True)
            if args.report_file:
                write_report(args.report_file, cases,
                             [done[i] for i in ids if i in done], len(ids))

    print()
    report(cases, [done[i] for i in ids if i in done])


if __name__ == "__main__":
    main()
