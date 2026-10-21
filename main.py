from graph import State, build_graph, qualification


def print_update(node: str, update: State) -> None:
    if node == "analyst":
        summary = update["summary"]
        print("\n[Аналитик] Юридическая суть:")
        print(summary.legal_essence)
        for sign in summary.qualifying_signs:
            print(f"  - {sign}")

    elif node == "compare":
        print("\n[Сравнение] Статьи-кандидаты:")
        for article in update["candidates"]:
            parts = ", ".join(part or "—" for part in article["matched_parts"])
            print(
                f"  - ст. {article['number']}. {article['title']}"
                f" (части: {parts}; расстояние: {article['distance']:.4f})"
            )

    elif node == "lawyer":
        print("\n[Юрист] Выбранные статьи:")
        if not update["selected"]:
            print("  Ни одна статья не набрала достаточной уверенности.")
        for article in update["selected"]:
            print(
                f"  - {qualification(article.article_number, article.part, article.points)}"
                f" — уверенность {article.confidence:.2f}"
            )
            print(f"    {article.reasoning}")
        if update.get("missing_facts"):
            print("  Нужно выяснить:")
            for fact in update["missing_facts"]:
                print(f"    • {fact}")

    elif node == "judge":
        decision = update["decision"]
        print("\n=== Решение судьи ===")
        for verdict in decision.verdicts:
            print(f"\nСтатья {verdict.article_number}, {verdict.part}")
            print(f"Описание: {verdict.description}")
            print(f"Наказание: {verdict.punishment}")
            print("Возможные смягчающие обстоятельства:")
            for item in verdict.mitigating_circumstances:
                print(f"  - {item}")
        print(f"\nРезюме: {decision.conclusion}")


def main() -> None:
    app = build_graph()

    print("=== Предварительный анализ по УК РФ ===")
    print("Для выхода введи 'выход'.")

    while True:
        situation = input("\nОпиши ситуацию:\n> ").strip()

        if situation.lower() in {"выход", "exit", "quit"}:
            break

        if not situation:
            print("Описание не должно быть пустым.")
            continue

        try:
            for chunk in app.stream(
                {"situation": situation},
                stream_mode="updates",
            ):
                for node, update in chunk.items():
                    print_update(node, update)
        except Exception as error:
            print(f"\nОшибка: {error}")


if __name__ == "__main__":
    main()
