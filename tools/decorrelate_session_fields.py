"""
Reassigns session_turn_number and prior_context_tokens on the LLM-generated
queries to random values, independent of label_mode. Fixes a confound
where GPT tended to write "thinking" examples as later-turn follow-ups
and "fast" examples as fresh turn-1 questions - a correlation with the
label that has nothing to do with actual query content, but was strong
enough that the classifier leaned on it instead of the query itself.

Run once:
    python -m tools.decorrelate_session_fields
"""
import csv
import os
import random

PATH = os.path.join(os.path.dirname(__file__), "..", "data", "llm_generated_queries.csv")


def main():
    rng = random.Random(11)
    with open(os.path.abspath(PATH), encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        # Random turn number 1-6, independent of label_mode.
        turn = rng.choice([1, 1, 1, 2, 2, 3, 4, 5, 6])  # weighted toward early turns
        r["session_turn_number"] = str(turn)
        r["prior_context_tokens"] = "0" if turn == 1 else str(rng.choice([50, 90, 140, 200, 300, 450]))

    with open(os.path.abspath(PATH), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "label_mode", "session_turn_number", "prior_context_tokens"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Reassigned session_turn_number and prior_context_tokens for {len(rows)} rows, "
          f"decorrelated from label_mode. Query text and labels unchanged.")


if __name__ == "__main__":
    main()