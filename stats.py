import json
import sys
from statistics import mean

def process_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    total_processed = len(data)
    failed = 0
    ok_entries = []

    for item in data:
        if item.get("status") != "ok":
            failed += 1
        else:
            ok_entries.append(item)

    total_ok = len(ok_entries)

    outliers_removed = sum(len(item.get("outliers", [])) for item in ok_entries)

    initial_scores = [item["initial_score"] for item in ok_entries if item.get("initial_score") is not None]
    final_scores = [item["final_score"] for item in ok_entries if item.get("final_score") is not None]

    # 🔥 FIX HERE
    lin_scores = [item["lin_score"] for item in ok_entries if item.get("lin_score") is not None]

    mean_initial = mean(initial_scores) if initial_scores else 0
    mean_final = mean(final_scores) if final_scores else 0
    mean_lin = mean(lin_scores) if lin_scores else 0

    high_coherence_count = sum(1 for item in ok_entries if item.get("final_score", 0) >= 7)
    high_coherence_percentage = (high_coherence_count / total_ok * 100) if total_ok else 0

    communities_with_outliers = sum(1 for item in ok_entries if len(item.get("outliers", [])) > 0)

    print(f"Total processed              : {total_processed}")
    print(f"Failed                       : {failed}")
    print(f"Outliers removed             : {outliers_removed}")
    print(f"Mean LLM score before        : {mean_initial:.4f}")
    print(f"Mean LLM score after         : {mean_final:.4f}")
    print(f"Mean Lin score               : {mean_lin:.4f}")
    print(f"High coherence (count)       : {high_coherence_count}")
    print(f"High coherence (%)           : {high_coherence_percentage:.2f}")
    print(f"Communities with outliers    : {communities_with_outliers}")


process_json('llm_judge_se2_results_level2.json')