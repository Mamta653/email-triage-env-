
import random
from pydantic import BaseModel
from typing import Optional

# ─── BASELINE SCRIPT ───
# This script runs a random agent against all 3 tasks
# and produces reproducible baseline scores

def run_baseline():
    print(" Email Triage Environment - Baseline Script")
    print("=" * 50)

    env = EmailTriageEnvironment()
    results = {}

    for task in ["easy", "medium", "hard"]:
        print(f"\n Testing {task.upper()} task...")
        scores = []

        for episode in range(20):
            obs = env.reset(task=task)

            # Random agent picks random action
            if task == "easy":
                label = random.choice(["spam", "not_spam"])
            elif task == "medium":
                label = random.choice([
                    "spam", "work",
                    "finance", "personal",
                    "emergency"
                ])
            else:
                label = random.choice([
                    "high_reply", "high_delete",
                    "medium_reply", "medium_archive",
                    "low_delete", "low_archive"
                ])

            action = EmailAction(task=task, label=label)
            result = env.step(action)
            scores.append(result.reward)

        avg = sum(scores) / len(scores)
        results[task] = avg

        bar = "█" * int(avg * 20)
        print(f"   Average score: {avg:.2f}")
        print(f"   [{bar:<20}] {avg*100:.1f}%")

    print("\n" + "=" * 50)
    print(" BASELINE SUMMARY:")
    print("=" * 50)
    for task, score in results.items():
        print(f"   {task:8s} → {score:.2f}")

    print("\n Baseline complete!")
    return results

# Run it!
results = run_baseline()
