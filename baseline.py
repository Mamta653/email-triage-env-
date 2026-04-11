import random
from email_triage_env import EmailTriageEnvironment, EmailAction

def run_baseline():
    print("Email Triage Environment - Baseline Script")
    print("=" * 55)
    print("Random agent vs Smart agent comparison")
    print("Each episode = 5 emails to triage\n")

    env = EmailTriageEnvironment()
    results = {}

    EPISODES = 20

    # ─── RANDOM AGENT ───
    print("RANDOM AGENT (20 episodes per task)")
    print("-" * 55)
    random_results = {}

    for task in ["easy", "medium", "hard"]:
        episode_scores = []

        for _ in range(EPISODES):
            obs = env.reset(task=task)
            episode_reward = 0.0
            steps = 0

            while not obs.done:
                if task == "easy":
                    label = random.choice(["spam", "not_spam"])
                elif task == "medium":
                    label = random.choice([
                        "spam", "work", "finance", "personal", "emergency"
                    ])
                else:
                    label = random.choice([
                        "high_reply", "high_escalate", "high_delete",
                        "medium_reply", "medium_forward", "medium_archive",
                        "low_archive", "low_delete", "low_ignore",
                    ])
                obs = env.step(EmailAction(task=task, label=label))
                if obs.reward:
                    episode_reward += obs.reward
                steps += 1

            avg_episode = episode_reward / steps if steps > 0 else 0.15
            episode_scores.append(round(avg_episode, 4))

        avg = sum(episode_scores) / len(episode_scores)
        random_results[task] = avg
        bar = "█" * int(avg * 30)
        print(f"  {task:8s} → avg: {avg:.3f}  [{bar:<30}] {avg*100:.1f}%")

    # ─── SMART AGENT (rule-based) ───
    print("\nSMART AGENT (rule-based, 20 episodes per task)")
    print("-" * 55)
    smart_results = {}

    def smart_action(task, subject, body, sender, urgency):
        subject_lower = subject.lower()
        body_lower = body.lower()
        sender_lower = sender.lower()

        # Detect spam signals
        spam_signals = ["won", "prize", "free", "click", "claim",
                        "cheap", "deal", "discount", "selected"]
        is_spam = any(s in subject_lower or s in body_lower for s in spam_signals)

        # Detect emergency signals
        emergency_signals = ["urgent", "critical", "down", "breach",
                             "alert", "action required", "immediately"]
        is_emergency = any(s in subject_lower or s in body_lower
                           for s in emergency_signals) or urgency == "high"

        # Detect finance signals
        finance_signals = ["bill", "invoice", "payment", "bank",
                           "salary", "tax", "credit"]
        is_finance = any(s in subject_lower or s in body_lower
                         for s in finance_signals)

        # Detect personal signals
        personal_senders = ["gmail.com", "yahoo.com", "hotmail.com"]
        is_personal = any(s in sender_lower for s in personal_senders)

        if task == "easy":
            return "spam" if is_spam else "not_spam"

        elif task == "medium":
            if is_spam:
                return "spam"
            if is_emergency:
                return "emergency"
            if is_finance:
                return "finance"
            if is_personal:
                return "personal"
            return "work"

        else:  # hard
            if is_spam:
                return "low_delete"
            if is_emergency:
                return "high_reply"
            if is_finance:
                return "medium_archive"
            if is_personal:
                return "low_archive"
            return "medium_reply"

    for task in ["easy", "medium", "hard"]:
        episode_scores = []

        for _ in range(EPISODES):
            obs = env.reset(task=task)
            episode_reward = 0.0
            steps = 0

            while not obs.done:
                label = smart_action(
                    task=task,
                    subject=obs.subject,
                    body=obs.body,
                    sender=obs.sender,
                    urgency=obs.urgency,
                )
                obs = env.step(EmailAction(task=task, label=label))
                if obs.reward:
                    episode_reward += obs.reward
                steps += 1

            avg_episode = episode_reward / steps if steps > 0 else 0.15
            episode_scores.append(round(avg_episode, 4))

        avg = sum(episode_scores) / len(episode_scores)
        smart_results[task] = avg
        bar = "█" * int(avg * 30)
        print(f"  {task:8s} → avg: {avg:.3f}  [{bar:<30}] {avg*100:.1f}%")

    # ─── SUMMARY ───
    print("\n" + "=" * 55)
    print("BASELINE SUMMARY")
    print("=" * 55)
    print(f"  {'Task':<10} {'Random':>10} {'Smart':>10} {'Gap':>10}")
    print("-" * 55)
    for task in ["easy", "medium", "hard"]:
        r = random_results[task]
        s = smart_results[task]
        gap = s - r
        print(f"  {task:<10} {r:>10.3f} {s:>10.3f} {gap:>+10.3f}")

    print("\nA well-trained RL agent should exceed the Smart agent scores.")
    print("Baseline complete!")

    return {"random": random_results, "smart": smart_results}


if __name__ == "__main__":
    results = run_baseline()
