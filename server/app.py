import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import random
import uvicorn

from email_triage_env import (
    EmailTriageEnvironment,
    EmailAction,
    INBOX_SIZE,
    EMAIL_TEMPLATES,
)

app = FastAPI(
    title="Email Triage RL Environment",
    description=(
        "A multi-step OpenEnv-compatible reinforcement learning environment "
        "for intelligent email inbox triage. An agent receives a realistic inbox "
        "of 5 emails per episode and must classify and prioritise each one. "
        "Tasks range from binary spam detection (easy) to full priority-action "
        "decisions with urgency-aware reward shaping (hard)."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = EmailTriageEnvironment()


# ─── HEALTH ───

@app.get("/health")
def health():
    return {"status": "healthy", "version": "2.0.0", "inbox_size": INBOX_SIZE}


# ─── CORE OPENENV ENDPOINTS ───

@app.post("/reset")
def reset(task: str = "easy"):
    if task not in ["easy", "medium", "hard"]:
        raise HTTPException(status_code=400, detail="task must be easy, medium, or hard")
    obs = env.reset(task=task)
    return obs.dict()


@app.post("/step")
def step(action: EmailAction):
    if env._done:
        raise HTTPException(
            status_code=400,
            detail="Episode is done. Call POST /reset to start a new episode."
        )
    result = env.step(action)
    return result.dict()


@app.get("/state")
def get_state():
    return env.state().dict()


# ─── TASK METADATA ───

@app.get("/tasks")
def get_tasks():
    return {
        "environment": "Email Triage RL Environment",
        "description": (
            f"Each episode presents an inbox of {INBOX_SIZE} emails. "
            "The agent must triage each email one at a time."
        ),
        "tasks": [
            {
                "id": "easy",
                "name": "Spam Detection",
                "description": f"For each of {INBOX_SIZE} emails, decide: spam or not_spam",
                "difficulty": "easy",
                "steps_per_episode": INBOX_SIZE,
                "actions": ["spam", "not_spam"],
                "reward_range": "(0.1, 0.9)",
                "scoring": "0.9 = correct, 0.1 = wrong",
            },
            {
                "id": "medium",
                "name": "Email Category Classification",
                "description": f"Classify each of {INBOX_SIZE} emails into the correct category",
                "difficulty": "medium",
                "steps_per_episode": INBOX_SIZE,
                "actions": ["spam", "work", "finance", "personal", "emergency"],
                "reward_range": "(0.05, 0.95)",
                "scoring": (
                    "0.95 = emergency correct, 0.85 = correct, "
                    "0.45 = near miss, 0.05 = missed emergency, 0.1 = wrong"
                ),
            },
            {
                "id": "hard",
                "name": "Priority and Action Decision",
                "description": (
                    f"For each of {INBOX_SIZE} emails, assign a combined "
                    "priority_action decision"
                ),
                "difficulty": "hard",
                "steps_per_episode": INBOX_SIZE,
                "actions": [
                    "high_reply", "high_escalate", "high_delete",
                    "medium_reply", "medium_forward", "medium_archive",
                    "low_archive", "low_delete", "low_ignore",
                ],
                "reward_range": "(0.05, 0.95)",
                "scoring": (
                    "Partial credit for priority (0.45) + action (0.4) + base (0.1). "
                    "Emergency mishandled is capped at 0.25."
                ),
            },
        ],
    }


# ─── DATASET INFO ───

@app.get("/dataset")
def get_dataset():
    from collections import Counter
    label_counts = Counter(e["label"] for e in EMAIL_TEMPLATES)
    urgency_counts = Counter(e["urgency"] for e in EMAIL_TEMPLATES)
    return {
        "total_email_templates": len(EMAIL_TEMPLATES),
        "inbox_size_per_episode": INBOX_SIZE,
        "label_distribution": dict(label_counts),
        "urgency_distribution": dict(urgency_counts),
        "categories": ["spam", "work", "finance", "personal", "emergency"],
    }


# ─── BASELINE ───

@app.get("/baseline")
def baseline():
    """Run a random agent for 10 episodes per task to establish baselines."""
    results = {}
    test_env = EmailTriageEnvironment()

    for task in ["easy", "medium", "hard"]:
        episode_scores = []

        for _ in range(10):
            obs = test_env.reset(task=task)
            episode_reward = 0.0

            while not obs.done:
                if task == "easy":
                    label = random.choice(["spam", "not_spam"])
                elif task == "medium":
                    label = random.choice(["spam", "work", "finance", "personal", "emergency"])
                else:
                    label = random.choice([
                        "high_reply", "high_escalate", "high_delete",
                        "medium_reply", "medium_forward", "medium_archive",
                        "low_archive", "low_delete", "low_ignore",
                    ])
                obs = test_env.step(EmailAction(task=task, label=label))
                if obs.reward:
                    episode_reward += obs.reward

            episode_scores.append(round(episode_reward / INBOX_SIZE, 4))

        results[task] = {
            "average_episode_score": round(sum(episode_scores) / len(episode_scores), 4),
            "episode_scores": episode_scores,
        }

    return {"random_agent_baseline": results}


# ─── GRADER INFO ───

@app.get("/grader")
def grader():
    return {
        "episode_id": env._episode_id,
        "task_level": env._task_level,
        "step_count": env._step_count,
        "emails_handled": env._current_index,
        "inbox_size": INBOX_SIZE,
        "done": env._done,
        "cumulative_reward": round(env._cumulative_reward, 4),
        "urgent_handled_correctly": env._urgent_handled_correctly,
        "urgent_missed": env._urgent_missed,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
