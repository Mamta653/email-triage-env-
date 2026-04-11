"""
Email Triage Environment - Client

Usage:
    # Connect to deployed HF Space
    from client import EmailTriageClient, EmailAction

    client = EmailTriageClient(base_url="https://mamta24-email-triage-env.hf.space")

    # Run a full episode
    obs = client.reset(task="medium")
    while not obs["done"]:
        action = EmailAction(task="medium", label="work")
        obs = client.step(action)
        print(f"Reward: {obs['reward']} | {obs['message']}")
"""

import requests
from typing import Optional, Dict, Any
from pydantic import BaseModel


# ─── ACTION MODEL ───

class EmailAction(BaseModel):
    task: str   # "easy", "medium", or "hard"
    label: str  # agent's chosen action


# ─── CLIENT ───

class EmailTriageClient:
    """
    Typed HTTP client for the Email Triage RL Environment.
    Follows the OpenEnv client interface standard.
    """

    def __init__(self, base_url: str = "http://localhost:7860", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> Dict[str, Any]:
        """Check if the environment server is running."""
        response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def reset(self, task: str = "easy") -> Dict[str, Any]:
        """
        Start a new episode with a fresh inbox of 5 emails.

        Args:
            task: Difficulty level — "easy", "medium", or "hard"

        Returns:
            EmailObservation dict with first email and task instructions
        """
        if task not in ["easy", "medium", "hard"]:
            raise ValueError(f"task must be easy, medium, or hard. Got: {task}")

        response = requests.post(
            f"{self.base_url}/reset",
            params={"task": task},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def step(self, action: EmailAction) -> Dict[str, Any]:
        """
        Submit a triage decision for the current email.

        Args:
            action: EmailAction with task and label fields

        Returns:
            EmailObservation dict with reward, feedback, and next email
        """
        response = requests.post(
            f"{self.base_url}/step",
            json=action.dict(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def state(self) -> Dict[str, Any]:
        """Get current episode state metadata."""
        response = requests.get(f"{self.base_url}/state", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def tasks(self) -> Dict[str, Any]:
        """Get available tasks and their descriptions."""
        response = requests.get(f"{self.base_url}/tasks", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def baseline(self) -> Dict[str, Any]:
        """Run random agent baseline for all tasks."""
        response = requests.get(f"{self.base_url}/baseline", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def dataset(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        response = requests.get(f"{self.base_url}/dataset", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def run_episode(self, task: str = "medium", agent_fn=None) -> Dict[str, Any]:
        """
        Run a complete episode (5 steps) using an optional agent function.

        Args:
            task: "easy", "medium", or "hard"
            agent_fn: callable(obs) -> label string. Uses random if None.

        Returns:
            dict with rewards list, avg_reward, and final state
        """
        import random

        EASY_ACTIONS = ["spam", "not_spam"]
        MEDIUM_ACTIONS = ["spam", "work", "finance", "personal", "emergency"]
        HARD_ACTIONS = [
            "high_reply", "high_escalate", "high_delete",
            "medium_reply", "medium_forward", "medium_archive",
            "low_archive", "low_delete", "low_ignore",
        ]

        obs = self.reset(task=task)
        rewards = []

        while not obs["done"]:
            if agent_fn:
                label = agent_fn(obs)
            else:
                if task == "easy":
                    label = random.choice(EASY_ACTIONS)
                elif task == "medium":
                    label = random.choice(MEDIUM_ACTIONS)
                else:
                    label = random.choice(HARD_ACTIONS)

            obs = self.step(EmailAction(task=task, label=label))
            if obs.get("reward") is not None:
                rewards.append(obs["reward"])

        avg = sum(rewards) / len(rewards) if rewards else 0.0
        final_state = self.state()

        return {
            "task": task,
            "rewards": rewards,
            "avg_reward": round(avg, 4),
            "steps": len(rewards),
            "urgent_handled": final_state.get("urgent_handled_correctly", 0),
            "urgent_missed": final_state.get("urgent_missed", 0),
        }


# ─── QUICK TEST ───

if __name__ == "__main__":
    client = EmailTriageClient(base_url="http://localhost:7860")

    print("Health check:", client.health())
    print()

    for task in ["easy", "medium", "hard"]:
        result = client.run_episode(task=task)
        print(f"Task={task} | Avg reward={result['avg_reward']} | "
              f"Urgent handled={result['urgent_handled']} | "
              f"Missed={result['urgent_missed']}")
