import os
import random
from openai import OpenAI
from email_triage_env import EmailTriageEnvironment, EmailAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")

if API_KEY is None:
    raise ValueError("API_KEY or HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def log_start(task, env_name, model):
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_step(step, action, reward, done, error=None):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.4f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}", flush=True)


def get_agent_action(task, subject, body, sender, urgency, email_index, inbox_size, pending_urgent):
    if task == "easy":
        choices = "spam or not_spam"
        extra = ""
    elif task == "medium":
        choices = "spam, work, finance, personal, or emergency"
        extra = "Note: if it looks like an emergency/critical alert, use 'emergency'."
    else:
        choices = ("high_reply, high_escalate, high_delete, "
                   "medium_reply, medium_forward, medium_archive, "
                   "low_archive, low_delete, low_ignore")
        extra = (
            "Priority guide: high=emergency/critical, medium=work/finance, low=spam/personal. "
            "Action guide: reply=needs response, escalate=needs escalation, "
            "delete=discard, forward=send to someone else, archive=save for later, ignore=skip."
        )

    prompt = f"""You are an intelligent email triage agent managing a work inbox.

Email {email_index + 1} of {inbox_size}:
From: {sender}
Subject: {subject}
Body: {body}
Urgency signal: {urgency}
Pending urgent emails: {pending_urgent}

Task: {task}
Your choices: {choices}
{extra}

Reply with ONLY one choice from the list. No explanation."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=15,
        )
        return response.choices[0].message.content.strip().lower().split()[0]
    except Exception as e:
        print(f"[DEBUG] Model call failed: {e}", flush=True)
        if task == "easy":
            return random.choice(["spam", "not_spam"])
        elif task == "medium":
            return random.choice(["spam", "work", "finance", "personal", "emergency"])
        else:
            return random.choice(["high_reply", "medium_archive", "low_delete"])


def run_episode(env, task):
    obs = env.reset(task=task)
    log_start(task=task, env_name="email-triage", model=MODEL_NAME)

    rewards = []
    steps_taken = 0
    success = False

    try:
        while not obs.done:
            step = obs.email_index + 1
            try:
                action_label = get_agent_action(
                    task=task,
                    subject=obs.subject,
                    body=obs.body,
                    sender=obs.sender,
                    urgency=obs.urgency,
                    email_index=obs.email_index,
                    inbox_size=obs.inbox_size,
                    pending_urgent=obs.pending_urgent,
                )
                action = EmailAction(task=task, label=action_label)
                obs = env.step(action)
                reward = obs.reward if obs.reward is not None else 0.15
                rewards.append(reward)
                steps_taken = step
                log_step(step=step, action=action_label, reward=reward, done=obs.done)
            except Exception as e:
                obs_reward = 0.15
                rewards.append(obs_reward)
                steps_taken = step
                log_step(step=step, action="null", reward=obs_reward, done=True, error=str(e))
                break

        score = sum(rewards) / len(rewards) if rewards else 0.15
        score = round(min(max(score, 0.05), 0.95), 4)
        success = score > 0.5

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return rewards


def main():
    env = EmailTriageEnvironment()
    for task in ["easy", "medium", "hard"]:
        print(f"\n{'='*50}")
        print(f"Task: {task.upper()}")
        print(f"{'='*50}")
        rewards = run_episode(env, task)
        avg = sum(rewards) / len(rewards) if rewards else 0
        print(f"Episode complete. Avg reward: {avg:.4f}")


if __name__ == "__main__":
    main()
