import os
import random
from openai import OpenAI
from email_triage_env import EmailTriageEnvironment, EmailAction

# ─── ENVIRONMENT VARIABLES ───
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = os.getenv("API_KEY") or HF_TOKEN  # ← USE API_KEY!
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

if API_KEY is None:
    raise ValueError("API_KEY or HF_TOKEN environment variable is required")

# ─── OPENAI CLIENT ───
# Use API_BASE_URL and API_KEY as injected by the platform
client = OpenAI(
    base_url=os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1"),
    api_key=os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
)

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def get_agent_action(task, subject, body, sender):
    if task == "easy":
        choices = "spam or not_spam"
    elif task == "medium":
        choices = "spam, work, finance, personal, or emergency"
    else:
        choices = "high_reply, high_delete, medium_reply, medium_archive, low_delete, or low_archive"

    prompt = f"""You are an email classification agent.
Email:
From: {sender}
Subject: {subject}
Body: {body}
Task: {task}
Your choices: {choices}
Reply with ONLY one choice. Nothing else."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"[DEBUG] Model failed: {e}", flush=True)
        if task == "easy":
            return random.choice(["spam", "not_spam"])
        elif task == "medium":
            return random.choice(["spam", "work", "finance", "personal", "emergency"])
        else:
            return random.choice(["high_reply", "low_delete", "medium_reply"])

def run_episode(env, task):
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    obs = env.reset(task=task)
    log_start(task=task, env="email-triage", model=MODEL_NAME)

    try:
        step = 0
        done = False
        while not done:
            step += 1
            try:
                action_label = get_agent_action(
                    task=task,
                    subject=obs.subject,
                    body=obs.body,
                    sender=obs.sender
                )
                action = EmailAction(task=task, label=action_label)
                result = env.step(action)
                reward = result.reward if result.reward else 0.0
                done = result.done
                rewards.append(reward)
                steps_taken = step
                obs = result
                log_step(step=step, action=action_label, reward=reward, done=done)
            except Exception as e:
                done = True
                rewards.append(0.0)
                steps_taken = step
                log_step(step=step, action="null", reward=0.0, done=True, error=str(e))

        score = sum(rewards)/len(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score > 0.0

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return rewards

def main():
    env = EmailTriageEnvironment()
    for task in ["easy", "medium", "hard"]:
        print(f"\n--- Task: {task.upper()} ---")
        run_episode(env, task)

if __name__ == "__main__":
    main()
