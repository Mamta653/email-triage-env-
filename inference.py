
import os
import random
from openai import OpenAI

# ─── ENVIRONMENT VARIABLES ───
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ─── OPENAI CLIENT ───
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# ─── GET AGENT DECISION ───
def get_agent_action(task, email_subject, email_body, email_sender):
    if task == "easy":
        choices = "spam or not_spam"
    elif task == "medium":
        choices = "spam, work, finance, personal, or emergency"
    else:
        choices = "high_reply, high_delete, medium_reply, medium_archive, low_delete, or low_archive"

    prompt = f"""You are an email classification agent.

Email:
From: {email_sender}
Subject: {email_subject}
Body: {email_body}

Task: {task}
Your choices: {choices}

Reply with ONLY one of the choices above. Nothing else."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=10
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        # Fallback to random if API fails
        if task == "easy":
            return random.choice(["spam", "not_spam"])
        elif task == "medium":
            return random.choice(["spam", "work", "finance",
                                 "personal", "emergency"])
        else:
            return random.choice(["high_reply", "low_delete",
                                 "medium_reply"])

# ─── RUN ONE EPISODE ───
def run_episode(env, task, model_name):
    rewards = []
    last_error = None

    # START
    obs = env.reset(task=task)
    print(f"[START] task={task} env=email-triage model={model_name}")

    step = 0
    done = False

    while not done:
        step += 1

        try:
            # Get agent decision
            action_label = get_agent_action(
                task=task,
                email_subject=obs.subject,
                email_body=obs.body,
                email_sender=obs.sender
            )

            # Take action
            action = EmailAction(task=task, label=action_label)
            result = env.step(action)

            reward = result.reward if result.reward else 0.0
            done = result.done
            rewards.append(reward)
            last_error = None

            # STEP output
            print(f"[STEP] step={step} action={action_label} reward={reward:.2f} done={str(done).lower()} error=null")

            obs = result

        except Exception as e:
            last_error = str(e)
            done = True
            rewards.append(0.0)
            print(f"[STEP] step={step} action=null reward=0.00 done=true error={last_error}")

    # END
    success = any(r > 0 for r in rewards)
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={str(success).lower()} steps={step} rewards={rewards_str}")

    return rewards

# ─── MAIN ───
def main():
    env = EmailTriageEnvironment()

    for task in ["easy", "medium", "hard"]:
        print(f"\n{'='*50}")
        rewards = run_episode(env, task, MODEL_NAME)
        avg = sum(rewards) / len(rewards) if rewards else 0
        print(f"Average reward: {avg:.2f}")

if __name__ == "__main__":
    main()
