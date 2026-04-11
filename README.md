#  Email Triage RL Environment

A **multi-step reinforcement learning environment** built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework. An AI agent receives a realistic inbox of **5 emails per episode** and must triage each one — classifying, prioritising, and deciding the correct action under time pressure.

[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces/mamta24/email-triage-env)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)

---

##  Why This Environment?

Email triage is a **real-world sequential decision problem** that every professional faces daily. It requires:

- **Context awareness** — urgency signals in subject, body, and sender
- **Priority reasoning** — emergencies must be handled before low-priority emails
- **Action selection** — not just *what* an email is, but *what to do* with it
- **Sequential thinking** — decisions affect the state of a growing inbox

This makes it an ideal environment for training and evaluating LLM agents on practical reasoning tasks.

---

##  Environment Design

### Multi-Step Episodes
Each episode consists of **5 emails** drawn from a diverse dataset. The agent handles them one by one, accumulating reward across the episode. This forces genuine sequential reasoning rather than isolated single-step decisions.

```
Episode start → Email 1 → Email 2 → Email 3 → Email 4 → Email 5 → Done
                  ↓          ↓          ↓          ↓          ↓
               reward     reward     reward     reward     reward
                                                        cumulative score
```

### Three Task Levels

| Task | Description | Actions | Reward Range |
|------|-------------|---------|--------------|
| `easy` | Binary spam detection | `spam`, `not_spam` | 0.1 – 0.9 |
| `medium` | 5-class category classification | `spam`, `work`, `finance`, `personal`, `emergency` | 0.05 – 0.95 |
| `hard` | Priority + action decision | `high_reply`, `medium_archive`, `low_delete`, ... | 0.05 – 0.95 |

### Urgency-Aware Reward Shaping

The reward function is designed to teach agents real-world priorities:

-  Correctly catching an **emergency** email → `0.95` (bonus)
-  Missing an **emergency** email → `0.05` (heavy penalty)
-  Correct category → `0.85`
-  Near-miss (e.g. work vs finance) → `0.35–0.45` partial credit
-  Wrong → `0.1`

On the **hard task**, rewards have two components:
- Priority correctness (up to `+0.45`)
- Action correctness (up to `+0.40`)
- Base score (`0.10`) so reward is never exactly 0

---

## 📊 Dataset

30 email templates across 5 categories, with urgency tags:

| Category | Count | Urgency |
|----------|-------|---------|
| spam | 6 | low |
| work | 7 | medium |
| finance | 6 | medium/low |
| personal | 5 | low |
| emergency | 6 | high |

Each episode guarantees at least **1 emergency email** in medium/hard tasks, so the agent always faces high-stakes decisions.

---

##  Quick Start

### Connect to the deployed environment

```python
from client import EmailTriageClient, EmailAction

client = EmailTriageClient(base_url="https://mamta24-email-triage-env.hf.space")

# Start a medium-difficulty episode
obs = client.reset(task="medium")

while not obs["done"]:
    print(f"Email: {obs['subject']}")
    print(f"Urgency: {obs['urgency']} | Remaining: {obs['emails_remaining']}")

    # Your agent decides
    action = EmailAction(task="medium", label="emergency")
    obs = client.step(action)

    print(f"Reward: {obs['reward']} | {obs['message']}\n")

print(f"Episode done! Total score: {obs['cumulative_reward']}")
```

### Run the baseline comparison

```bash
python baseline.py
```

Expected output:
```
RANDOM AGENT  easy: ~0.50 | medium: ~0.22 | hard: ~0.27
SMART AGENT   easy: ~0.85 | medium: ~0.72 | hard: ~0.68
```

A trained RL agent should exceed the smart agent scores.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server health check |
| POST | `/reset?task=easy` | Start new episode |
| POST | `/step` | Submit action for current email |
| GET | `/state` | Current episode metadata |
| GET | `/tasks` | Task descriptions and action spaces |
| GET | `/dataset` | Dataset statistics |
| GET | `/baseline` | Random agent baseline scores |
| GET | `/grader` | Current grader state |

### Example: Reset

```bash
curl -X POST "https://mamta24-email-triage-env.hf.space/reset?task=hard"
```

```json
{
  "subject": "URGENT: Production server is DOWN",
  "body": "Production server crashed at 2:15am! 10,000 users affected.",
  "sender": "alerts@company.com",
  "urgency": "high",
  "task": "hard",
  "email_index": 0,
  "inbox_size": 5,
  "emails_remaining": 5,
  "pending_urgent": 0,
  "message": "Triage your inbox of 5 emails...",
  "done": false,
  "reward": null,
  "cumulative_reward": 0.0
}
```

### Example: Step

```bash
curl -X POST "https://mamta24-email-triage-env.hf.space/step" \
  -H "Content-Type: application/json" \
  -d '{"task": "hard", "label": "high_reply"}'
```

---

##  Project Structure

```
email-triage-env/
├── email_triage_env.py   # Core environment logic
├── client.py             # Typed HTTP client
├── baseline.py           # Random + smart agent baselines
├── inference.py          # LLM agent runner
├── openenv.yaml          # Environment metadata
├── pyproject.toml        # Package config
├── requirements.txt      # Dependencies
├── Dockerfile            # Container setup
└── server/
    └── app.py            # FastAPI server
```

---

##  Reward Design Philosophy

The reward function is carefully designed to:

1. **Never return exactly 0 or 1** — all scores are strictly in (0.05, 0.95)
2. **Provide partial credit** — near-misses get some reward to guide learning
3. **Penalise missed emergencies heavily** — reflects real-world cost of ignoring critical alerts
4. **Reward urgency awareness** — agents learn to scan for emergency signals

---

##  Local Development

```bash
git clone https://github.com/Mamta653/email-triage-env
cd email-triage-env
pip install -r requirements.txt

# Run server
python server/app.py

# Test baseline
python baseline.py
```

---

Built for the **Meta PyTorch OpenEnv Hackathon x Scaler School of Technology 2026**
