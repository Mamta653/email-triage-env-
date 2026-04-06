---
title: Email Triage Env
emoji: 
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

#  Email Triage Environment

##  Overview and Motivation
Email triage is a real-world task where humans sort and 
prioritize incoming emails. This OpenEnv environment trains 
AI agents to classify emails automatically — saving time 
and improving productivity.

##  Three Tasks

| Task | Difficulty | Description | Max Score |
|------|-----------|-------------|-----------|
| easy | ⭐ Easy | Spam vs Not Spam | 1.0 |
| medium | ⭐⭐ Medium | 5-way classification | 1.0 |
| hard | ⭐⭐⭐ Hard | Priority + Action | 1.0 |

##  Action Space

### Easy Task:
- `spam` — email is spam
- `not_spam` — email is not spam

### Medium Task:
- `spam` — unwanted email
- `work` — professional email
- `finance` — bills/invoices
- `personal` — friends/family
- `emergency` — urgent issues

### Hard Task:
- `high_reply` — urgent, needs reply
- `high_delete` — urgent spam
- `medium_reply` — normal, needs reply
- `medium_archive` — save for later
- `low_delete` — not important, delete
- `low_archive` — not important, save

## Observation Space
| Field | Type | Description |
|-------|------|-------------|
| subject | string | Email subject line |
| body | string | Email body text |
| sender | string | Sender email address |
| task | string | Current task level |
| message | string | Instruction/feedback |
| done | boolean | Is episode finished |
| reward | float | Score (None until step) |

##  Task Descriptions

### Easy — Spam Detection
Agent must classify email as spam or not_spam.
Reward: 1.0 for correct, 0.0 for wrong.
Expected difficulty: Beginner AI should score ~50%

### Medium — Email Category  
Agent must classify into 5 categories.
Reward: 1.0 for correct, 0.0 for wrong.
Expected difficulty: Random agent scores ~15-20%

### Hard — Priority and Action
Agent must provide priority AND action (e.g. high_reply).
Reward: 0.5 per correct component, max 1.0.
Expected difficulty: Random agent scores ~20-25%

##  Baseline Performance Scores

| Task | Random Agent | Description |
|------|-------------|-------------|
| Easy | 0.50 | 2 choices, ~50% random |
| Medium | 0.15 | 5 choices, ~20% random |
| Hard | 0.23 | 6 choices, partial rewards |

##  Setup and Usage

### Install:
```bash
pip install -r requirements.txt
```

### Run locally:
```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Docker:
```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env
```

### Connect:
```python
import requests

# Reset
obs = requests.post(
    "https://mamta24-email-triage-env.hf.space/reset?task=easy"
).json()

# Step  
result = requests.post(
    "https://mamta24-email-triage-env.hf.space/step",
    json={"task": "easy", "label": "spam"}
).json()
print(result["reward"])
```

##  API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| /health | GET | Health check |
| /reset | POST | Start new episode |
| /step | POST | Take action |
| /state | GET | Current state |
| /tasks | GET | List all tasks |
| /grader | GET | Get grade |
| /baseline | GET | Run baseline |

## 👤 Author
Mamta — Solo Warrior
OpenEnv Hackathon by Meta x Scaler
