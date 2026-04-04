#  Email Triage Environment

An OpenEnv environment for training AI agents to classify emails.

## What it does
Simulates real-world email triage — the task of sorting
and prioritizing emails. An AI agent reads emails and
learns to classify them correctly.

##  Three Tasks

| Task | Difficulty | Description | Max Score |
|------|-----------|-------------|-----------|
| easy | ⭐ Easy | Spam vs Not Spam | 1.0 |
| medium | ⭐⭐ Medium | 5-way classification | 1.0 |
| hard | ⭐⭐⭐ Hard | Priority + Action | 1.0 |

## Action Space

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

##  Observation Space
Each observation contains:
- `subject` — email subject line
- `body` — email body text
- `sender` — sender email address
- `task` — current task level
- `message` — instruction/feedback
- `done` — is episode finished
- `reward` — score (None until step)

##  Quick Start

### Install:
```bash
pip install -r requirements.txt
```

### Run locally:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Connect:
```python
import requests

# Reset
obs = requests.post("http://localhost:8000/reset?task=easy").json()
print(obs["subject"])

# Step
result = requests.post("http://localhost:8000/step",
    json={"task": "easy", "label": "spam"}).json()
print(result["reward"])
```

##  Baseline Scores
Random agent performance:
- Easy:   ~40%
- Medium: ~15%
- Hard:   ~23%

##  Docker
```bash
docker build -t email-triage-env .
docker run -p 8000:8000 email-triage-env
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

##  Author
Mamta — Solo Warrior
OpenEnv Hackathon by Meta x Scaler
