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

An OpenEnv environment for training AI agents to classify emails.

##  Overview
Simulates real-world email triage. An AI agent reads emails and classifies them.

##  Three Tasks
| Task | Difficulty | Description | Max Score |
|------|-----------|-------------|-----------|
| easy | ⭐ Easy | Spam vs Not Spam | 1.0 |
| medium | ⭐⭐ Medium | 5-way classification | 1.0 |
| hard | ⭐⭐⭐ Hard | Priority + Action | 1.0 |

##  Action Space
### Easy: spam, not_spam
### Medium: spam, work, finance, personal, emergency
### Hard: high_reply, high_delete, medium_reply, medium_archive, low_delete, low_archive

##  Observation Space
- subject, body, sender, task, message, done, reward

##  Baseline Scores
| Task | Score |
|------|-------|
| Easy | 0.50 |
| Medium | 0.15 |
| Hard | 0.23 |

## Setup
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

##  Endpoints
/health, /reset, /step, /state, /tasks, /grader, /baseline

## 👤 Author
Mamta — Solo Warrior — OpenEnv Hackathon by Meta x Scaler
