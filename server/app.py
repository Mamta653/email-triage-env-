import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
import random
from email_triage_env import EmailTriageEnvironment, EmailAction
import uvicorn

app = FastAPI(
    title="Email Triage Environment",
    description="OpenEnv environment for email classification",
    version="1.0.0"
)

env = EmailTriageEnvironment()

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/reset")
def reset(task: str = "easy"):
    obs = env.reset(task=task)
    return obs.dict()

@app.post("/step")
def step(action: EmailAction):
    result = env.step(action)
    return result.dict()

@app.get("/state")
def get_state():
    return env.state().dict()

@app.get("/tasks")
def get_tasks():
    return {"tasks": [
        {"id": "easy", "name": "Spam Detection", "difficulty": "easy",
         "description": "Classify email as spam or not_spam",
         "actions": ["spam", "not_spam"]},
        {"id": "medium", "name": "Email Category", "difficulty": "medium",
         "description": "Classify into categories",
         "actions": ["spam","work","finance","personal","emergency"]},
        {"id": "hard", "name": "Priority and Action", "difficulty": "hard",
         "description": "Give priority_action",
         "actions": ["high_reply","high_delete","medium_reply",
                    "medium_archive","low_delete","low_archive"]}
    ]}

@app.get("/grader")
def grader():
    return {"episode_id": env._episode_id,
            "task_level": env._task_level,
            "step_count": env._step_count,
            "done": env._done}

@app.get("/baseline")
def baseline():
    results = {}
    test_env = EmailTriageEnvironment()
    for task in ["easy", "medium", "hard"]:
        scores = []
        for _ in range(10):
            test_env.reset(task=task)
            if task == "easy":
                label = random.choice(["spam", "not_spam"])
            elif task == "medium":
                label = random.choice(["spam","work","finance",
                                      "personal","emergency"])
            else:
                label = random.choice(["high_reply","low_delete",
                                      "medium_reply","low_archive"])
            result = test_env.step(EmailAction(task=task, label=label))
            scores.append(result.reward)
        results[task] = {
            "average_score": sum(scores)/len(scores),
            "scores": scores
        }
    return {"baseline_results": results}

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
