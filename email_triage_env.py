
import random
import uuid
from typing import Optional
from pydantic import BaseModel

# ─── EMAIL DATASET ───
emails = [
    {"subject": "You won $1000000!!!", "body": "Click here to claim prize!", "sender": "spam@random.com", "correct_label": "spam"},
    {"subject": "Buy cheap medicines now", "body": "Best deals on pills!", "sender": "pharma@unknown.com", "correct_label": "spam"},
    {"subject": "FREE iPhone 15 for you!", "body": "You are selected! Click now!", "sender": "prize@fake.com", "correct_label": "spam"},
    {"subject": "Team meeting tomorrow", "body": "Hi, we have a meeting at 10am", "sender": "boss@company.com", "correct_label": "work"},
    {"subject": "Project deadline reminder", "body": "Please submit your report by Friday", "sender": "manager@company.com", "correct_label": "work"},
    {"subject": "Code review needed", "body": "Can you review my pull request?", "sender": "colleague@company.com", "correct_label": "work"},
    {"subject": "Your bill is due", "body": "Please pay your electricity bill", "sender": "bills@electric.com", "correct_label": "finance"},
    {"subject": "Invoice #1234", "body": "Your invoice of $500 is attached", "sender": "accounts@vendor.com", "correct_label": "finance"},
    {"subject": "Bank statement ready", "body": "Your monthly statement is available", "sender": "bank@hdfc.com", "correct_label": "finance"},
    {"subject": "Hey! How are you?", "body": "Long time no see! Lets catch up", "sender": "friend@gmail.com", "correct_label": "personal"},
    {"subject": "Birthday party invitation", "body": "Come to my birthday on Sunday!", "sender": "sister@gmail.com", "correct_label": "personal"},
    {"subject": "URGENT: Server is down!", "body": "Production server crashed! Fix immediately!", "sender": "alerts@company.com", "correct_label": "emergency"},
    {"subject": "CRITICAL: Security breach", "body": "Unauthorized access detected!", "sender": "security@company.com", "correct_label": "emergency"},
]

# ─── MODELS ───
class EmailAction(BaseModel):
    task: str
    label: str

class EmailObservation(BaseModel):
    subject: str
    body: str
    sender: str
    task: str
    attempts_remaining: int
    message: str
    done: bool
    reward: Optional[float]

class EmailState(BaseModel):
    episode_id: str
    step_count: int
    task_level: str
    total_episodes: int

# ─── ENVIRONMENT ───
class EmailTriageEnvironment:
    SUPPORTS_CONCURRENT_SESSIONS = True
    MAX_ATTEMPTS = 1

    def __init__(self):
        self._episode_id = ""
        self._step_count = 0
        self._task_level = "easy"
        self._current_email = None
        self._total_episodes = 0
        self._done = False

    def reset(self, task="easy"):
        self._task_level = task
        self._step_count = 0
        self._done = False
        self._episode_id = str(uuid.uuid4())
        self._total_episodes += 1
        self._current_email = random.choice(emails)
        return EmailObservation(
            subject=self._current_email["subject"],
            body=self._current_email["body"],
            sender=self._current_email["sender"],
            task=self._task_level,
            attempts_remaining=self.MAX_ATTEMPTS,
            message=self._get_task_instruction(),
            done=False,
            reward=None
        )

    def step(self, action: EmailAction):
        self._step_count += 1
        if self._task_level == "easy":
            reward = self._grade_easy(action.label)
        elif self._task_level == "medium":
            reward = self._grade_medium(action.label)
        else:
            reward = self._grade_hard(action.label)
        self._done = True
        if reward == 1.0:
            message = "Correct! Great job!"
        elif reward == 0.5:
            message = "Partially correct!"
        else:
            message = f"Wrong! Answer was: {self._current_email['correct_label']}"
        return EmailObservation(
            subject=self._current_email["subject"],
            body=self._current_email["body"],
            sender=self._current_email["sender"],
            task=self._task_level,
            attempts_remaining=0,
            message=message,
            done=True,
            reward=reward
        )

    def state(self):
        return EmailState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_level=self._task_level,
            total_episodes=self._total_episodes
        )

    def _get_task_instruction(self):
        if self._task_level == "easy":
            return "Is this email spam or not_spam?"
        elif self._task_level == "medium":
            return "Classify: spam/work/finance/personal/emergency"
        else:
            return "Give priority_action e.g. high_reply, low_delete"

    def _grade_easy(self, label):
        correct = self._current_email["correct_label"]
        is_spam = correct == "spam"
        agent_says_spam = label == "spam"
        return 1.0 if is_spam == agent_says_spam else 0.0

    def _grade_medium(self, label):
        correct = self._current_email["correct_label"]
        return 1.0 if label == correct else 0.0

    def _grade_hard(self, label):
        correct = self._current_email["correct_label"]
        score = 0.0
        if correct == "emergency" and "high" in label:
            score += 0.5
        elif correct == "work" and "medium" in label:
            score += 0.5
        elif correct in ["spam", "personal"] and "low" in label:
            score += 0.5
        if correct == "spam" and "delete" in label:
            score += 0.5
        elif correct == "emergency" and "reply" in label:
            score += 0.5
        elif correct == "work" and "reply" in label:
            score += 0.5
        elif correct == "finance" and "archive" in label:
            score += 0.5
        return score
