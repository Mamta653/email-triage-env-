import random
import uuid
from typing import Optional, List, Dict
from pydantic import BaseModel

# ─── EMAIL DATASET ───
EMAIL_TEMPLATES = [
    # SPAM
    {"subject": "You won $1000000!!!", "body": "Click here to claim your prize now!", "sender": "spam@random.com", "label": "spam", "urgency": "low"},
    {"subject": "Buy cheap medicines now", "body": "Best deals on pills, no prescription needed!", "sender": "pharma@unknown.com", "label": "spam", "urgency": "low"},
    {"subject": "FREE iPhone 16 for you!", "body": "You are selected! Claim in 24 hours!", "sender": "prize@fake.com", "label": "spam", "urgency": "low"},
    {"subject": "Congratulations! You are our winner", "body": "Send us your bank details to claim reward", "sender": "winner@scam.net", "label": "spam", "urgency": "low"},
    {"subject": "Hot deals just for you", "body": "Exclusive discounts expire tonight!", "sender": "deals@noreply.biz", "label": "spam", "urgency": "low"},
    {"subject": "Make money from home!", "body": "Earn $5000/week working from home. No experience needed.", "sender": "jobs@quickmoney.net", "label": "spam", "urgency": "low"},

    # WORK
    {"subject": "Team meeting tomorrow at 10am", "body": "Hi, we have our weekly sync tomorrow. Please come prepared with updates.", "sender": "boss@company.com", "label": "work", "urgency": "medium"},
    {"subject": "Project deadline reminder", "body": "Please submit your Q3 report by Friday EOD.", "sender": "manager@company.com", "label": "work", "urgency": "medium"},
    {"subject": "Code review needed", "body": "Can you review my pull request #247? It is blocking the release.", "sender": "colleague@company.com", "label": "work", "urgency": "medium"},
    {"subject": "Q3 performance review scheduled", "body": "Your annual review is next Tuesday at 3pm with HR.", "sender": "hr@company.com", "label": "work", "urgency": "medium"},
    {"subject": "New project assignment", "body": "You have been assigned to the new Acme Corp project starting Monday.", "sender": "pm@company.com", "label": "work", "urgency": "medium"},
    {"subject": "Client presentation feedback", "body": "The client loved the demo! They want a follow-up next week.", "sender": "sales@company.com", "label": "work", "urgency": "medium"},
    {"subject": "Onboarding new team member", "body": "Please help onboard Priya who joins us on Monday.", "sender": "hr@company.com", "label": "work", "urgency": "medium"},

    # FINANCE
    {"subject": "Your electricity bill is due", "body": "Your bill of Rs 2,400 is due on 15th. Pay now to avoid disconnection.", "sender": "bills@electric.com", "label": "finance", "urgency": "medium"},
    {"subject": "Invoice #1234 attached", "body": "Your invoice of $500 for services rendered in March is attached.", "sender": "accounts@vendor.com", "label": "finance", "urgency": "medium"},
    {"subject": "Bank statement ready", "body": "Your monthly statement for March 2026 is now available.", "sender": "bank@hdfc.com", "label": "finance", "urgency": "low"},
    {"subject": "Tax filing deadline approaching", "body": "Your ITR filing deadline is April 30. Please submit documents.", "sender": "tax@govt.in", "label": "finance", "urgency": "medium"},
    {"subject": "Salary credited for March", "body": "Your salary of Rs 85,000 has been credited to your account.", "sender": "payroll@company.com", "label": "finance", "urgency": "low"},
    {"subject": "Credit card payment due", "body": "Your HDFC credit card payment of Rs 12,500 is due in 3 days.", "sender": "alerts@hdfcbank.com", "label": "finance", "urgency": "medium"},

    # PERSONAL
    {"subject": "Hey! How are you?", "body": "Long time no see! Lets catch up this weekend over coffee?", "sender": "friend@gmail.com", "label": "personal", "urgency": "low"},
    {"subject": "Birthday party invitation!", "body": "Come celebrate my birthday this Sunday at 7pm. RSVP please!", "sender": "sister@gmail.com", "label": "personal", "urgency": "low"},
    {"subject": "College reunion this weekend", "body": "Our batch is meeting Saturday at Cafe Coffee Day at 6pm!", "sender": "rahul@gmail.com", "label": "personal", "urgency": "low"},
    {"subject": "Recipe you asked for", "body": "Here is the dal makhani recipe! Let me know how it turns out.", "sender": "mom@gmail.com", "label": "personal", "urgency": "low"},
    {"subject": "Movie tonight?", "body": "Avengers is showing at 8pm. Want to join? 3 tickets available.", "sender": "friend2@gmail.com", "label": "personal", "urgency": "low"},

    # EMERGENCY
    {"subject": "URGENT: Production server is DOWN", "body": "Production server crashed at 2:15am! 10,000 users affected. Need immediate fix!", "sender": "alerts@company.com", "label": "emergency", "urgency": "high"},
    {"subject": "CRITICAL: Security breach detected", "body": "Unauthorized access to customer database detected. Immediate action required!", "sender": "security@company.com", "label": "emergency", "urgency": "high"},
    {"subject": "ACTION REQUIRED: Database corruption", "body": "Main database has critical errors. Data loss is imminent. Please respond NOW!", "sender": "ops@company.com", "label": "emergency", "urgency": "high"},
    {"subject": "ALERT: Payment system down", "body": "Customers cannot checkout. We are losing Rs 50,000/minute in revenue!", "sender": "devops@company.com", "label": "emergency", "urgency": "high"},
    {"subject": "CRITICAL: API service failure", "body": "All API endpoints returning 500 errors. Mobile app is completely broken.", "sender": "monitoring@company.com", "label": "emergency", "urgency": "high"},
    {"subject": "URGENT: Data centre fire alarm", "body": "Fire alarm triggered in server room B. Evacuation in progress. Systems at risk.", "sender": "facilities@company.com", "label": "emergency", "urgency": "high"},
]

INBOX_SIZE = 5


# ─── MODELS ───

class EmailAction(BaseModel):
    task: str
    label: str


class EmailObservation(BaseModel):
    subject: str
    body: str
    sender: str
    urgency: str
    task: str
    email_index: int
    inbox_size: int
    emails_remaining: int
    pending_urgent: int
    message: str
    done: bool
    reward: Optional[float]
    cumulative_reward: float


class EmailState(BaseModel):
    episode_id: str
    step_count: int
    task_level: str
    total_episodes: int
    inbox_size: int
    emails_handled: int
    cumulative_reward: float
    urgent_handled_correctly: int
    urgent_missed: int


# ─── ENVIRONMENT ───

class EmailTriageEnvironment:
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._episode_id = ""
        self._step_count = 0
        self._task_level = "easy"
        self._inbox: List[Dict] = []
        self._current_index = 0
        self._total_episodes = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._urgent_handled_correctly = 0
        self._urgent_missed = 0
        self._pending_urgent = 0

    def reset(self, task: str = "easy"):
        self._task_level = task
        self._step_count = 0
        self._done = False
        self._episode_id = str(uuid.uuid4())
        self._total_episodes += 1
        self._cumulative_reward = 0.0
        self._urgent_handled_correctly = 0
        self._urgent_missed = 0
        self._pending_urgent = 0
        self._current_index = 0
        self._inbox = self._generate_inbox()
        return self._make_observation(message=self._get_task_instruction(), reward=None)

    def step(self, action: EmailAction):
        if self._done:
            return self._make_observation(
                message="Episode finished. Call reset() to start a new one.",
                reward=0.15
            )

        self._step_count += 1
        current_email = self._inbox[self._current_index]

        if current_email["label"] == "emergency":
            self._pending_urgent += 1

        reward = self._grade(action.label, current_email)
        self._cumulative_reward += reward

        if current_email["label"] == "emergency":
            if reward >= 0.7:
                self._urgent_handled_correctly += 1
                self._pending_urgent = max(0, self._pending_urgent - 1)
            else:
                self._urgent_missed += 1

        message = self._build_feedback(action.label, current_email, reward)
        self._current_index += 1

        if self._current_index >= len(self._inbox):
            self._done = True
            avg = round(self._cumulative_reward / len(self._inbox), 3)
            message += (f" | INBOX COMPLETE! Avg reward: {avg} | "
                        f"Emergencies handled: {self._urgent_handled_correctly} | "
                        f"Missed: {self._urgent_missed}")

        return self._make_observation(message=message, reward=reward)

    def state(self):
        return EmailState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_level=self._task_level,
            total_episodes=self._total_episodes,
            inbox_size=len(self._inbox),
            emails_handled=self._current_index,
            cumulative_reward=round(self._cumulative_reward, 4),
            urgent_handled_correctly=self._urgent_handled_correctly,
            urgent_missed=self._urgent_missed,
        )

    # ─── PRIVATE HELPERS ───

    def _generate_inbox(self) -> List[Dict]:
        inbox = []
        if self._task_level in ["medium", "hard"]:
            emergency_pool = [e for e in EMAIL_TEMPLATES if e["label"] == "emergency"]
            inbox.append(random.choice(emergency_pool).copy())

        remaining = INBOX_SIZE - len(inbox)
        pool = [e for e in EMAIL_TEMPLATES if e["label"] != "emergency"]
        sampled = random.sample(pool, min(remaining, len(pool)))
        inbox.extend([e.copy() for e in sampled])
        random.shuffle(inbox)
        return inbox[:INBOX_SIZE]

    def _make_observation(self, message: str, reward: Optional[float]) -> EmailObservation:
        if self._done and self._current_index >= len(self._inbox):
            idx = len(self._inbox) - 1
        else:
            idx = self._current_index

        idx = max(0, min(idx, len(self._inbox) - 1))
        email = self._inbox[idx]
        emails_remaining = max(0, len(self._inbox) - self._current_index)

        return EmailObservation(
            subject=email["subject"],
            body=email["body"],
            sender=email["sender"],
            urgency=email["urgency"],
            task=self._task_level,
            email_index=idx,
            inbox_size=len(self._inbox),
            emails_remaining=emails_remaining,
            pending_urgent=self._pending_urgent,
            message=message,
            done=self._done,
            reward=reward,
            cumulative_reward=round(self._cumulative_reward, 4),
        )

    def _get_task_instruction(self) -> str:
        if self._task_level == "easy":
            return (f"Triage your inbox of {INBOX_SIZE} emails. "
                    "For each email decide: spam or not_spam")
        elif self._task_level == "medium":
            return (f"Triage your inbox of {INBOX_SIZE} emails. "
                    "Classify each as: spam / work / finance / personal / emergency")
        else:
            return (f"Triage your inbox of {INBOX_SIZE} emails. "
                    "For each give a priority_action: "
                    "high_reply, high_escalate, high_delete, "
                    "medium_reply, medium_forward, medium_archive, "
                    "low_archive, low_delete, low_ignore")

    def _build_feedback(self, label: str, email: Dict, reward: float) -> str:
        correct = email["label"]
        remaining = max(0, len(self._inbox) - self._current_index - 1)

        if reward >= 0.85:
            quality = "Excellent!"
        elif reward >= 0.65:
            quality = "Good"
        elif reward >= 0.45:
            quality = "Partially correct"
        else:
            quality = f"Wrong (was: {correct})"

        urgency_note = ""
        if email["urgency"] == "high" and reward < 0.65:
            urgency_note = " MISSED EMERGENCY"

        return (f"{quality}{urgency_note} | "
                f"Email {self._current_index + 1}/{len(self._inbox)} | "
                f"{remaining} remaining | "
                f"Running score: {round(self._cumulative_reward, 2)}")

    def _grade(self, label: str, email: Dict) -> float:
        if self._task_level == "easy":
            return self._grade_easy(label, email)
        elif self._task_level == "medium":
            return self._grade_medium(label, email)
        else:
            return self._grade_hard(label, email)

    def _grade_easy(self, label: str, email: Dict) -> float:
        correct = email["label"]
        is_spam = correct == "spam"
        says_spam = label == "spam"
        return 0.9 if is_spam == says_spam else 0.1

    def _grade_medium(self, label: str, email: Dict) -> float:
        correct = email["label"]

        if label == correct:
            return 0.95 if correct == "emergency" else 0.85

        near_miss_map = {
            "emergency": {"work": 0.35},
            "work": {"finance": 0.45, "personal": 0.35},
            "finance": {"work": 0.45},
            "personal": {"work": 0.35},
            "spam": {},
        }
        partial = near_miss_map.get(correct, {}).get(label, None)
        if partial:
            return partial

        if correct == "emergency":
            return 0.05

        return 0.1

    def _grade_hard(self, label: str, email: Dict) -> float:
        correct = email["label"]

        ideal = {
            "emergency": ("high",   ["reply", "escalate"]),
            "work":       ("medium", ["reply", "forward"]),
            "finance":    ("medium", ["archive", "reply"]),
            "personal":   ("low",    ["archive", "ignore"]),
            "spam":       ("low",    ["delete"]),
        }

        ideal_priority, ideal_actions = ideal.get(correct, ("low", ["archive"]))

        parts = label.split("_", 1)
        if len(parts) != 2:
            return 0.1

        agent_priority, agent_action = parts[0], parts[1]

        # Priority score (max 0.45)
        if agent_priority == ideal_priority:
            priority_score = 0.45
        elif ideal_priority == "high" and agent_priority == "medium":
            priority_score = 0.2
        elif ideal_priority == "medium" and agent_priority in ["high", "low"]:
            priority_score = 0.15
        else:
            priority_score = 0.05

        # Action score (max 0.4)
        if agent_action in ideal_actions:
            action_score = 0.4
        elif agent_action in ["reply", "forward", "escalate", "archive", "delete", "ignore"]:
            action_score = 0.1
        else:
            action_score = 0.05

        base = 0.1
        total = base + priority_score + action_score

        # Heavy penalty for missing emergencies
        if correct == "emergency" and agent_priority != "high":
            total = min(total, 0.25)

        return round(min(max(total, 0.05), 0.95), 4)
