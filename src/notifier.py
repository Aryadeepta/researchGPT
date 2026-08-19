import os


class Notifier:
    def notify_transition(self, state, event_type, payload=None):
        raise NotImplementedError


class ConsoleNotifier(Notifier):
    def notify_transition(self, state, event_type, payload=None):
        key = f"{event_type}:{payload.get('decision_id') if isinstance(payload, dict) else ''}:{state.get('status')}"
        sent = state.setdefault("notifications_sent", [])
        if key in sent:
            return False
        sent.append(key)
        print(f"[notification] run={state.get('run_id')} event={event_type}")
        if isinstance(payload, dict) and payload.get("question"):
            print(f"decision_id={payload.get('decision_id')} question={payload.get('question')}")
            print(f"recommended={payload.get('recommended_option')}")
        return True


class GitHubNotifier(Notifier):
    def __init__(self, issue_file=None):
        self.issue_file = issue_file or os.environ.get("RESEARCH_GITHUB_NOTIFICATION_FILE")

    def notify_transition(self, state, event_type, payload=None):
        key = f"github:{event_type}:{payload.get('decision_id') if isinstance(payload, dict) else ''}:{state.get('status')}"
        sent = state.setdefault("notifications_sent", [])
        if key in sent:
            return False
        sent.append(key)
        if not self.issue_file:
            return False
        with open(self.issue_file, "a", encoding="utf-8") as f:
            f.write(f"run={state.get('run_id')} event={event_type}\n")
            if isinstance(payload, dict):
                f.write(f"decision={payload.get('decision_id')} question={payload.get('question')}\n")
                f.write(f"respond: /research decide {payload.get('decision_id')} <option>\n")
        return True


def notifier_from_env():
    if os.environ.get("RESEARCH_GITHUB_NOTIFICATION_FILE"):
        return GitHubNotifier()
    return ConsoleNotifier()
