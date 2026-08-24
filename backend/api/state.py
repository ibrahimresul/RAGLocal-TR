

import threading

from app.conversation import FollowUpContext
from app.conversation import SessionHistory

session_history = SessionHistory()
follow_up_context = FollowUpContext()

_source_filter_lock = threading.Lock()
_source_filter: str | None = None


def get_source_filter() -> str | None:
    with _source_filter_lock:
        return _source_filter


def set_source_filter(value: str | None) -> None:
    global _source_filter
    with _source_filter_lock:
        _source_filter = value


def reset() -> None:
    session_history.clear()
    follow_up_context.clear()
    set_source_filter(None)
