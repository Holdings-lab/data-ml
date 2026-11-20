"""Per-call diagnostics without changing legacy list/DataFrame return types."""
from contextlib import contextmanager
from contextvars import ContextVar
import logging

import requests


_errors = ContextVar("collection_errors", default=None)


def report_failure(exc=None, *, code=None):
    """Record safe frontend diagnostics; exception details remain in server logs."""
    if code is None:
        if isinstance(exc, requests.Timeout):
            code = "UPSTREAM_TIMEOUT"
        elif isinstance(exc, requests.RequestException):
            code = "UPSTREAM_REQUEST_FAILED"
        else:
            code = "COLLECTION_ERROR"
    error = {"code": code, "message": "뉴스 수집 중 오류가 발생했습니다."}
    current = _errors.get()
    if current is not None:
        current.append(error)
    return error


@contextmanager
def capture_errors():
    errors = []
    token = _errors.set(errors)
    try:
        yield errors
    finally:
        _errors.reset(token)


def propagate_errors(errors):
    current = _errors.get()
    if current is not None:
        current.extend(errors)


def attempt(callback):
    """Catch source failures while retaining diagnostics from skipped articles."""
    with capture_errors() as errors:
        try:
            records = callback()
        except Exception as exc:
            logging.getLogger(__name__).exception("News collection failed")
            report_failure(exc)
            records = []
    return records, errors


def collect_item(callback):
    """Keep other articles when a single detail request fails."""
    try:
        return callback()
    except Exception as exc:
        logging.getLogger(__name__).exception("News detail collection failed")
        report_failure(exc)
        return None


def summarize(statuses, count):
    statuses = list(statuses)
    if statuses and all(status == "failed" for status in statuses):
        return "failed"
    if any(status in ("failed", "partial_failure") for status in statuses):
        return "partial_failure"
    return "success" if count else "empty"
