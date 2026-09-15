"""LangSmith tracing.

LangChain and LangGraph trace themselves once the ``LANGSMITH_*`` variables are
in the environment, so there is nothing to sprinkle through the nodes: this
module is the one place that decides whether tracing is on, and turns the
agent's own settings into those variables.

Entry points call :func:`configure_tracing` before any model is built, wrap the
run in :func:`traced_run` to get a link to it, and :func:`flush_traces` before
a short-lived process exits.
"""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from .config import Settings

log = logging.getLogger(__name__)

# The current spelling, and the LANGCHAIN_* one older LangChain builds and a
# lot of existing shells still export. Both are written so the agent traces
# whichever the installed version happens to read.
_ENV_ALIASES = {
    "LANGSMITH_TRACING": "LANGCHAIN_TRACING_V2",
    "LANGSMITH_API_KEY": "LANGCHAIN_API_KEY",
    "LANGSMITH_PROJECT": "LANGCHAIN_PROJECT",
    "LANGSMITH_ENDPOINT": "LANGCHAIN_ENDPOINT",
}


def configure_tracing(settings: Settings) -> bool:
    """Point this process' LangChain tracer at LangSmith.

    Returns whether tracing ended up enabled, which is what the entry points
    tell the user.
    """
    if not settings.tracing_enabled:
        # A stale LANGSMITH_TRACING=true inherited from the surrounding shell
        # would make every model call queue an upload it has no key for, so
        # the switch is cleared rather than merely left alone.
        _unset("LANGSMITH_TRACING")
        return False

    _set("LANGSMITH_TRACING", "true")
    _set("LANGSMITH_API_KEY", settings.langsmith_api_key or "")
    _set("LANGSMITH_PROJECT", settings.langsmith_project)
    _set("LANGSMITH_ENDPOINT", settings.langsmith_endpoint)
    log.info("LangSmith tracing on, project %r", settings.langsmith_project)
    return True


@contextmanager
def traced_run(enabled: bool) -> Iterator[Callable[[], str]]:
    """Run a block, yielding a callable that returns its LangSmith URL.

    The URL is only knowable once the run has started, so it is handed back as
    a callable to invoke after the block — it returns ``''`` when tracing is
    off or the link could not be built.
    """
    if not enabled:
        yield lambda: ""
        return

    from langchain_core.tracers.context import collect_runs

    with collect_runs() as collector:
        yield lambda: trace_url(collector.traced_runs[0]) if collector.traced_runs else ""


def trace_url(run: Any) -> str:
    """The LangSmith URL for a traced run, or ``''`` if it cannot be built.

    A link is a convenience, never a reason to fail a run that has already
    done its work, so every failure here is swallowed.
    """
    try:
        from langsmith import Client

        with warnings.catch_warnings():
            # get_run_url is deprecated in favour of a call that wants the
            # project and trace ids we do not have on this side.
            warnings.simplefilter("ignore")
            return Client().get_run_url(run=run)
    except Exception as exc:  # noqa: BLE001 - a missing link is not an error
        log.debug("No LangSmith run URL: %s", exc)
        return ""


def flush_traces() -> None:
    """Wait for queued traces to upload.

    Traces are posted from a background thread in batches, so a process that
    exits as soon as the run ends — the CLI — can drop the tail of its own
    trace without this.
    """
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers

        wait_for_all_tracers()
    except Exception as exc:  # noqa: BLE001 - never fail a finished run
        log.debug("Could not flush traces: %s", exc)


def _set(name: str, value: str) -> None:
    os.environ[name] = value
    os.environ[_ENV_ALIASES[name]] = value


def _unset(name: str) -> None:
    os.environ.pop(name, None)
    os.environ.pop(_ENV_ALIASES[name], None)
