"""Headless agent runner for detached TUI sessions.

Spawned by ``session.detach`` as a fully-detached subprocess
(``start_new_session=True``).  Reads configuration from a JSON file written
by the gateway, runs the agent to completion, and persists results to the
session database so ``hermes --tui --resume`` can pick them up later.

Usage::

    python -m tui_gateway.detach_runner <task_id>
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

from hermes_constants import get_hermes_home


def _task_dir() -> Path:
    return get_hermes_home() / "detach-tasks"


def _config_path(task_id: str) -> Path:
    return _task_dir() / f"{task_id}.json"


def _result_path(task_id: str) -> Path:
    return _task_dir() / f"{task_id}.result.json"


def _write_result(task_id: str, success: bool, text: str) -> None:
    _result_path(task_id).write_text(
        json.dumps(
            {"task_id": task_id, "success": success, "text": text, "ts": time.time()},
            ensure_ascii=False,
        )
    )


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m tui_gateway.detach_runner <task_id>", file=sys.stderr)
        sys.exit(1)

    task_id = sys.argv[1]
    cfg_path = _config_path(task_id)

    if not cfg_path.exists():
        print(f"[detach-runner] config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[detach-runner] failed to read config: {exc}", file=sys.stderr)
        sys.exit(1)

    session_key = cfg.get("session_key", "")
    user_message = cfg.get("user_message", "")
    conversation_history = cfg.get("conversation_history", [])
    agent_kwargs = cfg.get("agent_kwargs", {})

    # Set session context so tools and DB writes land correctly.
    from tui_gateway.server import _set_session_context, _clear_session_context

    tokens = _set_session_context(session_key)
    try:
        from run_agent import AIAgent

        agent = AIAgent(**agent_kwargs)
        result = agent.run_conversation(
            user_message,
            conversation_history=conversation_history or None,
        )

        response = (
            result.get("final_response", str(result))
            if isinstance(result, dict)
            else str(result)
        )
        _write_result(task_id, success=True, text=response)
    except Exception as exc:
        trace = traceback.format_exc()
        print(f"[detach-runner] agent error: {trace}", file=sys.stderr)
        _write_result(task_id, success=False, text=f"error: {exc}")
    finally:
        _clear_session_context(tokens)

        # Clean up the config file; result file is kept for debugging.
        try:
            cfg_path.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
