"""Supervisor process for managing Ultima live trading loop via Telegram."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
STATE_FILE = Path(__file__).with_name("supervisor_state.json")
SETTINGS_CANDIDATES: tuple[Path, ...] = (
    ROOT_DIR / "telegram_settings.json",
    ROOT_DIR / "live/telegram_settings.json",
)
DEFAULT_POLL_INTERVAL = 2.5
LONG_POLL_TIMEOUT = 25


def _parse_allowed_ids(raw: Optional[str | Iterable[int]]) -> set[int]:
    ids: set[int] = set()
    if raw is None:
        return ids

    if isinstance(raw, (list, tuple, set)):
        for item in raw:
            try:
                ids.add(int(item))
            except (TypeError, ValueError):
                continue
        return ids

    for chunk in str(raw).split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            ids.add(int(chunk))
        except ValueError:
            continue
    return ids


def load_telegram_settings() -> tuple[str | None, set[int]]:
    token = os.getenv("ULTIMA_TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
    allowed_ids = _parse_allowed_ids(
        os.getenv("ULTIMA_TELEGRAM_ALLOWED_IDS") or os.getenv("TELEGRAM_ALLOWED_IDS")
    )

    for candidate in SETTINGS_CANDIDATES:
        if not candidate.exists():
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:  # pragma: no cover - runtime diagnostics
            print(f"⚠️  Failed to read Telegram settings from {candidate}: {exc}")
            continue

        token = token or payload.get("bot_token") or payload.get("token") or token
        allowed_payload = payload.get("allowed_user_ids") or payload.get("allowed_ids") or []
        allowed_ids.update(_parse_allowed_ids(allowed_payload))

    return token, allowed_ids


class UltimaSupervisor:
    def __init__(
        self,
        *,
        token: str,
        allowed_ids: Iterable[int],
        python_executable: str,
        script_path: Path,
        auto_start: bool = False,
    ) -> None:
        self.token = token.strip()
        self.api_base = f"https://api.telegram.org/bot{self.token}"
        self.session = requests.Session()
        self.allowed_ids: set[int] = {int(uid) for uid in allowed_ids}
        self.python_executable = python_executable
        self.script_path = script_path
        self.auto_start = auto_start
        self._auto_start_dispatched = False

        self.process: Optional[subprocess.Popen] = None
        self.known_chats: set[int] = set()
        self.update_offset: Optional[int] = None

        self._load_state()

    # ------------------------------------------------------------------
    # Persistence helpers

    def _load_state(self) -> None:
        if not STATE_FILE.exists():
            return
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return
        offset = payload.get("update_offset")
        if isinstance(offset, int):
            self.update_offset = offset

    def _save_state(self) -> None:
        try:
            with open(STATE_FILE, "w", encoding="utf-8") as handle:
                json.dump({"update_offset": self.update_offset}, handle)
        except Exception as exc:
            print(f"⚠️  Failed to persist supervisor state: {exc}")

    # ------------------------------------------------------------------
    # Telegram helpers

    def _get_updates(self, timeout: int = LONG_POLL_TIMEOUT) -> list[dict]:
        params: dict[str, object] = {"timeout": timeout}
        if self.update_offset is not None:
            params["offset"] = self.update_offset
        try:
            response = self.session.get(
                f"{self.api_base}/getUpdates",
                params=params,
                timeout=timeout + 5,
            )
        except requests.RequestException as exc:
            print(f"⚠️  Telegram getUpdates failed: {exc}")
            time.sleep(DEFAULT_POLL_INTERVAL)
            return []

        try:
            payload = response.json()
        except ValueError:
            print(f"⚠️  Telegram getUpdates returned non-JSON payload: {response.text[:200]}")
            return []

        if not payload.get("ok", False):
            print(f"⚠️  Telegram getUpdates returned error: {payload}")
            time.sleep(DEFAULT_POLL_INTERVAL)
            return []

        return payload.get("result", []) or []

    def _send_message(self, chat_id: int, text: str) -> None:
        payload = {"chat_id": chat_id, "text": text}
        try:
            response = self.session.post(
                f"{self.api_base}/sendMessage",
                data=payload,
                timeout=10,
            )
            if response.status_code >= 400:
                print(f"⚠️  Telegram sendMessage failed ({response.status_code}): {response.text}")
        except requests.RequestException as exc:
            print(f"⚠️  Telegram sendMessage error: {exc}")

    def _broadcast(self, text: str, *, exclude: Iterable[int] | None = None) -> None:
        exclude_set = {int(x) for x in exclude} if exclude else set()
        for chat_id in sorted(self.known_chats or self.allowed_ids):
            if chat_id in exclude_set:
                continue
            self._send_message(int(chat_id), text)

    @staticmethod
    def _parse_command(text: str) -> Optional[dict[str, object]]:
        if not text or not text.startswith('/'):
            return None
        parts = text.strip().split()
        raw_command = parts[0][1:]
        if '@' in raw_command:
            raw_command = raw_command.split('@', 1)[0]
        command = raw_command.lower()
        args = parts[1:]
        return {"name": command, "args": args}

    # ------------------------------------------------------------------
    # Process management

    def _start_child(self, chat_id: int | None = None) -> None:
        if self.process and self.process.poll() is None:
            if chat_id is not None:
                self._send_message(chat_id, "ℹ️ Live trading process already running.")
            return

        env = os.environ.copy()
        env.setdefault("ULTIMA_SUPERVISED_EXIT", "1")
        if self.update_offset is not None:
            env["ULTIMA_TELEGRAM_OFFSET"] = str(self.update_offset)
        elif "ULTIMA_TELEGRAM_OFFSET" in env:
            env.pop("ULTIMA_TELEGRAM_OFFSET", None)
        cmd = [self.python_executable, str(self.script_path)]
        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=str(ROOT_DIR),
                env=env,
            )
        except Exception as exc:
            print(f"❌ Failed to launch live trading process: {exc}")
            if chat_id is not None:
                self._send_message(chat_id, f"❌ Launch failed: {exc}")
            self.process = None
            return

        message = "▶️ Starting Ultima live trading process..."
        exclude_targets: set[int] = set()
        if chat_id is not None:
            self._send_message(chat_id, message)
            exclude_targets.add(int(chat_id))
        self._broadcast(message, exclude=exclude_targets)
        print("▶️ Live trading process launched by supervisor.")

    def _stop_child(self, *, reason: str = "Supervisor shutdown") -> None:
        if not self.process:
            return

        proc = self.process
        self.process = None

        try:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        finally:
            print(f"⏹️ Live process terminated ({reason}).")

    def _poll_child(self) -> None:
        if not self.process:
            return
        code = self.process.poll()
        if code is None:
            return

        self.process = None
        print(f"🔌 Live trading process exited with code {code}.")
        self._broadcast(
            f"🔌 Ultima live process stopped (exit {code}). Send /startbot to relaunch when ready."
        )

    # ------------------------------------------------------------------
    # Update handling

    def _handle_update(self, update: dict) -> None:
        update_id = update.get("update_id")
        if isinstance(update_id, int):
            self.update_offset = update_id + 1
            self._save_state()

        message = update.get("message") or update.get("edited_message") or update.get("channel_post")
        if not isinstance(message, dict):
            return

        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        sender = message.get("from") or {}
        user_id = sender.get("id")
        text = (message.get("text") or "").strip()

        if not isinstance(chat_id, int) or not isinstance(user_id, int):
            return

        if self.allowed_ids and user_id not in self.allowed_ids:
            print(f"🚫 Ignoring command from unauthorized user {user_id}.")
            return

        self.known_chats.add(int(chat_id))

        command = self._parse_command(text)
        if not command:
            return

        name = str(command.get("name") or "").lower()
        args = command.get("args") or []

        if name in {"start", "startbot"}:
            self._start_child(chat_id)
            return

        if name in {"status", "performance", "guards"}:
            self._send_message(
                chat_id,
                "ℹ️ Live process currently offline. Send /startbot to launch before requesting status.",
            )
            return

        if name in {"pause", "resume", "kill", "confirmkill", "cancelkill", "flatten", "scaleout"}:
            self._send_message(
                chat_id,
                "⚠️ Live process is offline, so this command could not be routed. Send /startbot first.",
            )
            return

        if name == "help":
            self._send_message(
                chat_id,
                "🤖 Ultima supervisor online. Send /startbot to launch the live trading loop. Other commands are processed once the bot is running.",
            )
            return

        if name == "stopbot":
            self._stop_child(reason="Manual stopbot command")
            self._send_message(chat_id, "⏹️ Live trading process stopped by supervisor.")
            return

        if name:
            self._send_message(
                chat_id,
                f"❓ Command '/{name}' is unavailable while the live process is offline.",
            )

    # ------------------------------------------------------------------

    def run(self) -> None:
        print("🛡️ Ultima supervisor started. Waiting for commands...")
        try:
            while True:
                self._poll_child()

                if self.process and self.process.poll() is None:
                    time.sleep(DEFAULT_POLL_INTERVAL)
                    continue

                if self.auto_start and not self._auto_start_dispatched:
                    self._start_child()
                    self._auto_start_dispatched = True
                    time.sleep(DEFAULT_POLL_INTERVAL)
                    continue

                updates = self._get_updates()
                if not updates:
                    continue

                for update in updates:
                    self._handle_update(update)
        except KeyboardInterrupt:
            print("🛑 Supervisor interrupted by user.")
        finally:
            self._stop_child(reason="Supervisor shutdown")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultima trading supervisor")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to launch the live trading process (default: current interpreter).",
    )
    parser.add_argument(
        "--script",
        default=str(ROOT_DIR / "live" / "demo_mt5.py"),
        help="Path to the live trading script to manage.",
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Automatically launch the live trading process when the supervisor starts.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    token, allowed_ids = load_telegram_settings()
    if not token:
        print("❌ Telegram bot token not configured. Update telegram_settings.json or environment variables.")
        return 1

    if not allowed_ids:
        print("❌ No authorized Telegram user IDs configured. Update telegram_settings.json or environment variables.")
        return 1

    supervisor = UltimaSupervisor(
        token=token,
        allowed_ids=allowed_ids,
        python_executable=args.python,
        script_path=Path(args.script).resolve(),
        auto_start=args.auto_start,
    )
    supervisor.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
