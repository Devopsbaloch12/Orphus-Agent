"""Windowed conversation history.

The prompt sent to the LLM is bounded on two independent axes, because either
one alone leaves a hole:

* ``history_max_turns`` bounds how far back the assistant remembers. Purely a
  product decision.
* ``history_max_chars`` bounds the prompt size. A single pathological turn (a
  user reading out a paragraph, an assistant that ignored the brevity
  instruction) can blow the context budget long before the turn limit is
  reached, and the resulting provider error would drop the whole conversation.

Trimming always removes *whole turns* from the oldest end. Cutting mid-turn
would leave an assistant reply with no preceding user message, which reads to
the model as if it spoke unprompted and measurably degrades the next reply. The
system prompt is never subject to either bound -- losing it changes the
assistant's persona mid-conversation, which is the most visible failure of all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from orphus.domain.types import Message, MessageRole
from orphus.observability.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from orphus.domain.types import ConversationTurn

logger = get_logger(__name__)

__all__ = ["History"]


class History:
    """A conversation transcript kept inside a turn and character budget.

    Args:
        system_prompt: Persona/instruction message pinned to the front of every
            prompt. ``None`` or empty means no system message is emitted.
        max_turns: Maximum user turns retained. A turn is a user message plus
            everything that follows it up to the next user message.
        max_chars: Maximum total characters across the system prompt and every
            retained message.

    Example:
        >>> history = History(system_prompt="Be brief.", max_turns=1, max_chars=1000)
        >>> history.add_user("first")
        >>> history.add_assistant("reply one")
        >>> history.add_user("second")
        >>> [message.content for message in history.messages()]
        ['Be brief.', 'second']
    """

    __slots__ = ("_max_chars", "_max_turns", "_messages", "_system_prompt", "_trimmed_turns")

    def __init__(
        self,
        *,
        system_prompt: str | None = None,
        max_turns: int = 20,
        max_chars: int = 8000,
    ) -> None:
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns}")
        if max_chars < 1:
            raise ValueError(f"max_chars must be >= 1, got {max_chars}")
        self._system_prompt = system_prompt or None
        self._max_turns = max_turns
        self._max_chars = max_chars
        self._messages: list[Message] = []
        self._trimmed_turns = 0

    # -- introspection ------------------------------------------------------

    @property
    def system_prompt(self) -> str | None:
        """The pinned system message, if any."""
        return self._system_prompt

    @property
    def turn_count(self) -> int:
        """User turns currently retained."""
        return sum(1 for message in self._messages if message.role is MessageRole.USER)

    @property
    def trimmed_turns(self) -> int:
        """Turns evicted by windowing over this history's lifetime."""
        return self._trimmed_turns

    @property
    def total_chars(self) -> int:
        """Characters across the system prompt and every retained message."""
        base = len(self._system_prompt) if self._system_prompt else 0
        return base + sum(len(message.content) for message in self._messages)

    def __len__(self) -> int:
        """Number of retained messages, excluding the system prompt."""
        return len(self._messages)

    # -- mutation -----------------------------------------------------------

    def set_system_prompt(self, prompt: str | None) -> None:
        """Replace the pinned system message and re-apply the char budget."""
        self._system_prompt = prompt or None
        self._trim()

    def add_user(self, text: str) -> None:
        """Append a user message and re-apply the window."""
        self.add(Message(MessageRole.USER, text))

    def add_assistant(self, text: str) -> None:
        """Append an assistant message and re-apply the window."""
        self.add(Message(MessageRole.ASSISTANT, text))

    def add(self, message: Message) -> None:
        """Append one message.

        Args:
            message: The message to append. A ``SYSTEM`` role is redirected to
                :meth:`set_system_prompt` rather than appended, so a stray
                system message cannot end up buried mid-transcript where most
                providers ignore it.
        """
        if message.role is MessageRole.SYSTEM:
            self.set_system_prompt(message.content)
            return
        self._messages.append(message)
        self._trim()

    def extend(self, messages: Iterable[Message]) -> None:
        """Append several messages, trimming once at the end."""
        for message in messages:
            if message.role is MessageRole.SYSTEM:
                self._system_prompt = message.content or None
            else:
                self._messages.append(message)
        self._trim()

    def add_turn(self, turn: ConversationTurn) -> None:
        """Append a completed exchange from a :class:`ConversationTurn`."""
        self.extend(turn.messages)

    def clear(self, *, keep_system_prompt: bool = True) -> None:
        """Drop the transcript.

        Args:
            keep_system_prompt: Retain the persona. Almost always ``True``;
                clearing it silently changes the assistant's behaviour.
        """
        self._messages.clear()
        if not keep_system_prompt:
            self._system_prompt = None

    # -- rendering ----------------------------------------------------------

    def messages(self) -> tuple[Message, ...]:
        """Render the prompt: system message first, then the retained window."""
        if self._system_prompt:
            return (Message(MessageRole.SYSTEM, self._system_prompt), *self._messages)
        return tuple(self._messages)

    def to_wire(self) -> list[dict[str, str]]:
        """Render the prompt in the OpenAI-compatible wire format."""
        return [message.to_wire() for message in self.messages()]

    # -- persistence --------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible mapping for the Redis hot store."""
        return {
            "system_prompt": self._system_prompt,
            "max_turns": self._max_turns,
            "max_chars": self._max_chars,
            "trimmed_turns": self._trimmed_turns,
            "messages": [message.to_wire() for message in self._messages],
        }

    @classmethod
    def restore(cls, snapshot: dict[str, Any]) -> History:
        """Rebuild a history from :meth:`snapshot` output.

        Unknown roles are dropped rather than raising: a cache entry written by
        an older build must never take down a live session.
        """
        history = cls(
            system_prompt=snapshot.get("system_prompt"),
            max_turns=int(snapshot.get("max_turns", 20)),
            max_chars=int(snapshot.get("max_chars", 8000)),
        )
        restored: list[Message] = []
        for raw in snapshot.get("messages", []):
            try:
                role = MessageRole(raw["role"])
            except (KeyError, ValueError):
                logger.warning("dropping unreadable cached message", extra={"raw": raw})
                continue
            if role is not MessageRole.SYSTEM:
                restored.append(Message(role, str(raw.get("content", ""))))
        history._messages = restored
        history._trimmed_turns = int(snapshot.get("trimmed_turns", 0))
        history._trim()
        return history

    # -- windowing ----------------------------------------------------------

    def _turn_starts(self) -> Sequence[int]:
        """Indices at which each retained turn begins."""
        return [
            index
            for index, message in enumerate(self._messages)
            if message.role is MessageRole.USER
        ]

    def _drop_oldest_turn(self) -> bool:
        """Remove everything up to the start of the second-oldest turn.

        Returns:
            ``False`` when only one turn remains, which is the floor: the
            current exchange must survive regardless of the budgets, because a
            prompt with no user message is not a prompt.
        """
        starts = self._turn_starts()
        if len(starts) < 2:
            return False
        del self._messages[: starts[1]]
        self._trimmed_turns += 1
        return True

    def _trim(self) -> None:
        """Enforce both budgets, oldest turn first."""
        while self.turn_count > self._max_turns and self._drop_oldest_turn():
            pass
        while self.total_chars > self._max_chars and self._drop_oldest_turn():
            pass
