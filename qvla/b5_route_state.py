"""Torch-independent, fail-closed lifecycle for one expert per action chunk."""
from __future__ import annotations

from dataclasses import dataclass


CHUNK_STEPS = 8


@dataclass(frozen=True)
class RouteToken:
    episode_key: str
    query_index: int
    expert_id: int
    serial: int


class ChunkRouteState:
    """A query selects an expert; eight executed actions admit the next query.

    ``end_query`` ends the model-forward scope, not the action execution scope.
    The executor must call ``record_executed_step`` after each environment step.
    An early-success episode can end with an unconsumed chunk. Instances are
    deliberately per evaluator and are not safe for concurrent model forwards.
    """

    def __init__(self, chunk_steps: int = CHUNK_STEPS):
        if type(chunk_steps) is not int or chunk_steps != CHUNK_STEPS:
            raise ValueError("B5 requires an eight-step action chunk")
        self.chunk_steps = chunk_steps
        self.episode_key: str | None = None
        self.next_query_index = 0
        self.executed_steps = 0
        self.remaining_actions = 0
        self._active: RouteToken | None = None
        self._serial = 0
        self._aborted = False

    @property
    def active_expert(self) -> int:
        if self._active is None:
            raise RuntimeError("Expert forward outside an active B5 query")
        return self._active.expert_id

    @property
    def active_token(self) -> RouteToken | None:
        return self._active

    def begin_episode(self, episode_key: str) -> None:
        if not isinstance(episode_key, str) or not episode_key:
            raise ValueError("Episode key must be a nonempty string")
        if self.episode_key is not None:
            raise RuntimeError("Previous B5 episode has not ended")
        self.episode_key = episode_key
        self.next_query_index = self.executed_steps = self.remaining_actions = 0
        self._active = None
        self._aborted = False

    def begin_query(self, episode_key: str, query_index: int, expert_id: int,
                    *, executed_step_start: int | None = None) -> RouteToken:
        if self.episode_key is None or episode_key != self.episode_key:
            raise RuntimeError("Query does not belong to the active episode")
        if self._aborted or self._active is not None or self.remaining_actions:
            raise RuntimeError("Query aborted, active, or prior chunk not consumed")
        if type(query_index) is not int or query_index != self.next_query_index:
            raise ValueError("Query index must advance exactly once per chunk")
        if type(expert_id) is not int or expert_id not in (0, 1):
            raise ValueError("Expert ID must be integer E0 or E1")
        if self.executed_steps != query_index * self.chunk_steps:
            raise RuntimeError("Query is not at an eight-step execution boundary")
        if executed_step_start is not None and (type(executed_step_start) is not int
                                                or executed_step_start != self.executed_steps):
            raise ValueError("Executor and router step counters disagree")
        self._serial += 1
        token = RouteToken(episode_key, query_index, expert_id, self._serial)
        self._active = token
        return token

    def _require_token(self, token: RouteToken) -> None:
        if self._active is None or token != self._active:
            raise RuntimeError("Stale or foreign B5 query token")

    def end_query(self, token: RouteToken, *, action_count: int = CHUNK_STEPS) -> None:
        self._require_token(token)
        if type(action_count) is not int or action_count != self.chunk_steps:
            raise ValueError("Policy query must produce exactly eight actions")
        self._active = None
        self.remaining_actions = action_count
        self.next_query_index += 1

    def abort_query(self, token: RouteToken) -> None:
        self._require_token(token)
        self._active = None
        self.remaining_actions = 0
        self._aborted = True

    def record_executed_step(self) -> None:
        if self.episode_key is None or self._active is not None or self._aborted:
            raise RuntimeError("Environment execution outside a valid completed query")
        if self.remaining_actions < 1:
            raise RuntimeError("Action execution exceeded the queued chunk")
        self.remaining_actions -= 1
        self.executed_steps += 1

    def end_episode(self) -> dict[str, int | str | bool]:
        if self.episode_key is None or self._active is not None:
            raise RuntimeError("No episode, or model query still active")
        receipt = {"episode_key": self.episode_key, "queries": self.next_query_index,
                   "executed_steps": self.executed_steps,
                   "discarded_actions": self.remaining_actions, "aborted": self._aborted}
        self.episode_key = None
        self.remaining_actions = 0
        self._aborted = False
        return receipt
