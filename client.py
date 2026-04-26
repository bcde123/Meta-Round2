"""GreenCodeEnv client — Gym-style API for the Green-Code Optimizer.

Provides client-side bindings that judges (and other training scripts) can
import to talk to a running environment server (locally or on Hugging Face
Spaces) without ever touching server-side internals. Mirrors OpenEnv's
`HTTPEnvClient` interface (RFC 001).

Example
-------

    from client import GreenCodeEnv

    env = GreenCodeEnv("https://s123hree-constrained-refactor-gauntlet-a100.hf.space")
    obs = env.reset(curriculum_level=2)
    print(obs.episode_id, len(obs.files))

    obs = env.step(action={"tool": "edit_file",
                           "args": {"filename": "math_utils.py",
                                    "content": new_code}})
    print(obs.reward, obs.done)

    state = env.state()
    rubric = env.rubric_tree()           # introspect the reward
    co2   = env.co2_dashboard(obs.episode_id)

    env.close()
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests


@dataclass
class Observation:
    """Gym-style observation returned by `reset()` / `step()`."""
    episode_id: str
    files: Dict[str, str] = field(default_factory=dict)
    reward: Optional[float] = None
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)
    steps_remaining: Optional[int] = None
    violation_report: Dict[str, Any] = field(default_factory=dict)


@dataclass
class State:
    """Gym-style state metadata returned by `state()`."""
    episode_id: str
    step_count: int
    steps_remaining: int
    done: bool
    hack_detected: bool
    n_files: int
    curriculum_level: Optional[int] = None


class GreenCodeEnv:
    """Sync HTTP client for the Green-Code Optimizer environment.

    Mirrors OpenEnv's `HTTPEnvClient` API (`reset`, `step`, `state`, `close`)
    so it composes naturally with TRL / Unsloth / RLlib training loops.

    Server URL examples:
      • Local Docker:        "http://localhost:7860"
      • Hugging Face Space:  "https://<user>-<space>.hf.space"
    """

    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._episode_id: Optional[str] = None
        self._session = requests.Session()

    # ── Gym-style API ─────────────────────────────────────────────────────
    def reset(self, curriculum_level: Optional[int] = None) -> Observation:
        """Start a new episode. Returns the initial observation."""
        body: Dict[str, Any] = {}
        if curriculum_level is not None:
            body["curriculum_level"] = curriculum_level
        r = self._session.post(f"{self.base_url}/reset", json=body, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        self._episode_id = data["episode_id"]
        obs = data["observation"]
        return Observation(
            episode_id=data["episode_id"],
            files=obs.get("files", {}),
            steps_remaining=obs.get("steps_remaining"),
            violation_report=obs.get("violation_report", {}),
            info=data.get("info", {}),
        )

    def step(self, action: Dict[str, Any], episode_id: Optional[str] = None) -> Observation:
        """Submit an action; receive new observation + reward + done flag."""
        eid = episode_id or self._episode_id
        if eid is None:
            raise RuntimeError("Call reset() before step()")
        r = self._session.post(
            f"{self.base_url}/step",
            json={"episode_id": eid, "action": action},
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        obs = data.get("observation", {})
        return Observation(
            episode_id=eid,
            files=obs.get("files", {}),
            reward=data.get("reward"),
            done=data.get("done", False),
            info=data.get("info", {}),
            steps_remaining=obs.get("steps_remaining"),
            violation_report=obs.get("violation_report", {}),
        )

    def state(self, episode_id: Optional[str] = None) -> State:
        """Return episode metadata (Gym API)."""
        eid = episode_id or self._episode_id
        if eid is None:
            raise RuntimeError("Call reset() before state()")
        r = self._session.get(f"{self.base_url}/state/{eid}", timeout=self.timeout)
        r.raise_for_status()
        d = r.json()
        return State(
            episode_id=d["episode_id"],
            step_count=d["step_count"],
            steps_remaining=d["steps_remaining"],
            done=d["done"],
            hack_detected=d["hack_detected"],
            n_files=d["n_files"],
            curriculum_level=d.get("curriculum_level"),
        )

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()
        self._episode_id = None

    # ── Environment-specific helpers ─────────────────────────────────────
    def rubric_tree(self) -> Dict[str, Any]:
        """Fetch the composable rubric tree (named children + formula)."""
        r = self._session.get(f"{self.base_url}/rubric", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def co2_dashboard(self, episode_id: Optional[str] = None) -> Dict[str, Any]:
        """Fetch the CO₂ savings dashboard JSON for an episode."""
        eid = episode_id or self._episode_id
        if eid is None:
            raise RuntimeError("Call reset() before co2_dashboard()")
        r = self._session.get(
            f"{self.base_url}/dashboard/co2/{eid}",
            headers={"Accept": "application/json"},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def health(self) -> Dict[str, Any]:
        """Server health check — gpu_available, version, etc."""
        r = self._session.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()


# Aliases for nicer imports
EnvClient = GreenCodeEnv


def main() -> None:
    """Tiny CLI smoke test:  `python client.py http://localhost:7860`."""
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860"
    env = GreenCodeEnv(url)
    print("→ /health", json.dumps(env.health(), indent=2))
    print("→ /rubric", json.dumps(env.rubric_tree(), indent=2))
    obs = env.reset()
    print(f"→ reset: episode_id={obs.episode_id} files={len(obs.files)} "
          f"steps={obs.steps_remaining}")
    state = env.state()
    print(f"→ state: step_count={state.step_count} done={state.done}")
    env.close()


if __name__ == "__main__":
    main()
