"""Fake or HTTP-backed decisions over the existing MCP chemistry tools."""

from __future__ import annotations

import json
import os
from typing import Any, Protocol

import anyio
import httpx

from physlab.chem.visual_titration import dose_protocol, initial_protocol, unsafe_protocols
from physlab.mcp_server.server import build_server
from physlab.mcp_server.session import LabSession


class DoseLLM(Protocol):
    """Decision interface shared by offline and HTTP clients."""

    def choose(self, task: str, observations: list[tuple[float, float]]) -> float:
        """Choose 10 or 2 uL from observations.

        Args:
            task: Natural language experiment request.
            observations: Cumulative base volume and measured pH.

        Returns:
            Next dose in microliters.
        """
        ...


class FakeLLM:
    """Deterministic offline decision fixture with no network access."""

    def choose(self, task: str, observations: list[tuple[float, float]]) -> float:
        """Choose a conservative fine dose as the pH begins rising.

        Args:
            task: Natural language task, accepted for interface compatibility.
            observations: Measured titration history.

        Returns:
            A coarse or fine dose; no hidden equivalence volume is used.
        """
        return 2.0 if observations[-1][1] > 5.0 else 10.0


class HttpLLM:
    """OpenAI-compatible HTTP client using only httpx."""

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com",
        key_env: str = "PHYS_LAB_LLM_API_KEY",
    ) -> None:
        """Configure the endpoint and read only the selected key variable.

        Args:
            model: Provider model name.
            base_url: API origin, optionally ending in /v1.
            key_env: Environment variable containing the API key.

        Raises:
            ValueError: The selected variable is missing.
        """
        self.key = os.environ.get(key_env)
        if not self.key:
            raise ValueError(f"Real LLM requires {key_env}; set it or use the default FakeLLM.")
        self.model = model
        self.url = base_url.rstrip("/")
        if not self.url.endswith("/v1"):
            self.url += "/v1"

    def choose(self, task: str, observations: list[tuple[float, float]]) -> float:
        """Ask the provider for a bounded JSON dose decision.

        Args:
            task: Natural language request.
            observations: Base volume and measured pH, with no acid concentration.

        Returns:
            Validated dose of 10 or 2 uL.

        Raises:
            ValueError: The response requests an unsupported dose.
        """
        response = httpx.post(
            self.url + "/chat/completions",
            timeout=60,
            headers={"Authorization": f"Bearer {self.key}"},
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You control a simulated 300 uL well titration of unknown acetic "
                            "acid with phenolphthalein and 0.1 M NaOH. Choose coarse 10 uL "
                            "or fine 2 uL near the pH jump. Return only JSON "
                            '{"dose_ul": 10} or {"dose_ul": 2}.'
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps({"task": task, "volume_ph": observations}),
                    },
                ],
            },
        )
        response.raise_for_status()
        dose = float(json.loads(response.json()["choices"][0]["message"]["content"])["dose_ul"])
        if dose not in (2, 10):
            raise ValueError("LLM dose must be 2 or 10 uL")
        return dose


async def run_agent(
    session: LabSession,
    llm: DoseLLM,
    task: str = "Visually titrate unknown acetic acid with phenolphthalein.",
) -> dict[str, Any]:
    """Run checked microprotocols through registered MCP tools and await jobs.

    Args:
        session: Chemistry gantry session.
        llm: Offline fixture or real HTTP decision client.
        task: Natural language experiment objective.

    Returns:
        Observations, captured animation frames, safety log and report path.

    Raises:
        RuntimeError: A check, job, or bounded endpoint fails.
    """
    server = build_server(session)
    logs: list[str] = []
    observations: list[tuple[float, float]] = []
    frames = []

    async def call(name: str, request: dict[str, Any]) -> dict[str, Any]:
        result = await server.call_tool(name, {"request": request})
        envelope = json.loads(next(block.text for block in result.content if block.type == "text"))
        if not envelope["ok"]:
            raise RuntimeError(f"{name}: {envelope['error']}")
        return envelope["data"]

    for label, protocol in unsafe_protocols().items():
        checked = await call("check_chem_protocol", {"protocol": protocol.model_dump(mode="json")})
        if checked["valid"]:
            raise RuntimeError(f"Unsafe fixture unexpectedly accepted: {label}")
        logs.append(
            f"BLOCKED {label}: " + "; ".join(checked["issues"] + checked["safety"]["violations"])
        )

    protocol = initial_protocol()
    for _ in range(30):
        payload = {"protocol": protocol.model_dump(mode="json")}
        checked = await call("check_chem_protocol", payload)
        if not checked["valid"]:
            raise RuntimeError(str(checked))
        logs.append("PASS check_chem_protocol: " + protocol.id)
        started = await call("run_chem_protocol", {**payload, "question": task})
        with anyio.fail_after(120):
            while True:
                job = await call("get_job", {"job_id": started["job_id"]})
                if job["status"] in ("done", "error"):
                    break
                await anyio.sleep(0.02)
        if job["status"] != "done":
            raise RuntimeError(str(job["error"]))
        run_id = job["result"]["run_id"]
        trial = session.runs[run_id].trials[-1]
        observations.append((trial.values["titrant_ul"], trial.measurements["ph"]))
        await call("capture_image", {"camera": "chem_overhead"})
        bridge = session.chem_bridge
        for shot in bridge.frames:
            frames.append((shot, list(observations), list(logs)))
        bridge.frames.clear()
        frames.append((bridge.capture(), list(observations), list(logs)))
        if observations[-1][1] >= 9.5:
            break
        protocol = dose_protocol(llm.choose(task, observations))
    else:
        raise RuntimeError("Titration exceeded the bounded 30-step budget")
    fits = await call("fit_models", {"run_id": run_id, "x": "titrant_ul", "y": "ph"})
    report = await call("get_report", {"run_id": run_id, "x": "titrant_ul", "y": "ph"})
    return {
        "observations": observations,
        "frames": frames,
        "logs": logs,
        "report": report,
        "fits": fits,
        "directory": job["result"]["directory"],
    }
