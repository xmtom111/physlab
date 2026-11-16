"""Run the offline chemistry gantry demo, optionally using an HTTP LLM."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import anyio

from physlab.chem.demo_gif import write_demo_gif
from physlab.chem.llm_agent import FakeLLM, HttpLLM, run_agent
from physlab.core.config import load_config
from physlab.mcp_server.session import LabSession
from physlab.sim.gl import close_renderers


def main() -> None:
    """Run checked titration and print artifact paths.

    Raises:
        SystemExit: Real LLM configuration is missing or invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-llm", action="store_true")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--base-url", default="https://api.openai.com")
    parser.add_argument("--key-env", default="PHYS_LAB_LLM_API_KEY")
    parser.add_argument("--write-docs-gif", action="store_true")
    args = parser.parse_args()
    try:
        llm = HttpLLM(args.model, args.base_url, args.key_env) if args.real_llm else FakeLLM()
    except ValueError as exc:
        parser.error(str(exc))
    session = LabSession(load_config(), "chem_gantry")
    try:
        result = anyio.run(run_agent, session, llm)
    finally:
        close_renderers()
    path = Path(result["directory"]) / "demo_chem_gantry.gif"
    write_demo_gif(result["frames"], path)
    if args.write_docs_gif:
        Path("docs").mkdir(exist_ok=True)
        shutil.copyfile(path, "docs/demo_chem_gantry.gif")
    print("GIF:", path)
    print("Report:", result["report"]["path"])
    print("Final (uL, pH):", result["observations"][-1])


if __name__ == "__main__":
    main()
