"""AgentAPI command line interface."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from agentapi.config.settings import SUPPORTED_PROVIDERS


MAIN_TEMPLATE = '''from agentapi import AgentAPI, Agent\n\napp = AgentAPI()\n\nagent = Agent(\n    system_prompt="You are a helpful assistant",\n    provider="{provider}",\n)\n\n\n@app.chat("/chat")\nasync def chat(message: str):\n    return await agent.run(message)\n\n\n@app.chat("/stream")\nasync def stream_chat(message: str):\n    return agent.stream(message)\n'''

TOOLS_TEMPLATE = '''from agentapi import tool\n\n\n@tool\ndef get_weather(city: str) -> str:\n    """Get weather information for a city."""\n    return f"Weather in {city}: sunny"\n'''

AGENTS_TEMPLATE = '''from agentapi import Agent\nfrom tools import get_weather\n\nassistant = Agent(\n    system_prompt="You are a helpful assistant",\n    provider="{provider}",\n    tools=[get_weather],\n)\n'''

ENV_TEMPLATE = '''OPENAI_API_KEY=\nGEMINI_API_KEY=\nOPENROUTER_API_KEY=\nDEFAULT_PROVIDER={provider}\n'''


def _write_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _prompt_with_default(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def _prompt_provider() -> str:
    """
    Prompt the user to select a supported provider.
    Loops until a valid choice is entered.
    """
    providers = sorted(SUPPORTED_PROVIDERS)
    provider_list = ", ".join(providers)

    print(f"\nSupported providers: {provider_list}")

    while True:
        choice = input(f"Enter provider [{providers[0]}]: ").strip().lower()

        # Accept empty input → use default (first alphabetically)
        if not choice:
            choice = providers[0]

        if choice in SUPPORTED_PROVIDERS:
            return choice

        print(
            f'  ✗ "{choice}" is not a supported provider.\n'
            f"  Supported: {provider_list}\n"
        )


def _collect_new_project_config(args: argparse.Namespace) -> tuple[str, str]:
    project_name = args.project_name
    provider = args.provider.lower()

    if args.interactive or not project_name:
        print("AgentAPI project configuration")
        print("Press Enter to accept defaults.")

        while True:
            project_name = _prompt_with_default("Project name", project_name or "myproject")
            if project_name:
                break
            print("Project name is required.")

        # ── UPDATED: Use the new constrained provider prompt ──
        provider = _prompt_provider()

    if not project_name:
        raise ValueError("Project name is required")

    return project_name, provider


def cmd_new(args: argparse.Namespace) -> int:
    try:
        project_name, provider = _collect_new_project_config(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    project_path = Path(project_name).resolve()

    if project_path.exists():
        print(f"Error: directory already exists: {project_path}", file=sys.stderr)
        return 1

    project_path.mkdir(parents=True, exist_ok=False)

    _write_file(project_path / "main.py", MAIN_TEMPLATE.format(provider=provider))
    _write_file(project_path / "tools.py", TOOLS_TEMPLATE)
    _write_file(project_path / "agents.py", AGENTS_TEMPLATE.format(provider=provider))
    _write_file(project_path / ".env", ENV_TEMPLATE.format(provider=provider))

    print(f"Created AgentAPI project at: {project_path}")
    print("Next steps:")
    print(f"  cd {project_path}")
    print("  uvicorn main:app --reload")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    app_target = args.app
    host = args.host
    port = str(args.port)

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        app_target,
        "--host",
        host,
        "--port",
        port,
    ]

    if args.reload:
        cmd.append("--reload")

    if args.workers is not None:
        cmd.extend(["--workers", str(args.workers)])

    env = os.environ.copy()
    try:
        completed = subprocess.run(cmd, env=env, check=False)
        return int(completed.returncode)
    except KeyboardInterrupt:
        return 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentapi", description="AgentAPI CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    new_parser = subparsers.add_parser("new", help="Create a new AgentAPI project")
    new_parser.add_argument("project_name", nargs="?", help="Directory name for the new project")
    new_parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "gemini", "openrouter"],
        help="Default provider to scaffold in generated files",
    )
    new_parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for project configuration values",
    )
    new_parser.set_defaults(func=cmd_new)

    run_parser = subparsers.add_parser("run", help="Run a FastAPI app via uvicorn")
    run_parser.add_argument("--app", default="main:app", help="ASGI app target (default: main:app)")
    run_parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    run_parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    run_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    run_parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    run_parser.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())