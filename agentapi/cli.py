"""AgentAPI command line interface."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


MAIN_TEMPLATE = '''from agentapi import AgentApp, Agent\n\napp = AgentApp()\n\nagent = Agent(\n    system_prompt="You are a helpful assistant",\n    provider="{provider}",\n)\n\n\n@app.chat("/chat")\nasync def chat(message: str):\n    return await agent.run(message)\n\n\n@app.chat("/stream")\nasync def stream_chat(message: str):\n    return agent.stream(message)\n'''

TOOLS_TEMPLATE = '''from agentapi import tool\n\n\n@tool\ndef get_weather(city: str) -> str:\n    """Get weather information for a city."""\n    return f"Weather in {city}: sunny"\n'''

AGENTS_TEMPLATE = '''from agentapi import Agent\nfrom tools import get_weather\n\nassistant = Agent(\n    system_prompt="You are a helpful assistant",\n    provider="{provider}",\n    tools=[get_weather],\n)\n'''

ENV_TEMPLATE = '''OPENAI_API_KEY=\nGEMINI_API_KEY=\nOPENROUTER_API_KEY=\nDEFAULT_PROVIDER={provider}\n'''


def _write_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _prompt_with_default(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def _collect_new_project_config(args: argparse.Namespace) -> tuple[str, str]:
    provider_choices = ["openai", "gemini", "openrouter"]

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

        while True:
            print("Provider options:")
            for index, choice in enumerate(provider_choices, start=1):
                marker = " (default)" if choice == provider else ""
                print(f"  {index}. {choice}{marker}")

            selection = input(f"Select provider [default: {provider}]: ").strip().lower()
            if not selection:
                break

            if selection.isdigit():
                selected_index = int(selection)
                if 1 <= selected_index <= len(provider_choices):
                    provider = provider_choices[selected_index - 1]
                    break

            if selection in provider_choices:
                provider = selection
                break

            choices = ", ".join(f"{i + 1}:{name}" for i, name in enumerate(provider_choices))
            print(f"Invalid provider selection. Choose one of: {choices}")

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
