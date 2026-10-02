"""Regression tests for the project scaffolding CLI."""

from types import SimpleNamespace

from agentapi.cli import cmd_new


def test_generated_main_uses_shared_assistant_from_agents(tmp_path):
    """The scaffold should import and use the assistant defined in agents.py."""
    project_dir = tmp_path / "myproject"

    exit_code = cmd_new(
        SimpleNamespace(
            project_name=str(project_dir),
            provider="openai",
            interactive=False,
        )
    )

    assert exit_code == 0

    main_content = (project_dir / "main.py").read_text(encoding="utf-8")
    agents_content = (project_dir / "agents.py").read_text(encoding="utf-8")

    assert "from agents import assistant" in main_content
    assert "Agent(" not in main_content
    assert "assistant.run(message)" in main_content
    assert "assistant.stream(message)" in main_content
    assert "tools=[get_weather]" in agents_content
