# Contributing to AgentAPI

Thanks for your interest in contributing to AgentAPI! This document outlines the development setup, guidelines, and a list of active issues that need your help.

---

## 1. Development Setup

1. **Fork and clone** the repository.
2. **Create and activate** a virtual environment:
   ```bash
   python -m venv .venv
   
   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   
   # macOS/Linux
   source .venv/bin/activate
   ```
3. **Upgrade pip and install dependencies** in editable mode:
   ```bash
   pip install -U pip
   pip install -e .
   ```

---

## 2. Local Runs & Testing

### Run the example app
```bash
uvicorn examples.main:app --reload
```

### Run the tests
Ensure you have `pytest` installed and run it from the root directory:
```bash
python -m pytest
```

---

## 3. Code Style & PR Guidelines

* **Keep changes focused and minimal**: Submit separate PRs for unrelated bug fixes.
* **Docstrings**: Add docstrings to new public methods/classes.
* **Python Compatibility**: Maintain compatibility with Python 3.10+.
* **Commit Messages**: Use semantic, short messages. Examples:
  * `fix: handle streaming http errors safely`
  * `feat: add trace logging middleware`
  * `docs: fix typo in README`

---

## 4. Active Issues for Contributors (Help Wanted)

Below is a curated list of open bugs/issues that are ready to be picked up. Each issue contains details and a suggested patch strategy.

---


### Issue #2: `parse_tool_args` crashes on non-string inputs

* **Affected Files**: [agentapi/agent/tools.py](agentapi/agent/tools.py)
* **Severity**: Medium
* **Description**: `parse_tool_args` calls `args_json.strip()`. If `args_json` is not a string (e.g. if the provider parsed it as a dictionary already, or if it is `None`), it raises an `AttributeError` instead of handling it gracefully.
* **Recommended Fix**: Safely handle `None`, dictionaries, or other non-string types.
* **Suggested Patch**:
  Update `parse_tool_args` in [agentapi/agent/tools.py](agentapi/agent/tools.py):
  ```python
  def parse_tool_args(args_json: str | dict[str, Any] | None) -> dict[str, Any]:
      if args_json is None:
          return {}
      if isinstance(args_json, dict):
          return args_json
      if not isinstance(args_json, str):
          args_json = str(args_json)
      if not args_json.strip():
          return {}
      try:
          return json.loads(args_json)
      except json.JSONDecodeError as exc:
          raise AgentProviderError(
              f"Failed to parse tool arguments as JSON: {exc}. Raw input: {args_json[:200]!r}",
              status_code=422,
          ) from exc
  ```

---

### Issue #3: Silent exception swallowing in `_safe_error_detail` implementations

* **Affected Files**: [agentapi/providers/openai_compatible.py](agentapi/providers/openai_compatible.py), [agentapi/providers/gemini.py](agentapi/providers/gemini.py)
* **Severity**: Medium
* **Description**: Both `_safe_error_detail` and `_safe_error_detail_sync` catch all exceptions (`except Exception: pass`) without logging them. This masks issues reading response bodies or parsing errors during diagnostics.
* **Recommended Fix**: Log exceptions using a module-level logger before falling back to reason phrases.
* **Suggested Patch**:
  Update `_safe_error_detail` in [agentapi/providers/openai_compatible.py](agentapi/providers/openai_compatible.py):
  ```python
      async def _safe_error_detail(self, response: httpx.Response) -> str:
          try:
              raw = await response.aread()
              if raw:
                  return raw.decode(errors="replace").strip()[:500]
          except Exception as exc:
              logger.debug("Failed to read response body for error detail: %s", exc)
  
          try:
              text = response.text
              if text:
                  return text.strip()[:500]
          except Exception as exc:
              logger.debug("Failed to read response text for error detail: %s", exc)
  
          return response.reason_phrase or "Unknown error"
  ```

---

### Issue #4: Broad exception handlers that silence stack traces in tool execution

* **Affected Files**: [agentapi/agent/agent.py](agentapi/agent/agent.py)
* **Severity**: Medium
* **Description**: Generic `except Exception` blocks catch all errors during tool execution and return sanitized error strings without logging the full stack trace, making troubleshooting difficult.
* **Recommended Fix**: Log the exception with `logger.exception(...)` before returning the sanitized error.
* **Suggested Patch**:
  Update `_execute_tool_calls` in [agentapi/agent/agent.py](agentapi/agent/agent.py):
  ```python
              except Exception as exc:  # noqa: BLE001
                  logger.exception("Failed to execute tool '%s'", call.name)
                  output = f"Tool execution failed: {exc}"
  ```

---

### Issue #5: Constructing `Agent` with provider instance results in incorrect `provider_name`

* **Affected Files**: [agentapi/agent/agent.py](agentapi/agent/agent.py)
* **Severity**: Low/Medium
* **Description**: When `Agent` is passed a provider instance, `self.provider_name` is derived from `provider.__class__.__name__.lower()`. This evaluates to names like `"openaiprovider"` or `"geminiprovider"`, which fail to match default model selection maps (e.g. `_default_model_for` checks for `"openai"`, `"gemini"`).
* **Recommended Fix**: Strip the trailing `"provider"` suffix from the class name.
* **Suggested Patch**:
  Update `__init__` in [agentapi/agent/agent.py](agentapi/agent/agent.py):
  ```python
          if isinstance(provider, BaseProvider):
              raw_name = provider.__class__.__name__.lower()
              if raw_name.endswith("provider"):
                  self.provider_name = raw_name[:-8]
              else:
                  self.provider_name = raw_name
  ```

---

### Issue #6: Tests fail in some local runs due to lack of editable package installation in PYTHONPATH

* **Affected Files**: [tests/test_inmemory_conversations.py](tests/test_inmemory_conversations.py)
* **Severity**: Medium (Developer Experience)
- **Description**: Running test suites directly or through some IDE test runners can raise `ModuleNotFoundError: No module named 'agentapi'` because the package is not installed in editable mode or added to `sys.path`.
- **Recommended Fix**: Add instructions or a `pytest.ini` / `conftest.py` containing path insertion, or update CI steps to install in editable mode `pip install -e .` first.

---

### Issue #7: Potential information leak from raw exception messages in `AgentAPIProviderError`

* **Affected Files**: [agentapi/agent/agent.py](agentapi/agent/agent.py)
* **Severity**: Medium (Security-sensitive)
* **Description**: Upstream LLM provider exceptions may contain sensitive information (such as API keys, authorization headers, or database strings in debug dumps). Including `str(exc)` in client-facing exception messages could leak credentials.
* **Recommended Fix**: Log the original error server-side and raise a sanitized message.
* **Suggested Patch**:
  Update `run` in [agentapi/agent/agent.py](agentapi/agent/agent.py):
  ```python
              except Exception as exc:
                  logger.exception("[AgentAPI] Provider error during chat completion")
                  raise AgentAPIProviderError(
                      "Provider call failed. Check server logs for details.",
                      original=exc,
                  ) from exc
  ```

---

### Issue #8: Anthropic tool calling crashes on subsequent turns due to missing preceding `tool_use` blocks in message history

* **Affected Files**: [agentapi/providers/anthropic.py](agentapi/providers/anthropic.py)
* **Severity**: High / Critical
* **Description**: 
  In `_format_messages`, assistant messages are formatted as `{"role": msg["role"], "content": msg["content"]}`. This completely discards the `tool_calls` list inside the assistant's message.
  
  When the subsequent `tool` response message is formatted, it creates a `tool_result` block. However, the Anthropic API requires that if a message has `tool_result` blocks, the immediately preceding assistant message *must* contain the corresponding `tool_use` blocks. Discarding `tool_calls` causes Anthropic to reject the request with an API error.
  
* **Recommended Fix**: Format assistant messages as a list of content blocks, translating OpenAI-style `tool_calls` into Anthropic `tool_use` blocks.
* **Suggested Patch**:
  Update `_format_messages` in [agentapi/providers/anthropic.py](agentapi/providers/anthropic.py):
  ```python
      def _format_messages(self, messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
          system_prompt = ""
          anthropic_messages = []
          
          for msg in messages:
              if msg["role"] == "system":
                  system_prompt = msg["content"]
              elif msg["role"] == "user":
                  anthropic_messages.append({"role": "user", "content": msg["content"]})
              elif msg["role"] == "assistant":
                  content_blocks = []
                  if msg.get("content"):
                      content_blocks.append({"type": "text", "text": msg["content"]})
                  
                  for call in msg.get("tool_calls") or []:
                      try:
                          input_data = json.loads(call["function"]["arguments"]) if isinstance(call["function"]["arguments"], str) else call["function"]["arguments"]
                      except Exception:
                          input_data = {}
                      content_blocks.append({
                          "type": "tool_use",
                          "id": call["id"],
                          "name": call["function"]["name"],
                          "input": input_data,
                      })
                  anthropic_messages.append({"role": "assistant", "content": content_blocks})
              elif msg["role"] == "tool":
                  anthropic_messages.append({
                      "role": "user", 
                      "content": [{"type": "tool_result", "tool_use_id": msg.get("tool_call_id"), "content": msg["content"]}]
                  })
          
          return system_prompt, anthropic_messages
  ```

---

### Issue #9: Gemini provider crashes with 400 Bad Request when tools have parameters with default values (type array compatibility)

* **Affected Files**: [agentapi/providers/gemini.py](agentapi/providers/gemini.py)
* **Severity**: High
* **Description**: 
  OpenAI schemas generated for functions with default arguments represent optional types as lists, e.g. `type: ["string", "null"]`. Gemini's function declarations schema does not support arrays for the `type` property and will return a `400 Bad Request` API error.
  
* **Recommended Fix**: Recursively traverse the schema parameters and clean up type arrays (e.g. converting `["string", "null"]` to `"string"` and setting `nullable: true`) for compatibility with Gemini's API.
* **Suggested Patch**:
  Update `_to_function_declarations` in [agentapi/providers/gemini.py](agentapi/providers/gemini.py):
  ```python
      def _clean_schema_for_gemini(self, schema: dict[str, Any]) -> dict[str, Any]:
          if not isinstance(schema, dict):
              return schema
          cleaned = dict(schema)
          if "type" in cleaned:
              t = cleaned["type"]
              if isinstance(t, list):
                  non_null = [x for x in t if x != "null"]
                  cleaned["type"] = non_null[0] if non_null else "string"
                  cleaned["nullable"] = True
          if "properties" in cleaned and isinstance(cleaned["properties"], dict):
              cleaned["properties"] = {
                  k: self._clean_schema_for_gemini(v) for k, v in cleaned["properties"].items()
              }
          if "items" in cleaned and isinstance(cleaned["items"], dict):
              cleaned["items"] = self._clean_schema_for_gemini(cleaned["items"])
          return cleaned
  
      def _to_function_declarations(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
          declarations: list[dict[str, Any]] = []
          for tool in tools:
              function_schema = tool.get("function")
              if not function_schema:
                  continue
              params = function_schema.get("parameters", {"type": "object", "properties": {}})
              declaration = {
                  "name": function_schema.get("name", ""),
                  "description": function_schema.get("description", ""),
                  "parameters": self._clean_schema_for_gemini(params),
              }
              declarations.append(declaration)
          return declarations
  ```

---

### Issue #10: OpenAICompatibleProvider ignores error chunks during streaming, leading to silent failures

* **Affected Files**: [agentapi/providers/openai_compatible.py](agentapi/providers/openai_compatible.py)
* **Severity**: Medium
* **Description**: If an upstream API returns an error block inside a stream chunk (e.g. `{"error": {"message": "..."}}`), `OpenAICompatibleProvider.stream` checks for `chunk.get("choices")`. Finding it empty, it silently calls `continue`. The client gets no indication of error, and the stream ends abruptly.
* **Recommended Fix**: Check for `"error"` in the parsed chunk and raise an `AgentProviderError`.
* **Suggested Patch**:
  Update `stream` in [agentapi/providers/openai_compatible.py](agentapi/providers/openai_compatible.py):
  ```python
                          try:
                              chunk = json.loads(data)
                          except json.JSONDecodeError:
                              continue
  
                          if "error" in chunk:
                              err_msg = chunk["error"].get("message") or str(chunk["error"])
                              raise AgentProviderError(
                                  f"Upstream stream error: {err_msg}",
                                  status_code=500,
                              )
  ```

---

### Issue #11: `InMemoryMemory` instances with the same `conversation_id` do not share state (Isolation Bug)

* **Affected Files**: [agentapi/agent/memory.py](agentapi/agent/memory.py)
* **Severity**: Medium / High (Architectural)
* **Description**: 
  `InMemoryMemory` stores message lists in `self._conversations` which is an instance attribute. In stateless contexts like a FastAPI app, constructing `InMemoryMemory(conversation_id=some_id)` on each request creates a brand new instance with a fresh, empty history. This prevents the developer from retrieving conversation history by ID.
  
* **Recommended Fix**: Store conversations in a class-level dictionary `_global_conversations` so different instances using the same `conversation_id` share history.
  *(Note: The test `test_multiple_agents_same_conversation` asserts that separate instances remain isolated. That test should be updated, as isolating matching IDs defeats the purpose of looking up a conversation by ID.)*
* **Suggested Patch**:
  Update `InMemoryMemory` in [agentapi/agent/memory.py](agentapi/agent/memory.py):
  ```python
  class InMemoryMemory(MemoryBackend):
      _global_conversations: dict[str, list[dict[str, Any]]] = {}
  
      def __init__(self, conversation_id: str | None = None) -> None:
          if conversation_id is not None:
              self.conversation_id = str(UUID(conversation_id))
          else:
              self.conversation_id = create_conversation_id()
  
          if self.conversation_id not in self._global_conversations:
              self._global_conversations[self.conversation_id] = []
  
      @property
      def messages(self) -> list[dict[str, Any]]:
          return self._global_conversations.get(self.conversation_id, [])
  
      def add(self, message: dict[str, Any]) -> None:
          if self.conversation_id not in self._global_conversations:
              self._global_conversations[self.conversation_id] = []
          self._global_conversations[self.conversation_id].append(message)
  
      def reset(self) -> None:
          self._global_conversations[self.conversation_id] = []
  ```

---

### Issue #12: Documentation mismatch (missing `multi_user_example.py` in `examples`)

* **Affected Files**: [README.md](README.md)
* **Severity**: Low
* **Description**: The `README.md` references a file `examples/multi_user_example.py` explaining how to build conversation-aware FastAPI apps. This file is missing from the repository.
* **Recommended Fix**: Scaffolding the missing `examples/multi_user_example.py` or updating the docs.

---

### Issue #13: Streaming mode accepts `tools` parameter but does not parse or run tool calls

* **Affected Files**: [agentapi/agent/agent.py](agentapi/agent/agent.py)
* **Severity**: High
* **Description**: 
  The streaming path `_stream_generator` passes `tools=self._tool_schemas()` to `provider.stream`. However, neither the provider stream parsers nor `_stream_generator` contain logic to yield, execute, or recurse on tool calls during streaming. If a model attempts a tool call in streaming mode, it is completely swallowed or outputs empty tokens, hanging or breaking the agent loop.
  
* **Recommended Fix**: Clear tools/tool_calling from streaming requests, or raise an error/warning if the developer attempts to use tools with streaming.
* **Suggested Patch**:
  Update `_stream_generator` in [agentapi/agent/agent.py](agentapi/agent/agent.py):
  ```python
              async for token in provider.stream(
                  conversation_messages,
                  tools=None, # Tools not supported in streaming
                  tool_calling=None,
              ):
  ```

---

## 5. Recently Resolved Issues

### Issue #1: OpenAI Strict Schema overrides parameter default arguments with `None` (Resolved)

* **Affected Files**: [agentapi/agent/tools.py](agentapi/agent/tools.py), [agentapi/agent/agent.py](agentapi/agent/agent.py)
* **Severity**: High
* **Description**: 
  In `_build_openai_tool_schema`, all parameter names are appended to the `required` list to satisfy OpenAI's strict mode (`"strict": true`). If a python parameter has a default value (e.g. `format: str = "celsius"`), the schema marks it as required but adds `"null"` to the allowed types (`["string", "null"]`). 
  
  When calling the tool, the LLM will explicitly generate `"format": null` if it doesn't specify a value. When `func(**args)` is executed in `_execute_tool_calls`, python binds `format=None`, which completely overrides the function's default value (`"celsius"`). This leads to runtime type errors or incorrect tool behaviors.
  
* **Recommended Fix**: Keep all properties in the `required` list for strict mode, but inside `_execute_tool_calls`, filter out or resolve parameters where the value is `None` but the function signature has a defined default parameter that is not `inspect._empty`.
* **Suggested Patch**:
  Update [agentapi/agent/agent.py](agentapi/agent/agent.py) inside `_execute_tool_calls`:
  ```python
              try:
                  args = parse_tool_args(call.arguments)
                  # Resolve default parameters if the LLM sent null/None
                  sig = inspect.signature(tool_def.func)
                  for param_name, param in sig.parameters.items():
                      if param.default is not inspect._empty and args.get(param_name) is None:
                          args[param_name] = param.default
  
                  result = tool_def.func(**args)
  ```
