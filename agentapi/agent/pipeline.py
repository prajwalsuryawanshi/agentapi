"""Orchestration pipelines for chaining multiple agents."""

from __future__ import annotations

import logging
from typing import AsyncIterator

from fastapi.responses import StreamingResponse

from agentapi.agent.agent import Agent, AgentAPIProviderError

logger = logging.getLogger(__name__)


class SequentialPipeline:
    """Orchestrates a sequential Planner -> Executor agent pipeline.
    
    The Planner Agent accepts the initial objective and produces a strategy or plan.
    The Executor Agent ingests the Planner's output as context and performs the final operations.
    """

    def __init__(self, planner: Agent, executor: Agent) -> None:
        self.planner = planner
        self.executor = executor

    async def run(self, message: str) -> str:
        """Run the pipeline sequentially without streaming."""
        plan = await self.planner.run(message)
        
        executor_context = (
            f"Original Request: {message}\n\n"
            f"Execution Plan:\n{plan}"
        )
        return await self.executor.run(executor_context)

    def stream(self, message: str) -> StreamingResponse:
        """Run the pipeline sequentially and stream the Executor's response.
        
        This captures the full output of the Planner agent asynchronously,
        constructs the runtime context for the Executor, and then yields
        the Executor's token stream back to the client.
        """
        async def _sse_generator() -> AsyncIterator[str]:
            try:
                # Step 1: Run the planner to completion internally
                plan = await self.planner.run(message)
                
                # Step 2: Formulate the executor context
                executor_context = (
                    f"Original Request: {message}\n\n"
                    f"Execution Plan:\n{plan}"
                )
                
                # Step 3: Stream the executor response
                async for token in self.executor._stream_generator(executor_context):
                    for line in str(token).replace("\r", "").split("\n"):
                        yield f"data: {line}\n"
                    yield "\n"
                    
            except AgentAPIProviderError:
                logger.exception("[AgentAPI] Streaming error surfaced in SSE generator.")
                yield "data: [ERROR] Streaming failed. Check server logs for details.\n\n"
            except Exception as exc:  # noqa: BLE001
                logger.exception("[AgentAPI] Pipeline error surfaced in SSE generator.")
                yield f"data: [ERROR] Pipeline failed: {exc}\n\n"

        return StreamingResponse(_sse_generator(), media_type="text/event-stream")
