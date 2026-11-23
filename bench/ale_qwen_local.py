"""Run the ALE office protocol with Qwen's native SGLang tool calling."""

import functools
import runpy

import litellm
from agent.decorators import register_agent
from ale_run.agents.ale_claw.harness.model.unified_loop import UnifiedAgentConfig

MODEL = "openai/qwen3.8-27b"
ENDPOINT = "http://127.0.0.1:30019/v1"


@register_agent(models=r"^openai/qwen3\.8-27b$", priority=30)
class QwenLocalConfig(UnifiedAgentConfig):
    """Reuse ALE-Claw's Chat Completions tooling for the local Qwen profile."""


def pin_local_endpoint(call):
    @functools.wraps(call)
    async def local_call(*args, **kwargs):
        if kwargs.get("model") != MODEL:
            raise ValueError("ALE helper selected a model outside the local Qwen profile")
        kwargs.update(api_base=ENDPOINT, api_key="local-eval")
        return await call(*args, **kwargs)

    return local_call


if __name__ == "__main__":
    litellm.acompletion = pin_local_endpoint(litellm.acompletion)
    litellm.aresponses = pin_local_endpoint(litellm.aresponses)
    runpy.run_module("ale_run", run_name="__main__")
