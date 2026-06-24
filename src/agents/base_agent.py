"""
src/agents/base_agent.py — Base class for all LLM agents in the multi-agent pipeline.

Each agent has:
- A specific role and system prompt
- Input/output validation
- LLM call with error handling
- Execution trace logging for transparency
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import GEMINI_MODEL, AGENT_MAX_TOKENS, AGENT_TEMPERATURE, get_api_key

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Base class for role-based LLM agents.

    Subclasses must implement:
        - role_name: str
        - system_prompt: str
        - build_user_message(context) -> str
        - parse_response(raw_text) -> dict
    """

    role_name: str = "BaseAgent"
    system_prompt: str = "You are a helpful assistant."

    def __init__(self):
        self.trace: dict[str, Any] = {}

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute this agent's task.

        1. Build the user message from context
        2. Call the LLM with the role-specific system prompt
        3. Parse the response
        4. Record execution trace

        Returns parsed output dict with 'success' key.
        """
        start_time = time.time()
        self.trace = {
            "agent": self.role_name,
            "start_time": start_time,
            "input_keys": list(context.keys()),
        }

        try:
            user_message = self.build_user_message(context)
            self.trace["prompt_length"] = len(user_message)

            raw_response = self._call_llm(user_message)
            self.trace["response_length"] = len(raw_response)

            parsed = self.parse_response(raw_response)
            parsed["success"] = True
            parsed["raw_response"] = raw_response

            elapsed = time.time() - start_time
            self.trace["elapsed_seconds"] = round(elapsed, 2)
            self.trace["status"] = "success"

            logger.info("%s completed in %.2fs", self.role_name, elapsed)
            return parsed

        except Exception as exc:
            elapsed = time.time() - start_time
            self.trace["elapsed_seconds"] = round(elapsed, 2)
            self.trace["status"] = "error"
            self.trace["error"] = str(exc)[:200]
            logger.error("%s failed: %s", self.role_name, exc)

            return {
                "success": False,
                "error": str(exc)[:200],
                "raw_response": "",
            }

    def build_user_message(self, context: dict[str, Any]) -> str:
        """Build the user message from context. Override in subclass."""
        raise NotImplementedError

    def parse_response(self, raw_text: str) -> dict[str, Any]:
        """Parse LLM response into structured output. Override in subclass."""
        return {"text": raw_text}

    def _call_llm(self, user_message: str) -> str:
        """Call Gemini API with this agent's system prompt."""
        from openai import OpenAI

        api_key = get_api_key()
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")

        client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

        response = client.chat.completions.create(
            model=GEMINI_MODEL,
            max_tokens=AGENT_MAX_TOKENS,
            temperature=AGENT_TEMPERATURE,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content

    def get_trace(self) -> dict[str, Any]:
        """Return execution trace for transparency/debugging."""
        return self.trace.copy()
