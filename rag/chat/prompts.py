"""Load centrally-managed system prompts from the package JSON file."""

import json
from importlib.resources import files


_PROMPTS = json.loads(
    files(__package__).joinpath("prompts.json").read_text(encoding="utf-8")
)

RAG_SYSTEM_PROMPT = _PROMPTS["rag_system_prompt"].strip()
CHAT_SYSTEM_PROMPT = _PROMPTS["chat_system_prompt"].strip()
COMPARISON_SYSTEM_PROMPT = _PROMPTS["comparison_system_prompt"].strip()
SUMMARY_SYSTEM_PROMPT = _PROMPTS["summary_system_prompt"].strip()
