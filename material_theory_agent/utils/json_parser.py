"""
utils/json_parser.py — Robust JSON extraction from LLM outputs.

LLMs sometimes wrap JSON in markdown fences or add preamble text.
This module handles all those edge cases so nodes never crash on
imperfect model output.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to extract a JSON object from raw LLM text.

    Strategy (in order):
    1. Direct JSON parse (model returned clean JSON)
    2. Strip ```json ... ``` markdown fences
    3. Find first { ... } block with regex
    4. Return None if all strategies fail
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown fences
    fence_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    fenced = re.findall(fence_pattern, text)
    for block in fenced:
        try:
            return json.loads(block.strip())
        except json.JSONDecodeError:
            continue

    # Strategy 3: find first { } block
    brace_pattern = r"\{[\s\S]*\}"
    matches = re.findall(brace_pattern, text)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    return None


def safe_json_parse(text: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract JSON from text; return fallback dict if parsing fails.
    Logs a warning so you can spot model output issues during dev.
    """
    result = extract_json(text)
    if result is None:
        print(f"[JSON Parser] WARNING: Could not parse JSON from:\n{text[:300]}...")
        return fallback
    return result
