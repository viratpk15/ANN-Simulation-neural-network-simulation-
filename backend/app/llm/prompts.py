"""Prompt construction for the explanation layer.

The strict separation of concerns is enforced *here*: the model only ever sees
numbers and findings that the deterministic engine already computed. It is
told explicitly that it must not compute or invent any of them.

Nothing sensitive is included: the payload is aggregate statistics and the
engine's findings, never raw user data, prompts or credentials.
"""
from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = """\
You are an educational neural-network analysis assistant helping a student \
understand what happened during a real training run in "NeuroSim Lab".

You will receive STRUCTURED DIAGNOSTIC FINDINGS that were computed \
deterministically by the lab's analysis engine from real recorded numbers \
(loss curves, metrics, activation statistics, gradient statistics, weight \
statistics, architecture and dataset summaries).

Your role is to:
- explain the detected issues in plain, accurate language
- explain the relevant machine-learning concepts behind them
- explain the possible causes and why they produce those symptoms
- explain the suggested experiments or remedies and what to watch for
- help the student learn and reason, not just receive a verdict

Hard rules:
1. Never invent, recompute, estimate or round any metric. Only use numbers \
supplied to you. If a number is not in the input, do not mention it.
2. Never invent or guess architecture details, dataset statistics, parameter \
counts, or configuration values. Only describe what was supplied.
3. Never claim certainty. The findings are heuristics; use hedged language \
("this suggests", "a likely cause is", "consider checking") when the evidence \
is suggestive rather than conclusive.
4. Clearly separate what was OBSERVED (reported metrics/findings) from what is \
HYPOTHESIS (your interpretation of possible causes).
5. Prefer technically grounded, educational explanations over generic advice.
6. Do not replace or contradict the deterministic diagnosis. The structured \
findings are the source of truth; you are explaining them, not disputing them.
7. If the findings report a healthy run, say so plainly rather than \
manufacturing problems.

Write for a learner: clear, concrete, and encouraging. Use short paragraphs or \
bullets. Aim for at most 250 words.
"""

# Keys the model is explicitly told are already known to it. Used to build the
# prompt header and to document the contract.
STRUCTURED_SECTIONS = (
    "architecture_summary",
    "dataset_summary",
    "training_metrics",
    "loss_history_summary",
    "activation_statistics",
    "gradient_statistics",
    "weight_statistics",
    "diagnostic_findings",
    "possible_causes",
    "suggested_checks",
)


def build_user_prompt(payload: dict[str, Any], findings: list[dict]) -> str:
    """Render the structured payload + findings as the single user message.

    ``payload`` is the sanitised context assembled by the API layer; only the
    sections the engine actually produced are included, so the prompt never
    implies data that was not measured.
    """
    parts: list[str] = [
        "Below are deterministic findings computed from a completed training run.",
        "Explain them for the student.",
    ]

    context = payload.get("context") or {}
    if context:
        parts.append(f"\nRun context (observed):\n{_json(context, 900)}")

    for section in ("architecture_summary", "dataset_summary", "training_metrics",
                    "loss_history_summary", "activation_statistics",
                    "gradient_statistics", "weight_statistics"):
        value = payload.get(section)
        if value:
            parts.append(f"\n{section} (observed):\n{_json(value, 700)}")

    if findings:
        lines: list[str] = []
        for f in findings:
            lines.append(f"- [{str(f.get('severity', 'info')).upper()}] "
                         f"{f.get('title', '')} (code: {f.get('code', '')})")
            if f.get("explanation"):
                lines.append(f"  What the engine observed: {f['explanation']}")
            if f.get("suggestions"):
                lines.append("  Engine-suggested checks: " + "; ".join(f["suggestions"]))
        parts.append("\nDeterministic findings (source of truth):\n" + "\n".join(lines))

    causes = payload.get("possible_causes")
    if causes:
        parts.append(f"\npossible_causes (engine hypotheses):\n{_json(causes, 600)}")
    checks = payload.get("suggested_checks")
    if checks:
        parts.append(f"\nsuggested_checks (engine recommendations):\n{_json(checks, 600)}")

    parts.append(
        "\nReminder: use only the numbers above, hedge appropriately, and do not "
        "introduce any metric that is not present."
    )
    return "\n".join(parts)


def _json(value: Any, limit: int) -> str:
    """Compact JSON, truncated to keep the prompt (and latency) bounded."""
    try:
        text = json.dumps(value, default=str, separators=(",", ":"))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return str(value)[:limit]
    return text if len(text) <= limit else text[:limit] + " …(truncated)"
