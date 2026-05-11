import json
from typing import Any

from agent_knowledge_harvester.agents.blueprint import MEMORY_SYNTHESIS_PROMPT
from agent_knowledge_harvester.memory.pack import (
    build_compact_memory_pack,
    render_compact_memory_pack,
)
from agent_knowledge_harvester.schemas.memory import AgentMemoryPack, CompactMemoryPack
from agent_knowledge_harvester.utils.files import write_json


async def synthesize_compact_memory_with_llm(
    pack: AgentMemoryPack,
    llm_client: object,
    out_dir,
) -> CompactMemoryPack:
    """Use the memory-synthesis expert to produce a compact agent memory layer."""
    fallback = build_compact_memory_pack(pack, budget="llm_compact")
    warning_path = out_dir / "agent_memory_pack.llm_compact.warning.json"
    if warning_path.exists():
        warning_path.unlink()
    try:
        result = await llm_client.chat_json(
            "linking",
            system_prompt=build_memory_synthesis_system_prompt(),
            user_prompt=build_memory_synthesis_user_prompt(pack),
            temperature=0.1,
        )
        compact = parse_compact_memory_payload(result.payload, pack)
    except Exception as exc:  # noqa: BLE001 - the pipeline must survive provider timeouts.
        compact = fallback
        write_json(
            warning_path,
            {
                "stage": "memory_synthesis",
                "fallback": "deterministic_compact_memory",
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
                "input_entries": len(pack.entries),
            },
        )
    write_json(out_dir / "agent_memory_pack.llm_compact.json", compact.model_dump(mode="json"))
    (out_dir / "agent_memory_pack.llm_compact.md").write_text(
        render_compact_memory_pack(compact),
        encoding="utf-8",
    )
    return compact


def build_memory_synthesis_system_prompt() -> str:
    return (
        MEMORY_SYNTHESIS_PROMPT.strip()
        + "\n\nReturn strict JSON only. Optimize for agents using this as context, "
        "not for human tutorial reading."
    )


def build_memory_synthesis_user_prompt(pack: AgentMemoryPack) -> str:
    entries = build_memory_synthesis_entries(pack)
    entries_json = json.dumps(entries, ensure_ascii=False)
    return (
        "Create a compact agent-development memory pack from these source-grounded entries.\n"
        "This is one layer of a larger knowledge base, not the whole knowledge store.\n"
        "Keep only high-frequency rules, current patterns, anti-patterns, and retrieval pointers.\n"
        "Prefer short imperative wording that another coding agent can act on.\n"
        "Do not include long explanations. Preserve source URLs in patterns or pointers.\n"
        "Return JSON with keys: core_rules, current_patterns, anti_patterns, "
        "retrieval_pointers, watchlist, memory_operations.\n\n"
        f"Entries: {entries_json}"
    )


def build_memory_synthesis_entries(
    pack: AgentMemoryPack,
    max_entries: int = 24,
    max_evidence_items: int = 1,
    max_text_chars: int = 360,
) -> list[dict[str, Any]]:
    """Prepare a bounded handoff payload for the LLM memory expert."""
    entries = [
        {
            "title": entry.card_title,
            "claim": trim_text(entry.claim, max_text_chars),
            "agent_move": trim_text(entry.agent_move, max_text_chars),
            "topics": [topic.value for topic in entry.topics],
            "priority_score": entry.priority_score,
            "source_url": str(entry.source_url),
            "evidence": [
                trim_text(evidence, max_text_chars)
                for evidence in entry.evidence[:max_evidence_items]
            ],
        }
        for entry in sorted(pack.entries, key=lambda item: item.priority_score, reverse=True)[
            :max_entries
        ]
    ]
    return entries


def trim_text(value: str, limit: int) -> str:
    """Trim long text fields while preserving complete enough semantic cues."""
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "..."


def parse_compact_memory_payload(
    payload: dict[str, Any],
    source_pack: AgentMemoryPack,
) -> CompactMemoryPack:
    fallback = build_compact_memory_pack(source_pack, budget="llm_compact")
    return CompactMemoryPack(
        source_pack="agent_memory_pack.json",
        budget="llm_compact",
        core_rules=clamp_string_list(payload.get("core_rules"), fallback.core_rules, 8),
        current_patterns=clamp_string_list(
            payload.get("current_patterns"),
            fallback.current_patterns,
            12,
        ),
        anti_patterns=clamp_string_list(payload.get("anti_patterns"), fallback.anti_patterns, 8),
        retrieval_pointers=clamp_string_list(
            payload.get("retrieval_pointers"),
            fallback.retrieval_pointers,
            8,
        ),
        watchlist=clamp_string_list(payload.get("watchlist"), fallback.watchlist, 6),
        memory_operations=clamp_string_list(
            payload.get("memory_operations"),
            fallback.memory_operations,
            8,
        ),
    )


def clamp_string_list(value: object, fallback: list[str], limit: int) -> list[str]:
    if not isinstance(value, list):
        return fallback[:limit]
    items = [str(item).strip() for item in value if str(item).strip()]
    return [item[:500] for item in items[:limit]] or fallback[:limit]
