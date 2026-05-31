import re

from agent_knowledge_harvester.config import Settings
from agent_knowledge_harvester.schemas.ingestion import CleanDocument, RawDocument
from agent_knowledge_harvester.utils.token_counter import estimate_cost_usd, estimate_tokens

TECHNICAL_TERMS = {
    "agent",
    "agents",
    "ai agent",
    "langgraph",
    "langchain",
    "mcp",
    "model context protocol",
    "multi-agent",
    "tool calling",
    "function calling",
    "rag",
    "workflow",
    "orchestration",
    "llm",
    "prompt",
    "retrieval",
    "memory",
    "planner",
    "executor",
}

NOISE_PATTERNS = [
    r"^\s*(sign in|sign up|subscribe|newsletter|cookie|privacy policy|terms of service)\s*$",
    r"^\s*(share|tweet|like|follow|advertisement|sponsored)\s*$",
    r"^\s*(skip to content|open navigation|close navigation)\s*$",
    r"^\s*\[!\[[^\]]*\]\([^)]+\)\]\([^)]+\)\s*$",
]

TRUNCATION_MARKER = "<!-- AKH_TRUNCATED_FOR_TOKEN_BUDGET -->"


class MarkdownPreprocessor:
    """Trim non-technical page chrome while preserving source-attributable content."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def clean(self, raw: RawDocument) -> CleanDocument:
        normalized = normalize_markdown(raw.markdown)
        image_urls = extract_image_urls(normalized)
        table_snippets = extract_markdown_tables(normalized)
        reader_title = extract_jina_reader_title(normalized)
        markdown = strip_jina_reader_preamble(normalized)
        markdown = remove_leading_documentation_index(markdown)
        markdown = remove_noise_lines(markdown)
        markdown = collapse_repeated_navigation(markdown)
        was_truncated = len(markdown) > self.settings.max_markdown_chars
        markdown = trim_to_budget(markdown, self.settings.max_markdown_chars)
        score = technical_signal_score(markdown)
        summary_hint = build_summary_hint(markdown)

        tokens = estimate_tokens(markdown)
        stats = raw.stats.model_copy(
            update={
                "output_chars": len(markdown),
                "estimated_tokens": tokens,
                "estimated_cost_usd": estimate_cost_usd(tokens),
            }
        )

        return CleanDocument(
            source_url=raw.source_url,
            final_url=raw.final_url,
            title=extract_title(markdown) or reader_title or raw.title,
            markdown=markdown,
            summary_hint=summary_hint,
            technical_signal_score=score,
            stats=stats,
            metadata=raw.metadata
            | {
                "preprocess_version": "phase1-v2",
                "original_chars": len(raw.markdown),
                "normalized_chars": len(normalized),
                "clean_chars": len(markdown),
                "was_truncated": was_truncated,
                "image_urls": image_urls,
                "table_snippets": table_snippets,
            },
        )


def normalize_markdown(markdown: str) -> str:
    markdown = markdown.replace("\r\n", "\n").replace("\r", "\n")
    markdown = re.sub(r"\n{4,}", "\n\n\n", markdown)
    markdown = re.sub(r"[ \t]+$", "", markdown, flags=re.MULTILINE)
    return markdown.strip()


def extract_image_urls(markdown: str, limit: int = 12) -> list[str]:
    """Extract remote Markdown image URLs without downloading them."""
    urls: list[str] = []
    for match in re.finditer(r"!\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", markdown):
        url = match.group(1).strip()
        if url.startswith(("http://", "https://")) and url not in urls:
            urls.append(url)
        if len(urls) >= limit:
            break
    return urls


def extract_markdown_tables(
    markdown: str,
    *,
    max_tables: int = 4,
    max_chars: int = 1200,
) -> list[str]:
    """Extract compact Markdown table blocks without interpreting their semantics."""
    tables: list[str] = []
    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if not looks_like_table_row(line):
            index += 1
            continue
        block = [line.rstrip()]
        index += 1
        while index < len(lines) and looks_like_table_row(lines[index]):
            block.append(lines[index].rstrip())
            index += 1
        if is_markdown_table(block):
            table = "\n".join(block).strip()
            if len(table) > max_chars:
                table = table[: max_chars - 1].rstrip() + "..."
            tables.append(table)
            if len(tables) >= max_tables:
                break
    return tables


def looks_like_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def is_markdown_table(lines: list[str]) -> bool:
    if len(lines) < 2:
        return False
    separator_pattern = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
    return any(separator_pattern.match(line) for line in lines[1:3])


def extract_jina_reader_title(markdown: str) -> str | None:
    match = re.search(r"^\s*Title:\s*(.+)$", markdown, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def strip_jina_reader_preamble(markdown: str) -> str:
    lines = markdown.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)

    if not lines or not lines[0].strip().startswith("Title:"):
        return markdown

    cursor = 0
    while cursor < len(lines):
        stripped = lines[cursor].strip()
        if not stripped:
            cursor += 1
            continue
        if stripped.startswith(("Title:", "URL Source:", "Markdown Content:")):
            cursor += 1
            continue
        break

    return "\n".join(lines[cursor:]).strip()


def remove_leading_documentation_index(markdown: str) -> str:
    lines = markdown.splitlines()
    cursor = 0
    while cursor < len(lines) and not lines[cursor].strip():
        cursor += 1

    if cursor >= len(lines) or "Documentation Index" not in lines[cursor]:
        return markdown

    end = cursor
    while end < len(lines):
        stripped = lines[end].strip()
        if stripped.startswith(">") or not stripped:
            end += 1
            continue
        break

    block = "\n".join(lines[cursor:end]).lower()
    if "llms.txt" not in block:
        return markdown

    return "\n".join(lines[:cursor] + lines[end:]).strip()


def remove_noise_lines(markdown: str) -> str:
    cleaned: list[str] = []
    compiled = [re.compile(pattern, re.IGNORECASE) for pattern in NOISE_PATTERNS]
    in_code_fence = False

    for line in markdown.splitlines():
        stripped = line.strip()
        if is_code_fence_line(stripped):
            in_code_fence = not in_code_fence
            cleaned.append(line)
            continue
        if in_code_fence:
            cleaned.append(line)
            continue
        if any(pattern.match(stripped) for pattern in compiled):
            continue
        if is_link_farm_line(stripped):
            continue
        cleaned.append(line)

    return "\n".join(cleaned).strip()


def is_link_farm_line(line: str) -> bool:
    if len(line) < 20:
        return False
    link_count = len(re.findall(r"\[[^\]]+\]\([^)]+\)", line))
    word_count = len(re.findall(r"\w+", line))
    return link_count >= 4 and word_count <= link_count * 3


def collapse_repeated_navigation(markdown: str) -> str:
    seen_short_lines: set[str] = set()
    output: list[str] = []
    in_code_fence = False

    for line in markdown.splitlines():
        stripped = line.strip()
        if is_code_fence_line(stripped):
            in_code_fence = not in_code_fence
            output.append(line)
            continue
        if in_code_fence:
            output.append(line)
            continue

        key = line.strip().lower()
        if key and len(key) <= 80 and key in seen_short_lines and not key.startswith("#"):
            continue
        if key and len(key) <= 80:
            seen_short_lines.add(key)
        output.append(line)

    return "\n".join(output).strip()


def trim_to_budget(markdown: str, max_chars: int) -> str:
    if len(markdown) <= max_chars:
        return markdown

    head_budget = int(max_chars * 0.72)
    tail_budget = max_chars - head_budget
    return (
        markdown[:head_budget].rstrip()
        + f"\n\n{TRUNCATION_MARKER}\n\n"
        + markdown[-tail_budget:].lstrip()
    )


def technical_signal_score(markdown: str) -> float:
    text = markdown.lower()
    if not text:
        return 0.0

    term_hits = sum(1 for term in TECHNICAL_TERMS if term in text)
    code_blocks = len(re.findall(r"^(?:```|~~~)", markdown, flags=re.MULTILINE)) // 2
    headings = len(re.findall(r"^#{1,4}\s+", markdown, flags=re.MULTILINE))
    score = min(1.0, (term_hits * 0.06) + (code_blocks * 0.08) + (headings * 0.01))
    return round(score, 3)


def is_code_fence_line(line: str) -> bool:
    return line.startswith("```") or line.startswith("~~~")


def build_summary_hint(markdown: str, max_chars: int = 700) -> str:
    lines = [
        line.strip(" #")
        for line in markdown.splitlines()
        if line.strip() and not line.strip().startswith(("!", "["))
    ]
    hint = " ".join(lines[:8])
    return hint[:max_chars].strip()


def extract_title(markdown: str) -> str | None:
    match = re.search(r"^#\s+(.+)$", markdown, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None
