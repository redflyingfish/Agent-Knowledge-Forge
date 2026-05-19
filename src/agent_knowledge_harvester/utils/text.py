COMMON_MOJIBAKE_REPLACEMENTS = {
    "\u00e2\u20ac\u201c": "-",
    "\u00e2\u20ac\u201d": "-",
    "\u00e2\u20ac\u02dc": "'",
    "\u00e2\u20ac\u2122": "'",
    "\u00e2\u20ac\u0153": '"',
    "\u00e2\u20ac\u009d": '"',
    "\u00e2\u20ac\u00a6": "...",
    "\u00c2 ": " ",
    "\u00c2": "",
}


def clean_display_text(text: str) -> str:
    """Repair common encoding artifacts before writing human-readable artifacts."""
    cleaned = text
    for bad, replacement in COMMON_MOJIBAKE_REPLACEMENTS.items():
        cleaned = cleaned.replace(bad, replacement)
    return cleaned
