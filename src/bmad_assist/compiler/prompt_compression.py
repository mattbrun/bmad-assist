"""LLM-based prompt compression for context documents.

Replaces simple truncation with intelligent compression using the helper
provider (e.g., haiku). Preserves key technical information while reducing
token count to fit within budgets.

Compression results are cached to `.bmad-assist/cache/compressed/` using a
content-hash key (SHA-256 of source content + target_tokens). Cache hits
skip the LLM call entirely.

Two entry points:
- compress_document(): Compress a single document to a target token count.
- compress_context_files(): Post-compilation pass to compress <file> elements
  in compiled XML when the total prompt exceeds budget.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from bmad_assist.compiler.shared_utils import estimate_tokens

if TYPE_CHECKING:
    from bmad_assist.providers.base import BaseProvider

logger = logging.getLogger(__name__)

# Compression system prompt template
COMPRESSION_PROMPT = """\
You are a technical document compressor. Compress the following document to \
approximately {target_tokens} tokens (~{target_chars} characters).

RULES:
- Preserve ALL: file paths, API contracts, class/function names, constraints, \
architectural decisions, naming conventions, technology choices
- Remove: verbose explanations, examples that repeat the same pattern, \
redundant descriptions
- Output: self-contained markdown, same heading structure where possible
- Do NOT add commentary or meta-text about the compression

DOCUMENT:
{content}"""


# ---------------------------------------------------------------------------
# Compression cache helpers
# ---------------------------------------------------------------------------

def _compression_cache_dir() -> Path | None:
    """Return the compression cache directory, or None if paths not initialized."""
    try:
        from bmad_assist.core.paths import get_paths
        return get_paths().cache_dir / "compressed"
    except Exception:
        return None


def _cache_key(content: str, target_tokens: int) -> str:
    """Compute a cache key from content hash and target tokens."""
    h = hashlib.sha256()
    h.update(content.encode("utf-8", errors="replace"))
    h.update(f"|target={target_tokens}".encode())
    return h.hexdigest()[:16]


def _load_cached(content: str, target_tokens: int) -> tuple[str, int] | None:
    """Load compressed content from cache if it exists and source matches."""
    cache_dir = _compression_cache_dir()
    if cache_dir is None:
        return None

    key = _cache_key(content, target_tokens)
    cache_file = cache_dir / f"{key}.json"
    if not cache_file.exists():
        return None

    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        compressed = data["compressed"]
        actual_tokens = data["actual_tokens"]
        logger.info(
            "Compression cache hit: key=%s, tokens=%d",
            key, actual_tokens,
        )
        return compressed, actual_tokens
    except (json.JSONDecodeError, KeyError, OSError) as e:
        logger.debug("Compression cache read failed for %s: %s", key, e)
        return None


def _save_cached(
    content: str, target_tokens: int, compressed: str, actual_tokens: int
) -> None:
    """Save compressed content to cache."""
    cache_dir = _compression_cache_dir()
    if cache_dir is None:
        return

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = _cache_key(content, target_tokens)
        cache_file = cache_dir / f"{key}.json"
        cache_file.write_text(
            json.dumps(
                {"compressed": compressed, "actual_tokens": actual_tokens},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        logger.debug("Compression cache saved: key=%s", key)
    except OSError as e:
        logger.debug("Compression cache write failed: %s", e)


# ---------------------------------------------------------------------------
# Core compression
# ---------------------------------------------------------------------------

def compress_document(
    content: str,
    target_tokens: int,
    provider: BaseProvider,
    model: str,
    timeout: int = 120,
) -> tuple[str, int]:
    """Compress a document using the helper LLM, with caching.

    Checks `.bmad-assist/cache/compressed/` for a cached result keyed by
    SHA-256(content + target_tokens). On cache miss, invokes the helper
    provider and saves the result for future runs.

    Args:
        content: Full document content to compress.
        target_tokens: Target token count for the compressed output.
        provider: Provider instance (must have .invoke() method).
        model: Model identifier for the provider.
        timeout: Timeout in seconds for the provider call.

    Returns:
        Tuple of (compressed_content, actual_tokens).

    Raises:
        RuntimeError: If the provider fails and no fallback is desired.
            Callers should catch this and fall back to truncation.

    """
    # Check cache first
    cached = _load_cached(content, target_tokens)
    if cached is not None:
        return cached

    target_chars = target_tokens * 4
    prompt = COMPRESSION_PROMPT.format(
        target_tokens=target_tokens,
        target_chars=target_chars,
        content=content,
    )

    result = provider.invoke(
        prompt,
        model=model,
        timeout=timeout,
        disable_tools=True,
    )

    if result.exit_code != 0:
        error_msg = result.stderr[:200] if result.stderr else "Unknown error"
        raise RuntimeError(f"Compression provider failed: {error_msg}")

    compressed = result.stdout.strip()
    if not compressed:
        raise RuntimeError("Compression returned empty content")

    actual_tokens = estimate_tokens(compressed)

    # Save to cache for future runs
    _save_cached(content, target_tokens, compressed, actual_tokens)

    return compressed, actual_tokens


# Regex to match <file ...>...</file> elements in compiled XML
_FILE_PATTERN = re.compile(
    r'(<file\b[^>]*>)\s*(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?\s*(</file>)',
    re.DOTALL,
)


def compress_context_files(
    xml_prompt: str,
    target_tokens: int,
    provider: BaseProvider,
    model: str,
    timeout: int = 120,
    min_file_tokens: int = 2000,
    compression_ratio: float = 0.5,
) -> tuple[str, int]:
    """Compress <file> elements in compiled XML to fit within token budget.

    Post-compilation safety net: parses the compiled XML to find large
    <file> elements, compresses them via LLM, and replaces the content.

    Args:
        xml_prompt: Full compiled XML prompt string.
        target_tokens: Target total token count for the prompt.
        provider: Provider instance for LLM compression.
        model: Model identifier.
        timeout: Per-file compression timeout.
        min_file_tokens: Only compress files larger than this.
        compression_ratio: Target ratio (0.5 = compress to 50% of original).

    Returns:
        Tuple of (compressed_xml, new_token_estimate).

    """
    current_tokens = estimate_tokens(xml_prompt)

    if current_tokens <= target_tokens:
        return xml_prompt, current_tokens

    # Find all <file> elements and their sizes
    matches = list(_FILE_PATTERN.finditer(xml_prompt))
    if not matches:
        logger.debug("No <file> elements found in prompt for compression")
        return xml_prompt, current_tokens

    # Build list of (match, content, tokens) sorted by size descending
    file_entries = []
    for match in matches:
        file_content = match.group(2)
        tokens = estimate_tokens(file_content)
        if tokens >= min_file_tokens:
            file_entries.append((match, file_content, tokens))

    file_entries.sort(key=lambda x: x[2], reverse=True)

    if not file_entries:
        logger.debug("No files exceed min_file_tokens=%d for compression", min_file_tokens)
        return xml_prompt, current_tokens

    # Compress files largest-first until within budget
    result_xml = xml_prompt
    compressed_count = 0

    for match, file_content, tokens in file_entries:
        file_target = int(tokens * compression_ratio)

        try:
            compressed_content, actual_tokens = compress_document(
                file_content, file_target, provider, model, timeout
            )
        except RuntimeError as e:
            logger.warning("Skipping file compression: %s", e)
            continue

        # Build replacement: re-wrap in CDATA
        opening_tag = match.group(1)
        closing_tag = match.group(3)
        old_full = match.group(0)
        new_full = f"{opening_tag}<![CDATA[{compressed_content}]]>{closing_tag}"

        result_xml = result_xml.replace(old_full, new_full, 1)
        compressed_count += 1

        # Recalculate and check if within budget
        current_tokens = estimate_tokens(result_xml)
        logger.info(
            "Compressed file (%d -> %d tokens), total now ~%d tokens",
            tokens,
            actual_tokens,
            current_tokens,
        )

        if current_tokens <= target_tokens:
            break

    if compressed_count > 0:
        logger.info(
            "Post-compilation compression: %d files compressed, total ~%d tokens",
            compressed_count,
            current_tokens,
        )
    else:
        logger.warning("Post-compilation compression: no files could be compressed")

    return result_xml, current_tokens
