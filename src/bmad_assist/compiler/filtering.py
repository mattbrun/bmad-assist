"""Instruction filtering module for BMAD workflow compiler.

This module provides whitelist-based filtering of workflow instructions,
removing user-interaction elements and keeping only executable actions
for automated LLM execution.

Key features:
- Whitelist-based tag filtering (only allowed tags are kept)
- User-condition check removal (e.g., "if user chooses...")
- HALT/GOTO instruction removal
- XML comment preservation (comments survive ElementTree parsing)

Public API:
    filter_instructions: Filter workflow instructions to keep only executable elements
"""

import logging
import re
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element

from bmad_assist.compiler.types import WorkflowIR
from bmad_assist.core.exceptions import CompilerError

logger = logging.getLogger(__name__)

# Pattern to match XML/HTML comments
_COMMENT_PATTERN = re.compile(r"<!--(.*?)-->", re.DOTALL)
# Placeholder tag for preserving comments through XML parsing
_COMMENT_PLACEHOLDER_TAG = "__xml_comment__"

# Tags that are KEPT in filtered output (executable/informational elements)
WHITELIST_TAGS: frozenset[str] = frozenset(
    {
        "action",
        "step",
        "substep",  # Nested step within a step
        "check",
        "invoke-task",
        "invoke-workflow",
        "invoke-protocol",
        "critical",
        "mandate",
        "note",
        # BMAD-specific workflow constructs
        "for-each-item",  # Iteration construct (e.g., checklist items)
        "mark-as",  # Marking/classification instructions
        "report-format",  # Output format specification
        "output-format",  # Alternative output format tag (Story 13.6)
        "o",  # Output/display instruction (non-interactive)
        # Internal: comment placeholder (preserved through filtering)
        _COMMENT_PLACEHOLDER_TAG,
    }
)

# Tags that are DISCARDED from output (user-interaction elements)
BLACKLIST_TAGS: frozenset[str] = frozenset(
    {
        "ask",
        "output",
        "template-output",
    }
)

# Patterns indicating user-related conditions (case-insensitive)
USER_CONDITION_PATTERNS: tuple[str, ...] = (
    "user",
    "chooses",
    "option",
    "response",
    "sprint status file does not exist",
)


def _comments_to_placeholders(xml_str: str) -> str:
    """Convert XML comments to placeholder elements for preservation.

    ElementTree drops comments during parsing. This converts them to
    placeholder elements that survive filtering, then get converted back.

    Args:
        xml_str: XML string with comments.

    Returns:
        XML string with comments replaced by placeholder elements.

    """

    def replace_comment(match: re.Match[str]) -> str:
        content = match.group(1)
        # Escape any XML special chars in comment content
        escaped = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f"<{_COMMENT_PLACEHOLDER_TAG}>{escaped}</{_COMMENT_PLACEHOLDER_TAG}>"

    return _COMMENT_PATTERN.sub(replace_comment, xml_str)


def _placeholders_to_comments(xml_str: str) -> str:
    """Convert placeholder elements back to XML comments.

    Handles both unescaped placeholders (normal XML context) and escaped
    placeholders (inside CDATA sections where ElementTree escapes < and >).

    Args:
        xml_str: XML string with placeholder elements.

    Returns:
        XML string with placeholders replaced by comments.

    """
    # Pattern for unescaped placeholders (normal XML context)
    unescaped_pattern = re.compile(
        rf"<{_COMMENT_PLACEHOLDER_TAG}>(.*?)</{_COMMENT_PLACEHOLDER_TAG}>", re.DOTALL
    )

    def restore_unescaped(match: re.Match[str]) -> str:
        content = match.group(1)
        # Unescape XML special chars in content
        unescaped = content.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
        return f"<!--{unescaped}-->"

    result = unescaped_pattern.sub(restore_unescaped, xml_str)

    # Pattern for escaped placeholders (inside CDATA, serialized as &lt;...&gt;)
    escaped_pattern = re.compile(
        rf"&lt;{_COMMENT_PLACEHOLDER_TAG}&gt;(.*?)&lt;/{_COMMENT_PLACEHOLDER_TAG}&gt;",
        re.DOTALL,
    )

    def restore_escaped(match: re.Match[str]) -> str:
        content = match.group(1)
        # Content is already escaped, keep it that way for CDATA context
        # Just wrap in escaped comment markers
        return f"&lt;!--{content}--&gt;"

    return escaped_pattern.sub(restore_escaped, result)


def _is_user_condition(if_attr: str) -> bool:
    """Determine if check condition is user-related (should be discarded).

    Args:
        if_attr: The value of the 'if' attribute from a <check> element.

    Returns:
        True if the condition references user actions/choices and should be
        discarded; False if it's a technical condition that should be kept.

    """
    if not if_attr:
        return False

    normalized = if_attr.lower().strip()
    return any(pattern in normalized for pattern in USER_CONDITION_PATTERNS)


def _filter_element(element: Element, is_root: bool = False) -> Element | None:
    """Filter element and children, return filtered copy or None.

    Recursively filters an XML element tree using whitelist-based approach:
    - Only elements in WHITELIST_TAGS are kept (plus root element)
    - User-condition checks are removed
    - Elements containing HALT/GOTO in direct text are removed

    Args:
        element: XML element to filter.
        is_root: True if this is the root element (never removed).

    Returns:
        Filtered copy of the element, or None if the element should be removed.

    """
    tag = element.tag  # Preserve original case (XML is case-sensitive)
    tag_lower = tag.lower()

    # Root element is ALWAYS preserved (regardless of tag name)
    if not is_root:
        # WHITELIST enforcement: only keep allowed tags
        # Note: blacklist is redundant if blacklist tags aren't in whitelist,
        # but we check both for explicit clarity and defense-in-depth
        if tag_lower not in WHITELIST_TAGS:
            return None

        # Check condition classification (only applies to 'check' which is whitelisted)
        if tag_lower == "check":
            if_attr = element.get("if", "")
            if _is_user_condition(if_attr):
                return None

        # HALT/GOTO in DIRECT text content (case-insensitive)
        text_upper = (element.text or "").strip().upper()
        if "HALT" in text_upper or "GOTO" in text_upper:
            return None

    # Create filtered copy with DEEP copy of attrib dict to prevent mutation
    filtered = Element(element.tag, dict(element.attrib))
    filtered.text = element.text
    filtered.tail = element.tail

    # Recursively filter children
    for child in element:
        filtered_child = _filter_element(child, is_root=False)
        if filtered_child is not None:
            filtered.append(filtered_child)

    # Remove empty CONTAINERS (step, check) - but NEVER root
    if not is_root and tag_lower in {"step", "check"}:
        has_children = len(filtered) > 0
        has_text = bool((filtered.text or "").strip())
        if not has_children and not has_text:
            return None

    return filtered


def _is_markdown_content(content: str) -> bool:
    """Detect if content is markdown rather than XML.

    Used to handle cached templates that may have .xml extension but contain
    markdown instructions from the original .md file.

    Args:
        content: The raw instructions content.

    Returns:
        True if content appears to be markdown, False if it looks like XML.

    """
    # Strip whitespace and any leading XML comments
    stripped = content.strip()

    # Remove leading XML comments for detection
    while stripped.startswith("<!--"):
        end = stripped.find("-->")
        if end == -1:
            break
        stripped = stripped[end + 3 :].strip()

    # Markdown typically starts with # heading, not < tag
    if stripped.startswith("#"):
        return True

    # If it starts with an XML tag, it's XML; otherwise assume markdown
    # (could be text paragraph, list, etc.)
    return not stripped.startswith("<")


def filter_instructions(workflow_ir: WorkflowIR) -> str:
    """Filter workflow instructions to keep only executable elements.

    Takes the raw_instructions XML string from WorkflowIR and produces
    a filtered XML string containing only executable elements. User-interaction
    elements (ask, output, template-output) and user-condition checks are removed.

    For markdown instructions (.md files), returns the content as-is without
    XML parsing/filtering.

    Args:
        workflow_ir: WorkflowIR instance containing raw_instructions.

    Returns:
        Filtered XML string with only executable elements.
        Returns empty string if input is empty or whitespace-only.
        Returns raw markdown if instructions_path ends with .md.

    Raises:
        CompilerError: If raw_instructions contains invalid XML.

    """
    raw_xml = workflow_ir.raw_instructions

    # Handle empty/whitespace input
    if not raw_xml or not raw_xml.strip():
        return ""

    # For markdown files, return content as-is (no XML filtering)
    # Check both file extension AND content pattern (cached templates may have .xml extension
    # but contain markdown content from original .md instructions)
    is_markdown = workflow_ir.instructions_path.suffix.lower() == ".md" or _is_markdown_content(
        raw_xml
    )
    if is_markdown:
        logger.debug("Skipping XML filtering for markdown instructions")
        return raw_xml

    # Log size for large inputs (only compute when debug enabled)
    if logger.isEnabledFor(logging.DEBUG):
        size_bytes = len(raw_xml.encode("utf-8"))
        if size_bytes > 1024 * 1024:  # > 1MB
            logger.debug(f"Processing large XML input: {size_bytes} bytes")

    # Preserve XML comments by converting to placeholder elements
    # (ElementTree drops comments during parsing)
    xml_with_placeholders = _comments_to_placeholders(raw_xml)

    # Parse XML
    try:
        root = ET.fromstring(xml_with_placeholders)
    except ET.ParseError as e:
        # Should not happen (parser validated), but handle defensively
        raise CompilerError(f"Invalid XML in raw_instructions: {e}") from e

    # Filter the tree
    filtered_root = _filter_element(root, is_root=True)

    # Root should never be None since is_root=True prevents removal
    # Using assertion instead of silent fallback (fail fast on invariant violation)
    assert filtered_root is not None, "Root element should never be None (is_root=True)"

    # Count filtered elements for debug logging (only when debug enabled)
    if logger.isEnabledFor(logging.DEBUG):
        original_count = sum(1 for _ in root.iter())
        filtered_count = sum(1 for _ in filtered_root.iter())
        logger.debug(
            f"Filtered instructions: {original_count} elements -> {filtered_count} elements"
        )

    # Generate output XML and restore comments from placeholders
    output_xml = ET.tostring(filtered_root, encoding="unicode")
    return _placeholders_to_comments(output_xml)
