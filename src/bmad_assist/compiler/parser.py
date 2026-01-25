"""BMAD workflow file parsing.

This module provides functions for parsing BMAD workflow files:
- parse_workflow_config: Parse workflow.yaml configuration
- parse_workflow_instructions: Parse and validate instructions.xml
- parse_workflow: Unified parsing returning WorkflowIR

All parsing is STRUCTURAL only - variable resolution is handled by Story 10.3.
Placeholders like {config_source}:output_folder are preserved as raw strings.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml

from bmad_assist.compiler.types import WorkflowIR
from bmad_assist.core.exceptions import ParserError


def parse_workflow_config(config_path: Path) -> dict[str, Any]:
    """Parse workflow.yaml configuration file.

    Parses YAML config and returns raw dictionary. All placeholders
    (e.g., {project-root}, {config_source}:) are preserved as strings.

    Args:
        config_path: Path to workflow.yaml file.

    Returns:
        Parsed YAML as dictionary. Empty dict for empty files.

    Raises:
        ParserError: If file not found or YAML syntax is invalid.

    """
    if not config_path.exists():
        raise ParserError(
            f"Configuration file not found: {config_path}\n"
            f"  Why it's needed: Contains workflow configuration and variable definitions\n"
            f"  How to fix: Ensure workflow.yaml exists in the workflow directory"
        )

    try:
        content = config_path.read_text(encoding="utf-8")
    except OSError as e:
        raise ParserError(
            f"Cannot read configuration file: {config_path}\n"
            f"  Error: {e}\n"
            f"  Suggestion: Check file permissions and ensure the file is accessible"
        ) from e

    # Empty file returns empty dict (AC6)
    if not content.strip():
        return {}

    try:
        result = yaml.safe_load(content)
        # yaml.safe_load returns None for empty/whitespace-only content
        if result is None:
            return {}
        if not isinstance(result, dict):
            raise ParserError(
                f"Invalid configuration in {config_path}:\n"
                f"  Root element must be a mapping (dict), got {type(result).__name__}\n"
                f"  Suggestion: Ensure workflow.yaml is key: value format, not a list or scalar"
            )
        return result
    except yaml.YAMLError as e:
        # Extract line number from YAML error if available
        line_info = ""
        if hasattr(e, "problem_mark") and e.problem_mark is not None:
            mark = e.problem_mark
            line_info = f"\n  Line {mark.line + 1}, column {mark.column + 1}"

        raise ParserError(
            f"Invalid YAML in {config_path}:{line_info}\n"
            f"  {e}\n"
            f"  Suggestion: Check YAML syntax (indentation, colons, quotes)"
        ) from e


def parse_workflow_instructions(instructions_path: Path) -> str:
    """Parse and validate instructions.xml file.

    Validates XML syntax using ElementTree, then returns raw XML content
    as string. The XML is NOT parsed into a tree - that happens in
    Story 10.5 (Instruction Filtering Engine).

    Args:
        instructions_path: Path to instructions.xml file.

    Returns:
        Raw XML content as string (validated for syntax).

    Raises:
        ParserError: If file not found or XML syntax is invalid.

    """
    if not instructions_path.exists():
        raise ParserError(
            f"Instructions file not found: {instructions_path}\n"
            f"  Why it's needed: Contains workflow execution steps and actions\n"
            f"  How to fix: Ensure instructions.xml exists in the workflow directory"
        )

    try:
        content = instructions_path.read_text(encoding="utf-8")
    except OSError as e:
        raise ParserError(
            f"Cannot read instructions file: {instructions_path}\n"
            f"  Error: {e}\n"
            f"  Suggestion: Check file permissions and ensure the file is accessible"
        ) from e

    # For .md files, skip XML validation (markdown may contain XML-like tags but isn't XML)
    if instructions_path.suffix.lower() == ".md":
        return content

    # Security: Reject XML with DOCTYPE/ENTITY declarations (XML bomb protection)
    if "<!DOCTYPE" in content or "<!ENTITY" in content:
        raise ParserError(
            f"Invalid XML in {instructions_path}:\n"
            f"  DOCTYPE and ENTITY declarations are not allowed\n"
            f"  Suggestion: Remove <!DOCTYPE> and <!ENTITY> declarations"
        )

    # Validate XML syntax by parsing (but we return the raw string)
    try:
        ET.fromstring(content)
    except ET.ParseError as e:
        # ParseError has position info: (msg, (line, column))
        line_info = ""
        if hasattr(e, "position") and e.position is not None:
            line, col = e.position
            line_info = f"\n  Line {line}, column {col}"

        raise ParserError(
            f"Invalid XML in {instructions_path}:{line_info}\n"
            f"  {e}\n"
            f"  Suggestion: Check XML syntax (tags, quotes, encoding)"
        ) from e

    return content


def parse_workflow(workflow_dir: Path) -> WorkflowIR:
    """Parse BMAD workflow directory into WorkflowIR.

    Loads and parses all workflow files from the directory:
    - workflow.yaml (required)
    - instructions.xml (required, path from config or convention)
    - template path (optional, stored as raw string)

    Args:
        workflow_dir: Directory containing workflow.yaml and instructions.xml.

    Returns:
        WorkflowIR with parsed content (placeholders preserved as strings).

    Raises:
        ParserError: If required files missing or parsing fails.

    """
    workflow_dir = workflow_dir.resolve()

    # Check workflow.yaml exists (AC6)
    config_path = workflow_dir / "workflow.yaml"
    if not config_path.exists():
        raise ParserError(
            f"workflow.yaml not found: {config_path}\n"
            f"  Why it's needed: Defines workflow configuration, variables, and file patterns\n"
            f"  How to fix: Ensure the workflow directory contains workflow.yaml"
        )

    # Parse config
    raw_config = parse_workflow_config(config_path)

    # Determine instructions path
    # If 'instructions' key contains placeholder like {installed_path},
    # assume instructions.xml in same directory (convention per AC4)
    instructions_key = raw_config.get("instructions", "")
    if isinstance(instructions_key, str) and "{" in instructions_key:
        # Placeholder present - use convention (try .xml first, then .md)
        instructions_path = workflow_dir / "instructions.xml"
        if not instructions_path.exists():
            instructions_path = workflow_dir / "instructions.md"
    elif isinstance(instructions_key, str) and instructions_key:
        # Explicit path (rare case - resolve relative to workflow_dir)
        # Security: Prevent path traversal attacks
        if ".." in instructions_key or instructions_key.startswith("/"):
            raise ParserError(
                f"Invalid instructions path in {config_path}:\n"
                f"  Path '{instructions_key}' contains path traversal\n"
                f"  Suggestion: Use relative path within workflow directory"
            )
        instructions_path = workflow_dir / instructions_key
    else:
        # No instructions key - use convention (try .xml first, then .md)
        instructions_path = workflow_dir / "instructions.xml"
        if not instructions_path.exists():
            instructions_path = workflow_dir / "instructions.md"

    # Check instructions file exists (AC6)
    if not instructions_path.exists():
        raise ParserError(
            f"instructions file not found: {instructions_path}\n"
            f"  Why it's needed: Contains workflow execution steps and actions\n"
            f"  How to fix: Ensure the workflow directory contains instructions.xml or instructions.md" # noqa: E501
        )

    # Parse instructions
    raw_instructions = parse_workflow_instructions(instructions_path)

    # Extract template path (AC3)
    template_value = raw_config.get("template")
    if template_value is False:
        # Explicit false means no template
        template_path: str | None = None
    elif isinstance(template_value, str):
        # Store as raw string with placeholders (NOT resolved)
        template_path = template_value
    else:
        # Key absent or other type - no template
        template_path = None

    # Extract validation/checklist path
    validation_value = raw_config.get("validation")
    if validation_value is False:
        # Explicit false means no validation
        validation_path: str | None = None
    elif isinstance(validation_value, str):
        # Store as raw string with placeholders (NOT resolved)
        validation_path = validation_value
    else:
        # Key absent or other type - no validation
        validation_path = None

    # Extract workflow name (AC4)
    # Priority: 'name' key in config, fallback to directory name
    name = raw_config.get("name")
    if not name or not isinstance(name, str):
        name = workflow_dir.name

    return WorkflowIR(
        name=name,
        config_path=config_path.resolve(),
        instructions_path=instructions_path.resolve(),
        template_path=template_path,
        validation_path=validation_path,
        raw_config=raw_config,
        raw_instructions=raw_instructions,
    )
