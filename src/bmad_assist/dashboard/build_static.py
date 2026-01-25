#!/usr/bin/env python3
"""Build dashboard static files by concatenating partials and copying JS.

Usage:
    python build_static.py          # Build index.html from partials + copy JS
    python build_static.py --watch  # Watch for changes and rebuild

Partials are in static-src/ and are concatenated in order to produce static/index.html.
JS files are copied from static-src/js/ to static/js/.
"""

import os
import shutil
import sys
from pathlib import Path

STATIC_SRC = Path(__file__).parent / "static-src"
STATIC_OUT = Path(__file__).parent / "static"
JS_SRC = STATIC_SRC / "js"
JS_OUT = STATIC_OUT / "js"

# Order of partials to concatenate
PARTIALS_ORDER = [
    "01-head.html",
    "02-sidebar.html",
    "03-main-header.html",
    "04-terminal.html",
    "05-settings-panel.html",
    "06-experiments-panel.html",
    "07-experiment-details.html",
    "08-comparison-panel.html",
    "09-footer.html",
    "10-modals.html",
    "11-tail.html",
]


def copy_js_files() -> int:
    """Copy JS files from static-src/js/ to static/js/ and sync deletions.

    Returns the number of files copied. Also removes orphaned files in
    JS_OUT that no longer exist in JS_SRC.
    """
    if not JS_SRC.exists():
        print(f"WARNING: JS source directory not found: {JS_SRC}")
        return 0

    # Ensure output directory exists
    JS_OUT.mkdir(parents=True, exist_ok=True)

    # Track expected destination files for orphan cleanup
    expected_files: set[Path] = set()

    # Copy all JS files recursively
    copied = 0
    for src_file in JS_SRC.rglob("*.js"):
        # Calculate relative path from JS_SRC
        rel_path = src_file.relative_to(JS_SRC)
        dst_file = JS_OUT / rel_path

        # Track expected file
        expected_files.add(dst_file)

        # Ensure parent directory exists
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        shutil.copy2(src_file, dst_file)
        copied += 1

    # Remove orphaned files (exist in output but not in source)
    orphans_removed = 0
    for dst_file in JS_OUT.rglob("*.js"):
        if dst_file not in expected_files:
            dst_file.unlink()
            print(f"Removed orphan: {dst_file.relative_to(JS_OUT)}")
            orphans_removed += 1

    # Clean up empty directories
    for dir_path in sorted(JS_OUT.rglob("*"), reverse=True):
        if dir_path.is_dir() and not any(dir_path.iterdir()):
            dir_path.rmdir()

    return copied


def build() -> None:
    """Concatenate all partials into index.html and copy JS files."""
    # Ensure output directory exists
    STATIC_OUT.mkdir(parents=True, exist_ok=True)

    output_lines: list[str] = []

    for partial_name in PARTIALS_ORDER:
        partial_path = STATIC_SRC / partial_name
        if not partial_path.exists():
            print(f"WARNING: Missing partial: {partial_name}")
            continue

        content = partial_path.read_text()
        # Note: Don't add HTML comments - they can break HTML when split
        # boundaries fall inside tags or attributes
        output_lines.append(content)
        if not content.endswith("\n"):
            output_lines.append("\n")

    output = "".join(output_lines)
    output_path = STATIC_OUT / "index.html"

    # Atomic write pattern (project_context.md requirement)
    temp_path = output_path.with_suffix(".tmp")
    temp_path.write_text(output)
    os.replace(temp_path, output_path)

    print(f"Built {output_path} ({len(output):,} bytes)")

    # Copy JS files
    js_count = copy_js_files()
    if js_count > 0:
        print(f"Copied {js_count} JS files to {JS_OUT}")


def watch() -> None:
    """Watch for changes and rebuild."""
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        print("Install watchdog for watch mode: pip install watchdog")
        sys.exit(1)

    class RebuildHandler(FileSystemEventHandler):  # type: ignore[misc]
        def _should_rebuild(self, event: object) -> bool:
            """Check if event should trigger rebuild."""
            if getattr(event, "is_directory", False):
                return False
            src_path = getattr(event, "src_path", "")
            return str(src_path).endswith((".html", ".js"))

        def on_modified(self, event: object) -> None:
            if self._should_rebuild(event):
                print(f"Changed: {getattr(event, 'src_path', '')}")
                build()

        def on_created(self, event: object) -> None:
            if self._should_rebuild(event):
                print(f"Created: {getattr(event, 'src_path', '')}")
                build()

        def on_deleted(self, event: object) -> None:
            if self._should_rebuild(event):
                print(f"Deleted: {getattr(event, 'src_path', '')}")
                build()

    observer = Observer()
    # Watch both HTML partials and JS files
    observer.schedule(RebuildHandler(), str(STATIC_SRC), recursive=True)
    observer.start()
    print(f"Watching {STATIC_SRC} for changes (HTML and JS)...")

    try:
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    if "--watch" in sys.argv:
        watch()
    else:
        build()
