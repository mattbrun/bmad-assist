"""Tests for the instruction filtering module.

Tests the filter_instructions function which removes user-interaction
elements from workflow instructions, keeping only executable actions.
"""

from pathlib import Path

import pytest

from bmad_assist.compiler.filtering import filter_instructions
from bmad_assist.compiler.types import WorkflowIR
from bmad_assist.core.exceptions import CompilerError


def create_test_workflow_ir(xml_content: str) -> WorkflowIR:
    """Create a WorkflowIR instance with the given raw_instructions XML."""
    return WorkflowIR(
        name="test-workflow",
        config_path=Path("/test/workflow.yaml"),
        instructions_path=Path("/test/instructions.xml"),
        template_path=None,
        validation_path=None,
        raw_config={},
        raw_instructions=xml_content,
    )


class TestWhitelistElements:
    """Tests for AC1: Whitelist element filtering."""

    def test_whitelist_elements_kept(self) -> None:
        """Whitelist elements are preserved in output."""
        xml = """<workflow>
            <action>Do something</action>
            <step n="1" goal="Test">
                <action>Nested action</action>
            </step>
            <critical>Important rule</critical>
            <mandate>Must do this</mandate>
            <note>Implementation note</note>
            <invoke-task>task_name</invoke-task>
            <invoke-workflow>workflow_name</invoke-workflow>
            <invoke-protocol name="discover_inputs"/>
        </workflow>"""

        workflow_ir = create_test_workflow_ir(xml)
        filtered = filter_instructions(workflow_ir)

        assert "<action>Do something</action>" in filtered
        assert "<step" in filtered
        assert 'n="1"' in filtered
        assert 'goal="Test"' in filtered
        assert "<critical>" in filtered
        assert "<mandate>" in filtered
        assert "<note>" in filtered
        assert "<invoke-task>" in filtered
        assert "<invoke-workflow>" in filtered
        assert "<invoke-protocol" in filtered


class TestBlacklistElements:
    """Tests for AC1: Blacklist element filtering."""

    def test_blacklist_elements_removed(self) -> None:
        """Blacklist elements are removed from output."""
        xml = """<workflow>
            <action>Keep this</action>
            <ask>Remove this prompt</ask>
            <output>Remove this output</output>
            <template-output>Remove this marker</template-output>
        </workflow>"""

        workflow_ir = create_test_workflow_ir(xml)
        filtered = filter_instructions(workflow_ir)

        assert "<action>Keep this</action>" in filtered
        assert "<ask>" not in filtered
        assert "<output>" not in filtered
        assert "<template-output>" not in filtered


class TestCheckConditionFiltering:
    """Tests for AC2: Conditional check filtering."""

    def test_user_condition_checks_removed(self) -> None:
        """Check elements with user conditions are removed."""
        xml = """<workflow>
            <check if="user chooses option 1">
                <action>User action</action>
            </check>
            <check if="{{story_path}} is provided by user">
                <action>Path action</action>
            </check>
            <check if="response='a'">
                <action>Response action</action>
            </check>
        </workflow>"""

        workflow_ir = create_test_workflow_ir(xml)
        filtered = filter_instructions(workflow_ir)

        assert "user chooses" not in filtered
        assert "provided by user" not in filtered
        assert "response=" not in filtered

    def test_technical_condition_checks_kept(self) -> None:
        """Check elements with technical conditions are preserved."""
        xml = """<workflow>
            <check if="story_num > 1">
                <action>Load previous</action>
            </check>
            <check if="previous story exists">
                <action>Analyze</action>
            </check>
            <check if="sharded pattern exists">
                <action>Load shards</action>
            </check>
            <check if="no matches found">
                <action>Handle empty</action>
            </check>
        </workflow>"""

        workflow_ir = create_test_workflow_ir(xml)
        filtered = filter_instructions(workflow_ir)

        # Note: XML escapes > to &gt; in attributes
        assert "story_num" in filtered and (">" in filtered or "&gt;" in filtered)
        assert "previous story exists" in filtered
        assert "sharded pattern exists" in filtered
        assert "no matches found" in filtered

    def test_option_keyword_check_removed(self) -> None:
        """Check with 'option' keyword in condition is removed."""
        xml = """<workflow>
            <check if="option A selected">
                <action>Do A</action>
            </check>
            <action>Keep this</action>
        </workflow>"""

        workflow_ir = create_test_workflow_ir(xml)
        filtered = filter_instructions(workflow_ir)

        assert "option A" not in filtered
        assert "Keep this" in filtered

    def test_sprint_status_does_not_exist_removed(self) -> None:
        """Check for 'sprint status file does NOT exist' is removed."""
        xml = """<workflow>
            <check if="sprint status file does NOT exist">
                <action>Show error</action>
            </check>
            <action>Keep this</action>
        </workflow>"""

        workflow_ir = create_test_workflow_ir(xml)
        filtered = filter_instructions(workflow_ir)

        assert "sprint status file does NOT exist" not in filtered
        assert "Keep this" in filtered


class TestHaltGotoFiltering:
    """Tests for HALT/GOTO filtering in AC1."""

    def test_halt_in_direct_text_removes_element(self) -> None:
        """Elements with HALT in direct text are removed."""
        xml = """<workflow>
            <action>HALT</action>
            <action>Keep this one</action>
        </workflow>"""

        workflow_ir = create_test_workflow_ir(xml)
        filtered = filter_instructions(workflow_ir)

        assert "HALT" not in filtered
        assert "Keep this one" in filtered

    def test_goto_in_direct_text_removes_element(self) -> None:
        """Elements with GOTO in direct text are removed."""
        xml = """<workflow>
            <action>GOTO step 2</action>
            <action>Keep this one</action>
        </workflow>"""

        workflow_ir = create_test_workflow_ir(xml)
        filtered = filter_instructions(workflow_ir)

        assert "GOTO" not in filtered
        assert "Keep this one" in filtered

    def test_halt_case_insensitive(self) -> None:
        """HALT/halt/Halt all trigger removal (case-insensitive)."""
        for keyword in ["HALT", "halt", "Halt", "HaLt"]:
            xml = f"""<workflow>
                <action>{keyword}</action>
                <action>Keep this</action>
            </workflow>"""

            workflow_ir = create_test_workflow_ir(xml)
            filtered = filter_instructions(workflow_ir)

            assert keyword not in filtered, f"Failed for keyword: {keyword}"
            assert "Keep this" in filtered

    def test_goto_case_insensitive(self) -> None:
        """GOTO/goto/Goto all trigger removal (case-insensitive)."""
        for keyword in ["GOTO", "goto", "Goto", "GoTo"]:
            xml = f"""<workflow>
                <action>{keyword} step 2</action>
                <action>Keep this</action>
            </workflow>"""

            workflow_ir = create_test_workflow_ir(xml)
            filtered = filter_instructions(workflow_ir)

            assert keyword not in filtered, f"Failed for keyword: {keyword}"
            assert "Keep this" in filtered

    def test_halt_in_child_does_not_propagate(self) -> None:
        """HALT in child doesn't propagate to remove parent.

        Only the specific element with HALT text is removed.
        If that leaves the parent empty, the empty container rule applies.
        """
        xml = """<workflow>
            <check if="some technical condition">
                <action>HALT - stop here</action>
                <action>Another action</action>
            </check>
        </workflow>"""

        workflow_ir = create_test_workflow_ir(xml)
        filtered = filter_instructions(workflow_ir)

        # HALT action removed, but check kept because "Another action" remains
        assert "HALT" not in filtered
        assert "Another action" in filtered
        assert "some technical condition" in filtered


class TestStructurePreservation:
    """Tests for AC3: Structure preservation."""

    def test_structure_preserved(self) -> None:
        """Element order and nesting is preserved."""
        xml = """<workflow>
            <step n="1" goal="First">
                <action>Action 1</action>
                <action>Action 2</action>
            </step>
            <step n="2" goal="Second">
                <action>Action 3</action>
            </step>
        </workflow>"""

        workflow_ir = create_test_workflow_ir(xml)
        filtered = filter_instructions(workflow_ir)

        # Check order preservation
        assert filtered.index("Action 1") < filtered.index("Action 2")
        assert filtered.index("Action 2") < filtered.index("Action 3")


class TestEmptyContainerRemoval:
    """Tests for AC3: Empty container removal."""

    def test_empty_step_containers_removed(self) -> None:
        """Steps that become empty after filtering are removed."""
        xml = """<workflow>
            <step n="1" goal="All filtered">
                <ask>Prompt removed</ask>
                <output>Output removed</output>
            </step>
            <step n="2" goal="Has content">
                <action>Kept action</action>
            </step>
        </workflow>"""

        workflow_ir = create_test_workflow_ir(xml)
        filtered = filter_instructions(workflow_ir)

        assert 'n="1"' not in filtered
        assert 'n="2"' in filtered

    def test_empty_check_containers_removed(self) -> None:
        """Checks that become empty after filtering are removed."""
        xml = """<workflow>
            <check if="story_num > 1">
                <ask>Prompt removed</ask>
                <output>Output removed</output>
            </check>
            <check if="sharded pattern exists">
                <action>Kept action</action>
            </check>
        </workflow>"""

        workflow_ir = create_test_workflow_ir(xml)
        filtered = filter_instructions(workflow_ir)

        # First check is empty after filtering - removed
        assert filtered.count("<check") == 1
        # Second check has content - kept
        assert "sharded pattern exists" in filtered


class TestAttributePreservation:
    """Tests for AC4: Attribute preservation."""

    def test_attributes_preserved(self) -> None:
        """All attributes on kept elements are preserved."""
        xml = """<workflow>
            <step n="5" goal="Important goal">
                <check if="template-workflow">
                    <action>Template action</action>
                </check>
                <invoke-protocol name="discover_inputs"/>
            </step>
        </workflow>"""

        workflow_ir = create_test_workflow_ir(xml)
        filtered = filter_instructions(workflow_ir)

        assert 'n="5"' in filtered
        assert 'goal="Important goal"' in filtered
        assert 'if="template-workflow"' in filtered
        assert 'name="discover_inputs"' in filtered


class TestVariablePlaceholders:
    """Tests for AC5: Variable placeholder preservation."""

    def test_variable_placeholders_preserved(self) -> None:
        """{{variable}} placeholders are not modified."""
        xml = """<workflow>
            <action>Set story Status to: "{{status}}"</action>
            <action>Load {{sprint_status}} file</action>
            <action>Extract Epic {{epic_num}}</action>
        </workflow>"""

        workflow_ir = create_test_workflow_ir(xml)
        filtered = filter_instructions(workflow_ir)

        assert "{{status}}" in filtered
        assert "{{sprint_status}}" in filtered
        assert "{{epic_num}}" in filtered


class TestErrorHandling:
    """Tests for AC6: Error handling."""

    def test_empty_input_returns_empty(self) -> None:
        """Empty raw_instructions returns empty string."""
        workflow_ir = create_test_workflow_ir("")
        filtered = filter_instructions(workflow_ir)

        assert filtered == ""

    def test_whitespace_only_returns_empty(self) -> None:
        """Whitespace-only raw_instructions returns empty string."""
        workflow_ir = create_test_workflow_ir("   \n\t  ")
        filtered = filter_instructions(workflow_ir)

        assert filtered == ""

    def test_invalid_xml_raises_compiler_error(self) -> None:
        """Invalid XML in raw_instructions raises CompilerError."""
        workflow_ir = create_test_workflow_ir("<workflow><unclosed>")

        with pytest.raises(CompilerError, match="Invalid XML"):
            filter_instructions(workflow_ir)


class TestRootElementHandling:
    """Tests for root element handling in AC1 and AC3."""

    def test_fully_filtered_workflow_returns_empty_root(self) -> None:
        """When all children are filtered, return empty root element."""
        xml = """<workflow>
            <ask>User prompt removed</ask>
            <output>Output removed</output>
            <template-output>Marker removed</template-output>
        </workflow>"""

        workflow_ir = create_test_workflow_ir(xml)
        filtered = filter_instructions(workflow_ir)

        # Root element preserved but empty
        assert "<workflow" in filtered
        # No children remain
        assert "<ask>" not in filtered
        assert "<output>" not in filtered

    def test_root_element_always_preserved(self) -> None:
        """Root element is preserved even if not in whitelist."""
        xml = """<instructions>
            <action>Keep this</action>
        </instructions>"""

        workflow_ir = create_test_workflow_ir(xml)
        filtered = filter_instructions(workflow_ir)

        assert "<instructions>" in filtered or "<instructions " in filtered
        assert "<action>" in filtered


class TestDeepNesting:
    """Tests for deeply nested structures in AC6."""

    def test_deeply_nested_structures(self) -> None:
        """Deeply nested structures are processed correctly."""
        # Build 30-level deep nesting
        xml = "<workflow>"
        for i in range(30):
            xml += f'<step n="{i}" goal="Level {i}">'
        xml += "<action>Deep action</action>"
        for _ in range(30):
            xml += "</step>"
        xml += "</workflow>"

        workflow_ir = create_test_workflow_ir(xml)
        filtered = filter_instructions(workflow_ir)

        assert "Deep action" in filtered
        assert 'n="29"' in filtered  # Deepest level preserved


class TestWhitelistEnforcement:
    """Tests for whitelist-based filtering (not just blacklist)."""

    def test_non_whitelisted_tags_removed(self) -> None:
        """Elements with tags not in whitelist are removed."""
        xml = """<workflow>
            <action>Keep this</action>
            <ui>Remove this unknown tag</ui>
            <random-tag>Also removed</random-tag>
            <malicious-script>Definitely removed</malicious-script>
        </workflow>"""

        workflow_ir = create_test_workflow_ir(xml)
        filtered = filter_instructions(workflow_ir)

        assert "Keep this" in filtered
        assert "<ui" not in filtered
        assert "random-tag" not in filtered
        assert "malicious-script" not in filtered

    def test_only_whitelist_tags_survive(self) -> None:
        """Verify comprehensive whitelist enforcement."""
        xml = """<workflow>
            <action>keep</action>
            <step n="1" goal="test"><action>keep</action></step>
            <check if="x > 1"><action>keep</action></check>
            <invoke-task>keep</invoke-task>
            <invoke-workflow>keep</invoke-workflow>
            <invoke-protocol name="test"/>
            <critical>keep</critical>
            <mandate>keep</mandate>
            <note>keep</note>
            <unknown>REMOVE</unknown>
            <foo>REMOVE</foo>
            <bar>REMOVE</bar>
        </workflow>"""

        workflow_ir = create_test_workflow_ir(xml)
        filtered = filter_instructions(workflow_ir)

        # All whitelist tags should be present
        assert "<action>" in filtered
        assert "<step" in filtered
        assert "<check" in filtered
        assert "<invoke-task>" in filtered
        assert "<invoke-workflow>" in filtered
        assert "<invoke-protocol" in filtered
        assert "<critical>" in filtered
        assert "<mandate>" in filtered
        assert "<note>" in filtered

        # Non-whitelist tags should be removed
        assert "<unknown" not in filtered
        assert "<foo" not in filtered
        assert "<bar" not in filtered


class TestEdgeCases:
    """Tests for various edge cases."""

    def test_self_closing_action_preserved(self) -> None:
        """Self-closing action element is preserved."""
        xml = """<workflow>
            <action/>
            <action>Normal action</action>
        </workflow>"""

        workflow_ir = create_test_workflow_ir(xml)
        filtered = filter_instructions(workflow_ir)

        # Self-closing action should be preserved
        assert "action" in filtered.lower()

    def test_unicode_content_preserved(self) -> None:
        """Unicode characters in text content are preserved."""
        xml = """<workflow>
            <action>Create story: 日本語テスト</action>
            <action>Polish: ąęćżźół</action>
        </workflow>"""

        workflow_ir = create_test_workflow_ir(xml)
        filtered = filter_instructions(workflow_ir)

        assert "日本語テスト" in filtered
        assert "ąęćżźół" in filtered

    def test_mixed_content_text_and_children(self) -> None:
        """Mixed content (text + children) is handled correctly."""
        xml = """<workflow>
            <step n="1" goal="Mixed">
                Text before
                <action>Action in middle</action>
                Text after
            </step>
        </workflow>"""

        workflow_ir = create_test_workflow_ir(xml)
        filtered = filter_instructions(workflow_ir)

        # Text content should be preserved
        assert "Action in middle" in filtered

    def test_check_without_if_attribute_kept(self) -> None:
        """Check element without if attribute is kept."""
        xml = """<workflow>
            <check>
                <action>Action inside</action>
            </check>
        </workflow>"""

        workflow_ir = create_test_workflow_ir(xml)
        filtered = filter_instructions(workflow_ir)

        # Check without if attr defaults to technical (kept)
        assert "<check>" in filtered or "<check " in filtered
        assert "Action inside" in filtered


class TestXmlCommentPreservation:
    """Tests for XML comment preservation through filtering."""

    def test_xml_comments_preserved(self) -> None:
        """XML comments survive filtering and are restored correctly."""
        xml = """<workflow>
            <!-- Important comment -->
            <action>Do something</action>
            <!-- Another comment -->
        </workflow>"""

        workflow_ir = create_test_workflow_ir(xml)
        filtered = filter_instructions(workflow_ir)

        assert "<!-- Important comment -->" in filtered
        assert "<!-- Another comment -->" in filtered

    def test_comments_in_output_format_cdata_preserved(self) -> None:
        """Comments inside CDATA (output-format) are restored as text markers.

        When XML comments appear inside CDATA sections (e.g., METRICS_JSON markers),
        they get escaped by ElementTree. The fix ensures escaped placeholders
        are converted back to escaped comments, preserving the text appearance.
        """
        # This tests the _placeholders_to_comments fix for escaped placeholders
        from bmad_assist.compiler.filtering import _placeholders_to_comments

        # Simulate what happens: comment in CDATA gets escaped after ElementTree
        escaped_placeholder = (
            "&lt;__xml_comment__&gt; METRICS_JSON_START &lt;/__xml_comment__&gt;"
        )
        result = _placeholders_to_comments(escaped_placeholder)

        # Should restore to escaped comment (preserves text in CDATA)
        assert result == "&lt;!-- METRICS_JSON_START --&gt;"

    def test_mixed_escaped_and_unescaped_placeholders(self) -> None:
        """Both escaped and unescaped placeholders are handled correctly."""
        from bmad_assist.compiler.filtering import _placeholders_to_comments

        mixed = (
            "<__xml_comment__> normal comment </__xml_comment__>"
            "some text"
            "&lt;__xml_comment__&gt; METRICS &lt;/__xml_comment__&gt;"
        )
        result = _placeholders_to_comments(mixed)

        assert "<!-- normal comment -->" in result
        assert "&lt;!-- METRICS --&gt;" in result
