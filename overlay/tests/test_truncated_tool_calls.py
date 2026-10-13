"""Tool calls cut off by the token limit: the streaming splitter must not send their partial arguments (the client
would get invalid JSON and retry the same oversized call), and truncated_call() must explain the cut."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qwen38_server as S  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "write", "parameters": {"type": "object", "properties": {
    "file_path": {"type": "string"}, "content": {"type": "string"}}}}}]

FULL = ("I will write it.\n<tool_call>\n<function=write>\n<parameter=file_path>\n/tmp/a.html\n</parameter>\n"
        "<parameter=content>\n<html>hi</html>\n</parameter>\n</function>\n</tool_call>")
CUT = ("I will write it.\n<tool_call>\n<function=write>\n<parameter=file_path>\n/tmp/a.html\n</parameter>\n"
       "<parameter=content>\n<html><body><script>const canvas = document.getElem")


def run_splitter(text, step=7):
    sp = S.StreamSplitter(False, TOOLS)
    out = []
    for i in range(0, len(text), step):
        out.append(sp.feed(text[i:i + step]))
    out.append(sp.feed("", final=True))
    return sp, out


class TruncatedToolCalls(unittest.TestCase):
    def test_complete_call_streams_valid_json(self):
        sp, out = run_splitter(FULL)
        args = "".join(d["function"].get("arguments", "") for o in out for d in o.get("tool_calls", []))
        self.assertEqual(json.loads(args), {"file_path": "/tmp/a.html", "content": "<html>hi</html>"})
        self.assertEqual(len(sp.calls), 1)

    def test_cut_call_sends_no_arguments(self):
        sp, out = run_splitter(CUT)
        self.assertEqual(sp.calls, [])
        self.assertFalse(any("tool_calls" in o for o in out))
        content = "".join(o.get("content", "") for o in out)
        self.assertEqual(content.strip(), "I will write it.")

    def test_truncated_call_note(self):
        note = S.truncated_call(CUT, "length")
        self.assertIn("`write`", note)
        self.assertIn("max_tokens", note)
        self.assertEqual(S.truncated_call(FULL, "length"), "")
        self.assertEqual(S.truncated_call(CUT, "stop"), "")
        self.assertEqual(S.truncated_call("plain text", "length"), "")


if __name__ == "__main__":
    unittest.main()


def _turn(cmd, result):
    return [{"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function", "function": {
                "name": "run_terminal_command", "arguments": {"command": cmd}}}]},
            {"role": "tool", "tool_call_id": "c", "content": result}]


class RepeatedFailedCalls(unittest.TestCase):
    """The loop seen in an agent session: grep with no match (exit 1) sent again and again at temperature 0."""
    BASE = [{"role": "system", "content": "s"}, {"role": "user", "content": "check the file"}]

    def test_three_identical_failures_detected(self):
        msgs = self.BASE + _turn("grep -n try a.html", "exit: 1 ") * 3
        self.assertEqual(S.repeated_failed_call(msgs, 3), ("run_terminal_command", 3))

    def test_reminder_between_turns_still_detected(self):
        msgs = (self.BASE + _turn("grep x", "exit: 1") * 2 + [{"role": "user", "content": "<system-reminder>"}]
                + _turn("grep x", "exit: 1"))
        self.assertEqual(S.repeated_failed_call(msgs, 3), ("run_terminal_command", 3))

    def test_not_triggered(self):
        self.assertIsNone(S.repeated_failed_call(self.BASE + _turn("grep x", "exit: 1") * 2, 3))   # only 2
        self.assertIsNone(S.repeated_failed_call(self.BASE + _turn("ls", "exit: 0\na.html") * 3, 3))  # succeeded
        mixed = self.BASE + _turn("grep x", "exit: 1") * 2 + _turn("grep y", "exit: 1")
        self.assertIsNone(S.repeated_failed_call(mixed, 3))                                        # different args
        self.assertIsNone(S.repeated_failed_call(self.BASE + _turn("grep x", "exit: 1") * 3, 0))   # guard off
        self.assertIsNone(S.repeated_failed_call(self.BASE + _turn("grep x", "exit: 1") * 3
                                                 + [{"role": "assistant", "content": "done"}], 3))  # model moved on
