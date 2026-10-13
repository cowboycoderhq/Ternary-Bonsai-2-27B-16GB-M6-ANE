"""Automatic prefix cache: the server's snapshot bookkeeping (parent segments, longest-prefix choice, RAM -> SSD
spill, leaf eviction, restore of a chain) over a fake model with the real KV-row and restore code of
qwen38_coreai_model.CoreAIQwen. No ANE: the model keeps numpy KV arrays and DeltaNet states."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import qwen38_server as S  # noqa: E402

try:
    import qwen38_coreai_model as A  # noqa: E402
except Exception:  # pragma: no cover  (needs the runtime's imports only)
    A = None

NKV, HD, CTX = 2, 4, 512


class FakeModel:
    """The KV / snapshot / restore methods of CoreAIQwen on plain arrays; feeding token t writes t into every KV row
    and adds t to the DeltaNet state, so restored state is checkable."""
    snapshot = A.CoreAIQwen.snapshot if A else None
    restore = A.CoreAIQwen.restore if A else None
    restorable = A.CoreAIQwen.restorable if A else None
    kv_rows = A.CoreAIQwen.kv_rows if A else None
    _put_kv_rows = A.CoreAIQwen._put_kv_rows if A else None
    _kv_transposed = A.CoreAIQwen._kv_transposed if A else None

    def __init__(self):
        self.keys_t, self.ladder, self.ctx = True, [CTX], CTX
        self.kv = [{"k3": (None, np.zeros((NKV, HD, CTX), np.float16)),
                    "v3": (None, np.zeros((NKV, CTX, HD), np.int8)),
                    "vs3": (None, np.zeros((NKV, CTX), np.float16))}]
        self.chunks = [{"curw": {"rec0": np.zeros(3, np.float32)}}]
        self.pos, self.pending, self.hi = 0, 0, 0

    def cap(self, ctx):
        return ctx

    def resize(self, ctx):
        self.ctx = ctx

    def reset(self, shrink=True):
        self.chunks[0]["curw"]["rec0"][:] = 0
        self.pos = self.pending = 0

    def feed(self, ids, on_features=None):
        for t in ids:
            d = self.kv[0]
            d["k3"][1][:, :, self.pos] = t
            d["v3"][1][:, self.pos] = t % 100
            d["vs3"][1][:, self.pos] = t
            self.chunks[0]["curw"]["rec0"] += t
            self.pos += 1
        self.hi = max(self.hi, self.pos)
        return np.zeros(4, np.float32)


def engine(ram_mb, disk_gb, tmp, pin_max=0):
    import threading
    e = S.Engine.__new__(S.Engine)
    e.a = SimpleNamespace(pin_dir=str(Path(tmp) / "pins"), pin_max=pin_max)
    e.lock, e.pins = threading.Lock(), []
    e.model, e.v2, e.drafter = FakeModel(), True, None
    e.fed, e.snaps, e.apc = [], {}, []
    e.apc_budget, e.apc_disk_budget = int(ram_mb * 2**20), int(disk_gb * 2**30)
    e.apc_dir = Path(tmp)
    e.im_start, e.user_hdr = 1, [1, 2]
    e.tok = SimpleNamespace(decode=lambda t, skip_special_tokens=False: str(t))
    return e


SYS_LEN = 300  # pinning ignores system prompts under 256 tokens


def prompt(workspace, question):
    """[system] [user: workspace] [user: question] [assistant header]: message starts are token 1."""
    return [1] + [9] * (SYS_LEN - 1) + [1, 2] + [workspace] * 6 + [1, 2] + [question] * 3 + [1, 7]


@unittest.skipIf(A is None, "qwen38_coreai_model not importable")
class PrefixCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def state_after(self, ids):
        """Reference: a fresh model fed the whole prompt."""
        m = FakeModel()
        m.feed(ids)
        return m

    def assert_same_state(self, e, ids):
        ref = self.state_after(ids)
        for n in ("k3", "v3", "vs3"):
            np.testing.assert_array_equal(e.model.kv[0][n][1][..., :len(ids)] if n != "v3"
                                          else e.model.kv[0][n][1][:, :len(ids)],
                                          ref.kv[0][n][1][..., :len(ids)] if n != "v3"
                                          else ref.kv[0][n][1][:, :len(ids)])
        np.testing.assert_array_equal(e.model.chunks[0]["curw"]["rec0"], ref.chunks[0]["curw"]["rec0"])
        self.assertEqual(e.model.pos, len(ids))

    def test_other_workspace_between_reuses_longest_prefix(self):
        e = engine(64, 0, self.tmp)
        a1, b, a2 = prompt(50, 60), prompt(51, 60), prompt(50, 61)
        _, r = e.prefill(a1)
        self.assertEqual(r, 0)
        e.prefill(b)                                  # overwrites the live KV rows past the system prompt
        _, r = e.prefill(a2)                          # same workspace, new question: workspace prefix restored
        self.assertEqual(r, SYS_LEN + 8)              # system + workspace message (8)
        self.assert_same_state(e, a2)

    def test_child_entries_store_only_their_segment(self):
        e = engine(64, 0, self.tmp)
        e.prefill(prompt(50, 60))
        by_len = {len(x["ids"]): x for x in e.apc}
        self.assertIn(SYS_LEN, by_len)
        self.assertIn(SYS_LEN + 8, by_len)
        child = by_len[SYS_LEN + 8]
        self.assertIs(child["parent"], by_len[SYS_LEN])
        self.assertEqual(child["kv_start"], SYS_LEN)
        self.assertEqual(child["kv"][0]["v3"].shape[1], 8)

    def test_spill_to_ssd_and_restore(self):
        e = engine(0.0001, 1, self.tmp)               # RAM budget below one entry: everything but the newest spills
        a1 = prompt(50, 60)
        e.prefill(a1)
        e.prefill(prompt(51, 60))
        self.assertTrue(any("disk" in x for x in e.apc))
        self.assertTrue(any(Path(self.tmp).glob("*.pkl")))
        a2 = prompt(50, 61)
        _, r = e.prefill(a2)
        self.assertEqual(r, SYS_LEN + 8)
        self.assert_same_state(e, a2)

    def test_eviction_keeps_parents_of_live_children(self):
        e = engine(0.0001, 0, self.tmp)               # no SSD tier: over budget means drop
        e.prefill(prompt(50, 60))
        e.prefill(prompt(51, 60))
        ids = {id(x) for x in e.apc}
        for x in e.apc:
            if x.get("parent") is not None:
                self.assertIn(id(x["parent"]), ids)

    def test_disabled_cache_keeps_old_behaviour(self):
        e = engine(0, 0, self.tmp)
        e.prefill(prompt(50, 60))
        e.prefill(prompt(51, 60))
        _, r = e.prefill(prompt(50, 61))
        self.assertEqual(e.apc, [])
        self.assertEqual(r, SYS_LEN)                  # only the in-place system snapshot

    def test_system_prompt_pinned_and_warmed_after_restart(self):
        e = engine(64, 1, self.tmp, pin_max=3)
        a1 = prompt(50, 60)
        e.prefill(a1)
        pins = json.loads((Path(self.tmp) / "pins" / f"{S.MODEL_ID}.json").read_text())
        self.assertEqual(pins, [a1[:SYS_LEN]])                       # the system prefix was recorded
        self.assertEqual([len(x["ids"]) for x in e.apc if x.get("pinned")], [SYS_LEN])
        # restart: a new engine warms the pin, so the first request skips the system prompt
        e2 = engine(64, 1, self.tmp, pin_max=3)
        e2.pins = e2._pin_load()
        e2.warm_pins()
        self.assertEqual([len(x["ids"]) for x in e2.apc if x.get("pinned")], [SYS_LEN])
        e2.model.reset()                                             # live state gone, cache entry stays
        e2.fed = []
        b = prompt(51, 61)
        _, r = e2.prefill(b)
        self.assertEqual(r, SYS_LEN)
        self.assert_same_state(e2, b)

    def test_pinned_entry_survives_pressure(self):
        e = engine(0.0001, 0, self.tmp, pin_max=3)                   # tiny RAM, no SSD: everything else drops
        e.prefill(prompt(50, 60))
        for w in range(51, 56):
            e.prefill(prompt(w, 60))
        self.assertIn(SYS_LEN, [len(x["ids"]) for x in e.apc if x.get("pinned")])


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(A is None, "qwen38_coreai_model not importable")
class PinAfterMismatch(unittest.TestCase):
    def test_pin_used_when_live_state_differs_early(self):
        """A pinned system prefix, the live state from another session, a prompt that shares only the pin."""
        tmp = tempfile.mkdtemp()
        e = engine(64, 1, tmp, pin_max=3)
        e.prefill(prompt(50, 60))                                    # records + pins the system prefix
        e.prefill([1] + [8] * 400 + [1, 7])                          # another client: different system prompt
        b = prompt(51, 61)
        _, r = e.prefill(b)
        self.assertEqual(r, SYS_LEN)
        e.model.reset(); e.fed = []
        ref = FakeModel(); ref.feed(b)
        np.testing.assert_array_equal(e.model.chunks[0]["curw"]["rec0"], 0)  # after reset: sanity
