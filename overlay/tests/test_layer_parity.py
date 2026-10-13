"""coreai_layer_parity: the per-layer error metric, and (when a saved ANE parity record is present) the recorded
per-layer parity of a Core AI build against the torch reference of the same export.

PARITY_RECORD points at the JSON lines written by `coreai_layer_parity.py ane` on the M6; skipped when absent."""
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import coreai_layer_parity as P  # noqa: E402

RECORDS = sorted((ROOT / "docs/results/m6-parity").glob("*-ane-parity.jsonl"))
if os.environ.get("PARITY_RECORD"):
    RECORDS = [Path(os.path.expanduser(os.environ["PARITY_RECORD"]))]


class LayerParityMetric(unittest.TestCase):
    def test_identical_rows(self):
        r = np.random.default_rng(0).normal(size=(8, 64))
        rel, cos = P.layer_errors(r, r)
        self.assertEqual(rel, 0.0)
        self.assertAlmostEqual(cos, 1.0, places=12)

    def test_scaled_and_noisy_rows(self):
        rng = np.random.default_rng(1)
        r = rng.normal(size=(16, 256))
        rel, cos = P.layer_errors(1.1 * r, r)
        self.assertAlmostEqual(rel, 0.1, places=9)
        self.assertAlmostEqual(cos, 1.0, places=9)
        rel, cos = P.layer_errors(-r, r)
        self.assertAlmostEqual(rel, 2.0, places=9)
        self.assertAlmostEqual(cos, -1.0, places=9)

    def test_median_ignores_one_bad_row(self):
        r = np.ones((5, 4))
        a = r.copy()
        a[0] = -5
        rel, cos = P.layer_errors(a, r)
        self.assertEqual(rel, 0.0)
        self.assertAlmostEqual(cos, 1.0)


@unittest.skipUnless(RECORDS, "no saved ANE parity record (docs/results/m6-parity or PARITY_RECORD)")
class RecordedAneParity(unittest.TestCase):
    def test_every_layer_within_two_percent_and_logits_agree(self):
        for rec in RECORDS:
            with self.subTest(record=rec.name):
                rows = [json.loads(l) for l in rec.read_text().splitlines() if l.startswith("{")]
                layers = [r for r in rows if "layer" in r]
                self.assertGreaterEqual(len(layers), 4)
                for r in layers:
                    self.assertLess(r["rel_l2"], 0.02, r)
                    self.assertGreater(r["cos"], 0.999, r)
                agree = [r for r in rows if "logits_top1_agree" in r][0]
                self.assertGreaterEqual(agree["logits_top1_agree"], 0.95)


if __name__ == "__main__":
    unittest.main()
