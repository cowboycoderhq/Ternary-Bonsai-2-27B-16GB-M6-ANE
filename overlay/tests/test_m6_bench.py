"""M6 benchmark harness (scripts/m6bench): energy integration over macmon samples, and the committed M6 results
(docs/results/m6-bench): ANE prefill and decode faster, and joules per prompt token and per
generated token lower, than llama.cpp Metal on the same M6, for both models and both prompt sizes."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "m6bench"))
import bench  # noqa: E402

RES = ROOT / "docs" / "results" / "m6-bench"


class EnergyIntegration(unittest.TestCase):
    def test_constant_power_window(self):
        s = [(0.0, 1.0, 2000.0, 1000.0, 500.0, 500.0), (1.0, 2.0, 2000.0, 1000.0, 500.0, 500.0)]
        self.assertEqual(bench.energy(s, 0.0, 2.0), [4.0, 2.0, 1.0, 1.0])
        self.assertEqual(bench.energy(s, 0.5, 1.5), [2.0, 1.0, 0.5, 0.5])   # partial overlap counts its share
        self.assertEqual(bench.energy(s, 3.0, 4.0), [0.0, 0.0, 0.0, 0.0])

    def test_real_macmon_samples(self):
        samples = bench.macmon_samples(RES / "macmon-sample.jsonl")
        self.assertGreater(len(samples), 30)
        for t0, t1, sys_mw, cpu, gpu, ane in samples:
            self.assertGreater(t1, t0)
            self.assertLess(t1 - t0, 0.5)          # 50 ms requested; must stay well under a phase window
            self.assertGreater(sys_mw, 0)          # whole-machine power, mW
        a, b = samples[0][0], samples[-1][1]
        mean_w = bench.energy(samples, a, b)[0] / (b - a)
        self.assertGreater(mean_w, 0.5)
        self.assertLess(mean_w, 60)               # an M6 mini draws a few to tens of watts


class CommittedResults(unittest.TestCase):
    def test_ane_beats_gpu_on_speed_and_energy(self):
        for gpu, ane in (("bonsai-gpu", "bonsai-ane-draft"), ("ornith-gpu", "ornith-ane")):
            runs = json.loads((RES / "summary.json").read_text())["runs"]
            G, A = runs[gpu], runs[ane]
            self.assertEqual(A["health"]["soc_class"], "m6")
            self.assertEqual(A["health"]["ane_bonded_compile_mode"], 2)
            self.assertEqual(set(A["placement"].values()), {"fully_ane"})
            self.assertTrue(A["power_source"].startswith("macmon") and G["power_source"].startswith("macmon"))
            for n in ("512", "2048"):
                g, a = G["median"][n], A["median"][n]
                with self.subTest(model=ane, prompt=n):
                    self.assertEqual(len([r for r in A["per_run"] if str(r["prompt_size"]) == n]), 3)
                    self.assertEqual(a["prompt_tokens"], g["prompt_tokens"])
                    self.assertEqual(a["gen_tokens"], g["gen_tokens"])
                    self.assertGreater(a["prefill_tps"], g["prefill_tps"])
                    self.assertGreater(a["decode_tps"], g["decode_tps"])
                    self.assertLess(a["J_per_prompt_token"], g["J_per_prompt_token"])
                    self.assertLess(a["J_per_gen_token"], g["J_per_gen_token"])


if __name__ == "__main__":
    unittest.main()
