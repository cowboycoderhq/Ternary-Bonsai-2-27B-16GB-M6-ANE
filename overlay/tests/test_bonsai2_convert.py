"""Bonsai 2 pack -> Forge export: the converter's ternary decode against the pack's own dequantization, the
per-row scale fit, the embedding un-rotation against the Prism runtime's inverse fwht, and the builder's
explicit-sign Hadamard against the Prism fwht. Runs on real pack tensors (BONSAI_PACK, default
~/models/bonsai2-mlx); skipped when the pack or its runtime dependencies are missing."""
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACK = Path(os.path.expanduser(os.environ.get("BONSAI_PACK", "~/models/bonsai2-mlx")))
HAVE = (PACK / "model.safetensors").exists()
sys.path.insert(0, str(ROOT / "scripts"))

try:
    from safetensors import safe_open
    from scipy.linalg import hadamard
    import torch
except ImportError:  # pragma: no cover
    HAVE = False

if HAVE:
    os.environ.setdefault("OUT_MODEL", tempfile.mkdtemp())
    os.environ.setdefault("OUT_EXPORT", tempfile.mkdtemp())
    import bonsai2_to_forge as C  # noqa: E402

PRE = "language_model.model.layers.0."


def pack_dequant(f, name):
    """The pack's affine 2-bit dequantization (mx.dequantize semantics): w = q * scale + bias per 128-group."""
    w, s, b = (f.get_tensor(f"{name}.{k}") for k in ("weight", "scales", "biases"))
    q = ((w[..., None] >> (2 * np.arange(16, dtype=np.uint32))) & 3).reshape(w.shape[0], -1, 128).astype(np.float32)
    return (q * s[..., None].astype(np.float32) + b[..., None].astype(np.float32)).reshape(w.shape[0], -1)


def prism_fwht(x, block, signs, inverse=False):
    """Prism runtime.fwht in numpy (normalized Sylvester-Walsh-Hadamard, explicit signs)."""
    h = hadamard(block) / math.sqrt(block)
    if not inverse:
        x = x * signs
    x = (x.reshape(*x.shape[:-1], -1, block) @ h).reshape(x.shape)
    if inverse:
        x = x * signs
    return x


@unittest.skipUnless(HAVE, "Bonsai 2 MLX pack or dependencies not available")
class Bonsai2Convert(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = safe_open(str(PACK / "model.safetensors"), framework="np")

    def test_ternary_codes_and_row_scale(self):
        for name in (PRE + "mlp.gate_proj", PRE + "linear_attn.out_proj"):
            q, snr = C.ternary_pcs(self.f, name)
            ref = pack_dequant(self.f, name)
            lut = q["lut"][:, 0].astype(np.float32)
            codes = lut[q["idx"]]
            # the codes are exactly the pack's ternary signs: sign(dequant) everywhere
            np.testing.assert_array_equal(codes, np.sign(ref))
            # rebuilt matrix = codes * row scale; its SNR matches the reported one and stays high
            rebuilt = codes * q["scale"].astype(np.float32)[:, None]
            err = ((ref - rebuilt) ** 2).sum() / (ref ** 2).sum()
            self.assertAlmostEqual(10 * math.log10(1 / err), snr, delta=0.05)
            self.assertGreater(snr, 18.0)
            # least-squares row scale: residual is orthogonal to the codes per row
            np.testing.assert_allclose(((ref - codes * q["scale"].astype(np.float64)[:, None]) * codes).sum(1)
                                       / np.abs(ref).sum(1), 0, atol=2e-3)

    def test_explicit_sign_hadamard_matches_prism(self):
        hj = json.loads((PACK / "hadamard.json").read_text())
        vals, off, signs = hj["prism.hadamard.sign_values"], 0, {}
        for wd in hj["prism.hadamard.sign_widths"]:
            signs[wd] = np.asarray(vals[off:off + wd], np.float64)
            off += wd
        with tempfile.TemporaryDirectory() as d:
            np.savez(Path(d) / "bonsai_signs.npz", **{str(C.SIGN_SEEDS[k]): v for k, v in signs.items()})
            (Path(d) / "config.json").write_text(json.dumps({"text_config": json.loads(
                (PACK / "config.json").read_text())["text_config"]}))
            os.environ["MODEL"] = d
            sys.path.insert(0, str(ROOT / "coreai"))
            import qwen38_ane_model as M  # another test may have imported it with a different MODEL
            prior = M.MODEL
            M.MODEL = Path(d)
            self.addCleanup(setattr, M, "MODEL", prior)
            import qwen38_coreai_build as B
            B.M.MODEL = Path(d)
            rng = np.random.default_rng(0)
            for n in (5120, 6144):
                x = rng.normal(size=(3, n))
                mod = B.Hadamard(n, C.SIGN_SEEDS[n])
                got = mod(torch.from_numpy(x.T.reshape(1, n, 1, 3)).half()).float().numpy().reshape(n, 3).T
                np.testing.assert_allclose(got, prism_fwht(x, 1024, signs[n]), atol=3e-2, rtol=0)
            self.assertIsNone(B.explicit_signs(12345, 5120))

    def test_embedding_unrotation(self):
        with tempfile.TemporaryDirectory() as d:
            name = "language_model.model.embed_tokens"
            w, s, b = (self.f.get_tensor(f"{name}.{k}")[:64] for k in ("weight", "scales", "biases"))
            es = self.f.get_tensor(f"{name}.signs").astype(np.float64)
            ref = prism_fwht((((w[..., None] >> (2 * np.arange(16, dtype=np.uint32))) & 3).reshape(64, -1, 128)
                              * s[..., None].astype(np.float32) + b[..., None].astype(np.float32)).reshape(64, -1),
                             1024, es, inverse=True)
            got = C.unembed_rows(w, s, b, es)
            np.testing.assert_allclose(got, ref, atol=2e-3, rtol=0)


if __name__ == "__main__":
    unittest.main()
