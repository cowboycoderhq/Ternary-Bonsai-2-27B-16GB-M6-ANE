"""AttnW on the host (fp32): the builder's verify path for one gated-attention layer against the streamed torch
reference (dflash2_target_ref.Target.attn) on the same weights and input rows, for any config the builder is
imported with (MODEL). Covers the query / output head layout that has to hold for grp == nkv (Ornith 9B, 16 / 4
heads) as well as grp != nkv (Qwen3.8 27B, 24 / 4). Skipped without torch, coremltools or a checkpoint."""
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(os.path.expanduser(os.environ.get("MODEL", "~/Models/Qwen3.8-27B"))) / "config.json"
INDEX = CONFIG.parent / "model.safetensors.index.json"
try:
    import torch
    sys.path.insert(0, str(ROOT / "coreai"))
    sys.path.insert(0, str(ROOT / "scripts"))
    HAVE = CONFIG.exists() and INDEX.exists()
    if HAVE:
        import qwen38_coreai_build as B  # noqa: E402
        import dflash2_target_ref as R  # noqa: E402
except ImportError:  # pragma: no cover
    HAVE = False


@unittest.skipUnless(HAVE, "torch / coremltools / checkpoint (MODEL) not available")
class AttnHeadLayout(unittest.TestCase):
    def test_attention_layer_matches_reference(self):
        c = R.text_cfg()
        L = c["layer_types"].index("full_attention")
        B.KV_CACHE_DTYPE, B.STABLE_ATTN = "fp16", False
        W = B.layer_arrays(B.M.Checkpoint(), L)
        att = B.AttnW(W, L).float().eval()
        tgt = R.Target.__new__(R.Target)
        tgt.cfg, tgt.eps = c, c["rms_norm_eps"]
        tgt.rot = int(c["head_dim"] * c["rope_parameters"]["partial_rotary_factor"])
        tgt.inv = 1.0 / c["rope_parameters"]["rope_theta"] ** (torch.arange(0, tgt.rot, 2, dtype=torch.float64) / tgt.rot)
        T, ctx = 8, 64
        torch.manual_seed(0)
        h = torch.randn(T, c["hidden_size"]) * 0.5
        w = R.Weights().layer(L)
        s_in, s_out = B.mixer_rotation(W, L)  # online mixer rotation (rotated exports): fold M into the weights,
        for seed, n, names in ((s_in, c["hidden_size"], ("q", "k", "v")),  # x M W^T = x (W M^T)^T
                               (s_out, B.nh * B.hd, ("o",))):
            if seed is None:
                continue
            with torch.no_grad():
                M = B.Hadamard(n, seed).float()(torch.eye(n).T.reshape(1, n, 1, n))[0, :, 0, :].T  # rows: e_i M
            for m in names:
                w[f"self_attn.{m}_proj.weight"] = w[f"self_attn.{m}_proj.weight"] @ M.T
        ref = tgt.attn(L, w, h, [(R.SeqState(c), 0, T, True)])
        f = torch.arange(0, T, dtype=torch.float64)[:, None] * tgt.inv[None]
        cos, sin = torch.cat([f, f], 1).cos().float(), torch.cat([f, f], 1).sin().float()
        k_st = torch.zeros(B.nkv, B.hd, ctx) if B.KV_KEYS_T else torch.zeros(B.nkv, ctx, B.hd)
        with torch.no_grad():
            y = att(h.T.reshape(1, -1, 1, T), cos, sin, torch.full((1, ctx), -1e4), k_st,
                    torch.zeros(B.nkv, ctx, B.hd), ctx, T)[0].reshape(-1, T).T
        rel = np.linalg.norm((y - ref).numpy(), axis=1) / np.linalg.norm(ref.numpy(), axis=1)
        self.assertLess(float(rel.max()), 2e-3, rel)

    def test_head_layout_is_head_major(self):
        """Rows of KV head j are its grp query heads j*grp .. j*grp+grp-1, each with T rows (GQA grouping)."""
        T = 8
        qh = torch.arange(T * B.nh * B.hd, dtype=torch.float32).reshape(T, B.nh, B.hd)
        q = qh.permute(1, 0, 2).reshape(B.nkv, B.grp * T, B.hd)
        for j in range(B.nkv):
            for g in range(B.grp):
                for t in range(T):
                    self.assertTrue(torch.equal(q[j, g * T + t], qh[t, j * B.grp + g]))


if __name__ == "__main__":
    unittest.main()
