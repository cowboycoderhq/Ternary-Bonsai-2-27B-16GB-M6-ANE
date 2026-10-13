"""Convert the prism-ml Ternary-Bonsai-2-27B MLX pack to a Forge MODEL dir + export dir (no GPTQ).

Each folded matrix W' (ternary codes x per-128-group scales, input side rotated by the pack's global
sign + Walsh-Hadamard transform) becomes a 2-bit LUT {-1, 0, +1} with one least-squares scale per output
row ("ternary + pcs"). The pack's rotation is kept: the export records seeds from SIGN_SEEDS, which the
patched builder maps back to the pack's explicit sign vectors. lm_head keeps its rotation (HEAD_ROT).
The embedding table is dequantized and un-rotated once into MODEL/embed_tokens_fp16.npy.

    PACK=~/models/bonsai2-mlx OUT_MODEL=... OUT_EXPORT=... python bonsai2_to_forge.py
"""
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
from scipy.linalg import hadamard

PACK = Path(os.path.expanduser(os.environ.get("PACK", "~/models/bonsai2-mlx")))
OUT_MODEL = Path(os.path.expanduser(os.environ["OUT_MODEL"]))
OUT_EXPORT = Path(os.path.expanduser(os.environ["OUT_EXPORT"]))
NLAYERS = int(os.environ.get("NLAYERS", "0"))
SIGN_SEEDS = {5120: 905120, 6144: 906144, 17408: 917408}  # keep in sync with qwen38_coreai_build.SIGN_BY_SEED
PRE = "language_model.model."
BLOCK = 1024
NORMS_SHIFTED = ("input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_norm.weight",
                 "self_attn.k_norm.weight")


def unpack_codes(w):
    """uint32 (rows, cols/16) MLX 2-bit -> int8 codes (rows, cols) in {0, 1, 2} (value = code - 1)."""
    shifts = (2 * np.arange(16, dtype=np.uint32))
    return ((w[..., None] >> shifts) & 3).reshape(w.shape[0], -1).astype(np.uint8)


def unembed_rows(w, s, b, signs):
    """Embedding rows of the pack in the model basis: dequantize (q * scale + bias per 128-group), then the pack's
    inverse transform (normalized Walsh-Hadamard per 1024 block, then the explicit signs)."""
    h = (hadamard(BLOCK) / math.sqrt(BLOCK)).astype(np.float32)
    c = unpack_codes(w).astype(np.float32)
    x = (c.reshape(c.shape[0], -1, 128) * s[..., None].astype(np.float32) + b[..., None].astype(np.float32))
    x = x.reshape(c.shape[0], -1)
    return (x.reshape(x.shape[0], -1, BLOCK) @ h).reshape(x.shape) * signs.astype(np.float32)


def ternary_pcs(f, name):
    w, s, b = (f.get_tensor(f"{name}.{k}") for k in ("weight", "scales", "biases"))
    if not np.allclose(b.astype(np.float32), -s.astype(np.float32)):
        raise ValueError(f"{name}: biases are not -scales (not a ternary pack)")
    idx = unpack_codes(w)
    if idx.max() > 2:
        raise ValueError(f"{name}: code 3 present")
    c = idx.astype(np.float32) - 1
    sg = np.repeat(s.astype(np.float32), 128, axis=1)
    nz = np.abs(c)
    row = (sg * nz).sum(1) / np.maximum(nz.sum(1), 1)          # least squares for W' ~ row * c
    err = ((c * sg - c * row[:, None]) ** 2).sum() / max(((c * sg) ** 2).sum(), 1e-30)
    lut = np.array([[-1.0], [0.0], [1.0], [0.0]], np.float16)
    return {"lut": lut, "idx": idx, "scale": row.astype(np.float16)}, 10 * math.log10(1 / max(err, 1e-30))


def main():
    cfg = json.loads((PACK / "config.json").read_text())
    text = dict(cfg["text_config"])
    nl = text["num_hidden_layers"] if not NLAYERS else NLAYERS
    OUT_MODEL.mkdir(parents=True, exist_ok=True)
    OUT_EXPORT.mkdir(parents=True, exist_ok=True)
    hj = json.loads((PACK / "hadamard.json").read_text())
    if hj["prism.hadamard.block_size"] != BLOCK or not hj["prism.hadamard.gdn_v_grouped"]:
        raise ValueError("unexpected hadamard.json")
    vals, off, signs = hj["prism.hadamard.sign_values"], 0, {}
    for wd in hj["prism.hadamard.sign_widths"]:
        signs[wd] = np.asarray(vals[off:off + wd], np.float32)
        off += wd
    np.savez(OUT_MODEL / "bonsai_signs.npz", **{str(SIGN_SEEDS[k]): v for k, v in signs.items()})
    f = safe_open(str(PACK / "model.safetensors"), framework="np")
    keys = set(f.keys())
    for k, v in signs.items():  # every folded matrix must use the global sign vector of its input width
        for name in [n for n in keys if n.endswith(".signs") and f.get_slice(n).get_shape()[0] == k][:6]:
            if not np.array_equal(f.get_tensor(name), v):
                raise ValueError(f"{name} differs from the global {k}-wide signs")

    small, log = {}, {"layers": []}
    for i in range(nl):
        p = f"{PRE}layers.{i}."
        for k in [k for k in keys if k.startswith(p) and not k.endswith((".scales", ".biases", ".signs"))]:
            sub = k[len(p):]
            if f.get_tensor(k).dtype == np.uint32:
                continue
            a = f.get_tensor(k).astype(np.float32)
            if sub in NORMS_SHIFTED:
                a = a - 1.0                                   # MLX stores 1 + w; the builder adds the 1
            if sub == "linear_attn.conv1d.weight":
                a = np.ascontiguousarray(a.transpose(0, 2, 1))  # MLX (C, K, 1) -> HF (C, 1, K)
            small[f"model.language_model.layers.{i}.{sub}"] = a.astype(np.float16) if a.ndim == 2 else a
        rec = {"layer": i}
        mlp = {}
        for m in ("gate", "up", "down"):
            q, snr = ternary_pcs(f, f"{p}mlp.{m}_proj")
            mlp.update({f"{m}.{k}": v for k, v in q.items()})
            rec[m] = round(snr, 2)
        save_file(mlp, str(OUT_EXPORT / f"layer_{i:02d}.safetensors"),
                  metadata={"format": "ternary + pcs", "basis": "online", "seed_in": str(SIGN_SEEDS[5120]),
                            "seed_mid": str(SIGN_SEEDS[17408])})
        mix = {}
        names = (("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj")
                 if f"{p}linear_attn.out_proj.weight" in keys else
                 ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"))
        for n in names:
            q, snr = ternary_pcs(f, p + n)
            mix.update({f"{n}.{k}": v for k, v in q.items()})
            rec[n.split(".")[-1]] = round(snr, 2)
        save_file(mix, str(OUT_EXPORT / f"layer_{i:02d}_mixer.safetensors"),
                  metadata={"format": "ternary + pcs", "basis": "online", "seed_in": str(SIGN_SEEDS[5120]),
                            "seed_out": str(SIGN_SEEDS[6144])})
        log["layers"].append(rec)
        print(json.dumps(rec), flush=True)

    q, snr = ternary_pcs(f, f"{PRE[:-6]}lm_head")
    save_file({f"lm_head.{k}": v for k, v in q.items()}, str(OUT_EXPORT / "lm_head.safetensors"),
              metadata={"format": "ternary + pcs", "rot_seed": str(SIGN_SEEDS[5120])})
    log["lm_head"] = round(snr, 2)
    print("lm_head", round(snr, 2), flush=True)

    small["model.language_model.norm.weight"] = f.get_tensor(f"{PRE}norm.weight").astype(np.float32) - 1.0
    save_file(small, str(OUT_MODEL / "model-small.safetensors"))
    wmap = {k: "model-small.safetensors" for k in small}
    (OUT_MODEL / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wmap}))

    # embedding: dequantize rows and undo the pack's inverse transform (out = fwht_inverse(dequant))
    emb = OUT_MODEL / "embed_tokens_fp16.npy"
    if not emb.exists():
        w, s = f.get_tensor(f"{PRE}embed_tokens.weight"), f.get_tensor(f"{PRE}embed_tokens.scales")
        b = f.get_tensor(f"{PRE}embed_tokens.biases")
        es = f.get_tensor(f"{PRE}embed_tokens.signs").astype(np.float32)
        out = np.lib.format.open_memmap(emb.with_suffix(".tmp.npy"), mode="w+", dtype=np.float16,
                                        shape=(w.shape[0], text["hidden_size"]))
        for a in range(0, w.shape[0], 8192):
            out[a:a + 8192] = unembed_rows(w[a:a + 8192], s[a:a + 8192], b[a:a + 8192], es).astype(np.float16)
        out.flush()
        del out
        emb.with_suffix(".tmp.npy").rename(emb)

    hf_cfg = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
              "text_config": text, "tie_word_embeddings": False,
              "bonsai_source": "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"}
    (OUT_MODEL / "config.json").write_text(json.dumps(hf_cfg, indent=1))
    for n in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        if (PACK / n).exists():
            shutil.copy2(PACK / n, OUT_MODEL / n)
    (OUT_EXPORT / "bonsai_convert.json").write_text(json.dumps(log, indent=1))
    print("done", nl, "layers")


if __name__ == "__main__":
    main()
