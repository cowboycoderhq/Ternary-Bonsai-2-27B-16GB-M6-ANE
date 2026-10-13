"""Per-layer parity of a Core AI target build against the torch reference of the same export.

    ref  (any Mac, torch):  MODEL=<ckpt> EXPORT_DIR=<export> python coreai_layer_parity.py ref --n 64 --out ref.npz
    ane  (M6, Core AI):     MODEL=<model> EMBED_NPY=... python coreai_layer_parity.py ane --build <build> --ref ref.npz

`ref` runs dflash2_target_ref.Target (the export dequantized, MLP rotated online) over the first n WikiText tokens
and saves the residual stream after every layer in TAPS (env, default every layer) plus the logits. `ane` feeds the
same tokens through the build in T=8 verify calls with the build's tap outputs and reports, per tapped layer, the
relative L2 error and cosine of the ANE hidden state against the reference. The build must be converted with the
same TAPS. Output: one JSON line per layer, then the logits' top-1 agreement.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def tokens(n):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ["MODEL"], local_files_only=True)
    text = Path(os.path.expanduser(os.environ.get("TEXT", "~/models/wikitext/wiki2_test.txt"))).read_text()
    return tok.encode(text, add_special_tokens=False)[:n]


def layer_errors(ane, ref):
    """Per row relative L2 error and cosine of ane against ref, (n, hidden) arrays -> (median rel, median cos)."""
    a, r = ane.astype(np.float64), ref.astype(np.float64)
    rel = np.linalg.norm(a - r, axis=1) / np.maximum(np.linalg.norm(r, axis=1), 1e-12)
    cos = (a * r).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(r, axis=1), 1e-12)
    return float(np.median(rel)), float(np.median(cos))


def cmd_ref(a):
    import torch
    import dflash2_target_ref as R
    n_layers = R.text_cfg()["num_hidden_layers"]
    taps = [int(x) for x in os.environ.get("TAPS", ",".join(map(str, range(n_layers)))).split(",") if x]
    R.TAPS = tuple(taps)
    ids = tokens(a.n)
    tgt = R.Target(n_layers=n_layers)
    st = R.SeqState(tgt.cfg)
    logits, feats = tgt.run([(st, ids, True)], [list(range(len(ids)))])
    np.savez(a.out, ids=np.array(ids), taps=np.array(taps), hidden=feats[0].float().numpy(),
             logits=logits[0].numpy().astype(np.float32))
    print(json.dumps({"ref": str(a.out), "tokens": len(ids), "taps": taps}))


def cmd_ane(a):
    import ane_compile_mode
    ane_compile_mode.apply(log=lambda m: None)
    os.environ.setdefault("COREAI_BRIDGE", "1")
    import qwen38_coreai_model as runtime
    ref = np.load(a.ref)
    ids, taps = ref["ids"].tolist(), ref["taps"].tolist()
    m = runtime.CoreAIQwenBridge(root=a.build, log=lambda *x: None, kv_cache_dtype="v8")
    if list(m.taps) != taps:
        raise SystemExit(f"build taps {m.taps} != reference taps {taps}")
    m.reset()
    hid, lg = [], []
    for i in range(0, len(ids), m.T):
        blk = ids[i:i + m.T]
        lg.append(m.call(blk)[:len(blk)])
        hid.append(m._features(len(blk)).reshape(len(blk), len(taps), -1))
        m.accept(len(blk))
    hid, lg = np.concatenate(hid), np.concatenate(lg)
    for j, layer in enumerate(taps):
        rel, cos = layer_errors(hid[:, j], ref["hidden"][:, j])
        rec = {"layer": layer, "rel_l2": round(rel, 4), "cos": round(cos, 5)}
        if os.environ.get("PER_BLOCK"):
            rec["per_block_rel"] = [round(layer_errors(hid[b:b + m.T, j], ref["hidden"][b:b + m.T, j])[0], 4)
                                    for b in range(0, len(ids), m.T)]
        print(json.dumps(rec), flush=True)
    agree = float((lg.argmax(-1) == ref["logits"].argmax(-1)).mean())
    print(json.dumps({"logits_top1_agree": round(agree, 4), "tokens": len(ids)}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("ref", "ane"))
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--build", type=Path)
    ap.add_argument("--ref", type=Path)
    a = ap.parse_args()
    {"ref": cmd_ref, "ane": cmd_ane}[a.cmd](a)


if __name__ == "__main__":
    main()
