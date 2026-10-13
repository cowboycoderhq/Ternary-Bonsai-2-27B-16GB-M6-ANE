"""Summary table of an M6 matrix run (bench.py JSON in DIR): speed (median of 3) and energy, ANE vs GPU, with the
pass/fail of each comparison.    python scripts/m6bench/summarize.py DIR"""
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
rows = {"bonsai": ("bonsai-gpu", "bonsai-ane-draft"), "ornith": ("ornith-gpu", "ornith-ane")}
ok_all = True
for model, (g, a) in rows.items():
    G, A = (json.loads((d / f"{t}.json").read_text()) for t in (g, a))
    print(f"== {model}: GPU={g}  ANE={a}  power source: {A.get('power_source')}")
    for n in ("512", "2048"):
        gm, am = G["median"][n], A["median"][n]
        for k, better in (("prefill_tps", "higher"), ("decode_tps", "higher"),
                          ("J_per_prompt_token", "lower"), ("J_per_gen_token", "lower")):
            ok = am[k] > gm[k] if better == "higher" else am[k] < gm[k]
            ok_all &= ok
            print(f"  prompt {n:>4}  {k:<19} GPU {gm[k]:9.4f}  ANE {am[k]:9.4f}  ratio {am[k] / gm[k]:6.2f}  "
                  f"{'PASS' if ok else 'FAIL'}")
    runs = A["runs"]
    if runs and runs[0].get("accept_pct") is not None:
        print("  drafter accept % per run:", [round(r["accept_pct"], 1) for r in runs],
              " tokens/verify:", [round(r["tokens_per_call"], 2) for r in runs])
    print("  ANE placement:", sorted(set((A.get("placement") or {}).values())),
          " health:", {k: A.get("health", {}).get(k) for k in ("soc_class", "ane_bonded_compile_mode")})
print("ALL SPEED+ENERGY CRITERIA:", "PASS" if ok_all else "FAIL")
