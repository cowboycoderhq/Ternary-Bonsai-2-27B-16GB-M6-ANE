# M6 ANE per-layer parity records

`coreai_layer_parity.py ane` output on the M6 (macOS 27.0.1, bonded compile mode 2), 64 WikiText-2 test tokens,
T=8 verify calls, cold state. Each line: median relative L2 error and cosine of the ANE residual stream against the
reference after that layer, then top-1 agreement of the logits.

- `ornith-ane-parity.jsonl`: Ornith-1.5-9B 3-bit export (build `forge-ornith15-taps`, taps 0-3,7,...,31) against
  `dflash2_target_ref.Target` on the same export (PyTorch fp32).
- `bonsai-ane-parity.jsonl`: Ternary Bonsai 2 27B (build `forge-bonsai2-taps`, taps 5,19,33,47,61) against the
  Prism MLX runtime with the per-row scales the converter writes.

`tests/test_layer_parity.py` checks every record: each layer under 2% error and cosine above 0.999, logits >= 95%.
