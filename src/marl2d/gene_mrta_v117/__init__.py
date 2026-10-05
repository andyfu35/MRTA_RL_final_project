"""V1.17 clean two-stage Gene MRTA training protocol.

Stage A:
    train from the complete known task semantics with independent base
    capability axes.

Stage B:
    analyze Stage-A hard worlds, add only evidence-supported robustness axes,
    and retrain/fuse a final policy.

V1.17 intentionally does not reuse V1.13 capability labels.
"""
