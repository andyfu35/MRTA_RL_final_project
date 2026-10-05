"""V1.20 public minmax-mTSP benchmark evolution.

Data is mirrored from:
https://github.com/pengfeihe-angers/mils

Primary phase:
- one shared 148-parameter Gene;
- variable city/task and robot counts;
- mutation-only paired parent/child evaluation;
- external reference = exact optimum when known, otherwise published BKS;
- public set S drives evolution/validation;
- public set L is protected for scale generalization;
- mating is deferred until the mutation-only Bank converges.
"""
