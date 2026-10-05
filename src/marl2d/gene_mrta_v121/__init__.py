"""V1.21 resident-world SEGB scheduler for the public minmax-mTSP benchmark.

Protocol:
- 34 fixed evolution benchmark instances are the external judges;
- each round contains exactly N logical candidate worlds (formal N=1000);
- every candidate Gene is evaluated on all 34 fixed instances;
- only after the whole round finishes is the Pareto Gene Bank rebuilt;
- rounds after round 0 inherit one parent from the Bank with squared total-score
  sampling, then mutate it;
- parent-child deltas are diagnostics only, never the admission gate;
- formal run is 1000 worlds x 34 instances x 50 rounds.
"""
