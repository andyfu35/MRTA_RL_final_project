# V1.7-T Direct-Time Scaling

This experiment isolates the architecture question raised by the V1.7 pilot.

It keeps the V1.7 direct autoregressive assignment policy with learned
STOP/WAIT and reuses the already proven V1.6-T-O cached MILP time-oracle
dataset.

Oracle data:

- 256 training worlds
- 64 probe worlds
- same V1.6-T environment
- T* is a proven global MILP optimum
- MILP is an external capability reference only and never supplies actions

The optimized capability is:

    O_T = mean_world( T_direct(world) / T_star(world) )

This is deliberately a Time-only scaling ablation. The six-axis V1.7
multi-objective experiment remains unchanged.

## Smoke

    git pull
    bash tools/run_gene_mrta_v17t_scale_mac.sh smoke

Smoke uses 10 generations, population 64 and eight oracle worlds per
generation. It bootstraps from the V1.7 pilot Time specialist and other V1.7
specialists only as initial evolutionary seeds.

## Long run

    bash tools/run_gene_mrta_v17t_scale_mac.sh long

Defaults:

- 2000 generations
- population 256
- archive 32
- HOF 64
- oracle batch schedule 16 -> 32 -> 64 -> 128
- 10% random immigrants
- checkpoint every 25 generations

The run is resumable:

    bash tools/run_gene_mrta_v17t_scale_mac.sh resume runs/gene_mrta_v17t_scale/<run>/checkpoint.json

## Held-out evaluation

The 97M worlds remain completely held out.

After training, run:

    python -m marl2d.gene_mrta_v17.global_time_test \
      --direct-run <V1.7-T run dir> \
      --v16to-run runs/gene_mrta_v16to/gene_mrta_v16to_20260930_141419_seed7 \
      --worlds 20 \
      --world-seed 97000000 \
      --time-limit 300 \
      --output-dir runs/gene_mrta_v17t_scale_global_test/heldout20

Interpretation:

- if probe and held-out retention both rise, the previous V1.7 gap was mostly
  data/optimization limited;
- if probe rises but held-out stays flat, generalization remains the main
  issue;
- if both plateau, the 116-parameter direct policy representation becomes the
  next target for expansion.
