# MILS minmax-mTSP benchmark provenance

Mirrored source repository:

https://github.com/pengfeihe-angers/mils

Source paper:

Pengfei He, Jin-Kao Hao, Jinhui Xia,
"Learning-guided iterated local search for the minmax multiple traveling
salesman problem",
Computers & Operations Research 185:107255, 2026.
DOI: 10.1016/j.cor.2025.107255

Mirrored source blobs:

- instances.zip: 878aeee12fc4453904ddb86d905baa82203c806d
- Certification.zip: f3d65efd832fdb89e88ad2c1fafe1aa5869bb27e
- README.md: ee921aa59868c59841f30c43154ff697c2a99bac

Mirror commit in MRTA_RL_final_project:

ae4e5f5fdc8ac1f01fd822ecb1f90c48f0a4fa0d

The raw archives are preserved byte-for-byte as Git blobs. Formal V1.20
training reads these mirrored files directly and does not fetch benchmark data
at runtime.

The source paper distinguishes known exact optimum values (marked by *) from
best-known solutions (BKS). V1.20 preserves that distinction.
