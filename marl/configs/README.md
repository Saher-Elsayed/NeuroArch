# MARL Configuration Files

YAML configs for QMIX training per building.

Key hyperparameters (matches Paper Appendix B):
- `lr: 1.0e-4`  (Adam)
- `buffer_size: 10000`
- `batch_size: 32`
- `gamma: 0.99`
- `lambda_E/C/P: 1.0/2.0/0.5`  (Pareto optimal, Table 12)
- `n_episodes: 2000`
