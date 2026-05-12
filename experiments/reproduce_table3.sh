#!/bin/bash
# Reproduce Table 3: SNN accuracy (all buildings + LOBO)
set -e
echo "=== Table 3: SNN Comfort Classification ==="
cd snn
for bld in medium_office residential mixed_use; do
    echo "--- $bld ---"
    python evaluate.py --building $bld --T 100
done
echo "Expected: NeuroArch 94.3% mean, 87.4% LOBO (paper Table 3)"
