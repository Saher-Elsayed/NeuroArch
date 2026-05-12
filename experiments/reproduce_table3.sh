#!/bin/bash
# Table 3: SNN accuracy per building + LOBO cross-validation
set -e; echo "=== Table 3: SNN Comfort Classification ==="
for bld in medium_office residential mixed_use; do
    echo "--- $bld ---"
    python -m snn.evaluate --building $bld
done
echo "Expected: 94.3% mean, 87.4% LOBO (Table 3)"
