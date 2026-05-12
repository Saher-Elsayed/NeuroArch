#!/usr/bin/env bash
# Reproduce Table 4 — Controller Comparison
set -e
echo "=== Table 4: Controller Comparison ==="
for building in medium_office residential mixed_use; do
    echo "--- $building ---"
    python -c "
import pandas as pd
df = pd.read_csv('data/energyplus/${building}/controller_comparison.csv')
print(df.to_string(index=False))
print()
"
done
