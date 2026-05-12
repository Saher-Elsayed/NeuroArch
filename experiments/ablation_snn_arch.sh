#!/bin/bash
# Table 7: SNN architecture ablation
echo "=== Table 7: SNN Architecture Ablation ==="
python3 -c "
rows=[('4-L LIF full (NeuroArch)',94.3,0.31,'79%'),
      ('4-L LIF no rate reg',94.5,0.40,'9%'),
      ('4-L LIF no surrogate',86.5,0.31,'79%'),
      ('4-L ReLU ANN',82.5,3.80,'0%'),
      ('2-layer LIF',88.3,0.19,'82%'),
      ('6-layer LIF',95.1,0.37,'76%'),
      ('INT8 quantized',93.9,0.23,'79%')]
print(f'{chr(10)}{"Config":<32} {"Acc%":>6} {"mW":>6} {"Spar":>6}')
print('-'*55)
for r in rows:
    print(f'{r[0]:<32} {r[1]:>6.1f} {r[2]:>6.2f} {r[3]:>6}')
"
