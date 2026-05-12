# FPGA Implementation Notes

## Target: Artix-7 XC7A35T

### Resources (Post-Implementation, Vivado 2023.2)
| Resource | Used | Available | Util |
|----------|------|-----------|------|
| LUT6 | 6,814 | 33,280 | 20.5% |
| Flip-Flops | 9,241 | 66,560 | 13.9% |
| BRAM 36K | 6 | 50 | 12.0% |
| DSP48E1 | 8 | 90 | 8.9% |

### Timing
- Target: 50 MHz (20 ns)
- Achieved Fmax: 52.8 MHz
- WNS: +2.8 ns

### Power
- Nominal (25°C): 0.31 mW
- Worst-case (85°C): 0.72 mW
- Budget (edge): 10 mW (92.8% margin)

### Inference Latency
- Per-timestep: 1 ms (50 MHz clock, 50 cycles)
- Full 100ms window: 4.1 ms mean (pipeline latency + BRAM access)
- Worst-case (p99): 12.0 ms (within 12ms budget)

## Build Instructions
```bash
cd fpga
vivado -mode batch -source scripts/build.tcl
```

## Simulation
```bash
cd fpga/sim
iverilog -o tb_lif_pe.vvp ../rtl/lif_pe.v tb_lif_pe.v
vvp tb_lif_pe.vvp
gtkwave tb_lif_pe.vcd
```
