# Arty A7-35T Constraints for NeuroArch
# Main 100MHz oscillator -> PLL -> 50MHz for SNN
set_property PACKAGE_PIN E3 [get_ports clk_50mhz]
set_property IOSTANDARD LVCMOS33 [get_ports clk_50mhz]
create_clock -period 20.000 -name sys_clk [get_ports clk_50mhz]
# Reset button
set_property PACKAGE_PIN C2 [get_ports rst_n]
set_property IOSTANDARD LVCMOS33 [get_ports rst_n]
# Timing constraints
set_max_delay -datapath_only 10.0 [get_cells -hier -filter {NAME =~ */lif_pe/*}]
