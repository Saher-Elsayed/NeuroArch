# Arty A7-35T Master Constraint File for NeuroArch
# Primary 100 MHz oscillator -> MMCM -> 50 MHz system clock

set_property PACKAGE_PIN E3 [get_ports clk_50mhz]
set_property IOSTANDARD LVCMOS33 [get_ports clk_50mhz]
create_clock -period 20.000 -name sys_clk [get_ports clk_50mhz]

# Active-low reset (BTN0)
set_property PACKAGE_PIN C2 [get_ports rst_n]
set_property IOSTANDARD LVCMOS33 [get_ports rst_n]

# Timing constraints
set_max_delay -datapath_only 10.0 [get_cells -hier -filter {NAME =~ */lif_pe/*}]
set_false_path -from [get_ports rst_n]

# Physical constraints
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
