# Artix-7 XC7A35T Timing Constraints for NeuroArch SNN
# Target: 50 MHz (20 ns period)
create_clock -period 20.000 -name clk_50mhz [get_ports clk_50mhz]
set_input_delay  -clock clk_50mhz -max 2.0 [get_ports {sensor_data[*] sensor_valid}]
set_output_delay -clock clk_50mhz -max 3.0 [get_ports {comfort_class[*] output_valid}]
set_false_path -from [get_ports rst_n]
