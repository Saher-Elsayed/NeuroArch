# Vivado build script for NeuroArch
# Usage: vivado -mode batch -source fpga/scripts/build.tcl
set project_name "neuroarch"
set device "xc7a35tcsg324-1"

create_project $project_name ./fpga/build -part $device -force
add_files -fileset sources_1 [glob fpga/rtl/*.v]
add_files -fileset constrs_1 fpga/constraints/arty_a7.xdc
add_files -fileset sim_1 [glob fpga/sim/*.v]

set_property top neuroarch_top [current_fileset]
set_property top tb_top [get_filesets sim_1]

# Synthesis
launch_runs synth_1 -jobs 8; wait_on_run synth_1
# Implementation
launch_runs impl_1 -to_step write_bitstream -jobs 8; wait_on_run impl_1
# Reports
open_run impl_1
report_utilization -file fpga/reports/utilization.rpt
report_timing_summary -file fpga/reports/timing.rpt
report_power -file fpga/reports/power.rpt
puts "Build complete."
