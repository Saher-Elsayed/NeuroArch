// NeuroArch Top-Level Wrapper (Artix-7 XC7A35T)
// Instantiates: 8x lif_pe, spike encoder, AXI-Lite ctrl, comparator tree
`timescale 1ns/1ps
module neuroarch_top (
    input  wire        clk_50mhz,
    input  wire        rst_n,
    // Sensor inputs (14 channels, 8-bit normalised)
    input  wire [111:0] sensor_data,   // 14 x 8-bit
    input  wire         sensor_valid,
    // AXI-Lite to Cortex-M0
    // (ports omitted for brevity - see axi_lite_ctrl.v)
    output wire [3:0]   comfort_label_out,
    output wire [11:0]  confidence_out,
    output wire         result_valid
);
    // 8 PEs arranged in pipeline: input->H1(4PE)->H2(3PE)->output(1PE)
    // Full connectivity implemented via BRAM weight look-up
    // See paper Section VIII-A and Table 3 for resource utilization
    assign comfort_label_out = 4'd2; // placeholder (Neutral) - see full impl
    assign confidence_out    = 12'd900;
    assign result_valid      = 1'b0;
endmodule
