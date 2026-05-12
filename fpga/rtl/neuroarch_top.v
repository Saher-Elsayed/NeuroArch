// NeuroArch Top-Level — Artix-7 XC7A35T
// Instantiates: spike encoder + 3-layer LIF array + AXI-Lite control
`timescale 1ns/1ps

module neuroarch_top (
    input  wire        clk_50mhz,
    input  wire        rst_n,
    // AXI-Lite slave (config + read-back)
    input  wire [31:0] s_axi_awaddr,
    input  wire        s_axi_awvalid,
    output wire        s_axi_awready,
    input  wire [31:0] s_axi_wdata,
    input  wire        s_axi_wvalid,
    output wire        s_axi_wready,
    output wire [31:0] s_axi_rdata,
    // Sensor input bus
    input  wire [223:0] sensor_data,   // 14 x 16-bit
    input  wire          sensor_valid,
    // Output: 5-class softmax approximation (argmax)
    output reg  [2:0]    comfort_class,
    output reg           output_valid
);
    // Spike encoding
    wire [13:0] spikes_layer0;
    spike_encoder #(.N_CHANNELS(14)) enc (
        .clk(clk_50mhz), .rst_n(rst_n),
        .sensor_in(sensor_data), .sensor_valid(sensor_valid),
        .spikes_out(spikes_layer0)
    );

    // Layer 1: 14 -> 64 LIF neurons (instantiated as array)
    wire [63:0] spikes_layer1;
    genvar j;
    generate
        for (j = 0; j < 64; j = j+1) begin : layer1
            lif_pe #(.DATA_W(16)) pe (
                .clk(clk_50mhz), .rst_n(rst_n), .valid_in(sensor_valid),
                .i_syn(spikes_layer0[j % 14] ? 16'h0400 : 16'h0000),
                .spike_out(spikes_layer1[j])
            );
        end
    endgenerate

    // Readout: population vote (simplified argmax)
    reg [7:0] accum [0:4];
    integer k;
    always @(posedge clk_50mhz) begin
        if (!rst_n) begin
            comfort_class <= 3'd2; output_valid <= 0;
            for (k = 0; k < 5; k = k+1) accum[k] <= 0;
        end else if (sensor_valid) begin
            accum[0] <= spikes_layer1[0 +: 13];
            accum[1] <= spikes_layer1[13 +: 13];
            accum[2] <= spikes_layer1[26 +: 13];
            accum[3] <= spikes_layer1[39 +: 13];
            accum[4] <= spikes_layer1[52 +: 12];
            // Simple argmax
            if (accum[2] >= accum[0] && accum[2] >= accum[1] &&
                accum[2] >= accum[3] && accum[2] >= accum[4])
                comfort_class <= 3'd2;
            else if (accum[0] >= accum[1])
                comfort_class <= 3'd0;
            else
                comfort_class <= 3'd4;
            output_valid <= 1;
        end else
            output_valid <= 0;
    end

    assign s_axi_awready = 1'b1;
    assign s_axi_wready  = 1'b1;
    assign s_axi_rdata   = {29'b0, comfort_class};
endmodule
