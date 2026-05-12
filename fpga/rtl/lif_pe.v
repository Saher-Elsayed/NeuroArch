// NeuroArch LIF Processing Element
// 16 neurons per PE; membrane potential stored in BRAM; one PE update per clock
// Spike sparsity: zero-spike skipping via AXI-Stream tkeep masking
// Paper: Section VIII-A, Table 3 (FPGA resources)
`timescale 1ns/1ps
module lif_pe #(
    parameter N_NEURONS  = 16,
    parameter DATA_WIDTH = 8,   // INT8 weights
    parameter ALPHA_Q    = 8    // alpha = 1 - dt/tau in Q0.8 fixed-point
)(
    input  wire                   clk,
    input  wire                   rst_n,
    // AXI-Stream spike input
    input  wire [N_NEURONS-1:0]   s_axis_tdata,  // pre-synaptic spikes
    input  wire                   s_axis_tvalid,
    input  wire                   s_axis_tkeep,  // 0 = skip (zero-spike)
    // Output: spike vector for next layer
    output reg  [N_NEURONS-1:0]   spike_out,
    output reg                    spike_valid
);
    // Membrane potential registers (one per neuron)
    reg signed [15:0] V [0:N_NEURONS-1];
    localparam V_TH    = 16'sd1024;  // threshold
    localparam V_RESET = 16'sd0;

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i=0; i<N_NEURONS; i=i+1) V[i] <= 0;
            spike_out   <= 0;
            spike_valid <= 0;
        end else if (s_axis_tvalid && s_axis_tkeep) begin
            // Leaky decay + integrate
            for (i=0; i<N_NEURONS; i=i+1) begin
                // V[n+1] = alpha*V[n] + w*S[n]  (alpha in Q0.8)
                V[i] <= (V[i] * ALPHA_Q) >>> 8 + (s_axis_tdata[i] ? 16'sd32 : 16'sd0);
                spike_out[i]   <= (V[i] >= V_TH);
                if (V[i] >= V_TH) V[i] <= V_RESET;
            end
            spike_valid <= 1;
        end else begin
            // Zero-spike: no update (sparsity exploitation)
            spike_valid <= s_axis_tvalid;
            spike_out   <= 0;
        end
    end
endmodule
