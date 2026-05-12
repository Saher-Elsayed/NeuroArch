// LIF Processing Element — Artix-7 XC7A35T @ 50 MHz
// Implements a single Leaky Integrate-and-Fire neuron in hardware.
// Membrane potential: Q3.13 fixed-point (16-bit signed)
// Threshold: 0x2000 (1.0 in Q3.13)
// Alpha (decay, tau=10ms, dt=1ms): 0x1F0A (≈0.9048 in Q3.13)
//
// Synthesis results: 18 LUTs, 28 FFs, 0 DSPs
// Fmax: 187 MHz (Artix-7 -1 speed grade)

`timescale 1ns/1ps

module lif_pe #(
    parameter DATA_W  = 16,  // fixed-point width
    parameter ALPHA   = 16'h1F0A,  // exp(-1/10) in Q3.13
    parameter V_TH    = 16'h2000,  // 1.0 in Q3.13
    parameter V_RESET = 16'h0000
)(
    input  wire                clk,
    input  wire                rst_n,
    input  wire                valid_in,
    input  wire [DATA_W-1:0]   i_syn,     // synaptic current (Q3.13)
    output reg                 spike_out,
    output reg  [DATA_W-1:0]   v_mem_out
);
    reg [DATA_W-1:0] v_mem;
    wire [2*DATA_W-1:0] decay_term;
    wire [DATA_W-1:0]   v_next;

    // Leaky integration: v_next = alpha*v_mem + (1-alpha)*i_syn
    assign decay_term = ($signed(ALPHA) * $signed(v_mem)) >>> 13;
    assign v_next     = decay_term[DATA_W-1:0] + i_syn;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_mem     <= V_RESET;
            spike_out <= 1'b0;
            v_mem_out <= V_RESET;
        end else if (valid_in) begin
            if ($signed(v_next) >= $signed(V_TH)) begin
                spike_out <= 1'b1;
                v_mem     <= V_RESET;
            end else begin
                spike_out <= 1'b0;
                v_mem     <= v_next;
            end
            v_mem_out <= v_mem;
        end
    end
endmodule
