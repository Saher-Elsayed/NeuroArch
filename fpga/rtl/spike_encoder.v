// Rate-coded spike encoder for 14 analog sensor channels
// Poisson rate coding: rate proportional to normalised input [0,1]
// Output: 14-bit spike vector per clock cycle
`timescale 1ns/1ps

module spike_encoder #(
    parameter N_CHANNELS = 14,
    parameter RATE_BITS  = 16
)(
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire [N_CHANNELS*16-1:0]   sensor_in,   // 14x Q3.13 normalized values
    output reg  [N_CHANNELS-1:0]       spikes_out
);
    // Linear Feedback Shift Register for pseudo-random threshold
    reg [RATE_BITS-1:0] lfsr;
    wire feedback = lfsr[15] ^ lfsr[14] ^ lfsr[12] ^ lfsr[3];

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lfsr       <= 16'hACE1;
            spikes_out <= {N_CHANNELS{1'b0}};
        end else begin
            lfsr <= {lfsr[RATE_BITS-2:0], feedback};
            for (i = 0; i < N_CHANNELS; i = i+1) begin
                spikes_out[i] <= (sensor_in[i*16 +: 16] > lfsr) ? 1'b1 : 1'b0;
            end
        end
    end
endmodule
