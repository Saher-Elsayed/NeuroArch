// Spike count accumulator + WTA comparator tree (5-class output)
`timescale 1ns/1ps
module comparator_tree #(parameter N_CLASSES=5, COUNT_W=8)(
    input  wire                      clk, rst_n, clear,
    input  wire [N_CLASSES-1:0]      spike_in,
    input  wire                      valid_in,
    output reg  [$clog2(N_CLASSES)-1:0] winner,
    output reg  [COUNT_W-1:0]           max_count,
    output reg                          result_valid
);
    reg [COUNT_W-1:0] counts [0:N_CLASSES-1];
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || clear) begin
            for(i=0;i<N_CLASSES;i=i+1) counts[i]<=0;
            winner<=0; max_count<=0; result_valid<=0;
        end else if (valid_in) begin
            for(i=0;i<N_CLASSES;i=i+1)
                if (spike_in[i] && counts[i] < {COUNT_W{1'b1}}) counts[i]<=counts[i]+1;
            // WTA: find max
            begin: wta
                integer j; reg [COUNT_W-1:0] mx; reg [$clog2(N_CLASSES)-1:0] wi;
                mx=0; wi=0;
                for(j=0;j<N_CLASSES;j=j+1) if(counts[j]>mx) begin mx=counts[j]; wi=j; end
                winner<=wi; max_count<=mx;
            end
            result_valid <= 1;
        end
    end
endmodule
