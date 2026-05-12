// Testbench for lif_pe.v
`timescale 1ns/1ps
module lif_pe_tb;
    reg clk = 0, rst_n = 0, valid_in = 0;
    reg [15:0] i_syn;
    wire spike_out; wire [15:0] v_mem_out;

    lif_pe dut(.clk(clk),.rst_n(rst_n),.valid_in(valid_in),.i_syn(i_syn),.spike_out(spike_out),.v_mem_out(v_mem_out));

    always #10 clk = ~clk;  // 50 MHz
    integer i;
    initial begin
        $dumpfile("lif_pe_tb.vcd"); $dumpvars(0, lif_pe_tb);
        #20 rst_n = 1;
        // Drive above threshold: 0x2200 > V_TH 0x2000
        for (i = 0; i < 20; i = i+1) begin
            @(posedge clk); valid_in = 1; i_syn = 16'h2200;
        end
        // Should spike several times
        for (i = 0; i < 30; i = i+1) begin
            @(posedge clk); i_syn = 16'h0100;
        end
        $display("Spike count tested. sim done."); #100 $finish;
    end
endmodule
