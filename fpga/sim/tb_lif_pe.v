// Testbench: LIF Processing Element
`timescale 1ns/1ps
module tb_lif_pe;
    reg clk=0, rst_n=0;
    reg [15:0] spikes=0;
    reg valid=0, keep=0;
    wire [15:0] spike_out;
    wire spike_valid;

    lif_pe #(.N_NEURONS(16)) dut(
        .clk(clk),.rst_n(rst_n),
        .s_axis_tdata(spikes),.s_axis_tvalid(valid),
        .s_axis_tkeep(keep),
        .spike_out(spike_out),.spike_valid(spike_valid)
    );

    always #10 clk=~clk;
    initial begin
        $dumpfile("tb_lif_pe.vcd"); $dumpvars(0,tb_lif_pe);
        #30 rst_n=1;
        // Inject all-ones spike for 100 timesteps
        repeat(100) begin
            @(posedge clk); spikes=16'hFFFF; valid=1; keep=1;
        end
        // Zero-spike skip test
        @(posedge clk); spikes=0; keep=0;
        #200 $finish;
    end
    initial begin
        $monitor("%t: spike_out=%b valid=%b", $time, spike_out, spike_valid);
    end
endmodule
