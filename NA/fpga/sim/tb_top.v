// Testbench: NeuroArch Top-Level
`timescale 1ns/1ps
module tb_top;
    reg clk=0, rst_n=0;
    reg [111:0] sensors=0;
    reg valid=0;
    wire [3:0] label;
    wire [11:0] conf;
    wire rv;

    neuroarch_top dut(.clk_50mhz(clk),.rst_n(rst_n),
        .sensor_data(sensors),.sensor_valid(valid),
        .comfort_label_out(label),.confidence_out(conf),.result_valid(rv));

    always #10 clk=~clk;
    initial begin
        $dumpfile("tb_top.vcd"); $dumpvars(0,tb_top);
        #30 rst_n=1;
        // Simulate Warm comfort event: high temp (0xD0), high RH (0xBF)
        sensors[7:0]=8'hD0; sensors[15:8]=8'hBF;
        valid=1; #10000 valid=0;
        #500 $finish;
    end
endmodule
