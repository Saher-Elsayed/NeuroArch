// AXI-Lite register bank (16 regs) - Cortex-M0 interface
// CTRL[0]: soft_reset, CTRL[1]: clock_gate, CTRL[2]: task_mode
// WIN_LEN: inference window (default 100), THRESHOLD: spike count alert
// STATUS: busy[0] done[1], RESULT: {confidence[11:0], label[3:0]}
`timescale 1ns/1ps
module axi_lite_ctrl #(parameter AW=6, DW=32)(
    input  wire          clk, rst_n,
    // Write channel
    input  wire [AW-1:0] awaddr,  input wire awvalid,  output reg awready,
    input  wire [DW-1:0] wdata,   input wire wvalid,   output reg wready,
    output reg  [1:0]    bresp,   output reg bvalid,   input wire bready,
    // Read channel
    input  wire [AW-1:0] araddr,  input wire arvalid,  output reg arready,
    output reg  [DW-1:0] rdata,   output reg [1:0] rresp, output reg rvalid,
    input  wire          rready,
    // SNN control
    output reg  [6:0]    win_len, output reg [11:0] threshold,
    output reg           soft_reset, output reg clock_gate,
    input  wire          inference_done,
    input  wire [3:0]    comfort_label, input wire [11:0] confidence
);
    reg [DW-1:0] regs [0:15];
    localparam REG_CTRL=0, REG_WIN=1, REG_THR=2, REG_STAT=3, REG_RESULT=4;
    initial begin
        regs[0]=0; regs[1]=100; regs[2]=10; regs[3]=0; regs[4]=0;
    end
    always @(posedge clk) begin
        regs[REG_STAT]   <= {30'h0, inference_done, 1'b0};
        regs[REG_RESULT] <= {16'h0, confidence, comfort_label};
    end
    assign win_len   = regs[REG_WIN][6:0];
    assign threshold = regs[REG_THR][11:0];
    assign soft_reset = regs[REG_CTRL][0];
    assign clock_gate = regs[REG_CTRL][1];
endmodule
