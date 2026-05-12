// NeuroArch AXI-Lite Control Register Bank (16 registers)
// Paper: Section VIII-C  — CTRL, WIN_LEN, THRESHOLD, STATUS, RESULT
`timescale 1ns/1ps
module axi_lite_ctrl #(parameter ADDR_W = 6, DATA_W = 32)(
    input  wire             clk, rst_n,
    // AXI-Lite slave
    input  wire [ADDR_W-1:0] s_axi_awaddr,
    input  wire              s_axi_awvalid,
    output reg               s_axi_awready,
    input  wire [DATA_W-1:0] s_axi_wdata,
    input  wire              s_axi_wvalid,
    output reg               s_axi_wready,
    output reg  [1:0]        s_axi_bresp,
    output reg               s_axi_bvalid,
    input  wire              s_axi_bready,
    input  wire [ADDR_W-1:0] s_axi_araddr,
    input  wire              s_axi_arvalid,
    output reg               s_axi_arready,
    output reg  [DATA_W-1:0] s_axi_rdata,
    output reg  [1:0]        s_axi_rresp,
    output reg               s_axi_rvalid,
    input  wire              s_axi_rready,
    // SNN control outputs
    output reg  [6:0]  win_len,        // default 100
    output reg  [11:0] threshold,      // spike count threshold
    output reg         soft_reset,
    // SNN status inputs
    input  wire        inference_done,
    input  wire [3:0]  comfort_label,
    input  wire [11:0] confidence_score
);
    // Registers: 0=CTRL, 1=WIN_LEN, 2=THRESHOLD, 3=STATUS, 4=RESULT
    reg [DATA_W-1:0] regs [0:15];
    initial begin
        regs[0] = 32'h0; regs[1] = 32'd100; regs[2] = 32'd10;
        regs[3] = 32'h0; regs[4] = 32'h0;
    end
    assign win_len   = regs[1][6:0];
    assign threshold = regs[2][11:0];
    assign soft_reset = regs[0][0];
    always @(posedge clk) begin
        regs[3] <= {31'h0, inference_done};
        regs[4] <= {16'h0, confidence_score, comfort_label};
    end
endmodule
