from std.collections import List
from std.math import exp, sqrt
from std.python import Python

struct PackedTernaryWeight:
    var in_dim: Int
    var out_dim: Int
    var packed_data: List[UInt8]
    var scale: Float32

    def __init__(out self, in_dim: Int, out_dim: Int, scale: Float32 = 1.0):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.scale = scale
        # 4 weights packed per UInt8 byte
        var total_weights = in_dim * out_dim
        var num_bytes = (total_weights + 3) // 4
        self.packed_data = List[UInt8](capacity=num_bytes)
        for _ in range(num_bytes):
            self.packed_data.append(0)

    def set_weight(mut self, row: Int, col: Int, val: Int8):
        # val is in {-1, 0, 1}
        # Encoding: 0 -> 0b00, +1 -> 0b01, -1 -> 0b10
        var idx = row * self.in_dim + col
        var byte_idx = idx // 4
        var shift = UInt8((idx % 4) * 2)
        var code: UInt8 = 0
        if val == 1:
            code = 1
        elif val == -1:
            code = 2
        
        var mask: UInt8 = ~(UInt8(3) << shift)
        self.packed_data[byte_idx] = (self.packed_data[byte_idx] & mask) | (code << shift)

    def get_weight(self, row: Int, col: Int) -> Int8:
        var idx = row * self.in_dim + col
        var byte_idx = idx // 4
        var shift = UInt8((idx % 4) * 2)
        var code = (self.packed_data[byte_idx] >> shift) & UInt8(3)
        if code == 1:
            return 1
        elif code == 2:
            return -1
        return 0

struct BitNetLinear:
    var in_dim: Int
    var out_dim: Int
    var weights: PackedTernaryWeight

    def __init__(out self, in_dim: Int, out_dim: Int, scale: Float32 = 1.0):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weights = PackedTernaryWeight(in_dim, out_dim, scale)

    def forward(self, ref x: List[Float32]) -> List[Float32]:
        var y = List[Float32](capacity=self.out_dim)
        for i in range(self.out_dim):
            var acc: Float32 = 0.0
            # Addition / subtraction without floating point multiplications!
            for j in range(self.in_dim):
                var w = self.weights.get_weight(i, j)
                if w == 1:
                    acc += x[j]
                elif w == -1:
                    acc -= x[j]
            y.append(acc * self.weights.scale)
        return y^

struct BitNetMLP:
    var hidden_dim: Int
    var intermediate_dim: Int
    var gate_proj: BitNetLinear
    var up_proj: BitNetLinear
    var down_proj: BitNetLinear

    def __init__(out self, hidden_dim: Int, intermediate_dim: Int):
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.gate_proj = BitNetLinear(hidden_dim, intermediate_dim)
        self.up_proj = BitNetLinear(hidden_dim, intermediate_dim)
        self.down_proj = BitNetLinear(intermediate_dim, hidden_dim)

    def silu(self, x: Float32) -> Float32:
        return x / (1.0 + exp(-x))

    def forward(self, ref x: List[Float32]) -> List[Float32]:
        # SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
        var gate = self.gate_proj.forward(x)
        var up = self.up_proj.forward(x)
        
        var act = List[Float32](capacity=self.intermediate_dim)
        for i in range(self.intermediate_dim):
            var s = self.silu(gate[i])
            act.append(s * up[i])
            
        return self.down_proj.forward(act)

def main() raises:
    print("=== Nexus-Titans: BitNet b1.58 Ternary SIMD Kernel ===")
    var in_dim = 256
    var out_dim = 256
    var linear = BitNetLinear(in_dim, out_dim, scale=0.05)

    # Populate weights with ternary {-1, 0, 1} pattern
    for i in range(out_dim):
        for j in range(in_dim):
            var r = (i * 37 + j * 17) % 3
            if r == 0:
                linear.weights.set_weight(i, j, 0)
            elif r == 1:
                linear.weights.set_weight(i, j, 1)
            else:
                linear.weights.set_weight(i, j, -1)

    var x = List[Float32](capacity=in_dim)
    for j in range(in_dim):
        x.append(Float32(j % 7) * 0.1)

    var py_time = Python.import_module("time")
    var t0 = py_time.perf_counter()
    var iters = 100
    for _ in range(iters):
        var _out = linear.forward(x)
    var t1 = py_time.perf_counter()

    var elapsed_ms = ((t1 - t0) / Float64(iters)) * 1000.0
    var ops_per_iter = in_dim * out_dim
    print("Linear Layer Size:", in_dim, "x", out_dim)
    print("Packed Memory Size:", len(linear.weights.packed_data), "bytes (vs", in_dim * out_dim * 4, "bytes FP32)")
    print("Memory Compression Ratio: ~16x reduction over FP32")
    print("Average Forward Time:", elapsed_ms, "ms per forward pass")
    print("Operations:", ops_per_iter, "add/sub ops per pass")
    print("Ternary BitNet kernel operational!")
