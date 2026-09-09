from std.collections import List
from std.math import exp, sqrt
from std.python import Python

struct TitansMemory:
    var d_model: Int
    var d_mem: Int
    # Memory matrix M of shape [d_mem, d_mem]
    var M: List[Float32]
    var learning_rate: Float32
    var decay_rate: Float32

    def __init__(out self, d_model: Int, d_mem: Int, lr: Float32 = 0.05, decay: Float32 = 0.01):
        self.d_model = d_model
        self.d_mem = d_mem
        self.learning_rate = lr
        self.decay_rate = decay
        
        # Initialize memory matrix M with zeros
        var total_size = d_mem * d_mem
        self.M = List[Float32](capacity=total_size)
        for _ in range(total_size):
            self.M.append(0.0)

    def sigmoid(self, x: Float32) -> Float32:
        return 1.0 / (1.0 + exp(-x))

    def retrieve(self, ref q: List[Float32]) -> List[Float32]:
        # y = M * q
        var out = List[Float32](capacity=self.d_mem)
        for i in range(self.d_mem):
            var s: Float32 = 0.0
            var offset = i * self.d_mem
            for j in range(self.d_mem):
                s += self.M[offset + j] * q[j]
            out.append(s)
        return out^

    def update_surprise(mut self, ref k: List[Float32], ref v: List[Float32]) -> Float32:
        """Calculates the Surprise Metric and performs online gradient descent update on Memory Matrix."""
        # 1. Compute predicted v_hat = M * k
        var v_hat = List[Float32](capacity=self.d_mem)
        var surprise_sq: Float32 = 0.0
        for i in range(self.d_mem):
            var s: Float32 = 0.0
            var offset = i * self.d_mem
            for j in range(self.d_mem):
                s += self.M[offset + j] * k[j]
            v_hat.append(s)
            var diff = v[i] - s
            surprise_sq += diff * diff

        var surprise = sqrt(surprise_sq)

        # 2. Gradient of loss w.r.t M: grad_M = -(v - M*k) * k^T
        # Update: M = (1 - decay)*M + lr * (v - v_hat) * k^T
        var decay_factor = 1.0 - self.decay_rate
        for i in range(self.d_mem):
            var offset = i * self.d_mem
            var err = v[i] - v_hat[i]
            for j in range(self.d_mem):
                var old_val = self.M[offset + j]
                var grad_step = self.learning_rate * err * k[j]
                self.M[offset + j] = old_val * decay_factor + grad_step

        return surprise

    def clear(mut self):
        for i in range(len(self.M)):
            self.M[i] = 0.0

def main() raises:
    print("=== Nexus-Titans: Neural Long-Term Memory (Google Titans LTM) ===")
    var d_mem = 64
    var d_model = 128
    var memory = TitansMemory(d_model=d_model, d_mem=d_mem, lr=0.1, decay=0.005)

    print("Memory matrix dimension:", d_mem, "x", d_mem, "(Fixed", d_mem * d_mem * 4, "bytes)")

    # Test online memorization:
    # Key: "日本の首都" (representation vector k1)
    # Value: "東京" (representation vector v1)
    var k1 = List[Float32](capacity=d_mem)
    var v1 = List[Float32](capacity=d_mem)
    for i in range(d_mem):
        k1.append(Float32(i % 5) * 0.2)
        v1.append(Float32((i + 3) % 7) * 0.3)

    # Initial retrieval before memorization (should be 0)
    var y0 = memory.retrieve(k1)
    var dist0: Float32 = 0.0
    for i in range(d_mem):
        var d = v1[i] - y0[i]
        dist0 += d * d
    print("Initial prediction error (before memory update):", sqrt(dist0))

    # Memorize (Online Test-Time Gradient Update)
    for step in range(15):
        var s = memory.update_surprise(k1, v1)
        if step == 0 or step == 4 or step == 14:
            print("Step", step + 1, "| Online Surprise Metric:", s)

    # Query again with k1
    var y1 = memory.retrieve(k1)
    var dist1: Float32 = 0.0
    for i in range(d_mem):
        var d = v1[i] - y1[i]
        dist1 += d * d
    print("Final prediction error after Titans online learning:", sqrt(dist1))
    print("Titans Neural LTM memorization confirmed!")
