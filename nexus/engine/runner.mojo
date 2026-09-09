from std.collections import List
from std.math import exp, sqrt
from std.python import Python, PythonObject

struct PackedWeight:
    var in_dim: Int
    var out_dim: Int
    var packed_data: List[UInt8]
    var scale: Float32

    def __init__(out self, in_dim: Int, out_dim: Int, scale: Float32 = 1.0):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.scale = scale
        var num_bytes = (in_dim * out_dim + 3) // 4
        self.packed_data = List[UInt8](capacity=num_bytes)
        for _ in range(num_bytes):
            self.packed_data.append(0)

    def set_weight(mut self, row: Int, col: Int, val: Int8):
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
    var weights: PackedWeight

    def __init__(out self, in_dim: Int, out_dim: Int, scale: Float32 = 1.0):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weights = PackedWeight(in_dim, out_dim, scale)

    def forward(self, ref x: List[Float32]) -> List[Float32]:
        var y = List[Float32](capacity=self.out_dim)
        for i in range(self.out_dim):
            var acc: Float32 = 0.0
            for j in range(self.in_dim):
                var w = self.weights.get_weight(i, j)
                if w == 1:
                    acc += x[j]
                elif w == -1:
                    acc -= x[j]
            y.append(acc * self.weights.scale)
        return y^

    def populate(mut self, in_d: Int, out_d: Int, sc: Float32, np_arr: PythonObject) raises:
        self.in_dim = in_d
        self.out_dim = out_d
        self.weights.in_dim = in_d
        self.weights.out_dim = out_d
        self.weights.scale = sc
        var byte_count = len(self.weights.packed_data)
        for i in range(byte_count):
            self.weights.packed_data[i] = UInt8(atol(String(np_arr[i])))

struct TitansLTM:
    var d_model: Int
    var d_mem: Int
    var M: List[Float32]
    var lr: Float32
    var decay: Float32

    def __init__(out self, d_model: Int, d_mem: Int, lr: Float32 = 0.05, decay: Float32 = 0.01):
        self.d_model = d_model
        self.d_mem = d_mem
        self.lr = lr
        self.decay = decay
        var sz = d_mem * d_mem
        self.M = List[Float32](capacity=sz)
        for _ in range(sz):
            self.M.append(0.0)

    def retrieve(self, ref q: List[Float32]) -> List[Float32]:
        var out = List[Float32](capacity=self.d_mem)
        for i in range(self.d_mem):
            var s: Float32 = 0.0
            var offset = i * self.d_mem
            for j in range(self.d_mem):
                s += self.M[offset + j] * q[j]
            out.append(s)
        return out^

    def update_surprise(mut self, ref k: List[Float32], ref v: List[Float32]) -> Float32:
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

        var decay_f = 1.0 - self.decay
        for i in range(self.d_mem):
            var offset = i * self.d_mem
            var err = v[i] - v_hat[i]
            for j in range(self.d_mem):
                self.M[offset + j] = self.M[offset + j] * decay_f + self.lr * err * k[j]

        return sqrt(surprise_sq)

struct NexusEngine:
    var vocab_size: Int
    var d_model: Int
    var d_mem: Int
    var d_ffn: Int
    var ltm: TitansLTM
    var q_proj: BitNetLinear
    var k_proj: BitNetLinear
    var v_proj: BitNetLinear
    var out_proj: BitNetLinear
    var ffn_gate: BitNetLinear
    var ffn_up: BitNetLinear
    var ffn_down: BitNetLinear
    var lm_head: BitNetLinear
    var embeddings: List[Float32]

    def __init__(out self, vocab_size: Int, d_model: Int, d_mem: Int, d_ffn: Int):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_mem = d_mem
        self.d_ffn = d_ffn
        self.ltm = TitansLTM(d_model, d_mem)
        self.q_proj = BitNetLinear(d_model, d_mem)
        self.k_proj = BitNetLinear(d_model, d_mem)
        self.v_proj = BitNetLinear(d_model, d_mem)
        self.out_proj = BitNetLinear(d_mem, d_model)
        self.ffn_gate = BitNetLinear(d_model, d_ffn)
        self.ffn_up = BitNetLinear(d_model, d_ffn)
        self.ffn_down = BitNetLinear(d_ffn, d_model)
        self.lm_head = BitNetLinear(d_model, vocab_size)
        var total_emb = vocab_size * d_model
        self.embeddings = List[Float32](capacity=total_emb)
        for _ in range(total_emb):
            self.embeddings.append(0.0)

    def load_binary_checkpoint(mut self, binary_path: String) raises:
        var py_struct = Python.import_module("struct")
        var py_builtins = Python.import_module("builtins")
        var f = py_builtins.open(binary_path, "rb")

        # 1. Header
        var magic = f.read(8)
        var header = py_struct.unpack("<IIII", f.read(16))
        var v_size = atol(String(header[0]))
        var d_m = atol(String(header[1]))
        var d_me = atol(String(header[2]))
        var d_f = atol(String(header[3]))

        # 2. Embeddings
        var emb_len = atol(String(py_struct.unpack("<I", f.read(4))[0]))
        var emb_bytes = f.read(emb_len)
        var py_np = Python.import_module("numpy")
        var emb_np = py_np.frombuffer(emb_bytes, dtype=py_np.float32)
        var total_emb = v_size * d_m
        for i in range(total_emb):
            self.embeddings[i] = Float32(atof(String(emb_np[i])))

        # 3. Packed layers
        var num_layers = atol(String(py_struct.unpack("<I", f.read(4))[0]))
        for _ in range(num_layers):
            var name_len = atol(String(py_struct.unpack("<H", f.read(2))[0]))
            var name_bytes = f.read(name_len)
            var layer_meta = py_struct.unpack("<IIf", f.read(12))
            var in_dim = atol(String(layer_meta[0]))
            var out_dim = atol(String(layer_meta[1]))
            var scale = Float32(atof(String(layer_meta[2])))
            var packed_len = atol(String(py_struct.unpack("<I", f.read(4))[0]))
            var packed_bytes = f.read(packed_len)
            var packed_np = py_np.frombuffer(packed_bytes, dtype=py_np.uint8)

            var name_str = String(name_bytes.decode("utf-8"))
            if name_str == "titans.q_proj":
                self.q_proj.populate(in_dim, out_dim, scale, packed_np)
            elif name_str == "titans.k_proj":
                self.k_proj.populate(in_dim, out_dim, scale, packed_np)
            elif name_str == "titans.v_proj":
                self.v_proj.populate(in_dim, out_dim, scale, packed_np)
            elif name_str == "titans.out_proj":
                self.out_proj.populate(in_dim, out_dim, scale, packed_np)
            elif name_str == "ffn.gate_proj":
                self.ffn_gate.populate(in_dim, out_dim, scale, packed_np)
            elif name_str == "ffn.up_proj":
                self.ffn_up.populate(in_dim, out_dim, scale, packed_np)
            elif name_str == "ffn.down_proj":
                self.ffn_down.populate(in_dim, out_dim, scale, packed_np)
            elif name_str == "lm_head":
                self.lm_head.populate(in_dim, out_dim, scale, packed_np)

        f.close()
        print("Successfully loaded trained Nexus-Titans binary checkpoint into Mojo memory!")

    def forward_token(mut self, token_id: Int, update_memory: Bool = True) -> List[Float32]:
        var offset = token_id * self.d_model
        var x = List[Float32](capacity=self.d_model)
        for i in range(self.d_model):
            x.append(self.embeddings[offset + i])

        # Titans LTM
        var q = self.q_proj.forward(x)
        var k = self.k_proj.forward(x)
        var v = self.v_proj.forward(x)
        var mem_retrieved = self.ltm.retrieve(q)
        if update_memory:
            var _ = self.ltm.update_surprise(k, v)

        var mem_out = self.out_proj.forward(mem_retrieved)
        var h = List[Float32](capacity=self.d_model)
        for i in range(self.d_model):
            h.append(x[i] + mem_out[i])

        # SwiGLU FFN
        var gate = self.ffn_gate.forward(h)
        var up = self.ffn_up.forward(h)
        var act = List[Float32](capacity=len(gate))
        for i in range(len(gate)):
            var silu = gate[i] / (1.0 + exp(-gate[i]))
            act.append(silu * up[i])
        var ffn_out = self.ffn_down.forward(act)

        var y = List[Float32](capacity=self.d_model)
        for i in range(self.d_model):
            y.append(h[i] + ffn_out[i])

        return self.lm_head.forward(y)

    def sample_token(self, ref logits: List[Float32], temperature: Float32 = 0.7) -> Int:
        var best_idx = 0
        var max_v = logits[0]
        for i in range(1, len(logits)):
            if logits[i] > max_v:
                max_v = logits[i]
                best_idx = i
        return best_idx

def main() raises:
    print("=== Nexus-Titans: High-Speed Native Mojo Inference Engine ===")
    var engine = NexusEngine(vocab_size=1000, d_model=128, d_mem=64, d_ffn=256)
    var model_path = String("/home/syuum/NexusAI/nexus/engine/nexus_titans.nexus")
    engine.load_binary_checkpoint(model_path)

    # Initialize Tokenizer via Python interop
    var py_sys = Python.import_module("sys")
    py_sys.path.insert(0, "/home/syuum/NexusAI")
    var py_tok_mod = Python.import_module("nexus.tokenizer.japanese_bpe")
    var tok = py_tok_mod.JapaneseBPETokenizer("/home/syuum/NexusAI/nexus/tokenizer/nexus_ja_bpe")
    tok.load()

    var prompt = String("人工知能とは、")
    var prompt_tokens_py = tok.encode(prompt)
    var prompt_len = atol(String(len(prompt_tokens_py)))
    print("\n[Inference] Prompt:", prompt)
    print("[Inference] Prompt Token Count:", prompt_len)

    # 1. Feed prompt tokens through model to prime Titans Long-Term Memory
    var prompt_tokens = List[Int](capacity=prompt_len)
    for i in range(prompt_len):
        prompt_tokens.append(atol(String(prompt_tokens_py[i])))

    var py_time = Python.import_module("time")
    var t0 = py_time.perf_counter()

    var last_token = prompt_tokens[0]
    for i in range(prompt_len):
        var _ = engine.forward_token(prompt_tokens[i], update_memory=True)
        last_token = prompt_tokens[i]

    # 2. Autoregressive token generation
    var gen_tokens_py = Python.import_module("builtins").list()
    var gen_count = 15
    for _ in range(gen_count):
        var logits = engine.forward_token(last_token, update_memory=True)
        var next_token = engine.sample_token(logits)
        gen_tokens_py.append(next_token)
        last_token = next_token

    var t1 = py_time.perf_counter()
    var total_time_sec = atof(String(t1 - t0))
    var total_tokens_processed = prompt_len + gen_count
    var tps = Float64(total_tokens_processed) / total_time_sec

    var generated_text = String(tok.decode(gen_tokens_py))
    print("\n[Inference] Generated Output:", generated_text)
    print("\n[Performance Benchmark]")
    print(" - Total tokens processed:", total_tokens_processed)
    print(" - Total time:", total_time_sec * 1000.0, "ms")
    print(" - Inference Throughput:", tps, "tokens/second")
    print(" - Memory used by Titans LTM: Exactly 16 KB (Fixed O(1) space)")
    print("Mojo Inference Engine execution complete!")
