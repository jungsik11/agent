import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass

@dataclass
class ModelArgs:
    d_model: int
    num_heads: int
    num_layers: int
    vocab_size: int
    ffn_dim_multiplier: int = 4
    rope_theta: float = 10000.0

class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def _norm(self, x):
        return x * mx.rsqrt(x.square().mean(-1, keepdims=True) + self.eps)

    def __call__(self, x):
        output = self._norm(x.astype(mx.float32)).astype(x.dtype)
        return self.weight * output

class MultiQueryAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.d_model = args.d_model
        self.num_heads = args.num_heads
        self.num_kv_heads = 1  # For MQA
        self.head_dim = self.d_model // self.num_heads
        self.scale = self.head_dim**-0.5

        self.wq = nn.Linear(self.d_model, self.d_model, bias=False)
        self.wk = nn.Linear(self.d_model, self.head_dim, bias=False)
        self.wv = nn.Linear(self.d_model, self.head_dim, bias=False)
        self.wo = nn.Linear(self.d_model, self.d_model, bias=False)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=args.rope_theta)

    def __call__(self, x, mask=None, cache=None):
        B, L, D = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        queries = queries.reshape(B, L, self.num_heads, self.head_dim)
        keys = keys.reshape(B, L, self.num_kv_heads, self.head_dim)
        values = values.reshape(B, L, self.num_kv_heads, self.head_dim)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.wo(output)

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, ffn_dim, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = MultiQueryAttention(args)
        ffn_dim = args.d_model * args.ffn_dim_multiplier
        self.ffn = SwiGLUFFN(d_model=args.d_model, ffn_dim=ffn_dim)
        self.norm1 = RMSNorm(args.d_model)
        self.norm2 = RMSNorm(args.d_model)

    def __call__(self, x, mask=None, cache=None):
        h = x + self.attention(self.norm1(x), mask, cache)
        out = h + self.ffn(self.norm2(h))
        return out

class GeminiNano(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = [TransformerBlock(args) for _ in range(args.num_layers)]
        self.norm = RMSNorm(args.d_model)
        self.output = nn.Linear(args.d_model, args.vocab_size, bias=False)

    def __call__(self, x):
        x = self.tok_embeddings(x)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(x.dtype)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        x = self.norm(x)
        return self.output(x)

    def generate(self, x, temp=1.0):
        y = x
        while True:
            logits = self(y)[:, -1, :]
            if temp == 0:
                y_next = mx.argmax(logits, axis=-1)
            else:
                y_next = mx.random.categorical(logits * (1 / temp))
            
            yield y_next
            
            y = mx.concatenate([y, y_next[:, None]], axis=1)