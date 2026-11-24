#!/usr/bin/env python3
"""Numpy port of the Opion act policy — the student's "playing brain".

Byte-mirrors opion/train/model.py (OpionPolicy) in pure numpy fp32:
  emb(85,128) + pos(41,128) + ctx_proj(LN17→Linear 17→128)
  → 4 × pre-LN transformer blocks (d=128, 4 heads, FFN 512, GELU)
  → ln_f → mean over the 41 token slots → Linear(128→6).

This is the SAME forward pass that ships (Step 4), so it is written for the
submission environment: stdlib + numpy only, single-threaded BLAS.

Weight sources:
  - canonical RL loop: policy_np.npz (exported from torch on Colab)
  - direct torch checkpoints: load_pt() reads final_act.pt WITHOUT torch
    (parses the zip/pickle storage format itself) — used for local tests.

GELU note: torch nn.GELU uses the exact erf form. numpy has no erf, so
_np_erf is the Abramowitz-Stegun 7.1.26 rational approximation
(|err| ≤ 1.5e-7). test_np_policy.py checks erf accuracy separately; the
diff-vs-torch gate tolerance (1e-3 on logits) absorbs it.

The optional value head (Linear 128→1, §9.2) loads when 'value.weight' /
'value.bias' are present in the weights dict; expose_value() computes it.
"""
import math
import os

# single-threaded BLAS BEFORE numpy import (§5 non-negotiable)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

N_TOK, N_CTX, N_ACT, VOCAB, D, HEADS, FFN, LAYERS = 41, 17, 6, 85, 128, 4, 512, 4
DH = D // HEADS


# ----------------------------------------------------------------- erf (A&S)
def _np_erf(x):
    """Abramowitz-Stegun 7.1.26 erf, max abs error 1.5e-7. Odd function."""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, \
        -1.453152027, 1.061405429
    p = 0.3275911
    z = np.abs(x)
    t = 1.0 / (1.0 + p * z)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t \
        * np.exp(-z * z)
    return np.sign(x) * y


def _gelu(x):
    return 0.5 * x * (1.0 + _np_erf(x * (1.0 / math.sqrt(2.0))))


def _layer_norm(x, w, b, eps=1e-5):
    m = x.mean(axis=-1, keepdims=True)
    v = x.var(axis=-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def _softmax_last(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _map_tokens(t):
    """Mirrors model.py exactly: t<53 unchanged, else 52+clip(t,1,32).
    NOTE: the raw data domain (cards 0..52, hist tokens 1..32) is entirely
    < 53, so the effective map is identity — hist tokens share embedding
    rows 1..32 with card ids (disambiguated by slot position embeddings).
    Kept byte-identical to model.py on purpose; do not 'fix' here alone."""
    t = np.asarray(t, dtype=np.int64)
    return np.where(t < 53, t, 52 + np.clip(t, 1, 32))


class NumpyPolicy:
    """Policy forward in numpy. Weights: flat dict name->fp32 ndarray using
    the torch state_dict key names. Configurable head/tokens so the same
    class serves the act policy (41/17/6, card-token mapping) and the disc
    policy (8/16/3, raw card ids, no mapping)."""

    def __init__(self, sd, n_tok=N_TOK, n_ctx=N_CTX, n_act=N_ACT,
                 map_tokens=True):
        self.sd = {k: np.ascontiguousarray(v, dtype=np.float32)
                   for k, v in sd.items()}
        self.n_tok, self.n_ctx, self.n_act = n_tok, n_ctx, n_act
        self.map_tokens = map_tokens
        self.has_value = "value.weight" in self.sd

    def forward(self, tokens, ctx):
        """tokens [B,n_tok] int, ctx [B,n_ctx] float -> logits [B,n_act]."""
        sd = self.sd
        tok = _map_tokens(tokens) if self.map_tokens \
            else np.asarray(tokens, dtype=np.int64)
        x = sd["emb.weight"][tok] + sd["pos.weight"][None, :self.n_tok, :]
        cx = ctx.astype(np.float32)
        cx = _layer_norm(cx, sd["ctx_proj.0.weight"], sd["ctx_proj.0.bias"])
        cx = cx @ sd["ctx_proj.1.weight"].T + sd["ctx_proj.1.bias"]
        x = x + cx[:, None, :]
        for i in range(LAYERS):
            p = f"blocks.{i}."
            h = _layer_norm(x, sd[p + "ln1.weight"], sd[p + "ln1.bias"])
            qkv = h @ sd[p + "qkv.weight"].T + sd[p + "qkv.bias"]
            B, T, _ = qkv.shape
            q, k, v = np.split(qkv, 3, axis=-1)
            # [B,T,D] -> [B,H,T,dh]
            q = q.reshape(B, T, HEADS, DH).transpose(0, 2, 1, 3)
            k = k.reshape(B, T, HEADS, DH).transpose(0, 2, 1, 3)
            v = v.reshape(B, T, HEADS, DH).transpose(0, 2, 1, 3)
            att = _softmax_last(q @ k.transpose(0, 1, 3, 2) / math.sqrt(DH))
            h = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
            x = x + h @ sd[p + "proj.weight"].T + sd[p + "proj.bias"]
            h = _layer_norm(x, sd[p + "ln2.weight"], sd[p + "ln2.bias"])
            h = _gelu(h @ sd[p + "fc1.weight"].T + sd[p + "fc1.bias"])
            x = x + h @ sd[p + "fc2.weight"].T + sd[p + "fc2.bias"]
        x = _layer_norm(x, sd["ln_f.weight"], sd["ln_f.bias"])
        pooled = x.mean(axis=1)
        return pooled @ sd["head.weight"].T + sd["head.bias"]

    def expose_value(self, pooled):
        """Optional value head on pooled trunk features (§9.2)."""
        assert self.has_value, "no value head in weights"
        return pooled @ self.sd["value.weight"].T + self.sd["value.bias"]

    def act_probs(self, tokens, ctx, legal):
        """Masked action distribution for play: [B,6] probs, 0 off-legal."""
        logits = self.forward(tokens, ctx).astype(np.float64)
        logits = np.where(np.asarray(legal) != 0, logits, -np.inf)
        return _softmax_last(logits)


# -------------------------------------------------- no-torch .pt reader ----
_DT = {"FloatStorage": np.float32, "DoubleStorage": np.float64,
       "HalfStorage": np.float16, "LongStorage": np.int64,
       "IntStorage": np.int32, "ShortStorage": np.int16,
       "CharStorage": np.int8, "ByteStorage": np.uint8,
       "BoolStorage": np.bool_}


def load_pt(path):
    """torch.save checkpoint -> dict, WITHOUT torch. Supports the plain
    zip format (data.pkl + raw storage files). State dicts of tensors only
    (no Parameters/views) — final_act.pt qualifies."""
    import pickle
    import zipfile

    zf = zipfile.ZipFile(path)
    pkl_name = [n for n in zf.namelist() if n.endswith("/data.pkl")][0]
    data_dir = pkl_name[:-len("data.pkl")]
    raw = {}
    cache = {}

    def persistent_load(pid):
        assert pid[0] == "storage", f"unsupported persistent id: {pid[0]}"
        _tag, storage_type, key, _loc, numel = pid[:5]
        if key not in cache:
            dtype = _DT[storage_type.__name__]
            buf = zf.read(data_dir + "data/" + key)
            assert len(buf) == numel * np.dtype(dtype).itemsize, \
                f"storage {key}: {len(buf)}B != {numel}x{dtype}"
            cache[key] = (np.frombuffer(buf, dtype=dtype), numel)
        return cache[key]

    class _U(pickle.Unpickler):
        def persistent_load(self, pid):
            return persistent_load(pid)

        def find_class(self, module, name):
            if name == "_rebuild_tensor_v2":
                return _rebuild
            if name == "_rebuild_parameter":
                return _rebuild
            if name == "OrderedDict":
                import collections
                return collections.OrderedDict
            if module == "torch" and name in _DT:
                return type(name, (), {})  # marker; __name__ drives the dtype map
            raise AssertionError(f"unsupported class {module}.{name}")

    def _rebuild(storage, offset, size, stride, *_a):
        flat, numel = storage
        need = 1
        for s in size:
            need *= s
        assert offset + need <= numel, "tensor exceeds storage"
        it = np.dtype(flat.dtype).itemsize
        arr = np.lib.stride_tricks.as_strided(
            flat[offset:], shape=tuple(size),
            strides=tuple(int(s) * it for s in stride))
        return np.ascontiguousarray(arr)

    with zf.open(pkl_name) as f:
        return _U(f).load()


def save_npz(sd, path):
    np.savez_compressed(
        path, **{k: np.ascontiguousarray(v, dtype=np.float32)
                 for k, v in sd.items()})


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
