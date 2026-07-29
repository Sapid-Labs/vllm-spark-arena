"""Page size 64 vs 256 on the DSv4 decode kernel, with the REAL page layout.

A page is structure-of-arrays, not array-of-structures:
    [pbs * IO_STRIDE bytes of KV][pbs * SCALE_BYTES_PER_TOKEN bytes of scales]
so the scale section starts at pbs*IO_STRIDE and the split point MOVES with the
page size. Building both layouts from one logical per-token source is the only
way the two page sizes see the same data.
"""
import torch
from flashinfer.mla import _sparse_mla_sm120 as M

IO_STRIDE, SCALE_BPT = 576, 8
bpt = M._bytes_per_token_for_model_type(M._MODEL_TYPE_DSV4)
assert bpt == IO_STRIDE + SCALE_BPT, (bpt, IO_STRIDE, SCALE_BPT)
dev = "cuda"

def build_pages(kv_data, scales, pbs):
    """kv_data [N,576], scales [N,8] -> [N/pbs, pbs*584] in page-major SoA."""
    N = kv_data.shape[0]
    pages = N // pbs
    buf = torch.empty(pages, pbs * bpt, dtype=torch.uint8, device=dev)
    buf[:, : pbs * IO_STRIDE] = kv_data.reshape(pages, pbs * IO_STRIDE)
    buf[:, pbs * IO_STRIDE :] = scales.reshape(pages, pbs * SCALE_BPT)
    return buf.contiguous()

def partials(pbs, kv_data, scales, q, idx, T, H, TOPK):
    DV = 512
    kv = build_pages(kv_data, scales, pbs)
    assert M._packed_kv_page_block_size(kv, model_type=M._MODEL_TYPE_DSV4, name="kv") == pbs
    splits = (TOPK + 63) // 64
    mo = torch.zeros(T, H, splits, DV, dtype=torch.bfloat16, device=dev)
    ml = torch.zeros(T, H, splits, dtype=torch.float32, device=dev)
    out = torch.zeros(T, H, DV, dtype=torch.bfloat16, device=dev)
    lse = torch.zeros(T, H, dtype=torch.float32, device=dev)
    M.sparse_mla_sm120_decode_dsv4(q, kv, idx, mo, ml, out, lse,
                                   sm_scale=1.0/(512**0.5), chunks_per_block=1)
    torch.cuda.synchronize()
    return mo, ml

cases = [("H=128 topk=1024", 4096, 1, 128, 1024),
         ("H=128 topk=512 ", 8192, 1, 128,  512),
         ("H=64  topk=128 ", 2048, 1,  64,  128),
         ("H=16  topk=1024", 16384, 1, 16, 1024)]
allok = True
for name, N, T, H, K in cases:
    for seed in (0, 1):
        torch.manual_seed(seed)
        # Bias the FP8 bytes away from NaN/Inf exponents so the maths is real.
        kv_data = torch.randint(0, 120, (N, IO_STRIDE), dtype=torch.uint8, device=dev)
        scales  = torch.randint(0, 60,  (N, SCALE_BPT), dtype=torch.uint8, device=dev)
        q = (torch.randn(T, H, 512, device=dev) * 0.1).to(torch.bfloat16)
        idx = torch.stack([torch.randperm(N, device=dev)[:K] for _ in range(T)]).to(torch.int32).contiguous()
        a_mo, a_ml = partials(64, kv_data, scales, q, idx, T, H, K)
        b_mo, b_ml = partials(256, kv_data, scales, q, idx, T, H, K)
        bits = int((a_mo.view(torch.int16) != b_mo.view(torch.int16)).sum())
        ok = bits == 0 and torch.equal(a_ml.view(torch.int32), b_ml.view(torch.int32))
        finite = bool(torch.isfinite(a_mo.float()).all())
        nz = bool((a_mo.view(torch.int16) != 0).any())
        allok &= (ok and nz)
        print(f"{name} seed={seed}  identical={ok}  ran={nz}  finite={finite}  differing={bits}")
print()
print("VERDICT:", "PASS — 256 is bit-identical to 64" if allok else "FAIL")
