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



N,T,H,K = 4096,1,128,1024
torch.manual_seed(0)
kv = torch.randint(0,120,(N,IO_STRIDE),dtype=torch.uint8,device=dev)
sc = torch.randint(0,60,(N,SCALE_BPT),dtype=torch.uint8,device=dev)
q  = (torch.randn(T,H,512,device=dev)*0.1).to(torch.bfloat16)
idx= torch.stack([torch.randperm(N,device=dev)[:K] for _ in range(T)]).to(torch.int32).contiguous()
f = lambda x,y: int((x.view(torch.int16)!=y.view(torch.int16)).sum())

base64,_  = partials(64,  kv, sc, q, idx, T,H,K)
base256,_ = partials(256, kv, sc, q, idx, T,H,K)
print("64 vs 256, same data          :", f(base64,base256), " (want 0)")

tok = int(idx[0,0].item())
k1 = kv.clone(); k1[tok,:] = (k1[tok,:].int()+53).remainder(120).to(torch.uint8)
p64,_  = partials(64,  k1, sc, q, idx, T,H,K)
p256,_ = partials(256, k1, sc, q, idx, T,H,K)
print("one WHOLE token changed @64   :", f(base64,p64),   " (want > 0 - control)")
print("one WHOLE token changed @256  :", f(base256,p256), " (want > 0 - control)")
print("perturbed 64 vs perturbed 256 :", f(p64,p256),     " (want 0 - equivalence holds under change)")
