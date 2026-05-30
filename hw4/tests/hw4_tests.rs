//! Tests for hw4 — Candle version.
//!
//! All module structs derive `Clone` so KV-cache consistency tests can
//! snapshot a model with its initial weights before any cache mutation.

use candle_core::{DType, Device, Result, Tensor};
use hw4::*;
use test_support::{as_f32_vec, assert_f32_slice_close, json, python_json};

// ---------- small helpers ----------

/// Flatten a tensor and pull it back to the host as a `Vec<f32>`.
fn to_vec_f32(t: &Tensor) -> Result<Vec<f32>> {
    t.flatten_all()?.to_vec1::<f32>()
}

// ---------- PyTorch ground-truth helpers (unchanged) ----------

fn torch_linear(
    x: Vec<f32>,
    x_shape: Vec<usize>,
    weight: Vec<f32>,
    weight_shape: Vec<usize>,
) -> Vec<f32> {
    as_f32_vec(python_json(
        r#"
import json
import sys
import torch
import torch.nn.functional as F

d = json.load(sys.stdin)
x = torch.tensor(d["x"], dtype=torch.float32).reshape(d["x_shape"])
w = torch.tensor(d["weight"], dtype=torch.float32).reshape(d["weight_shape"])
json.dump(F.linear(x, w).flatten().tolist(), sys.stdout)
"#,
        json!({ "x": x, "x_shape": x_shape, "weight": weight, "weight_shape": weight_shape }),
    ))
}

fn torch_silu(x: Vec<f32>, shape: Vec<usize>) -> Vec<f32> {
    as_f32_vec(python_json(
        r#"
import json
import sys
import torch
import torch.nn.functional as F

d = json.load(sys.stdin)
x = torch.tensor(d["x"], dtype=torch.float32).reshape(d["shape"])
json.dump(F.silu(x).flatten().tolist(), sys.stdout)
"#,
        json!({ "x": x, "shape": shape }),
    ))
}

fn torch_rms_norm(x: Vec<f32>, shape: Vec<usize>, weight: Option<Vec<f32>>, eps: f64) -> Vec<f32> {
    as_f32_vec(python_json(
        r#"
import json
import sys
import torch
import torch.nn.functional as F

d = json.load(sys.stdin)
x = torch.tensor(d["x"], dtype=torch.float32).reshape(d["shape"])
weight = None if d["weight"] is None else torch.tensor(d["weight"], dtype=torch.float32)
out = F.rms_norm(x, (x.shape[-1],), weight=weight, eps=d["eps"])
json.dump(out.flatten().tolist(), sys.stdout)
"#,
        json!({ "x": x, "shape": shape, "weight": weight, "eps": eps }),
    ))
}

fn torch_attention(
    q: Vec<f32>,
    q_shape: Vec<usize>,
    k: Vec<f32>,
    k_shape: Vec<usize>,
    v: Vec<f32>,
    v_shape: Vec<usize>,
    mask_len: Option<usize>,
) -> Vec<f32> {
    as_f32_vec(python_json(
        r#"
import json
import sys
import torch
import torch.nn.functional as F

d = json.load(sys.stdin)
q = torch.tensor(d["q"], dtype=torch.float32).reshape(d["q_shape"])
k = torch.tensor(d["k"], dtype=torch.float32).reshape(d["k_shape"])
v = torch.tensor(d["v"], dtype=torch.float32).reshape(d["v_shape"])
mask = None
if d["mask_len"] is not None:
    length = d["mask_len"]
    mask = torch.triu(torch.full((length, length), float("-inf")), diagonal=1)
if q.ndim == 2:
    out = F.scaled_dot_product_attention(
        q.unsqueeze(0).unsqueeze(0),
        k.unsqueeze(0).unsqueeze(0),
        v.unsqueeze(0).unsqueeze(0),
        attn_mask=mask,
        dropout_p=0.0,
    )[0, 0]
else:
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
json.dump(out.flatten().tolist(), sys.stdout)
"#,
        json!({
            "q": q,
            "q_shape": q_shape,
            "k": k,
            "k_shape": k_shape,
            "v": v,
            "v_shape": v_shape,
            "mask_len": mask_len,
        }),
    ))
}

#[allow(clippy::too_many_arguments)]
fn torch_mha(
    x: Vec<f32>,
    x_shape: Vec<usize>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wp: Vec<f32>,
    n_heads: usize,
    mask_len: Option<usize>,
) -> Vec<f32> {
    as_f32_vec(python_json(
        r#"
import json
import sys
import torch
import torch.nn as nn

d = json.load(sys.stdin)
dim = d["x_shape"][-1]
wq = torch.tensor(d["wq"], dtype=torch.float32).reshape(dim, dim)
wk = torch.tensor(d["wk"], dtype=torch.float32).reshape(dim, dim)
wv = torch.tensor(d["wv"], dtype=torch.float32).reshape(dim, dim)
wp = torch.tensor(d["wp"], dtype=torch.float32).reshape(dim, dim)

ref_attn = nn.MultiheadAttention(dim, d["n_heads"], bias=False, batch_first=True)
with torch.no_grad():
    ref_attn.in_proj_weight.copy_(torch.cat([wq, wk, wv], dim=0))
    ref_attn.out_proj.weight.copy_(wp)

x = torch.tensor(d["x"], dtype=torch.float32).reshape(d["x_shape"])
mask = None
if d["mask_len"] is not None:
    length = d["mask_len"]
    mask = torch.triu(torch.full((length, length), float("-inf")), diagonal=1)

with torch.no_grad():
    out = ref_attn(x, x, x, attn_mask=mask, need_weights=False)[0]

json.dump(out.flatten().tolist(), sys.stdout)
"#,
        json!({
            "x": x,
            "x_shape": x_shape,
            "wq": wq,
            "wk": wk,
            "wv": wv,
            "wp": wp,
            "n_heads": n_heads,
            "mask_len": mask_len,
        }),
    ))
}

// ---------- Linear ----------

#[test]
fn test_linear_shape() -> Result<()> {
    let device = Device::Cpu;
    let layer = Linear::new(10, 20, &device)?;
    let x = Tensor::randn(0f32, 1f32, (50, 10), &device)?;
    let out = layer.forward(&x)?;
    assert_eq!(out.dims(), &[50, 20]);
    Ok(())
}

#[test]
fn test_linear_batch_dims() -> Result<()> {
    let device = Device::Cpu;
    let layer = Linear::new(10, 20, &device)?;
    let x = Tensor::randn(0f32, 1f32, (7, 9, 10), &device)?;
    let out = layer.forward(&x)?;
    assert_eq!(out.dims(), &[7, 9, 20]);
    Ok(())
}

#[test]
fn test_linear_correctness() -> Result<()> {
    let device = Device::Cpu;
    let layer = Linear::new(10, 20, &device)?;
    let x = Tensor::randn(0f32, 1f32, (50, 10), &device)?;
    let out = layer.forward(&x)?;

    // `Linear` stores `weight` pre-transposed as [in, out]; PyTorch's
    // `F.linear` expects [out, in], so we transpose before comparing.
    let weight_pt = layer.weight.t()?.contiguous()?;
    let expected = torch_linear(
        to_vec_f32(&x)?,
        x.dims().to_vec(),
        to_vec_f32(&weight_pt)?,
        weight_pt.dims().to_vec(),
    );
    let actual = to_vec_f32(&out)?;
    assert_f32_slice_close(&actual, &expected, 1e-4);
    Ok(())
}

// ---------- Embedding ----------

#[test]
fn test_embedding_shape() -> Result<()> {
    let device = Device::Cpu;
    let layer = Embedding::new(200, 20, &device)?;
    // Candle requires u32 (or i64) indices for embedding.
    let y = Tensor::from_vec(vec![3u32, 7, 42, 100], (1, 4), &device)?;
    let out = layer.forward(&y)?;
    assert_eq!(out.dims(), &[1, 4, 20]);
    Ok(())
}

#[test]
fn test_embedding_batch_dims() -> Result<()> {
    let device = Device::Cpu;
    let layer = Embedding::new(200, 20, &device)?;
    let y = Tensor::from_vec(vec![0u32, 1, 2, 3, 4, 5, 6, 7, 8], (3, 3), &device)?;
    let out = layer.forward(&y)?;
    assert_eq!(out.dims(), &[3, 3, 20]);
    Ok(())
}

#[test]
fn test_embedding_correctness() -> Result<()> {
    let device = Device::Cpu;
    let layer = Embedding::new(8, 3, &device)?;
    let indices = Tensor::from_vec(vec![0u32, 3, 5], (1, 3), &device)?;
    let out = layer.forward(&indices)?;

    // Each row of output should be the corresponding row of the weight matrix.
    let w = &layer.weight;
    let row0 = to_vec_f32(&w.narrow(0, 0, 1)?.reshape(3)?)?;
    let row3 = to_vec_f32(&w.narrow(0, 3, 1)?.reshape(3)?)?;
    let row5 = to_vec_f32(&w.narrow(0, 5, 1)?.reshape(3)?)?;

    let out_squeezed = out.reshape((3, 3))?;
    let out0 = to_vec_f32(&out_squeezed.narrow(0, 0, 1)?.reshape(3)?)?;
    let out1 = to_vec_f32(&out_squeezed.narrow(0, 1, 1)?.reshape(3)?)?;
    let out2 = to_vec_f32(&out_squeezed.narrow(0, 2, 1)?.reshape(3)?)?;

    assert_eq!(out0, row0);
    assert_eq!(out1, row3);
    assert_eq!(out2, row5);
    Ok(())
}

// ---------- SiLU ----------

#[test]
fn test_silu() -> Result<()> {
    let device = Device::Cpu;
    let x = Tensor::from_vec(
        vec![-2.0f32, -0.5, 0.0, 1.0, 2.0, 3.0, -1.0, 0.25],
        (2, 4),
        &device,
    )?;
    let out = silu(&x)?;

    let expected = torch_silu(to_vec_f32(&x)?, x.dims().to_vec());
    let actual = to_vec_f32(&out)?;
    assert_f32_slice_close(&actual, &expected, 1e-5);
    Ok(())
}

#[test]
fn test_silu_multidim() -> Result<()> {
    let device = Device::Cpu;
    let x = Tensor::randn(0f32, 1f32, (3, 4, 5, 6), &device)?;
    let out = silu(&x)?;
    assert_eq!(out.dims(), x.dims());

    let expected = torch_silu(to_vec_f32(&x)?, x.dims().to_vec());
    let actual = to_vec_f32(&out)?;
    assert_f32_slice_close(&actual, &expected, 1e-4);
    Ok(())
}

// ---------- RMSNorm ----------

#[test]
fn test_rmsnorm_shape_and_init() -> Result<()> {
    let device = Device::Cpu;
    let layer = RMSNorm::new(20, 1e-3, &device)?;
    let x = Tensor::randn(0f32, 1f32, (100, 20), &device)?;
    let out = layer.forward(&x)?;
    assert_eq!(out.dims(), &[100, 20]);
    Ok(())
}

#[test]
fn test_rmsnorm_batch_dims() -> Result<()> {
    let device = Device::Cpu;
    let layer = RMSNorm::new(20, 1e-3, &device)?;
    let x = Tensor::randn(0f32, 1f32, (10, 7, 20), &device)?;
    let out = layer.forward(&x)?;
    assert_eq!(out.dims(), &[10, 7, 20]);
    Ok(())
}

#[test]
fn test_rmsnorm_correctness() -> Result<()> {
    // RMSNorm(x) = w * x / sqrt(mean(x^2) + eps), with weight initialised to ones.
    let device = Device::Cpu;
    let layer = RMSNorm::new(4, 1e-5, &device)?;
    let x = Tensor::from_vec(
        vec![1.0f32, -1.0, 0.5, 0.5, 2.0, 0.0, -2.0, 1.0],
        (2, 4),
        &device,
    )?;
    let out = layer.forward(&x)?;

    let expected = torch_rms_norm(to_vec_f32(&x)?, x.dims().to_vec(), None, 1e-5);
    let actual = to_vec_f32(&out)?;
    assert_f32_slice_close(&actual, &expected, 1e-5);
    Ok(())
}

// ---------- self_attention ----------

#[test]
fn test_self_attention_2d() -> Result<()> {
    let device = Device::Cpu;
    let q = Tensor::randn(0f32, 1f32, (5, 8), &device)?;
    let k = Tensor::randn(0f32, 1f32, (5, 8), &device)?;
    let v = Tensor::randn(0f32, 1f32, (5, 6), &device)?;
    let mask = build_causal_mask(5, &device)?;

    let out = self_attention(&q, &k, &v, Some(&mask))?;
    assert_eq!(out.dims(), &[5, 6]);

    let expected = torch_attention(
        to_vec_f32(&q)?,
        q.dims().to_vec(),
        to_vec_f32(&k)?,
        k.dims().to_vec(),
        to_vec_f32(&v)?,
        v.dims().to_vec(),
        Some(q.dims()[0]),
    );
    let actual = to_vec_f32(&out)?;
    assert_f32_slice_close(&actual, &expected, 1e-5);

    // First row attends only to itself under a causal mask, so it should equal V[0].
    let out_row0 = to_vec_f32(&out.narrow(0, 0, 1)?.reshape(6)?)?;
    let v_row0 = to_vec_f32(&v.narrow(0, 0, 1)?.reshape(6)?)?;
    for (a, b) in out_row0.iter().zip(v_row0.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "First row should equal V[0] with causal mask"
        );
    }
    Ok(())
}

#[test]
fn test_self_attention_batched() -> Result<()> {
    let device = Device::Cpu;
    let q = Tensor::randn(0f32, 1f32, (2, 3, 5, 8), &device)?;
    let k = Tensor::randn(0f32, 1f32, (2, 3, 5, 8), &device)?;
    let v = Tensor::randn(0f32, 1f32, (2, 3, 5, 4), &device)?;
    let mask = build_causal_mask(5, &device)?;

    let out = self_attention(&q, &k, &v, Some(&mask))?;
    assert_eq!(out.dims(), &[2, 3, 5, 4]);

    let expected = torch_attention(
        to_vec_f32(&q)?,
        q.dims().to_vec(),
        to_vec_f32(&k)?,
        k.dims().to_vec(),
        to_vec_f32(&v)?,
        v.dims().to_vec(),
        Some(q.dims()[2]),
    );
    let actual = to_vec_f32(&out)?;
    assert_f32_slice_close(&actual, &expected, 1e-5);
    Ok(())
}

#[test]
fn test_self_attention_no_mask() -> Result<()> {
    let device = Device::Cpu;
    let q = Tensor::randn(0f32, 1f32, (5, 8), &device)?;
    let k = Tensor::randn(0f32, 1f32, (5, 8), &device)?;
    let v = Tensor::randn(0f32, 1f32, (5, 6), &device)?;

    let out = self_attention(&q, &k, &v, None)?;
    assert_eq!(out.dims(), &[5, 6]);

    let expected = torch_attention(
        to_vec_f32(&q)?,
        q.dims().to_vec(),
        to_vec_f32(&k)?,
        k.dims().to_vec(),
        to_vec_f32(&v)?,
        v.dims().to_vec(),
        None,
    );
    let actual = to_vec_f32(&out)?;
    assert_f32_slice_close(&actual, &expected, 1e-5);
    Ok(())
}

// ---------- MultiHeadAttention ----------

#[test]
fn test_mha_shape() -> Result<()> {
    let device = Device::Cpu;
    let mut attn = MultiHeadAttention::new(12, 3, &device)?;
    let x = Tensor::randn(0f32, 1f32, (2, 5, 12), &device)?;
    let out = attn.forward(&x, None, 0, false)?;
    assert_eq!(out.dims(), &[2, 5, 12]);
    Ok(())
}

#[test]
fn test_mha_correctness() -> Result<()> {
    let device = Device::Cpu;
    let mut attn = MultiHeadAttention::new(12, 3, &device)?;
    let x = Tensor::randn(0f32, 1f32, (2, 5, 12), &device)?;
    let out = attn.forward(&x, None, 0, false)?;

    // Linear stores weights as [in, out]; PyTorch expects [out, in].
    let wq_pt = attn.wq.weight.t()?.contiguous()?;
    let wk_pt = attn.wk.weight.t()?.contiguous()?;
    let wv_pt = attn.wv.weight.t()?.contiguous()?;
    let wp_pt = attn.wp.weight.t()?.contiguous()?;
    let expected = torch_mha(
        to_vec_f32(&x)?,
        x.dims().to_vec(),
        to_vec_f32(&wq_pt)?,
        to_vec_f32(&wk_pt)?,
        to_vec_f32(&wv_pt)?,
        to_vec_f32(&wp_pt)?,
        attn.n_heads,
        None,
    );
    let actual = to_vec_f32(&out)?;
    assert_f32_slice_close(&actual, &expected, 1e-4);
    Ok(())
}

#[test]
fn test_mha_with_mask() -> Result<()> {
    let device = Device::Cpu;
    let mut attn = MultiHeadAttention::new(12, 3, &device)?;
    let x = Tensor::randn(0f32, 1f32, (2, 5, 12), &device)?;
    let mask = build_causal_mask(5, &device)?;
    let out = attn.forward(&x, Some(&mask), 0, false)?;

    let wq_pt = attn.wq.weight.t()?.contiguous()?;
    let wk_pt = attn.wk.weight.t()?.contiguous()?;
    let wv_pt = attn.wv.weight.t()?.contiguous()?;
    let wp_pt = attn.wp.weight.t()?.contiguous()?;
    let expected = torch_mha(
        to_vec_f32(&x)?,
        x.dims().to_vec(),
        to_vec_f32(&wq_pt)?,
        to_vec_f32(&wk_pt)?,
        to_vec_f32(&wv_pt)?,
        to_vec_f32(&wp_pt)?,
        attn.n_heads,
        Some(5),
    );
    let actual = to_vec_f32(&out)?;
    assert_f32_slice_close(&actual, &expected, 1e-4);
    Ok(())
}

// ---------- MultiHeadAttentionKVCache ----------

#[test]
fn test_mha_kvcache_no_cache() -> Result<()> {
    let device = Device::Cpu;
    let mut attn = MultiHeadAttentionKVCache::new(12, 3, 8, &device)?;
    let x = Tensor::randn(0f32, 1f32, (1, 5, 12), &device)?;
    let mask = build_causal_mask(5, &device)?;
    let out = attn.forward(&x, Some(&mask), 0, false)?;
    assert_eq!(out.dims(), &[1, 5, 12]);
    Ok(())
}

#[test]
fn test_mha_kvcache_consistency() -> Result<()> {
    // A full forward (no cache) on a length-5 input should match a length-3
    // prefix call followed by a length-2 tail call, both using the cache.
    let device = Device::Cpu;
    let mut attn = MultiHeadAttentionKVCache::new(12, 3, 8, &device)?;
    let x = Tensor::randn(0f32, 1f32, (1, 5, 12), &device)?;
    let mask = build_causal_mask(5, &device)?;

    let full = attn.forward(&x, Some(&mask), 0, false)?;
    assert_eq!(full.dims(), &[1, 5, 12]);

    // Snapshot the layer (and its zeroed cache) before any cached call mutates it.
    let mut attn2 = attn.clone();

    let prefix_mask = build_causal_mask(3, &device)?;
    let x_prefix = x.narrow(1, 0, 3)?;
    let prefix = attn2.forward(&x_prefix, Some(&prefix_mask), 0, true)?;

    // Tail mask: rows 3..5 of the full 5x5 mask, shape [2, 5].
    let tail_mask = mask.narrow(0, 3, 2)?;
    let x_tail = x.narrow(1, 3, 2)?;
    let tail = attn2.forward(&x_tail, Some(&tail_mask), 3, true)?;

    // Compare full[:, :3] against the cached prefix output.
    let full_prefix = to_vec_f32(&full.narrow(1, 0, 3)?)?;
    let prefix_vec = to_vec_f32(&prefix)?;
    for (a, b) in prefix_vec.iter().zip(full_prefix.iter()) {
        assert!((a - b).abs() < 1e-5, "KV cache prefix mismatch: {a} vs {b}");
    }

    // Compare full[:, 3:] against the cached tail output.
    let full_tail = to_vec_f32(&full.narrow(1, 3, 2)?)?;
    let tail_vec = to_vec_f32(&tail)?;
    for (a, b) in tail_vec.iter().zip(full_tail.iter()) {
        assert!((a - b).abs() < 1e-5, "KV cache tail mismatch: {a} vs {b}");
    }
    Ok(())
}

// ---------- GatedMLP ----------

#[test]
fn test_gated_mlp_shape() -> Result<()> {
    let device = Device::Cpu;
    let mlp = GatedMLP::new(8, 16, &device)?;
    let x = Tensor::randn(0f32, 1f32, (4, 8), &device)?;
    let out = mlp.forward(&x)?;
    assert_eq!(out.dims(), &[4, 8]);
    Ok(())
}

#[test]
fn test_gated_mlp_batch_dims() -> Result<()> {
    let device = Device::Cpu;
    let mlp = GatedMLP::new(8, 16, &device)?;
    let x = Tensor::randn(0f32, 1f32, (2, 4, 8), &device)?;
    let out = mlp.forward(&x)?;
    assert_eq!(out.dims(), &[2, 4, 8]);
    Ok(())
}

// ---------- TransformerBlock ----------

#[test]
fn test_transformer_block_shape() -> Result<()> {
    let device = Device::Cpu;
    let mut block = TransformerBlock::new(12, 3, 16, 8, &device)?;
    let x = Tensor::randn(0f32, 1f32, (1, 5, 12), &device)?;
    let mask = build_causal_mask(5, &device)?;
    let out = block.forward(&x, Some(&mask), 0, false)?;
    assert_eq!(out.dims(), &[1, 5, 12]);
    Ok(())
}

#[test]
fn test_transformer_block_residual() -> Result<()> {
    // We can't easily zero out the weights, so this just checks that the
    // output has the correct shape and is finite.
    let device = Device::Cpu;
    let mut block = TransformerBlock::new(8, 2, 12, 6, &device)?;
    let x = Tensor::randn(0f32, 1f32, (1, 4, 8), &device)?;
    let out = block.forward(&x, None, 0, false)?;
    assert_eq!(out.dims(), &[1, 4, 8]);

    let vals = to_vec_f32(&out)?;
    for v in &vals {
        assert!(
            v.is_finite(),
            "TransformerBlock output has non-finite values"
        );
    }
    Ok(())
}

// ---------- Llama3Simplified ----------

#[test]
fn test_llama3_shape() -> Result<()> {
    let device = Device::Cpu;
    let mut model = Llama3Simplified::new(5, 4, 2, 6, 6, 1, &device)?;
    let tokens = Tensor::from_vec(vec![2u32, 3, 4], (1, 3), &device)?;
    let out = model.forward(&tokens, 0, false)?;
    assert_eq!(out.dims(), &[1, 3, 5]);
    Ok(())
}

#[test]
fn test_llama3_kv_cache_consistency() -> Result<()> {
    let device = Device::Cpu;
    let mut model = Llama3Simplified::new(10, 8, 2, 16, 12, 2, &device)?;
    let tokens = Tensor::from_vec(vec![0u32, 1, 2, 3], (1, 4), &device)?;

    let full = model.forward(&tokens, 0, false)?;
    assert_eq!(full.dims()[1], 4);

    // Clone the model so model2 starts from identical weights and a zeroed cache.
    let mut model2 = model.clone();
    let prefix_tokens = Tensor::from_vec(vec![0u32, 1, 2], (1, 3), &device)?;
    let _prefix = model2.forward(&prefix_tokens, 0, true)?;
    let tail_tokens = Tensor::from_vec(vec![3u32], (1, 1), &device)?;
    let tail = model2.forward(&tail_tokens, 3, true)?;

    // Last position of the full forward should match the cached tail forward.
    let full_last = to_vec_f32(&full.narrow(1, 3, 1)?)?;
    let tail_vec = to_vec_f32(&tail)?;
    for (a, b) in tail_vec.iter().zip(full_last.iter()) {
        assert!(
            (a - b).abs() < 1e-3,
            "KV cache mismatch in Llama3: full_last={b}, cached_tail={a}"
        );
    }
    Ok(())
}

// ---------- generate ----------

#[test]
fn test_generate_basic() -> Result<()> {
    // A scripted "model" that emits token 3, then token 4 (which is the stop token).
    let device = Device::Cpu;
    let call_count = std::cell::RefCell::new(0usize);
    let next_tokens: Vec<u32> = vec![3, 4];

    let mut model_fn = |tokens: &Tensor, _seq_pos: usize, _use_cache: bool| -> Result<Tensor> {
        let (_b, seq_len) = tokens.dims2()?;
        let mut count = call_count.borrow_mut();
        let next_token = next_tokens[*count];
        *count += 1;
        // Build logits with the chosen next_token as the maximum at the last position.
        let mut logits_data = vec![-1e9f32; 6 * seq_len];
        logits_data[(seq_len - 1) * 6 + next_token as usize] = 0.0;
        Tensor::from_vec(logits_data, (1, seq_len, 6), &device)
    };

    let decode_fn = |tokens: &[u32]| -> String {
        tokens
            .iter()
            .map(|&t| match t {
                3 => "A",
                4 => "!",
                5 => "B",
                _ => "?",
            })
            .collect()
    };

    let config = GenerateConfig {
        decode_fn: &decode_fn,
        stop_tokens: &[4],
        temperature: 0.7,
        max_tokens: 5,
        verbose: false,
    };
    let generated = generate(&mut model_fn, &[1u32, 2], &config, &device)?;

    assert_eq!(generated, vec![3, 4]);
    Ok(())
}

#[test]
fn test_generate_max_tokens() -> Result<()> {
    // A model that always wants to emit token 3, never the stop token.
    let device = Device::Cpu;
    let mut model_fn = |tokens: &Tensor, _seq_pos: usize, _use_cache: bool| -> Result<Tensor> {
        let (_b, seq_len) = tokens.dims2()?;
        let mut logits_data = vec![-1e9f32; 6 * seq_len];
        logits_data[(seq_len - 1) * 6 + 3] = 0.0;
        Tensor::from_vec(logits_data, (1, seq_len, 6), &device)
    };

    let decode_fn = |_tokens: &[u32]| -> String { String::new() };
    let config = GenerateConfig {
        decode_fn: &decode_fn,
        stop_tokens: &[4],
        temperature: 0.7,
        max_tokens: 3,
        verbose: false,
    };
    let generated = generate(&mut model_fn, &[1u32, 2], &config, &device)?;

    assert_eq!(generated.len(), 3, "Should stop at max_tokens=3");
    Ok(())
}

// ---------- SiLU at zero ----------

#[test]
fn test_silu_at_zero() -> Result<()> {
    // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    let device = Device::Cpu;
    let x = Tensor::from_vec(vec![0.0f32], 1, &device)?;
    let out = silu(&x)?;
    let val = out.to_vec1::<f32>()?[0];
    assert!(val.abs() < 1e-7);
    Ok(())
}

// ---------- eval_llama3 (loads the actual pretrained model) ----------

#[test]
fn test_eval_llama3() -> Result<()> {
    // Use the configured device for the real model — this is the only test
    // where GPU acceleration meaningfully helps, since the model is ~1B params.
    let device = default_device()?;
    let mut model = eval_llama3(&device)?;

    let tokens = Tensor::from_vec(vec![0u32, 1, 2, 3], (1, 4), &device)?;
    let full = model.forward(&tokens, 0, false)?;

    // Shape: (1, 4, vocab_size).
    assert_eq!(full.dims()[0], 1);
    assert_eq!(full.dims()[1], 4);
    let vocab_size = full.dims()[2];
    assert!(vocab_size > 0);

    // Outputs should be finite (at least the first 16 logits per position).
    let check_len = 16.min(vocab_size);
    let first_logits = to_vec_f32(&full.narrow(2, 0, check_len)?)?;
    for v in &first_logits {
        assert!(v.is_finite(), "eval_llama3 output has non-finite logits");
    }

    // KV cache consistency: full[:, 3:] should match the cached tail forward.
    let mut model2 = eval_llama3(&device)?;
    let prefix_tokens = Tensor::from_vec(vec![0u32, 1, 2], (1, 3), &device)?;
    let prefix = model2.forward(&prefix_tokens, 0, true)?;
    assert_eq!(prefix.dims()[1], 3);

    let tail_tokens = Tensor::from_vec(vec![3u32], (1, 1), &device)?;
    let tail = model2.forward(&tail_tokens, 3, true)?;
    assert_eq!(tail.dims()[1], 1);

    let full_last = to_vec_f32(&full.narrow(1, 3, 1)?.reshape(vocab_size)?)?;
    let tail_vec = to_vec_f32(&tail.reshape(vocab_size)?)?;

    let max_diff: f32 = full_last
        .iter()
        .zip(tail_vec.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 3e-4,
        "KV cache inconsistency in eval_llama3: max_diff={max_diff}"
    );
    Ok(())
}

// ---------- Additional value-verification and edge-case tests ----------

#[test]
fn test_embedding_lookup_consistency() -> Result<()> {
    let device = Device::Cpu;
    let layer = Embedding::new(10, 4, &device)?;
    let idx1 = Tensor::from_vec(vec![3u32], (1, 1), &device)?;
    let idx2 = Tensor::from_vec(vec![3u32, 3], (1, 2), &device)?;

    let out1 = to_vec_f32(&layer.forward(&idx1)?.reshape(4)?)?;
    let out2_full = to_vec_f32(&layer.forward(&idx2)?.reshape(8)?)?;
    let out2_first = &out2_full[0..4];
    let out2_second = &out2_full[4..8];

    // Same index should give the same embedding row.
    assert_eq!(out1, out2_first);
    assert_eq!(out1, out2_second);
    Ok(())
}

#[test]
fn test_silu_properties() -> Result<()> {
    let device = Device::Cpu;

    // silu(0) = 0
    let zero = Tensor::from_vec(vec![0.0f32], 1, &device)?;
    let out_zero = silu(&zero)?.to_vec1::<f32>()?[0];
    assert!(out_zero.abs() < 1e-7, "silu(0) should be 0");

    // silu(x) > 0 for x > 0
    let pos = Tensor::from_vec(vec![1.0f32, 2.0, 5.0], 3, &device)?;
    let out_pos = silu(&pos)?.to_vec1::<f32>()?;
    for v in &out_pos {
        assert!(*v > 0.0, "silu(x) should be positive for x>0, got {v}");
    }

    // silu(x) < 0 for x < -1 (approximately)
    let neg = Tensor::from_vec(vec![-5.0f32], 1, &device)?;
    let out_neg = silu(&neg)?.to_vec1::<f32>()?[0];
    assert!(
        out_neg < 0.0,
        "silu(x) should be negative for very negative x, got {out_neg}"
    );
    Ok(())
}

#[test]
fn test_rmsnorm_unit_rms() -> Result<()> {
    // After RMSNorm with weight=1, the RMS of the output should be ≈ 1.
    let device = Device::Cpu;
    let layer = RMSNorm::new(8, 1e-5, &device)?;
    let x = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        (1, 8),
        &device,
    )?;
    let out = layer.forward(&x)?;
    let vals = to_vec_f32(&out)?;
    let rms: f32 = (vals.iter().map(|v| v * v).sum::<f32>() / vals.len() as f32).sqrt();
    assert!(
        (rms - 1.0).abs() < 1e-3,
        "RMSNorm output should have RMS ≈ 1.0, got {rms}"
    );
    Ok(())
}

#[test]
fn test_rmsnorm_zero_input() -> Result<()> {
    let device = Device::Cpu;
    let layer = RMSNorm::new(4, 1e-5, &device)?;
    let x = Tensor::zeros((1, 4), DType::F32, &device)?;
    let out = layer.forward(&x)?;
    let vals = to_vec_f32(&out)?;
    for v in &vals {
        assert!(
            v.is_finite(),
            "RMSNorm should handle zero input without NaN"
        );
    }
    Ok(())
}

#[test]
fn test_self_attention_output_is_weighted_v() -> Result<()> {
    // With Q = K = 0, attention weights are uniform after softmax,
    // so the output is the row-mean of V.
    let device = Device::Cpu;
    let n = 3;
    let d = 4;
    let q = Tensor::zeros((n, d), DType::F32, &device)?;
    let k = Tensor::zeros((n, d), DType::F32, &device)?;
    let v = Tensor::from_vec(
        vec![
            1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ],
        (3, 4),
        &device,
    )?;

    let out = self_attention(&q, &k, &v, None)?;
    let expected = torch_attention(
        to_vec_f32(&q)?,
        q.dims().to_vec(),
        to_vec_f32(&k)?,
        k.dims().to_vec(),
        to_vec_f32(&v)?,
        v.dims().to_vec(),
        None,
    );
    let vals = to_vec_f32(&out)?;
    assert_f32_slice_close(&vals, &expected, 1e-5);
    Ok(())
}

#[test]
fn test_transformer_block_output_finite() -> Result<()> {
    let device = Device::Cpu;
    let mut block = TransformerBlock::new(8, 2, 16, 10, &device)?;
    let x = Tensor::randn(0f32, 0.1f32, (1, 4, 8), &device)?;
    let mask = build_causal_mask(4, &device)?;
    let out = block.forward(&x, Some(&mask), 0, false)?;

    let vals = to_vec_f32(&out)?;
    for v in &vals {
        assert!(v.is_finite(), "TransformerBlock output must be finite");
    }

    // Residual: output should differ from input (since attn+MLP weights are random).
    let x_vals = to_vec_f32(&x)?;
    let differs = vals
        .iter()
        .zip(x_vals.iter())
        .any(|(a, b)| (a - b).abs() > 1e-7);
    assert!(
        differs,
        "TransformerBlock output should differ from input due to attention+MLP"
    );
    Ok(())
}

#[test]
fn test_generate_stops_at_stop_token() -> Result<()> {
    let device = Device::Cpu;
    let stop_token = 5u32;
    let mut call = 0;
    let mut model_fn = |tokens: &Tensor, _: usize, _: bool| -> Result<Tensor> {
        let (_b, seq_len) = tokens.dims2()?;
        let vocab = 8usize;
        let mut data = vec![f32::NEG_INFINITY; seq_len * vocab];
        // Always predict the stop token.
        data[(seq_len - 1) * vocab + stop_token as usize] = 0.0;
        call += 1;
        Tensor::from_vec(data, (1, seq_len, vocab), &device)
    };

    let decode_fn = |_: &[u32]| -> String { String::new() };
    let config = GenerateConfig {
        decode_fn: &decode_fn,
        stop_tokens: &[stop_token],
        temperature: 0.5,
        max_tokens: 100,
        verbose: false,
    };
    let generated = generate(&mut model_fn, &[1u32, 2], &config, &device)?;
    // Should stop after generating one token (the stop token).
    assert_eq!(generated.len(), 1);
    assert_eq!(generated[0], stop_token);
    Ok(())
}
