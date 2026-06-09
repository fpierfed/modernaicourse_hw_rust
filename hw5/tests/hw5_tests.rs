//! Tests for hw5 — Candle version.

use candle_core::{Device, Result, Tensor, Var};
use hw5::*;
use std::collections::HashMap;
use std::io::Write;
use test_support::{as_f32_vec, assert_f32_close, assert_f32_slice_close, json, python_json};

// ---------- small helpers ----------

fn to_vec_f32(t: &Tensor) -> Result<Vec<f32>> {
    t.flatten_all()?.to_vec1::<f32>()
}

fn to_vec_u32(t: &Tensor) -> Result<Vec<u32>> {
    t.flatten_all()?.to_vec1::<u32>()
}

// ---------- PyTorch ground-truth helpers ----------

// Rember that we store Linear.weight pre-transposed!
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
w = torch.tensor(d["weight"], dtype=torch.float32).reshape(d["weight_shape"]).T
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

fn torch_rms_norm(x: Vec<f32>, shape: Vec<usize>, eps: f64) -> Vec<f32> {
    as_f32_vec(python_json(
        r#"
import json
import sys
import torch
import torch.nn.functional as F

d = json.load(sys.stdin)
x = torch.tensor(d["x"], dtype=torch.float32).reshape(d["shape"])
out = F.rms_norm(x, (x.shape[-1],), eps=d["eps"])
json.dump(out.flatten().tolist(), sys.stdout)
"#,
        json!({ "x": x, "shape": shape, "eps": eps }),
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

fn torch_cross_entropy(logits: Vec<f32>, shape: Vec<usize>, targets: Vec<u32>) -> f32 {
    as_f32_vec(python_json(
        r#"
import json
import sys
import torch
import torch.nn.functional as F

d = json.load(sys.stdin)
logits = torch.tensor(d["logits"], dtype=torch.float32).reshape(d["shape"])
targets = torch.tensor(d["targets"], dtype=torch.long).reshape(d["shape"][:-1])
loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
json.dump([float(loss)], sys.stdout)
"#,
        json!({ "logits": logits, "shape": shape, "targets": targets }),
    ))[0]
}

fn causal_mask(length: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0.0f32; length * length];
    for i in 0..length {
        for j in (i + 1)..length {
            data[i * length + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(data, (length, length), device)
}

// ============================================================
// Part I: BPE Tokenization
// ============================================================

#[test]
fn test_text_to_corpus_simple() {
    let (corpus, counts) = text_to_corpus("a b b");
    assert_eq!(
        corpus,
        vec![
            vec!["a".to_string()],
            vec![" ".to_string(), "b".to_string()],
        ]
    );
    assert_eq!(counts, vec![1, 2]);
}

#[test]
fn test_text_to_corpus_multiline() {
    let (corpus, counts) = text_to_corpus("hi there\nthere");
    assert_eq!(
        corpus,
        vec![
            vec!["h".to_string(), "i".to_string()],
            vec![
                " ".to_string(),
                "t".to_string(),
                "h".to_string(),
                "e".to_string(),
                "r".to_string(),
                "e".to_string()
            ],
            vec![
                "\n".to_string(),
                "t".to_string(),
                "h".to_string(),
                "e".to_string(),
                "r".to_string(),
                "e".to_string()
            ],
        ]
    );
    assert_eq!(counts, vec![1, 1, 1]);
}

#[test]
fn test_most_common_pair_basic() {
    let corpus = vec![
        vec!["a".to_string(), "b".to_string(), "a".to_string()],
        vec!["a".to_string(), "b".to_string()],
        vec!["b".to_string(), "c".to_string()],
    ];
    let counts = vec![2, 1, 3];
    let pair = most_common_pair(&corpus, &counts);
    assert_eq!(pair, ("a".to_string(), "b".to_string()));
}

#[test]
fn test_most_common_pair_weighted() {
    let corpus = vec![
        vec![" ".to_string(), "x".to_string()],
        vec![" ".to_string(), "x".to_string(), "y".to_string()],
        vec!["x".to_string(), "y".to_string()],
    ];
    let counts = vec![4, 1, 1];
    let pair = most_common_pair(&corpus, &counts);
    assert_eq!(pair, (" ".to_string(), "x".to_string()));
}

#[test]
fn test_merge_pair_basic() {
    let mut corpus = vec![
        vec!["a".to_string(), "b".to_string(), "c".to_string()],
        vec!["c".to_string(), "a".to_string(), "b".to_string()],
    ];
    merge_pair(&mut corpus, &("a".to_string(), "b".to_string()));
    assert_eq!(
        corpus,
        vec![
            vec!["ab".to_string(), "c".to_string()],
            vec!["c".to_string(), "ab".to_string()],
        ]
    );
}

#[test]
fn test_merge_pair_partial() {
    let mut corpus = vec![
        vec!["x".to_string(), "y".to_string(), "z".to_string()],
        vec!["x".to_string(), "y".to_string()],
        vec!["y".to_string(), "z".to_string()],
    ];
    merge_pair(&mut corpus, &("y".to_string(), "z".to_string()));
    assert_eq!(
        corpus,
        vec![
            vec!["x".to_string(), "yz".to_string()],
            vec!["x".to_string(), "y".to_string()],
            vec!["yz".to_string()],
        ]
    );
}

#[test]
fn test_train_bpe() {
    let (tokens, merges) = train_bpe("aa aa aa", 258);
    assert_eq!(tokens["a"], 'a' as u32);
    assert_eq!(tokens[" "], ' ' as u32);
    assert_eq!(tokens["aa"], 256);
    assert_eq!(tokens[" aa"], 257);
    assert_eq!(
        merges,
        vec![
            ("a".to_string(), "a".to_string()),
            (" ".to_string(), "aa".to_string()),
        ]
    );
}

#[test]
fn test_bpe_encode() {
    let tokens: HashMap<String, u32> = [
        ("a".to_string(), 0),
        (" ".to_string(), 1),
        ("aa".to_string(), 2),
        (" aa".to_string(), 3),
    ]
    .into_iter()
    .collect();
    let merges = vec![
        ("a".to_string(), "a".to_string()),
        (" ".to_string(), "aa".to_string()),
    ];
    assert_eq!(bpe_encode("aa aa", &merges, &tokens), vec![2, 3]);
    assert_eq!(bpe_encode("aa", &merges, &tokens), vec![2]);
}

#[test]
fn test_bpe_decode() {
    let tokens: HashMap<String, u32> = [
        ("a".to_string(), 0),
        (" ".to_string(), 1),
        ("aa".to_string(), 2),
        (" aa".to_string(), 3),
    ]
    .into_iter()
    .collect();
    assert_eq!(bpe_decode(&[2, 3], &tokens), "aa aa");
    assert_eq!(bpe_decode(&[0, 1, 0], &tokens), "a a");
}

// ============================================================
// Part II: Transformer Architecture
// ============================================================

// ---------- Linear ----------

#[test]
fn test_linear() -> Result<()> {
    let device = default_device()?;
    let layer = Linear::new(10, 20, &device)?;

    let x = Tensor::randn(0f32, 1f32, (50, 10), &device)?;
    let out = layer.forward(&x)?;
    assert_eq!(out.dims(), &[50, 20]);

    let expected = torch_linear(
        to_vec_f32(&x)?,
        x.dims().to_vec(),
        to_vec_f32(&layer.weight)?,
        layer.weight.dims().to_vec(),
    );
    let actual = to_vec_f32(&out)?;
    assert_f32_slice_close(&actual, &expected, 1e-4);

    // Batch dims.
    let x = Tensor::randn(0f32, 1f32, (7, 9, 10), &device)?;
    let out = layer.forward(&x)?;
    assert_eq!(out.dims(), &[7, 9, 20]);
    Ok(())
}

#[test]
fn test_linear_kaiming_init() -> Result<()> {
    let device = default_device()?;
    let layer = Linear::new(100, 1000, &device)?;
    let w_data = to_vec_f32(&layer.weight)?;
    let n = w_data.len() as f32;
    let mean = w_data.iter().sum::<f32>() / n;
    let var = w_data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = var.sqrt();
    let expected_std = (2.0f32 / 100.0).sqrt();
    assert!(
        (std - expected_std).abs() < 3e-3,
        "Linear weight std {std} not close to expected {expected_std}"
    );
    Ok(())
}

// ---------- Embedding ----------

#[test]
fn test_embedding() -> Result<()> {
    let device = default_device()?;
    let layer = Embedding::new(200, 20, &device)?;

    let y = Tensor::from_vec(vec![0u32, 5, 10, 199], (1, 4), &device)?;
    let out = layer.forward(&y)?;
    assert_eq!(out.dims(), &[1, 4, 20]);

    // Correctness: first token's embedding == corresponding weight row.
    let out0 = to_vec_f32(&out.narrow(1, 0, 1)?.reshape(20)?)?;
    let w0 = to_vec_f32(&layer.weight.narrow(0, 0, 1)?.reshape(20)?)?;
    assert_f32_slice_close(&out0, &w0, 1e-6);

    // Batch dims.
    let y = Tensor::from_vec(vec![0u32, 1, 2, 3, 4, 5], (2, 3), &device)?;
    let out = layer.forward(&y)?;
    assert_eq!(out.dims(), &[2, 3, 20]);
    Ok(())
}

#[test]
fn test_embedding_std_init() -> Result<()> {
    let device = default_device()?;
    let layer = Embedding::new(1000, 100, &device)?;
    let w_data = to_vec_f32(&layer.weight)?;
    let n = w_data.len() as f32;
    let mean = w_data.iter().sum::<f32>() / n;
    let var = w_data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = var.sqrt();
    assert!(
        (std - 1.0).abs() < 3e-2,
        "Embedding weight std {std} not close to 1.0"
    );
    Ok(())
}

// ---------- SiLU ----------

#[test]
fn test_silu() -> Result<()> {
    let device = default_device()?;
    let x = Tensor::randn(0f32, 1f32, (10, 20), &device)?;
    let out = silu(&x)?;
    let expected = torch_silu(to_vec_f32(&x)?, x.dims().to_vec());
    let actual = to_vec_f32(&out)?;
    assert_f32_slice_close(&actual, &expected, 1e-6);

    // Multi-dim.
    let x = Tensor::randn(0f32, 1f32, (3, 4, 5, 6), &device)?;
    let out = silu(&x)?;
    assert_eq!(out.dims(), x.dims());
    Ok(())
}

// ---------- RMSNorm ----------

#[test]
fn test_rms_norm() -> Result<()> {
    let device = default_device()?;
    let x = Tensor::from_vec(
        vec![1.0f32, -1.0, 0.5, 0.5, 2.0, 0.0, -2.0, 1.0],
        (2, 4),
        &device,
    )?;
    let out = rms_norm(&x, 1e-5)?;

    let expected = torch_rms_norm(to_vec_f32(&x)?, x.dims().to_vec(), 1e-5);
    let actual = to_vec_f32(&out)?;
    assert_f32_slice_close(&actual, &expected, 1e-5);

    // Batch dims.
    let x = Tensor::randn(0f32, 1f32, (10, 7, 20), &device)?;
    let out = rms_norm(&x, 1e-3)?;
    assert_eq!(out.dims(), &[10, 7, 20]);
    Ok(())
}

// ---------- self_attention ----------

#[test]
fn test_self_attention_causal() -> Result<()> {
    let device = default_device()?;
    let q = Tensor::randn(0f32, 1f32, (5, 8), &device)?;
    let k = Tensor::randn(0f32, 1f32, (5, 8), &device)?;
    let v = Tensor::randn(0f32, 1f32, (5, 6), &device)?;
    let mask = causal_mask(5, &device)?;
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

    // First row with causal mask: only attends to position 0, so output = V[0].
    let out_row0 = to_vec_f32(&out.narrow(0, 0, 1)?.reshape(6)?)?;
    let v_row0 = to_vec_f32(&v.narrow(0, 0, 1)?.reshape(6)?)?;
    for (a, b) in out_row0.iter().zip(v_row0.iter()) {
        assert!((a - b).abs() < 1e-5);
    }
    Ok(())
}

#[test]
fn test_self_attention_batched() -> Result<()> {
    let device = default_device()?;
    let q = Tensor::randn(0f32, 1f32, (2, 3, 5, 8), &device)?;
    let k = Tensor::randn(0f32, 1f32, (2, 3, 5, 8), &device)?;
    let v = Tensor::randn(0f32, 1f32, (2, 3, 5, 4), &device)?;
    let mask = causal_mask(5, &device)?;
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

// ---------- MultiHeadAttentionKVCache ----------

#[test]
fn test_multi_head_attention_kv_cache() -> Result<()> {
    let device = default_device()?;
    let mut attn = MultiHeadAttentionKVCache::new(12, 3, 8, &device)?;
    let x = Tensor::randn(0f32, 1f32, (1, 5, 12), &device)?;
    let mask = causal_mask(5, &device)?;

    let full = attn.forward(&x, Some(&mask), 0, false)?;
    assert_eq!(full.dims(), &[1, 5, 12]);

    // Prefix + tail with cache.
    let mut attn2 = attn.clone();
    let prefix_mask = causal_mask(3, &device)?;
    let _prefix = attn2.forward(&x.narrow(1, 0, 3)?, Some(&prefix_mask), 0, true)?;
    let tail_mask = mask.narrow(0, 3, 2)?;
    let tail = attn2.forward(&x.narrow(1, 3, 2)?, Some(&tail_mask), 3, true)?;

    let full_tail = to_vec_f32(&full.narrow(1, 3, 2)?)?;
    let tail_vec = to_vec_f32(&tail)?;
    let max_diff: f32 = full_tail
        .iter()
        .zip(tail_vec.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-5, "KV cache mismatch: max diff = {max_diff}");
    Ok(())
}

// ---------- MLP ----------

#[test]
fn test_mlp() -> Result<()> {
    let device = default_device()?;
    let mlp = MLP::new(5, 7, &device)?;
    let x = Tensor::randn(0f32, 1f32, (4, 3, 5), &device)?;
    let out = mlp.forward(&x)?;
    assert_eq!(out.dims(), &[4, 3, 5]);
    Ok(())
}

// ---------- TransformerBlock ----------

#[test]
fn test_transformer_block() -> Result<()> {
    let device = default_device()?;
    let mut block = TransformerBlock::new(12, 3, 16, 8, &device)?;
    let x = Tensor::randn(0f32, 1f32, (1, 5, 12), &device)?;
    let mask = causal_mask(5, &device)?;

    let full = block.forward(&x, Some(&mask), 0, false)?;
    assert_eq!(full.dims(), &[1, 5, 12]);

    // KV cache consistency.
    let mut block2 = block.clone();
    let prefix_mask = causal_mask(3, &device)?;
    let _prefix = block2.forward(&x.narrow(1, 0, 3)?, Some(&prefix_mask), 0, true)?;
    let tail_mask = mask.narrow(0, 3, 2)?;
    let tail = block2.forward(&x.narrow(1, 3, 2)?, Some(&tail_mask), 3, true)?;

    let full_tail = to_vec_f32(&full.narrow(1, 3, 2)?)?;
    let tail_vec = to_vec_f32(&tail)?;
    let max_diff: f32 = full_tail
        .iter()
        .zip(tail_vec.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 3e-5,
        "TransformerBlock KV cache mismatch: {max_diff}"
    );
    Ok(())
}

// ---------- LLM ----------

#[test]
fn test_llm_hw4() -> Result<()> {
    let device = default_device()?;
    let mut model = LLM::new(10, 8, 2, 8, 12, 2, &device)?;
    let tokens = Tensor::from_vec(vec![0u32, 1, 2, 3], (1, 4), &device)?;

    let full = model.forward(&tokens, 0, false)?;
    assert_eq!(full.dims(), &[1, 4, 10]);

    let mut model2 = model.clone();
    let prefix_tokens = Tensor::from_vec(vec![0u32, 1, 2], (1, 3), &device)?;
    let _prefix = model2.forward(&prefix_tokens, 0, true)?;
    let tail_tokens = Tensor::from_vec(vec![3u32], (1, 1), &device)?;
    let tail = model2.forward(&tail_tokens, 3, true)?;

    let full_last = to_vec_f32(&full.narrow(1, 3, 1)?)?;
    let tail_vec = to_vec_f32(&tail)?;
    let max_diff: f32 = full_last
        .iter()
        .zip(tail_vec.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 3e-4, "LLM KV cache mismatch: {max_diff}");
    Ok(())
}

#[test]
fn test_llm() -> Result<()> {
    let device = default_device()?;
    let mut model = LLM::new(11, 12, 3, 8, 16, 2, &device)?;
    let tokens = Tensor::from_vec(vec![0u32, 1, 2, 3, 4], (1, 5), &device)?;

    let out = model.forward(&tokens, 0, false)?;
    assert_eq!(out.dims(), &[1, 5, 11]);

    // KV cache consistency.
    let mut model2 = model.clone();
    let _prefix = model2.forward(&tokens.narrow(1, 0, 3)?, 0, true)?;
    let tail = model2.forward(&tokens.narrow(1, 3, 2)?, 3, true)?;

    let out_tail = to_vec_f32(&out.narrow(1, 3, 2)?)?;
    let tail_vec = to_vec_f32(&tail)?;
    let max_diff: f32 = out_tail
        .iter()
        .zip(tail_vec.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 6e-5, "LLM KV cache mismatch: {max_diff}");
    Ok(())
}

// ============================================================
// Part III: Training
// ============================================================

// ---------- cross_entropy_loss ----------

#[test]
fn test_cross_entropy_loss_2d() -> Result<()> {
    let device = default_device()?;
    let logits_data = vec![2.0f32, 1.0, 0.0, 0.0, 2.0, 1.0];
    let targets = vec![0u32, 2];
    let shape = [2, 3];
    let logits = Tensor::from_vec(logits_data.clone(), (2, 3), &device)?;
    let y = Tensor::from_vec(targets.clone(), 2, &device)?;
    let loss = cross_entropy_loss(&logits, &y)?;
    let val: f32 = loss.to_scalar::<f32>()?;
    let expected = torch_cross_entropy(logits_data, shape.to_vec(), targets);
    assert_f32_close(val, expected, 1e-5);
    Ok(())
}

#[test]
fn test_cross_entropy_loss_3d() -> Result<()> {
    let device = default_device()?;
    let logits = Tensor::randn(0f32, 1f32, (4, 5, 7), &device)?;
    let y_data: Vec<u32> = (0..20).map(|i| i % 7).collect();
    let y = Tensor::from_vec(y_data.clone(), (4, 5), &device)?;
    let logits_flat = logits.reshape((20, 7))?;
    let y_flat = y.reshape(20)?;
    let loss = cross_entropy_loss(&logits_flat, &y_flat)?;
    let val: f32 = loss.to_scalar::<f32>()?;
    let expected = torch_cross_entropy(to_vec_f32(&logits)?, logits.dims().to_vec(), y_data);
    assert!(val.is_finite() && val > 0.0);
    assert_f32_close(val, expected, 1e-5);
    Ok(())
}

// ---------- pretokenize_data / DataLoader ----------

#[test]
fn test_pretokenize_data() {
    let tmp = tempfile::tempdir().unwrap();
    let in_path = tmp.path().join("sample.txt");
    let out_path = tmp.path().join("sample.bin");

    std::fs::write(&in_path, "abcdefg").unwrap();

    let encode_fn = |text: &str| -> Vec<u16> { text.chars().map(|c| c as u16).collect() };

    pretokenize_data(&encode_fn, &in_path, &out_path, 3, Some(2));

    let data = std::fs::read(&out_path).unwrap();
    let tokens: Vec<u16> = data
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();
    assert_eq!(tokens, vec![97, 98, 99, 100, 101, 102]);
}

#[test]
fn test_dataloader_file() -> Result<()> {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("tokens.bin");

    let mut f = std::fs::File::create(&path).unwrap();
    for i in 0u16..20 {
        f.write_all(&i.to_le_bytes()).unwrap();
    }
    drop(f);

    let loader = DataLoader::new(&path, 3, 2, &default_device()?)?;
    let batches: Vec<_> = loader.collect();

    assert_eq!(batches.len(), 3);

    let xb0 = to_vec_u32(&batches[0].0)?;
    assert_eq!(xb0, vec![0, 1, 2, 4, 5, 6]);

    let yb0 = to_vec_u32(&batches[0].1)?;
    assert_eq!(yb0, vec![1, 2, 3, 5, 6, 7]);
    Ok(())
}

// ---------- Adam ----------

#[test]
fn test_adam() -> Result<()> {
    let device = default_device()?;
    let layer = Linear::new(6, 3, &device)?;

    let params: Vec<Var> = vec![layer.weight.clone()];
    let mut opt = Adam::new(params.clone(), 1e-3, (0.9, 0.95), 1e-8)?;

    let w_before = to_vec_f32(&params[0])?;

    for _ in 0..5 {
        let x = Tensor::randn(0f32, 1f32, (16, 6), &device)?;
        let y_data: Vec<u32> = (0..16).map(|i| i % 3).collect();
        let y = Tensor::from_vec(y_data, 16, &device)?;

        let logits = layer.forward(&x)?;
        let loss = cross_entropy_loss(&logits, &y)?;
        let grads = loss.backward();
        opt.step(&grads);
    }

    let w_after = to_vec_f32(&params[0])?;
    let changed = w_after
        .iter()
        .zip(w_before.iter())
        .any(|(a, b)| (a - b).abs() > 1e-7);
    assert!(changed, "Adam did not update weights after 5 steps");
    Ok(())
}

#[test]
fn test_adam_zero_grad() -> Result<()> {
    let device = default_device()?;
    let layer = Linear::new(4, 3, &device)?;

    let params: Vec<Var> = vec![layer.weight.clone()];
    let mut opt = Adam::new(params.clone(), 1e-2, (0.9, 0.999), 1e-8)?;

    // Create gradients.
    let x = Tensor::randn(0f32, 1f32, (4, 4), &device)?;
    let y = Tensor::from_vec(vec![0u32, 1, 2, 0], 4, &device)?;
    let logits = layer.forward(&x)?;
    let loss = cross_entropy_loss(&logits, &y)?;
    let _grads = loss.backward();

    // zero_grad then step should be a no-op on weights (first step with zero grad).
    opt.zero_grad();
    let w_before = to_vec_f32(&params[0])?;
    let zero_loss = Tensor::from_vec(vec![0.0f32], 1, &device)?;
    let zero_grads = zero_loss.backward();
    opt.step(&zero_grads);
    let w_after = to_vec_f32(&params[0])?;
    for (a, b) in w_after.iter().zip(w_before.iter()) {
        assert!((a - b).abs() < 1e-7, "Adam zero_grad didn't prevent update");
    }
    Ok(())
}

// ---------- train_llm ----------

#[test]
fn test_train_llm() -> Result<()> {
    let device = default_device()?;
    let layer = Linear::new(4, 5, &device)?;

    let params: Vec<Var> = vec![layer.weight.clone()];
    let mut opt = Adam::new(params.clone(), 0.01, (0.9, 0.999), 1e-8)?;

    let w_before = to_vec_f32(&params[0])?;

    let x1 = Tensor::from_vec(vec![0u32, 1, 2, 1, 2, 3], (2, 3), &device)?;
    let y1 = Tensor::from_vec(vec![1u32, 2, 3, 2, 3, 4], (2, 3), &device)?;
    let x2 = Tensor::from_vec(vec![2u32, 3, 4, 0, 2, 4], (2, 3), &device)?;
    let y2 = Tensor::from_vec(vec![3u32, 4, 0, 2, 4, 1], (2, 3), &device)?;
    let loader = vec![(x1, y1), (x2, y2)];

    let mut model_fn = |tokens: &Tensor| -> Result<Tensor> {
        let dims = tokens.dims();
        let batch_size = dims[0];
        let seq_len = dims[1];

        let embed = Tensor::randn(0f32, 1f32, (5, 4), &device)?;
        let flat_tokens = to_vec_u32(tokens)?;
        let embed_data = to_vec_f32(&embed)?;

        let mut embedded_data = Vec::new();
        for &tok in &flat_tokens {
            let start = (tok as usize) * 4;
            embedded_data.extend_from_slice(&embed_data[start..start + 4]);
        }
        let x = Tensor::from_vec(embedded_data, (batch_size, seq_len, 4), &device)?;
        let x_2d = x.reshape((batch_size * seq_len, 4))?;
        let out_2d = layer.forward(&x_2d)?;
        out_2d.reshape((batch_size, seq_len, 5))
    };

    let _ = train_llm(&mut model_fn, loader, &mut opt)?;

    let w_after = to_vec_f32(&params[0])?;
    let changed = w_after
        .iter()
        .zip(w_before.iter())
        .any(|(a, b)| (a - b).abs() > 1e-7);
    assert!(changed, "train_llm did not update weights");
    Ok(())
}

// ---------- generate ----------

#[test]
fn test_generate() -> Result<()> {
    let device = default_device()?;
    let mut call_count = 0usize;
    let next_tokens: Vec<u32> = vec![3, 4];

    let mut model_fn = |tokens: &Tensor, _seq_pos: usize, _use_cache: bool| -> Result<Tensor> {
        let vocab_size = 6;
        let seq_len = tokens.dims()[1];
        let mut data = vec![f32::NEG_INFINITY; seq_len * vocab_size];
        let next = next_tokens[call_count] as usize;
        data[(seq_len - 1) * vocab_size + next] = 0.0;
        call_count += 1;
        Tensor::from_vec(data, (1, seq_len, vocab_size), &device)
    };

    let decode_fn = |tokens: &[u32]| -> String {
        tokens
            .iter()
            .map(|&t| match t {
                3 => 'A',
                4 => '!',
                5 => 'B',
                _ => '?',
            })
            .collect()
    };

    let generated = generate(
        &mut model_fn,
        &[1, 2],
        &decode_fn,
        4,
        0.7,
        5,
        false,
        &device,
    )?;
    assert_eq!(generated, vec![3, 4]);
    Ok(())
}

#[test]
fn test_generate_max_tokens() -> Result<()> {
    let device = default_device()?;
    let mut model_fn = |tokens: &Tensor, _seq_pos: usize, _use_cache: bool| -> Result<Tensor> {
        let vocab_size = 6;
        let seq_len = tokens.dims()[1];
        let mut data = vec![f32::NEG_INFINITY; seq_len * vocab_size];
        data[(seq_len - 1) * vocab_size + 3] = 0.0;
        Tensor::from_vec(data, (1, seq_len, vocab_size), &device)
    };

    let decode_fn = |_: &[u32]| -> String { String::new() };
    let generated = generate(
        &mut model_fn,
        &[1, 2],
        &decode_fn,
        4,
        0.7,
        3,
        false,
        &device,
    )?;
    assert_eq!(generated.len(), 3);
    Ok(())
}

// ============================================================
// End-to-end: eval_llm
// ============================================================

mod tiny_stories_eval {
    use hf_hub::api::sync::Api;
    use tokenizers::Tokenizer;

    pub fn load_gpt2_tokenizer() -> Tokenizer {
        let api = Api::new().unwrap();
        let repo = api.model("openai-community/gpt2".to_string());
        let tokenizer_path = repo.get("tokenizer.json").unwrap();
        Tokenizer::from_file(tokenizer_path).unwrap()
    }

    pub fn load_tiny_stories_text() -> String {
        let api = Api::new().unwrap();
        let repo = api.dataset("roneneldan/TinyStories".to_string());
        let path = repo.get("TinyStoriesV2-GPT4-train.txt").unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        content[..4096.min(content.len())].to_string()
    }

    pub fn get_eval_tokens(start_token: usize, num_tokens: usize) -> Vec<u32> {
        let tokenizer = load_gpt2_tokenizer();
        let text = load_tiny_stories_text();
        let encoding = tokenizer.encode(text.as_str(), false).unwrap();
        let ids: Vec<u32> = encoding.get_ids().to_vec();
        ids[start_token..start_token + num_tokens].to_vec()
    }
}

#[test]
fn test_eval_llm() -> Result<()> {
    let device = default_device()?;
    let mut model = eval_llm(&device)?;

    let eval_tokens = tiny_stories_eval::get_eval_tokens(0, 48);
    let tokens = Tensor::from_vec(eval_tokens.clone(), (1, 48), &device)?;

    // Compute sequence loss.
    let logits = model.forward(&tokens.narrow(1, 0, 47)?, 0, false)?;
    let vocab_size = logits.dims()[2];
    let logits_flat = logits.reshape((47, vocab_size))?;
    let targets = Tensor::from_vec(eval_tokens[1..].to_vec(), 47, &device)?;
    let phrase_loss: f32 = cross_entropy_loss(&logits_flat, &targets)?.to_scalar::<f32>()?;

    // Corrupted: reverse the token order (except first).
    let mut corrupted_tokens = eval_tokens.clone();
    corrupted_tokens[1..].reverse();
    let corrupted = Tensor::from_vec(corrupted_tokens.clone(), (1, 48), &device)?;
    let c_logits = model.forward(&corrupted.narrow(1, 0, 47)?, 0, false)?;
    let c_vocab_size = c_logits.dims()[2];
    let c_logits_flat = c_logits.reshape((47, c_vocab_size))?;
    let c_targets = Tensor::from_vec(corrupted_tokens[1..].to_vec(), 47, &device)?;
    let corrupted_loss: f32 = cross_entropy_loss(&c_logits_flat, &c_targets)?.to_scalar::<f32>()?;

    assert!(
        phrase_loss < 7.0,
        "eval_llm phrase_loss {phrase_loss} should be < 7.0"
    );
    assert!(
        phrase_loss < corrupted_loss,
        "phrase_loss {phrase_loss} should be < corrupted_loss {corrupted_loss}"
    );

    Ok(())
}

// ------------------------------------------------------------
// Additional edge case and value-verification tests
// ------------------------------------------------------------

#[test]
fn test_bpe_encode_decode_roundtrip() {
    let text = "hello world hello";
    let (tokens, merges) = train_bpe(text, 260);
    let encoded = bpe_encode(text, &merges, &tokens);
    let decoded = bpe_decode(&encoded, &tokens);
    assert_eq!(decoded, text, "BPE encode/decode should roundtrip");
}

#[test]
fn test_bpe_encode_single_char() {
    let tokens: HashMap<String, u32> = [("a".to_string(), 97)].into_iter().collect();
    let merges: Vec<(String, String)> = vec![];
    assert_eq!(bpe_encode("a", &merges, &tokens), vec![97]);
}

#[test]
fn test_text_to_corpus_empty() {
    let (corpus, counts) = text_to_corpus("");
    assert!(corpus.is_empty() || counts.iter().all(|&c| c == 0));
}

#[test]
fn test_most_common_pair_tie_breaking() {
    let corpus = vec![
        vec!["a".to_string(), "b".to_string()],
        vec!["c".to_string(), "d".to_string()],
    ];
    let counts = vec![1, 1];
    let pair = most_common_pair(&corpus, &counts);
    assert!(
        pair == ("a".to_string(), "b".to_string()) || pair == ("c".to_string(), "d".to_string()),
        "Should return one of the tied pairs"
    );
}

#[test]
fn test_linear_kaiming_std_various_sizes() -> Result<()> {
    let device = default_device()?;
    for in_f in [16, 64, 256] {
        let layer = Linear::new(in_f, 100, &device)?;
        let w_data = to_vec_f32(&layer.weight)?;
        let n = w_data.len() as f32;
        let mean = w_data.iter().sum::<f32>() / n;
        let var = w_data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
        let std = var.sqrt();
        let expected_std = (2.0f32 / in_f as f32).sqrt();
        assert!(
            (std - expected_std).abs() < 0.05,
            "Kaiming init failed for in_f={in_f}: std={std}, expected={expected_std}"
        );
    }
    Ok(())
}

#[test]
fn test_silu_matches_formula() -> Result<()> {
    let device = default_device()?;
    let x = Tensor::from_vec(vec![-2.0f32, -1.0, 0.0, 1.0, 2.0, 3.0], (2, 3), &device)?;
    let out = silu(&x)?;
    let expected = torch_silu(to_vec_f32(&x)?, x.dims().to_vec());
    let actual = to_vec_f32(&out)?;
    assert_f32_slice_close(&actual, &expected, 1e-5);
    Ok(())
}

#[test]
fn test_rms_norm_preserves_direction() -> Result<()> {
    let device = default_device()?;
    let x = Tensor::from_vec(vec![1.0f32, -2.0, 3.0, -4.0], (1, 4), &device)?;
    let out = rms_norm(&x, 1e-5)?;
    let x_vals = to_vec_f32(&x)?;
    let out_vals = to_vec_f32(&out)?;
    for i in 0..4 {
        assert_eq!(
            x_vals[i].signum(),
            out_vals[i].signum(),
            "rms_norm should preserve sign at index {i}"
        );
    }
    Ok(())
}

#[test]
fn test_cross_entropy_loss_hw5_stable() -> Result<()> {
    let device = default_device()?;
    let logits = Tensor::from_vec(
        vec![1000.0f32, 1001.0, 999.0, 0.0, 0.0, 0.0],
        (2, 3),
        &device,
    )?;
    let y = Tensor::from_vec(vec![1u32, 0], 2, &device)?;
    let loss: f32 = cross_entropy_loss(&logits, &y)?.to_scalar::<f32>()?;
    assert!(
        loss.is_finite(),
        "cross_entropy_loss must be stable for large logits"
    );
    assert!(loss >= 0.0);
    Ok(())
}

#[test]
fn test_adam_converges_faster_than_random() -> Result<()> {
    let device = default_device()?;
    let layer = Linear::new(4, 3, &device)?;
    let params: Vec<Var> = vec![layer.weight.clone()];
    let mut opt = Adam::new(params, 0.01, (0.9, 0.999), 1e-8)?;

    let x = Tensor::randn(0f32, 1f32, (8, 4), &device)?;
    let y = Tensor::from_vec(vec![0u32, 1, 2, 0, 1, 2, 0, 1], 8, &device)?;

    let logits = layer.forward(&x)?;
    let loss_before: f32 = cross_entropy_loss(&logits, &y)?.to_scalar::<f32>()?;

    for _ in 0..10 {
        let logits = layer.forward(&x)?;
        let loss = cross_entropy_loss(&logits, &y)?;
        let grads = loss.backward();
        opt.step(&grads);
    }

    let logits = layer.forward(&x)?;
    let loss_after: f32 = cross_entropy_loss(&logits, &y)?.to_scalar::<f32>()?;
    assert!(
        loss_after < loss_before,
        "Adam should reduce loss: before={loss_before}, after={loss_after}"
    );
    Ok(())
}
