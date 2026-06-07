//! Tests for hw6 — Candle version.

use candle_core::{DType, Device, Result, Tensor};
use hw6::*;
use test_support::{as_f32_vec, assert_f32_close, assert_f32_slice_close, json, python_json};

// ---------- small helpers ----------

fn to_vec_f32(t: &Tensor) -> Result<Vec<f32>> {
    t.flatten_all()?.to_vec1::<f32>()
}

fn to_vec_i32(t: &Tensor) -> Result<Vec<i32>> {
    t.flatten_all()?.to_vec1::<i32>()
}

// ---------- PyTorch ground-truth helpers ----------

fn torch_log_probs(logits: Vec<f32>, shape: [usize; 3], y: Vec<i32>, mask: Vec<f32>) -> Vec<f32> {
    as_f32_vec(python_json(
        r#"
import json
import sys
import torch

d = json.load(sys.stdin)
logits = torch.tensor(d["logits"], dtype=torch.float32).reshape(d["shape"])
y = torch.tensor(d["y"], dtype=torch.long).reshape(d["shape"][:2])
mask = torch.tensor(d["mask"], dtype=torch.bool).reshape(d["shape"][:2])
logp = logits.log_softmax(dim=-1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
json.dump((logp * mask).sum(dim=1).tolist(), sys.stdout)
"#,
        json!({ "logits": logits, "shape": shape, "y": y, "mask": mask }),
    ))
}

fn torch_softplus(x: Vec<f32>, beta: f64) -> Vec<f32> {
    as_f32_vec(python_json(
        r#"
import json
import sys
import torch

d = json.load(sys.stdin)
x = torch.tensor(d["x"], dtype=torch.float32)
out = torch.logaddexp(torch.zeros_like(x), x * d["beta"])
json.dump(out.tolist(), sys.stdout)
"#,
        json!({ "x": x, "beta": beta }),
    ))
}

// ============================================================
// Part I: Chat Format and SFT
// ============================================================

// ---------- messages_to_chat_format ----------

#[test]
fn test_convert_to_chat_format() {
    let messages = vec![
        ("user".to_string(), "Hello".to_string()),
        ("assistant".to_string(), "Hi!".to_string()),
        ("user".to_string(), "Bye".to_string()),
    ];
    let expected = "<USER>Hello</USER><ASSISTANT>Hi!</ASSISTANT><USER>Bye</USER>";
    assert_eq!(messages_to_chat_format(&messages), expected);
}

#[test]
fn test_convert_to_chat_format_multiline() {
    let messages = vec![
        ("user".to_string(), "Line one.\nLine two.".to_string()),
        ("assistant".to_string(), "Tabbed\tresponse.".to_string()),
    ];
    let expected = "<USER>Line one.\nLine two.</USER><ASSISTANT>Tabbed\tresponse.</ASSISTANT>";
    assert_eq!(messages_to_chat_format(&messages), expected);
}

#[test]
fn test_messages_to_chat_format_empty() {
    let messages: Vec<(String, String)> = vec![];
    let result = messages_to_chat_format(&messages);
    assert_eq!(result, "");
}

#[test]
fn test_messages_to_chat_format_single_user() {
    let messages = vec![("user".to_string(), "hello".to_string())];
    let result = messages_to_chat_format(&messages);
    assert!(result.contains("<USER>hello</USER>"));
    assert!(!result.contains("<ASSISTANT>"));
}

// ---------- pretokenize_chat ----------

#[test]
fn test_pretokenize_chat() {
    let dir = tempfile::tempdir().unwrap();
    let in_path = dir.path().join("chats.json");
    let out_path = dir.path().join("tokens.json");

    let input_json = r#"[
        [{"role": "user", "content": "One"}, {"role": "assistant", "content": "Two"}],
        [{"role": "user", "content": "Three"}]
    ]"#;
    std::fs::write(&in_path, input_json).unwrap();

    let encode_fn = |text: &str| -> Vec<u32> { text.chars().map(|c| c as u32).collect() };

    pretokenize_chat(&encode_fn, &in_path, &out_path);

    let output = std::fs::read_to_string(&out_path).unwrap();
    let tokens: Vec<Vec<u32>> = serde_json::from_str(&output).unwrap();

    assert_eq!(tokens.len(), 2);
    assert!(!tokens[0].is_empty());
    assert!(!tokens[1].is_empty());
    assert!(tokens[0].len() > tokens[1].len());
}

// ---------- get_loss_mask ----------

#[test]
fn test_get_loss_mask() {
    let assistant_start: u32 = 93;
    let assistant_end: u32 = 94;
    let tokens = vec![7, assistant_start, 1, 2, assistant_end, 8];
    let expected = vec![false, false, true, true, true, false];
    assert_eq!(
        get_loss_mask(&tokens, assistant_start, assistant_end),
        expected
    );
}

#[test]
fn test_get_loss_mask_multiple_regions() {
    let assistant_start: u32 = 93;
    let assistant_end: u32 = 94;
    let tokens = vec![
        91,
        10,
        assistant_start,
        20,
        21,
        assistant_end,
        assistant_start,
        30,
        assistant_end,
    ];
    let expected = vec![false, false, false, true, true, true, false, true, true];
    assert_eq!(
        get_loss_mask(&tokens, assistant_start, assistant_end),
        expected
    );
}

#[test]
fn test_get_loss_mask_no_assistant() {
    let assistant_start: u32 = 93;
    let assistant_end: u32 = 94;
    let tokens = vec![91, 7, 8, 92, 9];
    let expected = vec![false, false, false, false, false];
    assert_eq!(
        get_loss_mask(&tokens, assistant_start, assistant_end),
        expected
    );
}

#[test]
fn test_get_loss_mask_empty_assistant() {
    let assistant_start = 100u32;
    let assistant_end = 101u32;
    let tokens = vec![50, 51, 52, assistant_start, assistant_end, 53];
    let mask = get_loss_mask(&tokens, assistant_start, assistant_end);
    assert!(!mask[0]);
    assert!(!mask[1]);
    assert!(!mask[2]);
    assert!(!mask[3]); // assistant_start itself
    assert!(mask[4]); // assistant_end (included)
    assert!(!mask[5]);
}

// ---------- DataLoaderChat ----------

#[test]
fn test_dataloader_chat() -> Result<()> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chat_tokens.json");

    let assistant_start: u32 = 93;
    let assistant_end: u32 = 94;

    let chats: Vec<Vec<u32>> = vec![
        vec![1, assistant_start, 10, 11, assistant_end],
        vec![2, assistant_start, 12, assistant_end],
        vec![3, assistant_start, 13, 14, 15, assistant_end],
        vec![4, assistant_start, 16, assistant_end],
    ];

    std::fs::write(&path, serde_json::to_string(&chats).unwrap()).unwrap();

    let loader = DataLoaderChat::new(&path, 6, 2);
    let batches: Vec<_> = loader.collect();

    assert_eq!(batches.len(), 2);

    let (x0, y0, m0) = &batches[0];
    assert_eq!(x0.dims()[0], 2);
    assert_eq!(x0.dims()[1], 6);
    assert_eq!(y0.dims()[0], 2);
    assert_eq!(y0.dims()[1], 6);
    assert_eq!(m0.dims()[0], 2);
    assert_eq!(m0.dims()[1], 6);
    Ok(())
}

#[test]
fn test_dataloader_chat_reiterable() -> Result<()> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chat_tokens2.json");

    let chats: Vec<Vec<u32>> = vec![vec![1, 93, 10, 11, 94], vec![2, 93, 12, 94]];
    std::fs::write(&path, serde_json::to_string(&chats).unwrap()).unwrap();

    let loader1 = DataLoaderChat::new(&path, 6, 2);
    let batches1: Vec<_> = loader1.collect();

    let loader2 = DataLoaderChat::new(&path, 6, 2);
    let batches2: Vec<_> = loader2.collect();

    assert_eq!(batches1.len(), batches2.len());
    for ((x1, _y1, _m1), (x2, _y2, _m2)) in batches1.iter().zip(batches2.iter()) {
        let x1v = to_vec_i32(x1)?;
        let x2v = to_vec_i32(x2)?;
        assert_eq!(x1v, x2v);
    }
    Ok(())
}

// ---------- train_chat_sft ----------

#[test]
fn test_train_chat_sft() -> Result<()> {
    let device = Device::Cpu;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty_chats.json");
    std::fs::write(&path, "[]").unwrap();

    let mut loader = DataLoaderChat::new(&path, 6, 2);

    let model_fn = |tokens: &Tensor| -> Result<Tensor> {
        let batch = tokens.dims()[0];
        let seq = tokens.dims()[1];
        Tensor::zeros((batch, seq, 5), DType::F32, &device)
    };
    let mut optimizer_fn = || {};

    train_chat_sft(&model_fn, &mut loader, &mut optimizer_fn, Some(0));
    Ok(())
}

// ============================================================
// Part II: DPO
// ============================================================

// ---------- log_probs ----------

#[test]
fn test_log_probs() -> Result<()> {
    let device = Device::Cpu;
    let logits_data = vec![
        2.0f32, 0.0, -1.0, 0.5, 1.5, -0.5, 1.0, -1.0, 0.0, -0.5, 1.0, 0.0, 2.0, 0.0, -2.0, 0.25,
        0.25, 0.25,
    ];
    let y_data = vec![0i32, 1, 2, 1, 0, 2];
    let mask_data = vec![1.0f32, 0.0, 1.0, 1.0, 1.0, 0.0];
    let logits = Tensor::from_vec(logits_data.clone(), (2, 3, 3), &device)?;
    let y = Tensor::from_vec(y_data.clone(), (2, 3), &device)?;
    let mask = Tensor::from_vec(mask_data.clone(), (2, 3), &device)?;

    let out = log_probs(&logits, &y, &mask)?;
    assert_eq!(out.dims(), &[2]);

    let vals = to_vec_f32(&out)?;
    let expected = torch_log_probs(logits_data, [2, 3, 3], y_data, mask_data);
    assert_f32_slice_close(&vals, &expected, 1e-4);
    Ok(())
}

#[test]
fn test_log_probs_all_masked() -> Result<()> {
    let device = Device::Cpu;
    let logits = Tensor::randn(0f32, 1f32, (2, 3, 5), &device)?;
    let y = Tensor::from_vec(vec![0i32, 1, 2, 3, 4, 0], (2, 3), &device)?;
    let mask = Tensor::from_vec(vec![0.0f32; 6], (2, 3), &device)?;
    let lp = log_probs(&logits, &y, &mask)?;
    let vals = to_vec_f32(&lp)?;
    let expected = torch_log_probs(
        vec![0.0; 2 * 3 * 5],
        [2, 3, 5],
        vec![0, 1, 2, 3, 4, 0],
        vec![0.0; 6],
    );
    assert_f32_slice_close(&vals, &expected, 1e-6);
    Ok(())
}

// ---------- softplus ----------

#[test]
fn test_softplus() -> Result<()> {
    let device = Device::Cpu;
    let x_data = vec![-3.0f32, -0.5, 0.0, 2.0];
    let beta = 0.7;
    let x = Tensor::from_vec(x_data.clone(), 4, &device)?;
    let out = softplus(&x, beta)?;
    let expected = torch_softplus(x_data, beta);
    let vals = to_vec_f32(&out)?;
    assert_f32_slice_close(&vals, &expected, 1e-4);
    Ok(())
}

#[test]
fn test_softplus_2d() -> Result<()> {
    let device = Device::Cpu;
    let x_data = vec![1.0f32, -1.0, 0.25, -0.25];
    let x = Tensor::from_vec(x_data.clone(), 4, &device)?;
    let beta = 0.3;
    let out = softplus(&x, beta)?;
    let vals = to_vec_f32(&out)?;
    let expected = torch_softplus(x_data, beta);
    assert_f32_slice_close(&vals, &expected, 1e-5);
    Ok(())
}

#[test]
fn test_softplus_large_negative() -> Result<()> {
    let device = Device::Cpu;
    let x_data = vec![-100.0f32];
    let beta = 1.0;
    let x = Tensor::from_vec(x_data.clone(), 1, &device)?;
    let out: f32 = softplus(&x, beta)?.to_scalar::<f32>()?;
    let expected = torch_softplus(x_data, beta)[0];
    assert!(out.is_finite());
    assert_f32_close(out, expected, 1e-5);
    Ok(())
}

#[test]
fn test_softplus_large_positive() -> Result<()> {
    let device = Device::Cpu;
    let x_data = vec![100.0f32];
    let beta = 0.5;
    let x = Tensor::from_vec(x_data.clone(), 1, &device)?;
    let out: f32 = softplus(&x, beta)?.to_scalar::<f32>()?;
    let expected = torch_softplus(x_data, beta)[0];
    assert!(out.is_finite());
    assert_f32_close(out, expected, 1e-3);
    Ok(())
}

#[test]
fn test_softplus_at_zero() -> Result<()> {
    let device = Device::Cpu;
    let x_data = vec![0.0f32];
    let beta = 1.0;
    let x = Tensor::from_vec(x_data.clone(), 1, &device)?;
    let out: f32 = softplus(&x, beta)?.to_scalar::<f32>()?;
    let expected = torch_softplus(x_data, beta)[0];
    assert_f32_close(out, expected, 1e-5);
    Ok(())
}

// ---------- dpo_loss ----------

#[test]
fn test_dpo_loss_shape() -> Result<()> {
    let device = Device::Cpu;
    let model = |_x: &Tensor| -> Result<Tensor> { Tensor::zeros((2, 3, 3), DType::F32, &device) };
    let model_ref =
        |_x: &Tensor| -> Result<Tensor> { Tensor::zeros((2, 3, 3), DType::F32, &device) };

    let xp = Tensor::from_vec(vec![0i32, 1, 2, 1, 2, 0], (2, 3), &device)?;
    let yp = Tensor::from_vec(vec![1i32, 2, 0, 2, 0, 1], (2, 3), &device)?;
    let maskp = Tensor::from_vec(vec![1.0f32, 1.0, 0.0, 0.0, 1.0, 1.0], (2, 3), &device)?;
    let xn = Tensor::from_vec(vec![0i32, 2, 1, 2, 1, 0], (2, 3), &device)?;
    let yn = Tensor::from_vec(vec![2i32, 0, 1, 1, 0, 2], (2, 3), &device)?;
    let maskn = Tensor::from_vec(vec![1.0f32, 0.0, 1.0, 1.0, 1.0, 0.0], (2, 3), &device)?;

    let loss = dpo_loss(&model, &model_ref, &xp, &yp, &maskp, &xn, &yn, &maskn, 0.3)?;
    assert_eq!(loss.dims(), &[2]);
    let vals = to_vec_f32(&loss)?;
    let expected = torch_softplus(vec![0.0; vals.len()], 0.3);
    assert_f32_slice_close(&vals, &expected, 1e-5);
    for v in &vals {
        assert!(v.is_finite(), "DPO loss is not finite");
        assert!(
            *v >= 0.0,
            "DPO loss should be non-negative (softplus output)"
        );
    }
    Ok(())
}

// ============================================================
// End-to-end: eval_llm_chat and eval_llm_dpo
// ============================================================

#[test]
fn test_eval_llm_chat() -> Result<()> {
    let device = Device::Cpu;
    let mut model_fn = eval_llm_chat()?;

    let tokens = Tensor::from_vec(vec![0u32, 1, 2, 3], (1, 4), &device)?;
    let full = model_fn(&tokens, 0, false)?;

    assert_eq!(full.dims()[0], 1);
    assert_eq!(full.dims()[1], 4);
    let vocab_size = full.dims()[2];
    assert!(vocab_size > 0);

    // Outputs should be finite.
    let first_logits = to_vec_f32(&full.narrow(2, 0, 16.min(vocab_size))?)?;
    for v in &first_logits {
        assert!(v.is_finite(), "eval_llm_chat has non-finite logits");
    }

    // KV cache consistency.
    let mut model_fn2 = eval_llm_chat()?;
    let _prefix = model_fn2(&tokens.narrow(1, 0, 3)?, 0, true)?;
    let tail = model_fn2(&tokens.narrow(1, 3, 1)?, 3, true)?;

    let full_last = to_vec_f32(&full.narrow(1, 3, 1)?.reshape(vocab_size)?)?;
    let tail_vec = to_vec_f32(&tail.reshape(vocab_size)?)?;
    let max_diff: f32 = full_last
        .iter()
        .zip(tail_vec.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 3e-4,
        "eval_llm_chat KV cache mismatch: max_diff={max_diff}"
    );
    Ok(())
}

#[test]
fn test_eval_llm_dpo() -> Result<()> {
    let device = Device::Cpu;
    let mut model_fn = eval_llm_dpo()?;

    let tokens = Tensor::from_vec(vec![0u32, 1, 2, 3], (1, 4), &device)?;
    let full = model_fn(&tokens, 0, false)?;

    assert_eq!(full.dims()[0], 1);
    assert_eq!(full.dims()[1], 4);
    let vocab_size = full.dims()[2];
    assert!(vocab_size > 0);

    // Outputs should be finite.
    let first_logits = to_vec_f32(&full.narrow(2, 0, 16.min(vocab_size))?)?;
    for v in &first_logits {
        assert!(v.is_finite(), "eval_llm_dpo has non-finite logits");
    }

    // KV cache consistency.
    let mut model_fn2 = eval_llm_dpo()?;
    let _prefix = model_fn2(&tokens.narrow(1, 0, 3)?, 0, true)?;
    let tail = model_fn2(&tokens.narrow(1, 3, 1)?, 3, true)?;

    let full_last = to_vec_f32(&full.narrow(1, 3, 1)?.reshape(vocab_size)?)?;
    let tail_vec = to_vec_f32(&tail.reshape(vocab_size)?)?;
    let max_diff: f32 = full_last
        .iter()
        .zip(tail_vec.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 3e-4,
        "eval_llm_dpo KV cache mismatch: max_diff={max_diff}"
    );
    Ok(())
}
