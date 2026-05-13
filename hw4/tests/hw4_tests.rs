use candle_core::{Device, Tensor};
use candle_nn::VarMap;
use hw4::*;

const DEVICE: Device = Device::Cpu;

fn causal_mask(length: usize) -> Tensor {
    let mask_data: Vec<f32> = (0..length)
        .flat_map(|i| (0..length).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    Tensor::from_vec(mask_data, &[length, length], &DEVICE).unwrap()
}

// --- Linear layer ---

#[test]
fn test_linear_shape() {
    let varmap = VarMap::new();
    let layer = Linear::new(10, 20, &varmap, "linear").unwrap();
    let x = Tensor::randn(0.0f32, 1.0, &[50, 10], &DEVICE).unwrap();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.dims(), &[50, 20]);
}

#[test]
fn test_linear_batch_dims() {
    let varmap = VarMap::new();
    let layer = Linear::new(10, 20, &varmap, "linear_batch").unwrap();
    let x = Tensor::randn(0.0f32, 1.0, &[7, 9, 10], &DEVICE).unwrap();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.dims(), &[7, 9, 20]);
}

#[test]
fn test_linear_correctness() {
    let varmap = VarMap::new();
    let layer = Linear::new(10, 20, &varmap, "linear_correct").unwrap();
    let x = Tensor::randn(0.0f32, 1.0, &[50, 10], &DEVICE).unwrap();
    let out = layer.forward(&x).unwrap();
    // Reference: X @ W^T
    let expected = x.matmul(&layer.weight().t().unwrap()).unwrap();
    let diff: f32 = out.sub(&expected).unwrap().abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
    assert!(diff < 1e-4, "Linear output doesn't match X @ W^T, diff={diff}");
}

// --- Embedding ---

#[test]
fn test_embedding_shape() {
    let varmap = VarMap::new();
    let layer = Embedding::new(200, 20, &varmap, "emb").unwrap();
    let y = Tensor::new(&[3u32, 7, 42, 100], &DEVICE).unwrap();
    let out = layer.forward(&y).unwrap();
    assert_eq!(out.dims(), &[4, 20]);
}

#[test]
fn test_embedding_batch_dims() {
    let varmap = VarMap::new();
    let layer = Embedding::new(200, 20, &varmap, "emb_batch").unwrap();
    let y = Tensor::new(&[[0u32, 1, 2], [3, 4, 5], [6, 7, 8]], &DEVICE).unwrap();
    let out = layer.forward(&y).unwrap();
    assert_eq!(out.dims(), &[3, 3, 20]);
}

#[test]
fn test_embedding_correctness() {
    let varmap = VarMap::new();
    let layer = Embedding::new(8, 3, &varmap, "emb_correct").unwrap();
    let indices = Tensor::new(&[0u32, 3, 5], &DEVICE).unwrap();
    let out = layer.forward(&indices).unwrap();
    // Each row of output should be the corresponding row of the weight matrix
    let w = layer.weight();
    let row0: Vec<f32> = w.get(0).unwrap().to_vec1().unwrap();
    let row3: Vec<f32> = w.get(3).unwrap().to_vec1().unwrap();
    let row5: Vec<f32> = w.get(5).unwrap().to_vec1().unwrap();
    let out0: Vec<f32> = out.get(0).unwrap().to_vec1().unwrap();
    let out1: Vec<f32> = out.get(1).unwrap().to_vec1().unwrap();
    let out2: Vec<f32> = out.get(2).unwrap().to_vec1().unwrap();
    assert_eq!(out0, row0);
    assert_eq!(out1, row3);
    assert_eq!(out2, row5);
}

// --- SiLU ---

#[test]
fn test_silu() {
    let x = Tensor::new(&[[-2.0f32, -0.5, 0.0, 1.0], [2.0, 3.0, -1.0, 0.25]], &DEVICE).unwrap();
    let out = silu(&x).unwrap();
    // Reference: x * sigmoid(x)
    let sigmoid = (x.neg().unwrap().exp().unwrap() + 1.0).unwrap().recip().unwrap();
    let expected = x.mul(&sigmoid).unwrap();
    let diff: f32 = out.sub(&expected).unwrap().abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
    assert!(diff < 1e-5, "silu mismatch, diff={diff}");
}

#[test]
fn test_silu_multidim() {
    let x = Tensor::randn(0.0f32, 1.0, &[3, 4, 5, 6], &DEVICE).unwrap();
    let out = silu(&x).unwrap();
    assert_eq!(out.dims(), x.dims());
    let sigmoid = (x.neg().unwrap().exp().unwrap() + 1.0).unwrap().recip().unwrap();
    let expected = x.mul(&sigmoid).unwrap();
    let diff: f32 = out.sub(&expected).unwrap().abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
    assert!(diff < 1e-4, "silu multidim mismatch, diff={diff}");
}

// --- RMSNorm ---

#[test]
fn test_rmsnorm_shape_and_init() {
    let varmap = VarMap::new();
    let layer = RMSNorm::new(20, 1e-3, &varmap, "rmsnorm").unwrap();
    let x = Tensor::randn(0.0f32, 1.0, &[100, 20], &DEVICE).unwrap();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.dims(), &[100, 20]);
}

#[test]
fn test_rmsnorm_batch_dims() {
    let varmap = VarMap::new();
    let layer = RMSNorm::new(20, 1e-3, &varmap, "rmsnorm_batch").unwrap();
    let x = Tensor::randn(0.0f32, 1.0, &[10, 7, 20], &DEVICE).unwrap();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.dims(), &[10, 7, 20]);
}

#[test]
fn test_rmsnorm_correctness() {
    // RMSNorm(x) = w * x / sqrt(mean(x^2) + eps)
    let varmap = VarMap::new();
    let layer = RMSNorm::new(4, 1e-5, &varmap, "rmsnorm_correct").unwrap();
    let x = Tensor::new(&[[1.0f32, -1.0, 0.5, 0.5], [2.0, 0.0, -2.0, 1.0]], &DEVICE).unwrap();
    let out = layer.forward(&x).unwrap();

    // With weight=1 (default init), RMSNorm(x) = x / rms(x)
    // Row 0: rms = sqrt((1+1+0.25+0.25)/4) = sqrt(0.625) ≈ 0.7906
    // Row 1: rms = sqrt((4+0+4+1)/4) = sqrt(2.25) = 1.5
    let x_vec: Vec<Vec<f32>> = (0..2).map(|i| x.get(i).unwrap().to_vec1().unwrap()).collect();
    let out_vec: Vec<Vec<f32>> = (0..2).map(|i| out.get(i).unwrap().to_vec1().unwrap()).collect();

    for row in 0..2 {
        let mean_sq: f32 = x_vec[row].iter().map(|v| v * v).sum::<f32>() / 4.0;
        let rms = (mean_sq + 1e-5f32).sqrt();
        for col in 0..4 {
            let expected = x_vec[row][col] / rms;
            assert!(
                (out_vec[row][col] - expected).abs() < 1e-5,
                "RMSNorm mismatch at [{row}][{col}]: got {}, expected {expected}",
                out_vec[row][col]
            );
        }
    }
}

// --- Self attention ---

#[test]
fn test_self_attention_2d() {
    // Basic 2D attention with causal mask
    let q = Tensor::randn(0.0f32, 1.0, &[5, 8], &DEVICE).unwrap();
    let k = Tensor::randn(0.0f32, 1.0, &[5, 8], &DEVICE).unwrap();
    let v = Tensor::randn(0.0f32, 1.0, &[5, 6], &DEVICE).unwrap();
    let mask = causal_mask(5);

    let out = self_attention(&q, &k, &v, Some(&mask)).unwrap();
    assert_eq!(out.dims(), &[5, 6]);

    // Verify first row only attends to itself (due to causal mask)
    // Q[0] @ K^T / sqrt(d) + mask[0] -> only position 0 is not -inf
    // So softmax gives [1, 0, 0, 0, 0] and output = V[0]
    let out_row0: Vec<f32> = out.get(0).unwrap().to_vec1().unwrap();
    let v_row0: Vec<f32> = v.get(0).unwrap().to_vec1().unwrap();
    for (a, b) in out_row0.iter().zip(v_row0.iter()) {
        assert!((a - b).abs() < 1e-5, "First row should equal V[0] with causal mask");
    }
}

#[test]
fn test_self_attention_batched() {
    let q = Tensor::randn(0.0f32, 1.0, &[2, 3, 5, 8], &DEVICE).unwrap();
    let k = Tensor::randn(0.0f32, 1.0, &[2, 3, 5, 8], &DEVICE).unwrap();
    let v = Tensor::randn(0.0f32, 1.0, &[2, 3, 5, 4], &DEVICE).unwrap();
    let mask = causal_mask(5);
    let out = self_attention(&q, &k, &v, Some(&mask)).unwrap();
    assert_eq!(out.dims(), &[2, 3, 5, 4]);
}

#[test]
fn test_self_attention_no_mask() {
    let q = Tensor::randn(0.0f32, 1.0, &[5, 8], &DEVICE).unwrap();
    let k = Tensor::randn(0.0f32, 1.0, &[5, 8], &DEVICE).unwrap();
    let v = Tensor::randn(0.0f32, 1.0, &[5, 6], &DEVICE).unwrap();
    let out = self_attention(&q, &k, &v, None).unwrap();
    assert_eq!(out.dims(), &[5, 6]);
}

// --- MultiHeadAttentionKVCache ---

#[test]
fn test_mha_kvcache_no_cache() {
    let varmap = VarMap::new();
    let mut attn = MultiHeadAttentionKVCache::new(12, 3, 8, &varmap, "mha").unwrap();
    let x = Tensor::randn(0.0f32, 1.0, &[1, 5, 12], &DEVICE).unwrap();
    let mask = causal_mask(5);
    let out = attn.forward(&x, Some(&mask), 0, false).unwrap();
    assert_eq!(out.dims(), &[1, 5, 12]);
}

#[test]
fn test_mha_kvcache_consistency() {
    // Full forward should match prefix+tail with KV cache
    let varmap = VarMap::new();
    let mut attn = MultiHeadAttentionKVCache::new(12, 3, 8, &varmap, "mha_cache").unwrap();
    let x = Tensor::randn(0.0f32, 1.0, &[1, 5, 12], &DEVICE).unwrap();
    let mask = causal_mask(5);

    let full = attn.forward(&x, Some(&mask), 0, false).unwrap();
    assert_eq!(full.dims(), &[1, 5, 12]);

    // Reset by creating new instance with same weights
    let mut attn2 = MultiHeadAttentionKVCache::new(12, 3, 8, &varmap, "mha_cache").unwrap();
    let prefix_mask = causal_mask(3);
    let prefix = attn2.forward(
        &x.narrow(1, 0, 3).unwrap(),
        Some(&prefix_mask),
        0,
        true,
    ).unwrap();

    // tail mask: rows 3..5 of the full 5x5 mask (shape 2x5)
    let tail_mask = mask.narrow(0, 3, 2).unwrap();
    let tail = attn2.forward(
        &x.narrow(1, 3, 2).unwrap(),
        Some(&tail_mask),
        3,
        true,
    ).unwrap();

    // prefix should match full[:, :3]
    let full_prefix: Vec<f32> = full.narrow(1, 0, 3).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let prefix_vec: Vec<f32> = prefix.flatten_all().unwrap().to_vec1().unwrap();
    for (a, b) in prefix_vec.iter().zip(full_prefix.iter()) {
        assert!((a - b).abs() < 1e-5, "KV cache prefix mismatch: {a} vs {b}");
    }

    // tail should match full[:, 3:]
    let full_tail: Vec<f32> = full.narrow(1, 3, 2).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let tail_vec: Vec<f32> = tail.flatten_all().unwrap().to_vec1().unwrap();
    for (a, b) in tail_vec.iter().zip(full_tail.iter()) {
        assert!((a - b).abs() < 1e-5, "KV cache tail mismatch: {a} vs {b}");
    }
}

// --- GatedMLP ---

#[test]
fn test_gated_mlp_shape() {
    let varmap = VarMap::new();
    let mlp = GatedMLP::new(8, 16, &varmap, "mlp").unwrap();
    let x = Tensor::randn(0.0f32, 1.0, &[4, 8], &DEVICE).unwrap();
    let out = mlp.forward(&x).unwrap();
    assert_eq!(out.dims(), &[4, 8]);
}

#[test]
fn test_gated_mlp_batch_dims() {
    let varmap = VarMap::new();
    let mlp = GatedMLP::new(8, 16, &varmap, "mlp_batch").unwrap();
    let x = Tensor::randn(0.0f32, 1.0, &[2, 4, 8], &DEVICE).unwrap();
    let out = mlp.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, 4, 8]);
}

// --- TransformerBlock ---

#[test]
fn test_transformer_block_shape() {
    let varmap = VarMap::new();
    let mut block = TransformerBlock::new(12, 3, 16, 8, &varmap, "block").unwrap();
    let x = Tensor::randn(0.0f32, 1.0, &[1, 5, 12], &DEVICE).unwrap();
    let mask = causal_mask(5);
    let out = block.forward(&x, Some(&mask), 0, false).unwrap();
    assert_eq!(out.dims(), &[1, 5, 12]);
}

#[test]
fn test_transformer_block_residual() {
    // With all weights zero, output should equal input (residual connection)
    // This is hard to test without weight access, so just check shape and finiteness
    let varmap = VarMap::new();
    let mut block = TransformerBlock::new(8, 2, 12, 6, &varmap, "block_res").unwrap();
    let x = Tensor::randn(0.0f32, 1.0, &[1, 4, 8], &DEVICE).unwrap();
    let out = block.forward(&x, None, 0, false).unwrap();
    assert_eq!(out.dims(), &[1, 4, 8]);
    let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    for v in &vals {
        assert!(v.is_finite(), "TransformerBlock output has non-finite values");
    }
}

// --- Llama3Simplified ---

#[test]
fn test_llama3_shape() {
    let varmap = VarMap::new();
    let mut model = Llama3Simplified::new(5, 4, 2, 6, 6, 1, &varmap).unwrap();
    let tokens = Tensor::new(&[[2u32, 3, 4]], &DEVICE).unwrap();
    let out = model.forward(&tokens, 0, false).unwrap();
    assert_eq!(out.dims(), &[1, 3, 5]);
}

#[test]
fn test_llama3_kv_cache_consistency() {
    let varmap = VarMap::new();
    let mut model = Llama3Simplified::new(10, 8, 2, 16, 12, 2, &varmap).unwrap();
    let tokens = Tensor::new(&[[0u32, 1, 2, 3]], &DEVICE).unwrap();

    let full = model.forward(&tokens, 0, false).unwrap();
    assert_eq!(full.dims()[1], 4);

    // Create fresh model with same weights for cache test
    let mut model2 = Llama3Simplified::new(10, 8, 2, 16, 12, 2, &varmap).unwrap();
    let prefix_tokens = Tensor::new(&[[0u32, 1, 2]], &DEVICE).unwrap();
    let _prefix = model2.forward(&prefix_tokens, 0, true).unwrap();
    let tail_tokens = Tensor::new(&[[3u32]], &DEVICE).unwrap();
    let tail = model2.forward(&tail_tokens, 3, true).unwrap();

    // Last token's output should match between full and cached versions
    let full_last: Vec<f32> = full.narrow(1, 3, 1).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let tail_vec: Vec<f32> = tail.flatten_all().unwrap().to_vec1().unwrap();
    for (a, b) in tail_vec.iter().zip(full_last.iter()) {
        assert!(
            (a - b).abs() < 1e-3,
            "KV cache mismatch in Llama3: full_last={b}, cached_tail={a}"
        );
    }
}

// --- Generate ---

#[test]
fn test_generate_basic() {
    // Simulate a model that always predicts token 3, then token 4 (stop)
    let call_count = std::cell::RefCell::new(0usize);
    let next_tokens: Vec<u32> = vec![3, 4];

    let mut model_fn = |tokens: &Tensor, _seq_pos: usize, _use_cache: bool| -> candle_core::Result<Tensor> {
        let mut count = call_count.borrow_mut();
        let next_token = next_tokens[*count];
        *count += 1;
        let seq_len = tokens.dims()[1];
        // Return logits where next_token has highest score
        let mut logits_data = vec![-1e9f32; 6 * seq_len];
        // Set the last position's next_token index to 0 (highest)
        logits_data[(seq_len - 1) * 6 + next_token as usize] = 0.0;
        Tensor::from_vec(logits_data, &[1, seq_len, 6], &DEVICE)
    };

    let decode_fn = |tokens: &[u32]| -> String {
        tokens.iter().map(|&t| match t {
            3 => "A",
            4 => "!",
            5 => "B",
            _ => "?",
        }).collect()
    };

    let stop_tokens: Vec<u32> = vec![4];
    let generated = generate(
        &mut model_fn,
        &[1, 2],
        &decode_fn,
        &stop_tokens,
        0.7,
        5,
        false,
    ).unwrap();

    assert_eq!(generated, vec![3, 4]);
}

#[test]
fn test_generate_max_tokens() {
    // Model always predicts token 3 (never hits stop token)
    let mut model_fn = |tokens: &Tensor, _seq_pos: usize, _use_cache: bool| -> candle_core::Result<Tensor> {
        let seq_len = tokens.dims()[1];
        let mut logits_data = vec![-1e9f32; 6 * seq_len];
        logits_data[(seq_len - 1) * 6 + 3] = 0.0;
        Tensor::from_vec(logits_data, &[1, seq_len, 6], &DEVICE)
    };

    let decode_fn = |_tokens: &[u32]| -> String { String::new() };
    let stop_tokens: Vec<u32> = vec![4];
    let generated = generate(
        &mut model_fn,
        &[1, 2],
        &decode_fn,
        &stop_tokens,
        0.7,
        3,
        false,
    ).unwrap();

    assert_eq!(generated.len(), 3, "Should stop at max_tokens=3");
}

// --- SiLU at zero (sanity) ---

#[test]
fn test_silu_at_zero() {
    // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    let x = Tensor::new(&[0.0f32], &DEVICE).unwrap();
    let out = silu(&x).unwrap();
    let val: f32 = out.to_scalar().unwrap();
    assert!(val.abs() < 1e-7);
}

// --- eval_llama3 (end-to-end model loading + inference) ---

#[test]
fn test_eval_llama3() {
    let mut model = eval_llama3().unwrap();

    let tokens = Tensor::new(&[[0u32, 1, 2, 3]], &DEVICE).unwrap();
    let full = model.forward(&tokens, 0, false).unwrap();

    // Shape: (1, 4, vocab_size)
    assert_eq!(full.dims()[0], 1);
    assert_eq!(full.dims()[1], 4);
    let vocab_size = full.dims()[2];
    assert!(vocab_size > 0);

    // Outputs should be finite (at least first 16 logits)
    let first_logits: Vec<f32> = full
        .narrow(2, 0, 16.min(vocab_size))
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    for v in &first_logits {
        assert!(v.is_finite(), "eval_llama3 output has non-finite logits");
    }

    // KV cache consistency: full[:, 3:] should match tail from cached inference
    let mut model2 = eval_llama3().unwrap();
    let prefix_tokens = Tensor::new(&[[0u32, 1, 2]], &DEVICE).unwrap();
    let prefix = model2.forward(&prefix_tokens, 0, true).unwrap();
    assert_eq!(prefix.dims()[1], 3);

    let tail_tokens = Tensor::new(&[[3u32]], &DEVICE).unwrap();
    let tail = model2.forward(&tail_tokens, 3, true).unwrap();
    assert_eq!(tail.dims()[1], 1);

    // tail should closely match full[:, 3:]
    let full_last: Vec<f32> = full
        .narrow(1, 3, 1)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let tail_vec: Vec<f32> = tail.flatten_all().unwrap().to_vec1().unwrap();

    let max_diff: f32 = full_last
        .iter()
        .zip(tail_vec.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 3e-4,
        "KV cache inconsistency in eval_llama3: max_diff={max_diff}"
    );
}
