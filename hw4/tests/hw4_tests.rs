use burn::backend::ndarray::NdArrayDevice;
use burn::tensor::{Distribution, Int, Tensor, TensorData};
use hw4::*;

const DEVICE: NdArrayDevice = NdArrayDevice::Cpu;

fn causal_mask(length: usize) -> Tensor<B, 2> {
    let mask_data: Vec<f32> = (0..length)
        .flat_map(|i| (0..length).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    Tensor::<B, 2>::from_data(TensorData::new(mask_data, [length, length]), &DEVICE)
}

// --- Linear layer ---

#[test]
fn test_linear_shape() {
    let layer = Linear::new(10, 20, &DEVICE);
    let x: Tensor<B, 2> = Tensor::random([50, 10], Distribution::Normal(0.0, 1.0), &DEVICE);
    let out = layer.forward(x);
    assert_eq!(out.dims(), [50, 20]);
}

#[test]
fn test_linear_batch_dims() {
    let layer = Linear::new(10, 20, &DEVICE);
    let x: Tensor<B, 3> = Tensor::random([7, 9, 10], Distribution::Normal(0.0, 1.0), &DEVICE);
    let out = layer.forward(x);
    assert_eq!(out.dims(), [7, 9, 20]);
}

#[test]
fn test_linear_correctness() {
    let layer = Linear::new(10, 20, &DEVICE);
    let x: Tensor<B, 2> = Tensor::random([50, 10], Distribution::Normal(0.0, 1.0), &DEVICE);
    let out = layer.forward(x.clone());
    // Reference: X @ W^T
    let expected = x.matmul(layer.weight().clone().transpose());
    let diff: f32 = (out - expected)
        .abs()
        .sum()
        .into_data()
        .to_vec::<f32>()
        .unwrap()[0];
    assert!(
        diff < 1e-4,
        "Linear output doesn't match X @ W^T, diff={diff}"
    );
}

// --- Embedding ---

#[test]
fn test_embedding_shape() {
    let layer = Embedding::new(200, 20, &DEVICE);
    let y: Tensor<B, 2, Int> =
        Tensor::from_data(TensorData::new(vec![3i32, 7, 42, 100], [1, 4]), &DEVICE);
    let out = layer.forward(y);
    assert_eq!(out.dims(), [1, 4, 20]);
}

#[test]
fn test_embedding_batch_dims() {
    let layer = Embedding::new(200, 20, &DEVICE);
    let y: Tensor<B, 2, Int> = Tensor::from_data(
        TensorData::new(vec![0i32, 1, 2, 3, 4, 5, 6, 7, 8], [3, 3]),
        &DEVICE,
    );
    let out = layer.forward(y);
    assert_eq!(out.dims(), [3, 3, 20]);
}

#[test]
fn test_embedding_correctness() {
    let layer = Embedding::new(8, 3, &DEVICE);
    let indices: Tensor<B, 2, Int> =
        Tensor::from_data(TensorData::new(vec![0i32, 3, 5], [1, 3]), &DEVICE);
    let out = layer.forward(indices);
    // Each row of output should be the corresponding row of the weight matrix
    let w = layer.weight().clone();
    let row0: Vec<f32> = w
        .clone()
        .narrow(0, 0, 1)
        .reshape([3])
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let row3: Vec<f32> = w
        .clone()
        .narrow(0, 3, 1)
        .reshape([3])
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let row5: Vec<f32> = w
        .narrow(0, 5, 1)
        .reshape([3])
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let out_squeezed: Tensor<B, 2> = out.reshape([3, 3]);
    let out0: Vec<f32> = out_squeezed
        .clone()
        .narrow(0, 0, 1)
        .reshape([3])
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let out1: Vec<f32> = out_squeezed
        .clone()
        .narrow(0, 1, 1)
        .reshape([3])
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let out2: Vec<f32> = out_squeezed
        .narrow(0, 2, 1)
        .reshape([3])
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(out0, row0);
    assert_eq!(out1, row3);
    assert_eq!(out2, row5);
}

// --- SiLU ---

#[test]
fn test_silu() {
    let x: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(vec![-2.0f32, -0.5, 0.0, 1.0, 2.0, 3.0, -1.0, 0.25], [2, 4]),
        &DEVICE,
    );
    let out = silu(x.clone());
    // Reference: x * sigmoid(x)
    let sigmoid = (x.clone().neg().exp() + 1.0).powf_scalar(-1.0);
    let expected = x * sigmoid;
    let diff: f32 = (out - expected)
        .abs()
        .sum()
        .into_data()
        .to_vec::<f32>()
        .unwrap()[0];
    assert!(diff < 1e-5, "silu mismatch, diff={diff}");
}

#[test]
fn test_silu_multidim() {
    let x: Tensor<B, 4> = Tensor::random([3, 4, 5, 6], Distribution::Normal(0.0, 1.0), &DEVICE);
    let out = silu(x.clone());
    assert_eq!(out.dims(), x.dims());
    let sigmoid = (x.clone().neg().exp() + 1.0).powf_scalar(-1.0);
    let expected = x * sigmoid;
    let diff: f32 = (out - expected)
        .abs()
        .sum()
        .into_data()
        .to_vec::<f32>()
        .unwrap()[0];
    assert!(diff < 1e-4, "silu multidim mismatch, diff={diff}");
}

// --- RMSNorm ---

#[test]
fn test_rmsnorm_shape_and_init() {
    let layer = RMSNorm::new(20, 1e-3, &DEVICE);
    let x: Tensor<B, 2> = Tensor::random([100, 20], Distribution::Normal(0.0, 1.0), &DEVICE);
    let out = layer.forward(x);
    assert_eq!(out.dims(), [100, 20]);
}

#[test]
fn test_rmsnorm_batch_dims() {
    let layer = RMSNorm::new(20, 1e-3, &DEVICE);
    let x: Tensor<B, 3> = Tensor::random([10, 7, 20], Distribution::Normal(0.0, 1.0), &DEVICE);
    let out = layer.forward(x);
    assert_eq!(out.dims(), [10, 7, 20]);
}

#[test]
fn test_rmsnorm_correctness() {
    // RMSNorm(x) = w * x / sqrt(mean(x^2) + eps)
    let layer = RMSNorm::new(4, 1e-5, &DEVICE);
    let x: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(vec![1.0f32, -1.0, 0.5, 0.5, 2.0, 0.0, -2.0, 1.0], [2, 4]),
        &DEVICE,
    );
    let out = layer.forward(x.clone());

    // With weight=1 (default init), RMSNorm(x) = x / rms(x)
    // Row 0: rms = sqrt((1+1+0.25+0.25)/4) = sqrt(0.625) ~ 0.7906
    // Row 1: rms = sqrt((4+0+4+1)/4) = sqrt(2.25) = 1.5
    let x_vec: Vec<f32> = x.into_data().to_vec::<f32>().unwrap();
    let out_vec: Vec<f32> = out.into_data().to_vec::<f32>().unwrap();

    for row in 0..2 {
        let mean_sq: f32 = (0..4)
            .map(|c| x_vec[row * 4 + c] * x_vec[row * 4 + c])
            .sum::<f32>()
            / 4.0;
        let rms = (mean_sq + 1e-5f32).sqrt();
        for col in 0..4 {
            let expected = x_vec[row * 4 + col] / rms;
            let got = out_vec[row * 4 + col];
            assert!(
                (got - expected).abs() < 1e-5,
                "RMSNorm mismatch at [{row}][{col}]: got {got}, expected {expected}",
            );
        }
    }
}

// --- Self attention ---

#[test]
fn test_self_attention_2d() {
    // Basic 2D attention with causal mask
    let q: Tensor<B, 2> = Tensor::random([5, 8], Distribution::Normal(0.0, 1.0), &DEVICE);
    let k: Tensor<B, 2> = Tensor::random([5, 8], Distribution::Normal(0.0, 1.0), &DEVICE);
    let v: Tensor<B, 2> = Tensor::random([5, 6], Distribution::Normal(0.0, 1.0), &DEVICE);
    let mask = causal_mask(5);

    let out = self_attention(q, k, v.clone(), Some(mask));
    assert_eq!(out.dims(), [5, 6]);

    // Verify first row only attends to itself (due to causal mask)
    // Q[0] @ K^T / sqrt(d) + mask[0] -> only position 0 is not -inf
    // So softmax gives [1, 0, 0, 0, 0] and output = V[0]
    let out_row0: Vec<f32> = out
        .narrow(0, 0, 1)
        .reshape([6])
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let v_row0: Vec<f32> = v
        .narrow(0, 0, 1)
        .reshape([6])
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    for (a, b) in out_row0.iter().zip(v_row0.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "First row should equal V[0] with causal mask"
        );
    }
}

#[test]
fn test_self_attention_batched() {
    let q: Tensor<B, 4> = Tensor::random([2, 3, 5, 8], Distribution::Normal(0.0, 1.0), &DEVICE);
    let k: Tensor<B, 4> = Tensor::random([2, 3, 5, 8], Distribution::Normal(0.0, 1.0), &DEVICE);
    let v: Tensor<B, 4> = Tensor::random([2, 3, 5, 4], Distribution::Normal(0.0, 1.0), &DEVICE);
    let mask = causal_mask(5);
    let out = self_attention_batched(q, k, v, Some(mask));
    assert_eq!(out.dims(), [2, 3, 5, 4]);
}

#[test]
fn test_self_attention_no_mask() {
    let q: Tensor<B, 2> = Tensor::random([5, 8], Distribution::Normal(0.0, 1.0), &DEVICE);
    let k: Tensor<B, 2> = Tensor::random([5, 8], Distribution::Normal(0.0, 1.0), &DEVICE);
    let v: Tensor<B, 2> = Tensor::random([5, 6], Distribution::Normal(0.0, 1.0), &DEVICE);
    let out = self_attention(q, k, v, None);
    assert_eq!(out.dims(), [5, 6]);
}

// --- MultiHeadAttentionKVCache ---

#[test]
fn test_mha_kvcache_no_cache() {
    let mut attn = MultiHeadAttentionKVCache::new(12, 3, 8, &DEVICE);
    let x: Tensor<B, 3> = Tensor::random([1, 5, 12], Distribution::Normal(0.0, 1.0), &DEVICE);
    let mask = causal_mask(5);
    let out = attn.forward(x, Some(mask), 0, false);
    assert_eq!(out.dims(), [1, 5, 12]);
}

#[test]
fn test_mha_kvcache_consistency() {
    // Full forward should match prefix+tail with KV cache
    let mut attn = MultiHeadAttentionKVCache::new(12, 3, 8, &DEVICE);
    let x: Tensor<B, 3> = Tensor::random([1, 5, 12], Distribution::Normal(0.0, 1.0), &DEVICE);
    let mask = causal_mask(5);

    let full = attn.forward(x.clone(), Some(mask.clone()), 0, false);
    assert_eq!(full.dims(), [1, 5, 12]);

    // Reset by creating new instance with same weights
    let mut attn2 = MultiHeadAttentionKVCache::new(12, 3, 8, &DEVICE);
    let prefix_mask = causal_mask(3);
    let prefix = attn2.forward(x.clone().narrow(1, 0, 3), Some(prefix_mask), 0, true);

    // tail mask: rows 3..5 of the full 5x5 mask (shape 2x5)
    let tail_mask: Tensor<B, 2> = mask.narrow(0, 3, 2);
    let tail = attn2.forward(x.narrow(1, 3, 2), Some(tail_mask), 3, true);

    // prefix should match full[:, :3]
    let full_prefix: Vec<f32> = full
        .clone()
        .narrow(1, 0, 3)
        .reshape([36])
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let prefix_vec: Vec<f32> = prefix.reshape([36]).into_data().to_vec::<f32>().unwrap();
    for (a, b) in prefix_vec.iter().zip(full_prefix.iter()) {
        assert!((a - b).abs() < 1e-5, "KV cache prefix mismatch: {a} vs {b}");
    }

    // tail should match full[:, 3:]
    let full_tail: Vec<f32> = full
        .narrow(1, 3, 2)
        .reshape([24])
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let tail_vec: Vec<f32> = tail.reshape([24]).into_data().to_vec::<f32>().unwrap();
    for (a, b) in tail_vec.iter().zip(full_tail.iter()) {
        assert!((a - b).abs() < 1e-5, "KV cache tail mismatch: {a} vs {b}");
    }
}

// --- GatedMLP ---

#[test]
fn test_gated_mlp_shape() {
    let mlp = GatedMLP::new(8, 16, &DEVICE);
    let x: Tensor<B, 2> = Tensor::random([4, 8], Distribution::Normal(0.0, 1.0), &DEVICE);
    let out = mlp.forward(x);
    assert_eq!(out.dims(), [4, 8]);
}

#[test]
fn test_gated_mlp_batch_dims() {
    let mlp = GatedMLP::new(8, 16, &DEVICE);
    let x: Tensor<B, 3> = Tensor::random([2, 4, 8], Distribution::Normal(0.0, 1.0), &DEVICE);
    let out = mlp.forward(x);
    assert_eq!(out.dims(), [2, 4, 8]);
}

// --- TransformerBlock ---

#[test]
fn test_transformer_block_shape() {
    let mut block = TransformerBlock::new(12, 3, 16, 8, &DEVICE);
    let x: Tensor<B, 3> = Tensor::random([1, 5, 12], Distribution::Normal(0.0, 1.0), &DEVICE);
    let mask = causal_mask(5);
    let out = block.forward(x, Some(mask), 0, false);
    assert_eq!(out.dims(), [1, 5, 12]);
}

#[test]
fn test_transformer_block_residual() {
    // With all weights zero, output should equal input (residual connection)
    // This is hard to test without weight access, so just check shape and finiteness
    let mut block = TransformerBlock::new(8, 2, 12, 6, &DEVICE);
    let x: Tensor<B, 3> = Tensor::random([1, 4, 8], Distribution::Normal(0.0, 1.0), &DEVICE);
    let out = block.forward(x, None, 0, false);
    assert_eq!(out.dims(), [1, 4, 8]);
    let vals: Vec<f32> = out.reshape([32]).into_data().to_vec::<f32>().unwrap();
    for v in &vals {
        assert!(
            v.is_finite(),
            "TransformerBlock output has non-finite values"
        );
    }
}

// --- Llama3Simplified ---

#[test]
fn test_llama3_shape() {
    let mut model = Llama3Simplified::new(5, 4, 2, 6, 6, 1, &DEVICE);
    let tokens: Tensor<B, 2, Int> =
        Tensor::from_data(TensorData::new(vec![2i32, 3, 4], [1, 3]), &DEVICE);
    let out = model.forward(tokens, 0, false);
    assert_eq!(out.dims(), [1, 3, 5]);
}

#[test]
fn test_llama3_kv_cache_consistency() {
    let mut model = Llama3Simplified::new(10, 8, 2, 16, 12, 2, &DEVICE);
    let tokens: Tensor<B, 2, Int> =
        Tensor::from_data(TensorData::new(vec![0i32, 1, 2, 3], [1, 4]), &DEVICE);

    let full = model.forward(tokens, 0, false);
    assert_eq!(full.dims()[1], 4);

    // Create fresh model with same weights for cache test
    let mut model2 = Llama3Simplified::new(10, 8, 2, 16, 12, 2, &DEVICE);
    let prefix_tokens: Tensor<B, 2, Int> =
        Tensor::from_data(TensorData::new(vec![0i32, 1, 2], [1, 3]), &DEVICE);
    let _prefix = model2.forward(prefix_tokens, 0, true);
    let tail_tokens: Tensor<B, 2, Int> =
        Tensor::from_data(TensorData::new(vec![3i32], [1, 1]), &DEVICE);
    let tail = model2.forward(tail_tokens, 3, true);

    // Last token's output should match between full and cached versions
    let full_last: Vec<f32> = full
        .narrow(1, 3, 1)
        .reshape([10])
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let tail_vec: Vec<f32> = tail.reshape([10]).into_data().to_vec::<f32>().unwrap();
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
    let next_tokens: Vec<i32> = vec![3, 4];

    let mut model_fn =
        |tokens: Tensor<B, 2, Int>, _seq_pos: usize, _use_cache: bool| -> Tensor<B, 3> {
            let mut count = call_count.borrow_mut();
            let next_token = next_tokens[*count];
            *count += 1;
            let seq_len = tokens.dims()[1];
            // Return logits where next_token has highest score
            let mut logits_data = vec![-1e9f32; 6 * seq_len];
            // Set the last position's next_token index to 0 (highest)
            logits_data[(seq_len - 1) * 6 + next_token as usize] = 0.0;
            Tensor::<B, 3>::from_data(TensorData::new(logits_data, [1, seq_len, 6]), &DEVICE)
        };

    let decode_fn = |tokens: &[i32]| -> String {
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

    let stop_tokens: Vec<i32> = vec![4];
    let generated = generate(
        &mut model_fn,
        &[1, 2],
        &decode_fn,
        &stop_tokens,
        0.7,
        5,
        false,
    );

    assert_eq!(generated, vec![3, 4]);
}

#[test]
fn test_generate_max_tokens() {
    // Model always predicts token 3 (never hits stop token)
    let mut model_fn =
        |tokens: Tensor<B, 2, Int>, _seq_pos: usize, _use_cache: bool| -> Tensor<B, 3> {
            let seq_len = tokens.dims()[1];
            let mut logits_data = vec![-1e9f32; 6 * seq_len];
            logits_data[(seq_len - 1) * 6 + 3] = 0.0;
            Tensor::<B, 3>::from_data(TensorData::new(logits_data, [1, seq_len, 6]), &DEVICE)
        };

    let decode_fn = |_tokens: &[i32]| -> String { String::new() };
    let stop_tokens: Vec<i32> = vec![4];
    let generated = generate(
        &mut model_fn,
        &[1, 2],
        &decode_fn,
        &stop_tokens,
        0.7,
        3,
        false,
    );

    assert_eq!(generated.len(), 3, "Should stop at max_tokens=3");
}

// --- SiLU at zero (sanity) ---

#[test]
fn test_silu_at_zero() {
    // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    let x: Tensor<B, 1> = Tensor::from_data(TensorData::new(vec![0.0f32], [1]), &DEVICE);
    let out = silu(x);
    let val: f32 = out.into_data().to_vec::<f32>().unwrap()[0];
    assert!(val.abs() < 1e-7);
}

// --- eval_llama3 (end-to-end model loading + inference) ---

#[test]
fn test_eval_llama3() {
    let mut model = eval_llama3();

    let tokens: Tensor<B, 2, Int> =
        Tensor::from_data(TensorData::new(vec![0i32, 1, 2, 3], [1, 4]), &DEVICE);
    let full = model.forward(tokens, 0, false);

    // Shape: (1, 4, vocab_size)
    assert_eq!(full.dims()[0], 1);
    assert_eq!(full.dims()[1], 4);
    let vocab_size = full.dims()[2];
    assert!(vocab_size > 0);

    // Outputs should be finite (at least first 16 logits)
    let check_len = 16.min(vocab_size);
    let first_logits: Vec<f32> = full
        .clone()
        .narrow(2, 0, check_len)
        .reshape([4 * check_len])
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    for v in &first_logits {
        assert!(v.is_finite(), "eval_llama3 output has non-finite logits");
    }

    // KV cache consistency: full[:, 3:] should match tail from cached inference
    let mut model2 = eval_llama3();
    let prefix_tokens: Tensor<B, 2, Int> =
        Tensor::from_data(TensorData::new(vec![0i32, 1, 2], [1, 3]), &DEVICE);
    let prefix = model2.forward(prefix_tokens, 0, true);
    assert_eq!(prefix.dims()[1], 3);

    let tail_tokens: Tensor<B, 2, Int> =
        Tensor::from_data(TensorData::new(vec![3i32], [1, 1]), &DEVICE);
    let tail = model2.forward(tail_tokens, 3, true);
    assert_eq!(tail.dims()[1], 1);

    // tail should closely match full[:, 3:]
    let full_last: Vec<f32> = full
        .narrow(1, 3, 1)
        .reshape([vocab_size])
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let tail_vec: Vec<f32> = tail
        .reshape([vocab_size])
        .into_data()
        .to_vec::<f32>()
        .unwrap();

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
