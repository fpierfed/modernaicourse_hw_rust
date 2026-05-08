use candle_core::{Device, Tensor};
use candle_nn::VarMap;
use hw5::*;

const DEVICE: Device = Device::Cpu;

fn causal_mask(length: usize) -> Tensor {
    let mut data = vec![0.0f32; length * length];
    for i in 0..length {
        for j in (i + 1)..length {
            data[i * length + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_slice(&data, &[length, length], &DEVICE).unwrap()
}

#[test]
fn test_linear() {
    let varmap = VarMap::new();
    let layer = Linear::new(10, 20, &varmap, "linear").unwrap();

    let x = Tensor::randn(0.0f32, 1.0, &[50, 10], &DEVICE).unwrap();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.dims(), &[50, 20]);

    let x = Tensor::randn(0.0f32, 1.0, &[7, 9, 10], &DEVICE).unwrap();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.dims(), &[7, 9, 20]);
}

#[test]
fn test_embedding() {
    let varmap = VarMap::new();
    let layer = Embedding::new(200, 20, &varmap, "embed").unwrap();

    let y = Tensor::new(&[0u32, 5, 10, 199], &DEVICE).unwrap();
    let out = layer.forward(&y).unwrap();
    assert_eq!(out.dims(), &[4, 20]);
}

#[test]
fn test_silu() {
    let x = Tensor::new(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &DEVICE).unwrap();
    let out = silu(&x).unwrap();
    // silu(0) = 0
    let vals: Vec<f32> = out.to_vec1().unwrap();
    assert!((vals[2] - 0.0).abs() < 1e-6);
    // silu(x) = x * sigmoid(x)
    // silu(1) = 1 * sigmoid(1) ≈ 0.7310586
    assert!((vals[3] - 0.7310586).abs() < 1e-5);
}

#[test]
fn test_rms_norm() {
    let varmap = VarMap::new();
    let layer = RMSNorm::new(4, 1e-5, &varmap, "norm").unwrap();
    let x = Tensor::new(&[[1.0f32, -1.0, 0.5, 0.5]], &DEVICE).unwrap();
    let out = layer.forward(&x).unwrap();
    // RMS of [1, -1, 0.5, 0.5] = sqrt((1+1+0.25+0.25)/4) = sqrt(0.625)
    // normalized = x / rms
    let vals: Vec<f32> = out.to_vec2::<f32>().unwrap()[0].clone();
    let rms = (0.625f32).sqrt();
    assert!((vals[0] - 1.0 / rms).abs() < 1e-5);
}

#[test]
fn test_self_attention_shape() {
    let q = Tensor::randn(0.0f32, 1.0, &[5, 8], &DEVICE).unwrap();
    let k = Tensor::randn(0.0f32, 1.0, &[5, 8], &DEVICE).unwrap();
    let v = Tensor::randn(0.0f32, 1.0, &[5, 6], &DEVICE).unwrap();
    let mask = causal_mask(5);
    let out = self_attention(&q, &k, &v, Some(&mask)).unwrap();
    assert_eq!(out.dims(), &[5, 6]);
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
fn test_multi_head_attention() {
    let varmap = VarMap::new();
    let attn = MultiHeadAttention::new(12, 3, &varmap, "mha").unwrap();
    let x = Tensor::randn(0.0f32, 1.0, &[2, 5, 12], &DEVICE).unwrap();
    let out = attn.forward(&x, None).unwrap();
    assert_eq!(out.dims(), &[2, 5, 12]);
}

#[test]
fn test_multi_head_attention_causal() {
    let varmap = VarMap::new();
    let attn = MultiHeadAttention::new(12, 3, &varmap, "mha_causal").unwrap();
    let x = Tensor::randn(0.0f32, 1.0, &[2, 5, 12], &DEVICE).unwrap();
    let mask = causal_mask(5);
    let out = attn.forward(&x, Some(&mask)).unwrap();
    assert_eq!(out.dims(), &[2, 5, 12]);
}

#[test]
fn test_kv_cache_consistency() {
    let varmap = VarMap::new();
    let mut attn = MultiHeadAttentionKVCache::new(12, 3, 8, &varmap, "mha_kv").unwrap();
    let x = Tensor::randn(0.0f32, 1.0, &[1, 5, 12], &DEVICE).unwrap();
    let mask = causal_mask(5);

    // Full forward without cache
    let full = attn.forward(&x, Some(&mask), 0, false).unwrap();
    assert_eq!(full.dims(), &[1, 5, 12]);

    // Prefix + tail with cache should match
    let prefix = attn
        .forward(
            &x.narrow(1, 0, 3).unwrap(),
            Some(&mask.narrow(0, 0, 3).unwrap().narrow(1, 0, 3).unwrap()),
            0,
            true,
        )
        .unwrap();
    let tail = attn
        .forward(
            &x.narrow(1, 3, 2).unwrap(),
            Some(&mask.narrow(0, 3, 2).unwrap()),
            3,
            true,
        )
        .unwrap();

    assert_eq!(prefix.dims(), &[1, 3, 12]);
    assert_eq!(tail.dims(), &[1, 2, 12]);
    // full[:, 3:] should be close to tail
    let full_tail = full.narrow(1, 3, 2).unwrap();
    let diff = full_tail
        .sub(&tail)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(diff < 1e-5, "KV cache mismatch: max diff = {diff}");
}

#[test]
fn test_gated_mlp() {
    let varmap = VarMap::new();
    let mlp = GatedMLP::new(4, 8, &varmap, "mlp").unwrap();
    let x = Tensor::randn(0.0f32, 1.0, &[3, 4], &DEVICE).unwrap();
    let out = mlp.forward(&x).unwrap();
    assert_eq!(out.dims(), &[3, 4]);
}

#[test]
fn test_transformer_block() {
    let varmap = VarMap::new();
    let mut block = TransformerBlock::new(12, 3, 16, 32, &varmap, "block").unwrap();
    let x = Tensor::randn(0.0f32, 1.0, &[1, 5, 12], &DEVICE).unwrap();
    let mask = causal_mask(5);
    let out = block.forward(&x, Some(&mask), 0, false).unwrap();
    assert_eq!(out.dims(), &[1, 5, 12]);
}

#[test]
fn test_llama3_simplified() {
    let varmap = VarMap::new();
    let mut model = Llama3Simplified::new(100, 12, 3, 32, 16, 2, &varmap).unwrap();
    let tokens = Tensor::new(&[[0u32, 1, 2, 3, 4]], &DEVICE).unwrap();
    let out = model.forward(&tokens, 0, false).unwrap();
    // Output shape: (1, 5, num_tokens=100)
    assert_eq!(out.dims(), &[1, 5, 100]);
}

#[test]
fn test_generate() {
    let mut call_count = 0usize;
    let next_tokens: Vec<u32> = vec![3, 4]; // A, then stop token

    let mut model_fn =
        |tokens: &Tensor, seq_pos: usize, use_kv_cache: bool| -> candle_core::Result<Tensor> {
            let vocab_size = 6;
            let seq_len = tokens.dims()[1];
            let mut data = vec![f32::NEG_INFINITY; seq_len * vocab_size];
            let next = next_tokens[call_count] as usize;
            data[(seq_len - 1) * vocab_size + next] = 0.0;
            call_count += 1;
            Tensor::from_slice(&data, &[1, seq_len, vocab_size], &Device::Cpu)
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

    let generated = generate(&mut model_fn, &[1, 2], &decode_fn, &[4], 0.7, 5, false).unwrap();

    assert_eq!(generated, vec![3, 4]);
}
