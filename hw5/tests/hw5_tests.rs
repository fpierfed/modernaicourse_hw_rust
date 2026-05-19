use burn::tensor::{Distribution, Int, Tensor, TensorData};
use hw5::*;
use std::collections::HashMap;
use std::io::Write;

fn causal_mask(length: usize) -> Tensor<B, 2> {
    let mut data = vec![0.0f32; length * length];
    for i in 0..length {
        for j in (i + 1)..length {
            data[i * length + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::<B, 2>::from_data(TensorData::new(data, [length, length]), &DEVICE)
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

#[test]
fn test_linear() {
    let layer = Linear::new(10, 20, &DEVICE);

    let x: Tensor<B, 2> = Tensor::random([50, 10], Distribution::Normal(0.0, 1.0), &DEVICE);
    let out = layer.forward(x.clone());
    assert_eq!(out.dims(), [50, 20]);

    // Correctness: output == X @ W^T
    let expected = x.matmul(layer.weight().clone().transpose());
    let diff: f32 = out.sub(expected).abs().sum().into_scalar();
    assert!(diff < 1e-4);

    // Batch dims
    let x: Tensor<B, 3> = Tensor::random([7, 9, 10], Distribution::Normal(0.0, 1.0), &DEVICE);
    let out = layer.forward(x);
    assert_eq!(out.dims(), [7, 9, 20]);
}

#[test]
fn test_linear_kaiming_init() {
    let layer = Linear::new(100, 1000, &DEVICE);
    let w = layer.weight().clone();
    let mean: f32 = w.clone().mean().into_scalar();
    let var: f32 = (w - mean).powf_scalar(2.0).mean().into_scalar();
    let std = (var as f64).sqrt();
    let expected_std = (2.0 / 100.0f64).sqrt();
    assert!(
        (std - expected_std).abs() < 3e-3,
        "Linear weight std {std} not close to expected {expected_std}"
    );
}

#[test]
fn test_embedding() {
    let layer = Embedding::new(200, 20, &DEVICE);

    // Use 2D input [1, 4] to test single-sequence embedding
    let y: Tensor<B, 2, Int> =
        Tensor::from_data(TensorData::new(vec![0i32, 5, 10, 199], [1, 4]), &DEVICE);
    let out = layer.forward(y);
    assert_eq!(out.dims(), [1, 4, 20]);

    // Correctness: first token's embedding == corresponding weight row
    let w = layer.weight().clone();
    let out0: Vec<f32> = out
        .clone()
        .narrow(1, 0, 1)
        .flatten::<1>(0, 2)
        .narrow(0, 0, 20)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let w0: Vec<f32> = w
        .clone()
        .narrow(0, 0, 1)
        .squeeze::<1>()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(out0, w0);

    // Batch dims
    let y: Tensor<B, 2, Int> =
        Tensor::from_data(TensorData::new(vec![0i32, 1, 2, 3, 4, 5], [2, 3]), &DEVICE);
    let out = layer.forward(y);
    assert_eq!(out.dims(), [2, 3, 20]);
}

#[test]
fn test_embedding_std_init() {
    let layer = Embedding::new(1000, 100, &DEVICE);
    let w = layer.weight().clone();
    let mean: f32 = w.clone().mean().into_scalar();
    let var: f32 = (w - mean).powf_scalar(2.0).mean().into_scalar();
    let std = (var as f64).sqrt();
    assert!(
        (std - 1.0).abs() < 3e-2,
        "Embedding weight std {std} not close to 1.0"
    );
}

#[test]
fn test_silu() {
    let x: Tensor<B, 2> = Tensor::random([10, 20], Distribution::Normal(0.0, 1.0), &DEVICE);
    let out = silu(x.clone());
    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    let sigmoid = x.clone().neg().exp().add_scalar(1.0).recip();
    let expected = x.clone().mul(sigmoid);
    let diff: f32 = out.sub(expected).abs().max().into_scalar();
    assert!(diff < 1e-6);

    // Multi-dim
    let x: Tensor<B, 4> = Tensor::random([3, 4, 5, 6], Distribution::Normal(0.0, 1.0), &DEVICE);
    let out = silu(x.clone());
    assert_eq!(out.dims(), x.dims());
}

#[test]
fn test_rms_norm() {
    let x: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(vec![1.0f32, -1.0, 0.5, 0.5, 2.0, 0.0, -2.0, 1.0], [2, 4]),
        &DEVICE,
    );
    let out = rms_norm(x.clone(), 1e-5);

    // Manual: row 0 rms = sqrt((1+1+0.25+0.25)/4) = sqrt(0.625)
    let x_data: Vec<f32> = x.into_data().to_vec::<f32>().unwrap();
    let out_data: Vec<f32> = out.into_data().to_vec::<f32>().unwrap();
    for row in 0..2 {
        let row_slice = &x_data[row * 4..(row + 1) * 4];
        let mean_sq: f32 = row_slice.iter().map(|v| v * v).sum::<f32>() / 4.0;
        let rms = (mean_sq + 1e-5f32).sqrt();
        for col in 0..4 {
            let expected = row_slice[col] / rms;
            assert!(
                (out_data[row * 4 + col] - expected).abs() < 1e-5,
                "rms_norm mismatch at [{row}][{col}]"
            );
        }
    }

    // Batch dims
    let x: Tensor<B, 3> = Tensor::random([10, 7, 20], Distribution::Normal(0.0, 1.0), &DEVICE);
    let out = rms_norm(x, 1e-3);
    assert_eq!(out.dims(), [10, 7, 20]);
}

#[test]
fn test_self_attention_causal() {
    let q: Tensor<B, 2> = Tensor::random([5, 8], Distribution::Normal(0.0, 1.0), &DEVICE);
    let k: Tensor<B, 2> = Tensor::random([5, 8], Distribution::Normal(0.0, 1.0), &DEVICE);
    let v: Tensor<B, 2> = Tensor::random([5, 6], Distribution::Normal(0.0, 1.0), &DEVICE);
    let mask = causal_mask(5);
    let out = self_attention(q, k, v.clone(), Some(mask));
    assert_eq!(out.dims(), [5, 6]);

    // First row with causal mask: only attends to position 0, so output = V[0]
    let out_row0: Vec<f32> = out
        .narrow(0, 0, 1)
        .squeeze::<1>()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let v_row0: Vec<f32> = v
        .narrow(0, 0, 1)
        .squeeze::<1>()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    for (a, b) in out_row0.iter().zip(v_row0.iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}

#[test]
fn test_self_attention_batched() {
    let q: Tensor<B, 4> = Tensor::random([2, 3, 5, 8], Distribution::Normal(0.0, 1.0), &DEVICE);
    let k: Tensor<B, 4> = Tensor::random([2, 3, 5, 8], Distribution::Normal(0.0, 1.0), &DEVICE);
    let v: Tensor<B, 4> = Tensor::random([2, 3, 5, 4], Distribution::Normal(0.0, 1.0), &DEVICE);
    let mask = causal_mask(5);
    let out = self_attention(q, k, v, Some(mask));
    assert_eq!(out.dims(), [2, 3, 5, 4]);
}

#[test]
fn test_multi_head_attention_kv_cache() {
    let mut attn = MultiHeadAttentionKVCache::new(12, 3, 8, &DEVICE);
    let x: Tensor<B, 3> = Tensor::random([1, 5, 12], Distribution::Normal(0.0, 1.0), &DEVICE);
    let mask = causal_mask(5);

    let full = attn.forward(x.clone(), Some(mask.clone()), 0, false);
    assert_eq!(full.dims(), [1, 5, 12]);

    // Prefix + tail with cache
    let mut attn2 = MultiHeadAttentionKVCache::new(12, 3, 8, &DEVICE);
    let prefix_mask = causal_mask(3);
    let _prefix = attn2.forward(x.clone().narrow(1, 0, 3), Some(prefix_mask), 0, true);
    let tail_mask: Tensor<B, 2> = mask.narrow(0, 3, 2);
    let tail = attn2.forward(x.narrow(1, 3, 2), Some(tail_mask), 3, true);

    let full_tail: Vec<f32> = full
        .narrow(1, 3, 2)
        .flatten::<1>(0, 2)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let tail_vec: Vec<f32> = tail.flatten::<1>(0, 2).into_data().to_vec::<f32>().unwrap();
    let max_diff: f32 = full_tail
        .iter()
        .zip(tail_vec.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-5, "KV cache mismatch: max diff = {max_diff}");
}

#[test]
fn test_mlp() {
    let mlp = MLP::new(5, 7, &DEVICE);
    let x: Tensor<B, 3> = Tensor::random([4, 3, 5], Distribution::Normal(0.0, 1.0), &DEVICE);
    let out = mlp.forward(x);
    assert_eq!(out.dims(), [4, 3, 5]);
}

#[test]
fn test_transformer_block() {
    let mut block = TransformerBlock::new(12, 3, 16, 8, &DEVICE);
    let x: Tensor<B, 3> = Tensor::random([1, 5, 12], Distribution::Normal(0.0, 1.0), &DEVICE);
    let mask = causal_mask(5);

    let full = block.forward(x.clone(), Some(mask.clone()), 0, false);
    assert_eq!(full.dims(), [1, 5, 12]);

    // KV cache consistency
    let mut block2 = TransformerBlock::new(12, 3, 16, 8, &DEVICE);
    let prefix_mask = causal_mask(3);
    let _prefix = block2.forward(x.clone().narrow(1, 0, 3), Some(prefix_mask), 0, true);
    let tail_mask: Tensor<B, 2> = mask.narrow(0, 3, 2);
    let tail = block2.forward(x.narrow(1, 3, 2), Some(tail_mask), 3, true);

    let full_tail: Vec<f32> = full
        .narrow(1, 3, 2)
        .flatten::<1>(0, 2)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let tail_vec: Vec<f32> = tail.flatten::<1>(0, 2).into_data().to_vec::<f32>().unwrap();
    let max_diff: f32 = full_tail
        .iter()
        .zip(tail_vec.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 3e-5,
        "TransformerBlock KV cache mismatch: {max_diff}"
    );
}

#[test]
fn test_llm() {
    let mut model = LLM::new(11, 12, 3, 8, 16, 2, &DEVICE);
    let tokens: Tensor<B, 2, Int> =
        Tensor::from_data(TensorData::new(vec![0i32, 1, 2, 3, 4], [1, 5]), &DEVICE);

    let out = model.forward(tokens.clone(), 0, false);
    assert_eq!(out.dims(), [1, 5, 11]);

    // KV cache consistency
    let mut model2 = LLM::new(11, 12, 3, 8, 16, 2, &DEVICE);
    let _prefix = model2.forward(tokens.clone().narrow(1, 0, 3), 0, true);
    let tail = model2.forward(tokens.narrow(1, 3, 2), 3, true);

    let out_tail: Vec<f32> = out
        .narrow(1, 3, 2)
        .flatten::<1>(0, 2)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let tail_vec: Vec<f32> = tail.flatten::<1>(0, 2).into_data().to_vec::<f32>().unwrap();
    let max_diff: f32 = out_tail
        .iter()
        .zip(tail_vec.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 6e-5, "LLM KV cache mismatch: {max_diff}");
}

// ============================================================
// Part III: Training
// ============================================================

#[test]
fn test_cross_entropy_loss_2d() {
    let logits: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(vec![2.0f32, 1.0, 0.0, 0.0, 2.0, 1.0], [2, 3]),
        &DEVICE,
    );
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from([0i32, 2]), &DEVICE);
    let loss = cross_entropy_loss(logits, y);
    let val: f32 = loss.into_scalar();
    assert!((val - 0.907_606).abs() < 1e-5);
}

#[test]
fn test_cross_entropy_loss_3d() {
    // Multi-dimensional logits: (batch, seq, classes) -> reshape to 2D for cross_entropy_loss
    let logits: Tensor<B, 3> = Tensor::random([4, 5, 7], Distribution::Normal(0.0, 1.0), &DEVICE);
    let y_data: Vec<i32> = (0..20).map(|i| i % 7).collect();
    let y: Tensor<B, 2, Int> = Tensor::from_data(TensorData::new(y_data, [4, 5]), &DEVICE);
    // Reshape logits to (20, 7) and targets to (20) for cross_entropy_loss
    let logits_flat: Tensor<B, 2> = logits.reshape([20, 7]);
    let y_flat: Tensor<B, 1, Int> = y.reshape([20]);
    let loss = cross_entropy_loss(logits_flat, y_flat);
    let val: f32 = loss.into_scalar();
    assert!(val.is_finite() && val > 0.0);
}

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
    // Should have tokenized "abc" and "def" (2 chunks of 3)
    assert_eq!(tokens, vec![97, 98, 99, 100, 101, 102]);
}

#[test]
fn test_dataloader_file() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("tokens.bin");

    // Write tokens 0..20 as u16
    let mut f = std::fs::File::create(&path).unwrap();
    for i in 0u16..20 {
        f.write_all(&i.to_le_bytes()).unwrap();
    }
    drop(f);

    let loader = DataLoader::new(&path, 3, 2);
    let batches: Vec<_> = loader.collect();

    assert_eq!(batches.len(), 3);

    // First batch: input [[0,1,2],[4,5,6]], target [[1,2,3],[5,6,7]]
    let xb0: Vec<i32> = batches[0].0.clone().into_data().to_vec::<i32>().unwrap();
    assert_eq!(xb0, vec![0, 1, 2, 4, 5, 6]);

    let yb0: Vec<i32> = batches[0].1.clone().into_data().to_vec::<i32>().unwrap();
    assert_eq!(yb0, vec![1, 2, 3, 5, 6, 7]);
}

#[test]
fn test_adam() {
    let layer = Linear::new(6, 3, &DEVICE);

    let params: Vec<Tensor<B, 2>> = vec![layer.weight().clone()];
    let mut opt = Adam::new(params.clone(), 1e-3, (0.9, 0.95), 1e-8);

    let w_before: Vec<f32> = params[0].clone().into_data().to_vec::<f32>().unwrap();

    // 5 training steps
    for _ in 0..5 {
        let x: Tensor<B, 2> = Tensor::random([16, 6], Distribution::Normal(0.0, 1.0), &DEVICE);
        let y_data: Vec<i32> = (0..16).map(|i| i % 3).collect();
        let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::new(y_data, [16]), &DEVICE);

        let logits = layer.forward(x);
        let loss = cross_entropy_loss(logits, y);
        let grads = loss.backward();
        opt.step(&grads);
    }

    let w_after: Vec<f32> = params[0].clone().into_data().to_vec::<f32>().unwrap();
    let changed = w_after
        .iter()
        .zip(w_before.iter())
        .any(|(a, b)| (a - b).abs() > 1e-7);
    assert!(changed, "Adam did not update weights after 5 steps");
}

#[test]
fn test_adam_zero_grad() {
    let layer = Linear::new(4, 3, &DEVICE);

    let params: Vec<Tensor<B, 2>> = vec![layer.weight().clone()];
    let mut opt = Adam::new(params.clone(), 1e-2, (0.9, 0.999), 1e-8);

    // Create gradients
    let x: Tensor<B, 2> = Tensor::random([4, 4], Distribution::Normal(0.0, 1.0), &DEVICE);
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from([0i32, 1, 2, 0]), &DEVICE);
    let logits = layer.forward(x);
    let loss = cross_entropy_loss(logits, y);
    let _grads = loss.backward();

    // zero_grad then step should be a no-op on weights (first step with zero grad)
    opt.zero_grad();
    let w_before: Vec<f32> = params[0].clone().into_data().to_vec::<f32>().unwrap();
    // Create dummy zero grads after zero_grad
    let zero_loss: Tensor<B, 1> =
        Tensor::from_data(TensorData::from([0.0f32]), &DEVICE).require_grad();
    let zero_grads = zero_loss.backward();
    opt.step(&zero_grads);
    let w_after: Vec<f32> = params[0].clone().into_data().to_vec::<f32>().unwrap();
    for (a, b) in w_after.iter().zip(w_before.iter()) {
        assert!((a - b).abs() < 1e-7, "Adam zero_grad didn't prevent update");
    }
}

#[test]
fn test_train_llm() {
    let layer = Linear::new(4, 5, &DEVICE);

    let params: Vec<Tensor<B, 2>> = vec![layer.weight().clone()];
    let mut opt = Adam::new(params.clone(), 0.01, (0.9, 0.999), 1e-8);

    let w_before: Vec<f32> = params[0].clone().into_data().to_vec::<f32>().unwrap();

    // Create loader data
    let x1: Tensor<B, 2, Int> =
        Tensor::from_data(TensorData::new(vec![0i32, 1, 2, 1, 2, 3], [2, 3]), &DEVICE);
    let y1: Tensor<B, 2, Int> =
        Tensor::from_data(TensorData::new(vec![1i32, 2, 3, 2, 3, 4], [2, 3]), &DEVICE);
    let x2: Tensor<B, 2, Int> =
        Tensor::from_data(TensorData::new(vec![2i32, 3, 4, 0, 2, 4], [2, 3]), &DEVICE);
    let y2: Tensor<B, 2, Int> =
        Tensor::from_data(TensorData::new(vec![3i32, 4, 0, 2, 4, 1], [2, 3]), &DEVICE);
    let loader = vec![(x1, y1), (x2, y2)];

    let model_fn = |tokens: Tensor<B, 2, Int>| -> Tensor<B, 3> {
        // Simple embedding-like model: just lookup and project
        let embed: Tensor<B, 2> = Tensor::random([5, 4], Distribution::Normal(0.0, 1.0), &DEVICE);
        // Gather embeddings for each token - flatten, gather, reshape
        let flat_tokens: Vec<i32> = tokens.clone().into_data().to_vec::<i32>().unwrap();
        let batch_size = tokens.dims()[0];
        let seq_len = tokens.dims()[1];
        let mut embedded_data = Vec::new();
        let embed_data: Vec<f32> = embed.into_data().to_vec::<f32>().unwrap();
        for &tok in &flat_tokens {
            let start = (tok as usize) * 4;
            embedded_data.extend_from_slice(&embed_data[start..start + 4]);
        }
        let x: Tensor<B, 3> = Tensor::from_data(
            TensorData::new(embedded_data, [batch_size, seq_len, 4]),
            &DEVICE,
        );
        // Project through linear: shape [batch, seq, 4] -> [batch, seq, 5]
        // We reshape to 2D, apply linear, reshape back
        let x_2d: Tensor<B, 2> = x.reshape([batch_size * seq_len, 4]);
        let out_2d = layer.forward(x_2d);
        out_2d.reshape([batch_size, seq_len, 5])
    };

    train_llm(&model_fn, &loader, &mut opt);

    let w_after: Vec<f32> = params[0].clone().into_data().to_vec::<f32>().unwrap();
    let changed = w_after
        .iter()
        .zip(w_before.iter())
        .any(|(a, b)| (a - b).abs() > 1e-7);
    assert!(changed, "train_llm did not update weights");
}

#[test]
fn test_generate() {
    let mut call_count = 0usize;
    let next_tokens: Vec<u32> = vec![3, 4];

    let mut model_fn =
        |tokens: Tensor<B, 2, Int>, _seq_pos: usize, _use_cache: bool| -> Tensor<B, 3> {
            let vocab_size = 6;
            let seq_len = tokens.dims()[1];
            let mut data = vec![f32::NEG_INFINITY; seq_len * vocab_size];
            let next = next_tokens[call_count] as usize;
            data[(seq_len - 1) * vocab_size + next] = 0.0;
            call_count += 1;
            Tensor::from_data(TensorData::new(data, [1, seq_len, vocab_size]), &DEVICE)
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

    let generated = generate(&mut model_fn, &[1, 2], &decode_fn, 4, 0.7, 5, false);
    assert_eq!(generated, vec![3, 4]);
}

#[test]
fn test_generate_max_tokens() {
    let mut model_fn =
        |tokens: Tensor<B, 2, Int>, _seq_pos: usize, _use_cache: bool| -> Tensor<B, 3> {
            let vocab_size = 6;
            let seq_len = tokens.dims()[1];
            let mut data = vec![f32::NEG_INFINITY; seq_len * vocab_size];
            // Always predict token 3 (never stop)
            data[(seq_len - 1) * vocab_size + 3] = 0.0;
            Tensor::from_data(TensorData::new(data, [1, seq_len, vocab_size]), &DEVICE)
        };

    let decode_fn = |_: &[u32]| -> String { String::new() };
    let generated = generate(&mut model_fn, &[1, 2], &decode_fn, 4, 0.7, 3, false);
    assert_eq!(generated.len(), 3);
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
fn test_eval_llm() {
    use burn::tensor::{Int, Tensor, TensorData};

    let mut model = eval_llm();

    let eval_tokens = tiny_stories_eval::get_eval_tokens(0, 48);
    let tokens_i32: Vec<i32> = eval_tokens.iter().map(|&t| t as i32).collect();
    let tokens: Tensor<B, 2, Int> =
        Tensor::from_data(TensorData::new(tokens_i32.clone(), [1, 48]), &DEVICE);

    // Compute sequence loss: cross-entropy of model predicting next tokens
    let logits = model.forward(tokens.clone().narrow(1, 0, 47), 0, false);
    let vocab_size = logits.dims()[2];
    let logits_flat: Tensor<B, 2> = logits.reshape([47, vocab_size]);
    let targets_i32: Vec<i32> = eval_tokens[1..].iter().map(|&t| t as i32).collect();
    let targets_t: Tensor<B, 1, Int> =
        Tensor::from_data(TensorData::new(targets_i32, [47]), &DEVICE);
    let phrase_loss: f32 = cross_entropy_loss(logits_flat, targets_t).into_scalar();

    // Corrupted: reverse the token order (except first)
    let mut corrupted_tokens = eval_tokens.clone();
    corrupted_tokens[1..].reverse();
    let corrupted_i32: Vec<i32> = corrupted_tokens.iter().map(|&t| t as i32).collect();
    let corrupted: Tensor<B, 2, Int> =
        Tensor::from_data(TensorData::new(corrupted_i32.clone(), [1, 48]), &DEVICE);
    let c_logits = model.forward(corrupted.narrow(1, 0, 47), 0, false);
    let c_vocab_size = c_logits.dims()[2];
    let c_logits_flat: Tensor<B, 2> = c_logits.reshape([47, c_vocab_size]);
    let c_targets_i32: Vec<i32> = corrupted_tokens[1..].iter().map(|&t| t as i32).collect();
    let c_targets_t: Tensor<B, 1, Int> =
        Tensor::from_data(TensorData::new(c_targets_i32, [47]), &DEVICE);
    let corrupted_loss: f32 = cross_entropy_loss(c_logits_flat, c_targets_t).into_scalar();

    assert!(
        phrase_loss < 7.0,
        "eval_llm phrase_loss {phrase_loss} should be < 7.0"
    );
    assert!(
        phrase_loss < corrupted_loss,
        "phrase_loss {phrase_loss} should be < corrupted_loss {corrupted_loss}"
    );

    // KV cache consistency
    let full = model.forward(tokens.clone().narrow(1, 0, 47), 0, false);
    let mut model2 = eval_llm();
    let _prefix = model2.forward(tokens.clone().narrow(1, 0, 46), 0, true);
    let tail = model2.forward(tokens.narrow(1, 46, 1), 46, true);

    let full_last: Vec<f32> = full
        .clone()
        .narrow(1, 46, 1)
        .flatten::<1>(0, 2)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let tail_vec: Vec<f32> = tail.flatten::<1>(0, 2).into_data().to_vec::<f32>().unwrap();
    let max_diff: f32 = full_last
        .iter()
        .zip(tail_vec.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 3e-4,
        "eval_llm KV cache mismatch: max_diff={max_diff}"
    );

    // Outputs should be finite
    let vocab_dim = full.dims()[2];
    let first_logits: Vec<f32> = full
        .narrow(2, 0, 16.min(vocab_dim))
        .flatten::<1>(0, 2)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    for v in &first_logits {
        assert!(v.is_finite(), "eval_llm has non-finite logits");
    }
}
