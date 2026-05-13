use candle_core::{Device, Tensor};
use candle_nn::VarMap;
use hw5::*;
use std::collections::HashMap;
use std::io::Write;

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
            vec![" ".to_string(), "t".to_string(), "h".to_string(), "e".to_string(), "r".to_string(), "e".to_string()],
            vec!["\n".to_string(), "t".to_string(), "h".to_string(), "e".to_string(), "r".to_string(), "e".to_string()],
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
    let varmap = VarMap::new();
    let layer = Linear::new(10, 20, &varmap, "linear").unwrap();

    let x = Tensor::randn(0.0f32, 1.0, &[50, 10], &DEVICE).unwrap();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.dims(), &[50, 20]);

    // Correctness: output == X @ W^T
    let expected = x.matmul(&layer.weight().t().unwrap()).unwrap();
    let diff: f32 = out.sub(&expected).unwrap().abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
    assert!(diff < 1e-4);

    // Batch dims
    let x = Tensor::randn(0.0f32, 1.0, &[7, 9, 10], &DEVICE).unwrap();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.dims(), &[7, 9, 20]);
}

#[test]
fn test_linear_kaiming_init() {
    let varmap = VarMap::new();
    let layer = Linear::new(100, 1000, &varmap, "linear_init").unwrap();
    let w = layer.weight();
    let var: f32 = w
        .broadcast_sub(&w.mean_all().unwrap()).unwrap()
        .sqr().unwrap()
        .mean_all().unwrap()
        .to_scalar().unwrap();
    let std = (var as f64).sqrt();
    let expected_std = (2.0 / 100.0f64).sqrt();
    assert!(
        (std - expected_std).abs() < 3e-3,
        "Linear weight std {std} not close to expected {expected_std}"
    );
}

#[test]
fn test_embedding() {
    let varmap = VarMap::new();
    let layer = Embedding::new(200, 20, &varmap, "embed").unwrap();

    let y = Tensor::new(&[0u32, 5, 10, 199], &DEVICE).unwrap();
    let out = layer.forward(&y).unwrap();
    assert_eq!(out.dims(), &[4, 20]);

    // Correctness: each row of output == corresponding weight row
    let w = layer.weight();
    let out0: Vec<f32> = out.get(0).unwrap().to_vec1().unwrap();
    let w0: Vec<f32> = w.get(0).unwrap().to_vec1().unwrap();
    assert_eq!(out0, w0);

    // Batch dims
    let y = Tensor::new(&[[0u32, 1, 2], [3, 4, 5]], &DEVICE).unwrap();
    let out = layer.forward(&y).unwrap();
    assert_eq!(out.dims(), &[2, 3, 20]);
}

#[test]
fn test_embedding_std_init() {
    let varmap = VarMap::new();
    let layer = Embedding::new(1000, 100, &varmap, "embed_init").unwrap();
    let w = layer.weight();
    let var: f32 = w
        .broadcast_sub(&w.mean_all().unwrap()).unwrap()
        .sqr().unwrap()
        .mean_all().unwrap()
        .to_scalar().unwrap();
    let std = (var as f64).sqrt();
    assert!(
        (std - 1.0).abs() < 3e-2,
        "Embedding weight std {std} not close to 1.0"
    );
}

#[test]
fn test_silu() {
    let x = Tensor::randn(0.0f32, 1.0, &[10, 20], &DEVICE).unwrap();
    let out = silu(&x).unwrap();
    let sigmoid = (x.neg().unwrap().exp().unwrap() + 1.0).unwrap().recip().unwrap();
    let expected = x.mul(&sigmoid).unwrap();
    let diff: f32 = out.sub(&expected).unwrap().abs().unwrap().max_all().unwrap().to_scalar().unwrap();
    assert!(diff < 1e-6);

    // Multi-dim
    let x = Tensor::randn(0.0f32, 1.0, &[3, 4, 5, 6], &DEVICE).unwrap();
    let out = silu(&x).unwrap();
    assert_eq!(out.dims(), x.dims());
}

#[test]
fn test_rms_norm() {
    let x = Tensor::new(&[[1.0f32, -1.0, 0.5, 0.5], [2.0, 0.0, -2.0, 1.0]], &DEVICE).unwrap();
    let out = rms_norm(&x, 1e-5).unwrap();

    // Manual: row 0 rms = sqrt((1+1+0.25+0.25)/4) = sqrt(0.625)
    let x_vec: Vec<Vec<f32>> = (0..2).map(|i| x.get(i).unwrap().to_vec1().unwrap()).collect();
    let out_vec: Vec<Vec<f32>> = (0..2).map(|i| out.get(i).unwrap().to_vec1().unwrap()).collect();
    for row in 0..2 {
        let mean_sq: f32 = x_vec[row].iter().map(|v| v * v).sum::<f32>() / 4.0;
        let rms = (mean_sq + 1e-5f32).sqrt();
        for col in 0..4 {
            let expected = x_vec[row][col] / rms;
            assert!(
                (out_vec[row][col] - expected).abs() < 1e-5,
                "rms_norm mismatch at [{row}][{col}]"
            );
        }
    }

    // Batch dims
    let x = Tensor::randn(0.0f32, 1.0, &[10, 7, 20], &DEVICE).unwrap();
    let out = rms_norm(&x, 1e-3).unwrap();
    assert_eq!(out.dims(), &[10, 7, 20]);
}

#[test]
fn test_self_attention_causal() {
    let q = Tensor::randn(0.0f32, 1.0, &[5, 8], &DEVICE).unwrap();
    let k = Tensor::randn(0.0f32, 1.0, &[5, 8], &DEVICE).unwrap();
    let v = Tensor::randn(0.0f32, 1.0, &[5, 6], &DEVICE).unwrap();
    let mask = causal_mask(5);
    let out = self_attention(&q, &k, &v, Some(&mask)).unwrap();
    assert_eq!(out.dims(), &[5, 6]);

    // First row with causal mask: only attends to position 0, so output = V[0]
    let out_row0: Vec<f32> = out.get(0).unwrap().to_vec1().unwrap();
    let v_row0: Vec<f32> = v.get(0).unwrap().to_vec1().unwrap();
    for (a, b) in out_row0.iter().zip(v_row0.iter()) {
        assert!((a - b).abs() < 1e-5);
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
fn test_multi_head_attention_kv_cache() {
    let varmap = VarMap::new();
    let mut attn = MultiHeadAttentionKVCache::new(12, 3, 8, &varmap, "mha_kv").unwrap();
    let x = Tensor::randn(0.0f32, 1.0, &[1, 5, 12], &DEVICE).unwrap();
    let mask = causal_mask(5);

    let full = attn.forward(&x, Some(&mask), 0, false).unwrap();
    assert_eq!(full.dims(), &[1, 5, 12]);

    // Prefix + tail with cache
    let mut attn2 = MultiHeadAttentionKVCache::new(12, 3, 8, &varmap, "mha_kv").unwrap();
    let prefix_mask = causal_mask(3);
    let _prefix = attn2.forward(&x.narrow(1, 0, 3).unwrap(), Some(&prefix_mask), 0, true).unwrap();
    let tail_mask = mask.narrow(0, 3, 2).unwrap();
    let tail = attn2.forward(&x.narrow(1, 3, 2).unwrap(), Some(&tail_mask), 3, true).unwrap();

    let full_tail: Vec<f32> = full.narrow(1, 3, 2).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let tail_vec: Vec<f32> = tail.flatten_all().unwrap().to_vec1().unwrap();
    let max_diff: f32 = full_tail.iter().zip(tail_vec.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_diff < 1e-5, "KV cache mismatch: max diff = {max_diff}");
}

#[test]
fn test_mlp() {
    let varmap = VarMap::new();
    let mlp = MLP::new(5, 7, &varmap, "mlp").unwrap();
    let x = Tensor::randn(0.0f32, 1.0, &[4, 3, 5], &DEVICE).unwrap();
    let out = mlp.forward(&x).unwrap();
    assert_eq!(out.dims(), &[4, 3, 5]);
}

#[test]
fn test_transformer_block() {
    let varmap = VarMap::new();
    let mut block = TransformerBlock::new(12, 3, 16, 8, &varmap, "block").unwrap();
    let x = Tensor::randn(0.0f32, 1.0, &[1, 5, 12], &DEVICE).unwrap();
    let mask = causal_mask(5);

    let full = block.forward(&x, Some(&mask), 0, false).unwrap();
    assert_eq!(full.dims(), &[1, 5, 12]);

    // KV cache consistency
    let mut block2 = TransformerBlock::new(12, 3, 16, 8, &varmap, "block").unwrap();
    let prefix_mask = causal_mask(3);
    let _prefix = block2.forward(&x.narrow(1, 0, 3).unwrap(), Some(&prefix_mask), 0, true).unwrap();
    let tail_mask = mask.narrow(0, 3, 2).unwrap();
    let tail = block2.forward(&x.narrow(1, 3, 2).unwrap(), Some(&tail_mask), 3, true).unwrap();

    let full_tail: Vec<f32> = full.narrow(1, 3, 2).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let tail_vec: Vec<f32> = tail.flatten_all().unwrap().to_vec1().unwrap();
    let max_diff: f32 = full_tail.iter().zip(tail_vec.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_diff < 3e-5, "TransformerBlock KV cache mismatch: {max_diff}");
}

#[test]
fn test_llm() {
    let varmap = VarMap::new();
    let mut model = LLM::new(11, 12, 3, 8, 16, 2, &varmap).unwrap();
    let tokens = Tensor::new(&[[0u32, 1, 2, 3, 4]], &DEVICE).unwrap();

    let out = model.forward(&tokens, 0, false).unwrap();
    assert_eq!(out.dims(), &[1, 5, 11]);

    // KV cache consistency
    let mut model2 = LLM::new(11, 12, 3, 8, 16, 2, &varmap).unwrap();
    let _prefix = model2.forward(&tokens.narrow(1, 0, 3).unwrap(), 0, true).unwrap();
    let tail = model2.forward(&tokens.narrow(1, 3, 2).unwrap(), 3, true).unwrap();

    let out_tail: Vec<f32> = out.narrow(1, 3, 2).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let tail_vec: Vec<f32> = tail.flatten_all().unwrap().to_vec1().unwrap();
    let max_diff: f32 = out_tail.iter().zip(tail_vec.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_diff < 6e-5, "LLM KV cache mismatch: {max_diff}");
}

// ============================================================
// Part III: Training
// ============================================================

#[test]
fn test_cross_entropy_loss_2d() {
    let logits = Tensor::new(&[[2.0f32, 1.0, 0.0], [0.0, 2.0, 1.0]], &DEVICE).unwrap();
    let y = Tensor::new(&[0u32, 2], &DEVICE).unwrap();
    let loss = cross_entropy_loss(&logits, &y).unwrap();
    let val: f32 = loss.to_scalar().unwrap();
    assert!((val - 0.9076060).abs() < 1e-5);
}

#[test]
fn test_cross_entropy_loss_3d() {
    // Multi-dimensional logits: (batch, seq, classes)
    let logits = Tensor::randn(0.0f32, 1.0, &[4, 5, 7], &DEVICE).unwrap();
    let y_data: Vec<u32> = (0..20).map(|i| i % 7).collect();
    let y = Tensor::from_vec(y_data, &[4, 5], &DEVICE).unwrap();
    let loss = cross_entropy_loss(&logits, &y).unwrap();
    let val: f32 = loss.to_scalar().unwrap();
    assert!(val.is_finite() && val > 0.0);
}

#[test]
fn test_pretokenize_data() {
    let tmp = tempfile::tempdir().unwrap();
    let in_path = tmp.path().join("sample.txt");
    let out_path = tmp.path().join("sample.bin");

    std::fs::write(&in_path, "abcdefg").unwrap();

    let encode_fn = |text: &str| -> Vec<u16> {
        text.chars().map(|c| c as u16).collect()
    };

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
    let xb0: Vec<Vec<u32>> = {
        let t = &batches[0].0;
        (0..t.dims()[0])
            .map(|i| t.get(i).unwrap().to_vec1::<u32>().unwrap())
            .collect()
    };
    assert_eq!(xb0, vec![vec![0, 1, 2], vec![4, 5, 6]]);

    let yb0: Vec<Vec<u32>> = {
        let t = &batches[0].1;
        (0..t.dims()[0])
            .map(|i| t.get(i).unwrap().to_vec1::<u32>().unwrap())
            .collect()
    };
    assert_eq!(yb0, vec![vec![1, 2, 3], vec![5, 6, 7]]);
}

#[test]
fn test_adam() {
    let varmap = VarMap::new();
    let layer = Linear::new(6, 3, &varmap, "adam_layer").unwrap();

    let params: Vec<Tensor> = varmap.all_vars().iter().map(|v| v.as_tensor().clone()).collect();
    let mut opt = Adam::new(params.clone(), 1e-3, (0.9, 0.95), 1e-8);

    let w_before: Vec<f32> = params[0].flatten_all().unwrap().to_vec1().unwrap();

    // 5 training steps
    for _ in 0..5 {
        let x = Tensor::randn(0.0f32, 1.0, &[16, 6], &DEVICE).unwrap();
        let y_data: Vec<u32> = (0..16).map(|i| i % 3).collect();
        let y = Tensor::from_vec(y_data, &[16], &DEVICE).unwrap();

        opt.zero_grad().unwrap();
        let logits = layer.forward(&x).unwrap();
        let loss = cross_entropy_loss(&logits, &y).unwrap();
        loss.backward().unwrap();
        opt.step().unwrap();
    }

    let w_after: Vec<f32> = params[0].flatten_all().unwrap().to_vec1().unwrap();
    let changed = w_after.iter().zip(w_before.iter()).any(|(a, b)| (a - b).abs() > 1e-7);
    assert!(changed, "Adam did not update weights after 5 steps");
}

#[test]
fn test_adam_zero_grad() {
    let varmap = VarMap::new();
    let layer = Linear::new(4, 3, &varmap, "adam_zg").unwrap();

    let params: Vec<Tensor> = varmap.all_vars().iter().map(|v| v.as_tensor().clone()).collect();
    let mut opt = Adam::new(params.clone(), 1e-2, (0.9, 0.999), 1e-8);

    // Create gradients
    let x = Tensor::randn(0.0f32, 1.0, &[4, 4], &DEVICE).unwrap();
    let y = Tensor::new(&[0u32, 1, 2, 0], &DEVICE).unwrap();
    let logits = layer.forward(&x).unwrap();
    let loss = cross_entropy_loss(&logits, &y).unwrap();
    loss.backward().unwrap();

    // zero_grad then step should be a no-op on weights (first step with zero grad)
    opt.zero_grad().unwrap();
    let w_before: Vec<f32> = params[0].flatten_all().unwrap().to_vec1().unwrap();
    opt.step().unwrap();
    let w_after: Vec<f32> = params[0].flatten_all().unwrap().to_vec1().unwrap();
    for (a, b) in w_after.iter().zip(w_before.iter()) {
        assert!((a - b).abs() < 1e-7, "Adam zero_grad didn't prevent update");
    }
}

#[test]
fn test_train_llm() {
    let varmap = VarMap::new();
    let layer = Linear::new(4, 5, &varmap, "train_layer").unwrap();

    let params: Vec<Tensor> = varmap.all_vars().iter().map(|v| v.as_tensor().clone()).collect();
    let mut opt = Adam::new(params.clone(), 0.01, (0.9, 0.999), 1e-8);

    let w_before: Vec<f32> = params[0].flatten_all().unwrap().to_vec1().unwrap();

    // Create loader data
    let x1 = Tensor::new(&[[0u32, 1, 2], [1, 2, 3]], &DEVICE).unwrap();
    let y1 = Tensor::new(&[[1u32, 2, 3], [2, 3, 4]], &DEVICE).unwrap();
    let x2 = Tensor::new(&[[2u32, 3, 4], [0, 2, 4]], &DEVICE).unwrap();
    let y2 = Tensor::new(&[[3u32, 4, 0], [2, 4, 1]], &DEVICE).unwrap();
    let loader = vec![(x1, y1), (x2, y2)];

    let model_fn = |tokens: &Tensor| -> candle_core::Result<Tensor> {
        // Simple embedding-like model: just lookup and project
        let embed = Tensor::randn(0.0f32, 1.0, &[5, 4], &DEVICE)?;
        let x = embed.index_select(tokens, 0)?;
        layer.forward(&x)
    };

    train_llm(&model_fn, &loader, &mut opt).unwrap();

    let w_after: Vec<f32> = params[0].flatten_all().unwrap().to_vec1().unwrap();
    let changed = w_after.iter().zip(w_before.iter()).any(|(a, b)| (a - b).abs() > 1e-7);
    assert!(changed, "train_llm did not update weights");
}

#[test]
fn test_generate() {
    let mut call_count = 0usize;
    let next_tokens: Vec<u32> = vec![3, 4];

    let mut model_fn =
        |tokens: &Tensor, _seq_pos: usize, _use_cache: bool| -> candle_core::Result<Tensor> {
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

    let generated = generate(&mut model_fn, &[1, 2], &decode_fn, 4, 0.7, 5, false).unwrap();
    assert_eq!(generated, vec![3, 4]);
}

#[test]
fn test_generate_max_tokens() {
    let mut model_fn =
        |tokens: &Tensor, _seq_pos: usize, _use_cache: bool| -> candle_core::Result<Tensor> {
            let vocab_size = 6;
            let seq_len = tokens.dims()[1];
            let mut data = vec![f32::NEG_INFINITY; seq_len * vocab_size];
            // Always predict token 3 (never stop)
            data[(seq_len - 1) * vocab_size + 3] = 0.0;
            Tensor::from_slice(&data, &[1, seq_len, vocab_size], &Device::Cpu)
        };

    let decode_fn = |_: &[u32]| -> String { String::new() };
    let generated = generate(&mut model_fn, &[1, 2], &decode_fn, 4, 0.7, 3, false).unwrap();
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
    let mut model = eval_llm().unwrap();

    let eval_tokens = tiny_stories_eval::get_eval_tokens(0, 48);
    let tokens = Tensor::from_vec(eval_tokens.clone(), &[1, 48], &DEVICE).unwrap();

    // Compute sequence loss: cross-entropy of model predicting next tokens
    let logits = model.forward(&tokens.narrow(1, 0, 47).unwrap(), 0, false).unwrap();
    let logits_flat = logits.reshape(&[47, logits.dims()[2]]).unwrap();
    let targets: Vec<u32> = eval_tokens[1..].to_vec();
    let targets_t = Tensor::from_vec(targets, &[47], &DEVICE).unwrap();
    let phrase_loss: f32 = cross_entropy_loss(&logits_flat, &targets_t)
        .unwrap()
        .to_scalar()
        .unwrap();

    // Corrupted: reverse the token order (except first)
    let mut corrupted_tokens = eval_tokens.clone();
    corrupted_tokens[1..].reverse();
    let corrupted = Tensor::from_vec(corrupted_tokens.clone(), &[1, 48], &DEVICE).unwrap();
    let c_logits = model.forward(&corrupted.narrow(1, 0, 47).unwrap(), 0, false).unwrap();
    let c_logits_flat = c_logits.reshape(&[47, c_logits.dims()[2]]).unwrap();
    let c_targets: Vec<u32> = corrupted_tokens[1..].to_vec();
    let c_targets_t = Tensor::from_vec(c_targets, &[47], &DEVICE).unwrap();
    let corrupted_loss: f32 = cross_entropy_loss(&c_logits_flat, &c_targets_t)
        .unwrap()
        .to_scalar()
        .unwrap();

    assert!(
        phrase_loss < 7.0,
        "eval_llm phrase_loss {phrase_loss} should be < 7.0"
    );
    assert!(
        phrase_loss < corrupted_loss,
        "phrase_loss {phrase_loss} should be < corrupted_loss {corrupted_loss}"
    );

    // KV cache consistency
    let full = model.forward(&tokens.narrow(1, 0, 47).unwrap(), 0, false).unwrap();
    let mut model2 = eval_llm().unwrap();
    let _prefix = model2.forward(&tokens.narrow(1, 0, 46).unwrap(), 0, true).unwrap();
    let tail = model2.forward(&tokens.narrow(1, 46, 1).unwrap(), 46, true).unwrap();

    let full_last: Vec<f32> = full.narrow(1, 46, 1).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let tail_vec: Vec<f32> = tail.flatten_all().unwrap().to_vec1().unwrap();
    let max_diff: f32 = full_last.iter().zip(tail_vec.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 3e-4,
        "eval_llm KV cache mismatch: max_diff={max_diff}"
    );

    // Outputs should be finite
    let first_logits: Vec<f32> = full.narrow(2, 0, 16.min(full.dims()[2])).unwrap()
        .flatten_all().unwrap().to_vec1().unwrap();
    for v in &first_logits {
        assert!(v.is_finite(), "eval_llm has non-finite logits");
    }
}
