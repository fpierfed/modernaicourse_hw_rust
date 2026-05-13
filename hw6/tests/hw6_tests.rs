use candle_core::{Device, Tensor};
use hw6::*;

#[allow(unused)]
const DEVICE: Device = Device::Cpu;

// ============================================================
// Part I: Chat Format and SFT
// ============================================================

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
fn test_pretokenize_chat() {
    let dir = tempfile::tempdir().unwrap();
    let in_path = dir.path().join("chats.json");
    let out_path = dir.path().join("tokens.json");

    // Write input: list of conversations, each conversation is a list of messages
    let input_json = r#"[
        [{"role": "user", "content": "One"}, {"role": "assistant", "content": "Two"}],
        [{"role": "user", "content": "Three"}]
    ]"#;
    std::fs::write(&in_path, input_json).unwrap();

    // Simple encoder: each character becomes its ASCII value
    let encode_fn = |text: &str| -> Vec<u32> {
        text.chars().map(|c| c as u32).collect()
    };

    pretokenize_chat(&encode_fn, &in_path, &out_path);

    let output = std::fs::read_to_string(&out_path).unwrap();
    let tokens: Vec<Vec<u32>> = serde_json::from_str(&output).unwrap();

    assert_eq!(tokens.len(), 2);
    // First conversation tokenizes "<USER>One</USER><ASSISTANT>Two</ASSISTANT>"
    // Second tokenizes "<USER>Three</USER>"
    assert!(!tokens[0].is_empty());
    assert!(!tokens[1].is_empty());
    assert!(tokens[0].len() > tokens[1].len());
}

#[test]
fn test_get_loss_mask() {
    let assistant_start: u32 = 93;
    let assistant_end: u32 = 94;
    // tokens: [7, <ASSISTANT>, 1, 2, </ASSISTANT>, 8]
    let tokens = vec![7, assistant_start, 1, 2, assistant_end, 8];
    let expected = vec![false, false, true, true, true, false];
    assert_eq!(get_loss_mask(&tokens, assistant_start, assistant_end), expected);
}

#[test]
fn test_get_loss_mask_multiple_regions() {
    let assistant_start: u32 = 93;
    let assistant_end: u32 = 94;
    // [<USER>, 10, <ASSISTANT>, 20, 21, </ASSISTANT>, <ASSISTANT>, 30, </ASSISTANT>]
    let tokens = vec![91, 10, assistant_start, 20, 21, assistant_end, assistant_start, 30, assistant_end];
    let expected = vec![false, false, false, true, true, true, false, true, true];
    assert_eq!(get_loss_mask(&tokens, assistant_start, assistant_end), expected);
}

#[test]
fn test_get_loss_mask_no_assistant() {
    let assistant_start: u32 = 93;
    let assistant_end: u32 = 94;
    let tokens = vec![91, 7, 8, 92, 9];
    let expected = vec![false, false, false, false, false];
    assert_eq!(get_loss_mask(&tokens, assistant_start, assistant_end), expected);
}

#[test]
fn test_dataloader_chat() {
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

    // Verify first batch shapes
    let (x0, y0, m0) = &batches[0];
    assert_eq!(x0.dims()[0], 2);
    assert_eq!(x0.dims()[1], 6);
    assert_eq!(y0.dims()[0], 2);
    assert_eq!(y0.dims()[1], 6);
    assert_eq!(m0.dims()[0], 2);
    assert_eq!(m0.dims()[1], 6);
}

#[test]
fn test_dataloader_chat_reiterable() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chat_tokens2.json");

    let chats: Vec<Vec<u32>> = vec![
        vec![1, 93, 10, 11, 94],
        vec![2, 93, 12, 94],
    ];
    std::fs::write(&path, serde_json::to_string(&chats).unwrap()).unwrap();

    let loader1 = DataLoaderChat::new(&path, 6, 2);
    let batches1: Vec<_> = loader1.collect();

    let loader2 = DataLoaderChat::new(&path, 6, 2);
    let batches2: Vec<_> = loader2.collect();

    assert_eq!(batches1.len(), batches2.len());
    for ((x1, _y1, _m1), (x2, _y2, _m2)) in batches1.iter().zip(batches2.iter()) {
        let x1v: Vec<f32> = x1.flatten_all().unwrap().to_vec1().unwrap();
        let x2v: Vec<f32> = x2.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(x1v, x2v);
    }
}

#[test]
fn test_train_chat_sft() {
    // Verify the train_chat_sft API is callable with correct types.
    // Full integration requires a working DataLoaderChat with real data.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty_chats.json");
    std::fs::write(&path, "[]").unwrap();

    let mut loader = DataLoaderChat::new(&path, 6, 2);

    let model_fn = |tokens: &Tensor| -> candle_core::Result<Tensor> {
        Tensor::randn(0.0f32, 1.0, &[tokens.dims()[0], tokens.dims()[1], 5], &Device::Cpu)
    };
    let mut optimizer_fn = || -> candle_core::Result<()> { Ok(()) };

    // With max_iter=0, should do nothing (no batches processed)
    let result = train_chat_sft(&model_fn, &mut loader, &mut optimizer_fn, Some(0));
    assert!(result.is_ok());
}

// ============================================================
// Part II: DPO
// ============================================================

#[test]
fn test_log_probs() {
    let logits = Tensor::new(
        &[
            [[2.0f32, 0.0, -1.0], [0.5, 1.5, -0.5], [1.0, -1.0, 0.0]],
            [[-0.5, 1.0, 0.0], [2.0, 0.0, -2.0], [0.25, 0.25, 0.25]],
        ],
        &DEVICE,
    )
    .unwrap();
    let y = Tensor::new(&[[0u32, 1, 2], [1, 0, 2]], &DEVICE).unwrap();
    let mask = Tensor::new(&[[1u8, 0, 1], [1, 1, 0]], &DEVICE).unwrap();

    let out = log_probs(&logits, &y, &mask).unwrap();
    assert_eq!(out.dims(), &[2]);

    let vals: Vec<f32> = out.to_vec1().unwrap();
    // Expected ≈ [-1.5775, -0.6073] from Python reference
    assert!(
        (vals[0] - (-1.5775)).abs() < 1e-3,
        "log_probs[0] = {}, expected ≈ -1.5775",
        vals[0]
    );
    assert!(
        (vals[1] - (-0.6073)).abs() < 1e-3,
        "log_probs[1] = {}, expected ≈ -0.6073",
        vals[1]
    );
}

#[test]
fn test_softplus() {
    let x = Tensor::new(&[-3.0f32, -0.5, 0.0, 2.0], &DEVICE).unwrap();
    let out = softplus(&x, 0.7).unwrap();
    let expected = vec![0.1155f32, 0.5334, 0.6931, 1.6204];
    let vals: Vec<f32> = out.to_vec1().unwrap();
    for (a, e) in vals.iter().zip(expected.iter()) {
        assert!((a - e).abs() < 1e-3, "softplus mismatch: got {a}, expected {e}");
    }
}

#[test]
fn test_softplus_2d() {
    let x = Tensor::new(&[[1.0f32, -1.0], [0.25, -0.25]], &DEVICE).unwrap();
    let out = softplus(&x, 0.3).unwrap();
    // softplus(x, beta) = log(1 + exp(beta*x))
    let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    let x_vals: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();
    for (v, &xv) in vals.iter().zip(x_vals.iter()) {
        let expected = (1.0 + (0.3 * xv).exp()).ln();
        assert!((v - expected).abs() < 1e-5, "softplus 2D mismatch");
    }
}

#[test]
fn test_dpo_loss_shape() {
    // Minimal DPO loss test: verify shape and differentiability
    let model = |x: &Tensor| -> candle_core::Result<Tensor> {
        // Simple identity-ish: return fixed logits
        let batch = x.dims()[0];
        let seq = x.dims()[1];
        Tensor::randn(0.0f32, 1.0, &[batch, seq, 3], &DEVICE)
    };
    let model_ref = |x: &Tensor| -> candle_core::Result<Tensor> {
        let batch = x.dims()[0];
        let seq = x.dims()[1];
        Tensor::randn(0.0f32, 1.0, &[batch, seq, 3], &DEVICE)
    };

    let xp = Tensor::new(&[[0u32, 1, 2], [1, 2, 0]], &DEVICE).unwrap();
    let yp = Tensor::new(&[[1u32, 2, 0], [2, 0, 1]], &DEVICE).unwrap();
    let maskp = Tensor::new(&[[1u8, 1, 0], [0, 1, 1]], &DEVICE).unwrap();
    let xn = Tensor::new(&[[0u32, 2, 1], [2, 1, 0]], &DEVICE).unwrap();
    let yn = Tensor::new(&[[2u32, 0, 1], [1, 0, 2]], &DEVICE).unwrap();
    let maskn = Tensor::new(&[[1u8, 0, 1], [1, 1, 0]], &DEVICE).unwrap();

    let loss = dpo_loss(&model, &model_ref, &xp, &yp, &maskp, &xn, &yn, &maskn, 0.3).unwrap();
    assert_eq!(loss.dims(), &[2]);
    let vals: Vec<f32> = loss.to_vec1().unwrap();
    for v in &vals {
        assert!(v.is_finite(), "DPO loss is not finite");
        assert!(*v >= 0.0, "DPO loss should be non-negative (softplus output)");
    }
}

// ============================================================
// End-to-end: eval_llm_chat and eval_llm_dpo
// ============================================================

#[test]
fn test_eval_llm_chat() {
    let mut model_fn = eval_llm_chat().unwrap();

    // Basic forward pass with some token IDs
    let tokens = Tensor::new(&[[0u32, 1, 2, 3]], &Device::Cpu).unwrap();
    let full = model_fn(&tokens, 0, false).unwrap();

    // Should produce logits of shape (1, 4, vocab_size)
    assert_eq!(full.dims()[0], 1);
    assert_eq!(full.dims()[1], 4);
    let vocab_size = full.dims()[2];
    assert!(vocab_size > 0);

    // Outputs should be finite
    let first_logits: Vec<f32> = full
        .narrow(2, 0, 16.min(vocab_size))
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    for v in &first_logits {
        assert!(v.is_finite(), "eval_llm_chat has non-finite logits");
    }

    // KV cache consistency: full[:, 3:] should match cached tail
    let mut model_fn2 = eval_llm_chat().unwrap();
    let _prefix = model_fn2(&tokens.narrow(1, 0, 3).unwrap(), 0, true).unwrap();
    let tail = model_fn2(&tokens.narrow(1, 3, 1).unwrap(), 3, true).unwrap();

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
        "eval_llm_chat KV cache mismatch: max_diff={max_diff}"
    );
}

#[test]
fn test_eval_llm_dpo() {
    let mut model_fn = eval_llm_dpo().unwrap();

    // Basic forward pass
    let tokens = Tensor::new(&[[0u32, 1, 2, 3]], &Device::Cpu).unwrap();
    let full = model_fn(&tokens, 0, false).unwrap();

    assert_eq!(full.dims()[0], 1);
    assert_eq!(full.dims()[1], 4);
    let vocab_size = full.dims()[2];
    assert!(vocab_size > 0);

    // Outputs should be finite
    let first_logits: Vec<f32> = full
        .narrow(2, 0, 16.min(vocab_size))
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    for v in &first_logits {
        assert!(v.is_finite(), "eval_llm_dpo has non-finite logits");
    }

    // KV cache consistency
    let mut model_fn2 = eval_llm_dpo().unwrap();
    let _prefix = model_fn2(&tokens.narrow(1, 0, 3).unwrap(), 0, true).unwrap();
    let tail = model_fn2(&tokens.narrow(1, 3, 1).unwrap(), 3, true).unwrap();

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
        "eval_llm_dpo KV cache mismatch: max_diff={max_diff}"
    );
}
