use burn::backend::ndarray::NdArrayDevice;
use burn::tensor::{Int, Tensor, TensorData};
use hw6::*;

#[allow(unused)]
const DEVICE: NdArrayDevice = NdArrayDevice::Cpu;

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
    let encode_fn = |text: &str| -> Vec<u32> { text.chars().map(|c| c as u32).collect() };

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
    assert_eq!(
        get_loss_mask(&tokens, assistant_start, assistant_end),
        expected
    );
}

#[test]
fn test_get_loss_mask_multiple_regions() {
    let assistant_start: u32 = 93;
    let assistant_end: u32 = 94;
    // [<USER>, 10, <ASSISTANT>, 20, 21, </ASSISTANT>, <ASSISTANT>, 30, </ASSISTANT>]
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

    let chats: Vec<Vec<u32>> = vec![vec![1, 93, 10, 11, 94], vec![2, 93, 12, 94]];
    std::fs::write(&path, serde_json::to_string(&chats).unwrap()).unwrap();

    let loader1 = DataLoaderChat::new(&path, 6, 2);
    let batches1: Vec<_> = loader1.collect();

    let loader2 = DataLoaderChat::new(&path, 6, 2);
    let batches2: Vec<_> = loader2.collect();

    assert_eq!(batches1.len(), batches2.len());
    for ((x1, _y1, _m1), (x2, _y2, _m2)) in batches1.iter().zip(batches2.iter()) {
        let x1v: Vec<i32> = x1.clone().into_data().to_vec::<i32>().unwrap();
        let x2v: Vec<i32> = x2.clone().into_data().to_vec::<i32>().unwrap();
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

    let model_fn = |tokens: Tensor<B, 2, Int>| -> Tensor<B, 3> {
        let batch = tokens.dims()[0];
        let seq = tokens.dims()[1];
        Tensor::<B, 3>::zeros([batch, seq, 5], &DEVICE)
    };
    let mut optimizer_fn = || {};

    // With max_iter=0, should do nothing (no batches processed)
    train_chat_sft(&model_fn, &mut loader, &mut optimizer_fn, Some(0));
}

// ============================================================
// Part II: DPO
// ============================================================

#[test]
fn test_log_probs() {
    let logits = Tensor::<B, 3>::from_data(
        TensorData::new(
            vec![
                2.0f32, 0.0, -1.0, 0.5, 1.5, -0.5, 1.0, -1.0, 0.0, -0.5, 1.0, 0.0, 2.0, 0.0, -2.0,
                0.25, 0.25, 0.25,
            ],
            [2, 3, 3],
        ),
        &DEVICE,
    );
    let y =
        Tensor::<B, 2, Int>::from_data(TensorData::new(vec![0i32, 1, 2, 1, 0, 2], [2, 3]), &DEVICE);
    let mask = Tensor::<B, 2>::from_data(
        TensorData::new(vec![1.0f32, 0.0, 1.0, 1.0, 1.0, 0.0], [2, 3]),
        &DEVICE,
    );

    let out = log_probs(logits, y, mask);
    assert_eq!(out.dims(), [2]);

    let vals: Vec<f32> = out.into_data().to_vec::<f32>().unwrap();
    // Expected ~ [-1.5775, -0.6073] from Python reference
    assert!(
        (vals[0] - (-1.5775)).abs() < 1e-3,
        "log_probs[0] = {}, expected ~ -1.5775",
        vals[0]
    );
    assert!(
        (vals[1] - (-0.6073)).abs() < 1e-3,
        "log_probs[1] = {}, expected ~ -0.6073",
        vals[1]
    );
}

#[test]
fn test_softplus() {
    let x = Tensor::<B, 1>::from_data(TensorData::from([-3.0f32, -0.5, 0.0, 2.0]), &DEVICE);
    let out = softplus(x, 0.7);
    #[allow(clippy::approx_constant)]
    let expected = [0.1155f32, 0.5334, 0.6931, 1.6204];
    let vals: Vec<f32> = out.into_data().to_vec::<f32>().unwrap();
    for (a, e) in vals.iter().zip(expected.iter()) {
        assert!(
            (a - e).abs() < 1e-3,
            "softplus mismatch: got {a}, expected {e}"
        );
    }
}

#[test]
fn test_softplus_2d() {
    // Test softplus on a flattened view (burn requires matching dimensions)
    let x_data = [1.0f32, -1.0, 0.25, -0.25];
    let x = Tensor::<B, 1>::from_data(TensorData::from(x_data.as_slice()), &DEVICE);
    let out = softplus(x, 0.3);
    let vals: Vec<f32> = out.into_data().to_vec::<f32>().unwrap();
    for (v, xv) in vals.iter().zip(x_data.iter()) {
        let expected = (1.0 + (0.3 * xv).exp()).ln();
        assert!((v - expected).abs() < 1e-5, "softplus 2D mismatch");
    }
}

#[test]
fn test_dpo_loss_shape() {
    // Minimal DPO loss test: verify shape and differentiability
    let model =
        |_x: Tensor<B, 2, Int>| -> Tensor<B, 3> { Tensor::<B, 3>::zeros([2, 3, 3], &DEVICE) };
    let model_ref =
        |_x: Tensor<B, 2, Int>| -> Tensor<B, 3> { Tensor::<B, 3>::zeros([2, 3, 3], &DEVICE) };

    let xp =
        Tensor::<B, 2, Int>::from_data(TensorData::new(vec![0i32, 1, 2, 1, 2, 0], [2, 3]), &DEVICE);
    let yp =
        Tensor::<B, 2, Int>::from_data(TensorData::new(vec![1i32, 2, 0, 2, 0, 1], [2, 3]), &DEVICE);
    let maskp = Tensor::<B, 2>::from_data(
        TensorData::new(vec![1.0f32, 1.0, 0.0, 0.0, 1.0, 1.0], [2, 3]),
        &DEVICE,
    );
    let xn =
        Tensor::<B, 2, Int>::from_data(TensorData::new(vec![0i32, 2, 1, 2, 1, 0], [2, 3]), &DEVICE);
    let yn =
        Tensor::<B, 2, Int>::from_data(TensorData::new(vec![2i32, 0, 1, 1, 0, 2], [2, 3]), &DEVICE);
    let maskn = Tensor::<B, 2>::from_data(
        TensorData::new(vec![1.0f32, 0.0, 1.0, 1.0, 1.0, 0.0], [2, 3]),
        &DEVICE,
    );

    let loss = dpo_loss(&model, &model_ref, xp, yp, maskp, xn, yn, maskn, 0.3);
    assert_eq!(loss.dims(), [2]);
    let vals: Vec<f32> = loss.into_data().to_vec::<f32>().unwrap();
    for v in &vals {
        assert!(v.is_finite(), "DPO loss is not finite");
        assert!(
            *v >= 0.0,
            "DPO loss should be non-negative (softplus output)"
        );
    }
}

// ============================================================
// End-to-end: eval_llm_chat and eval_llm_dpo
// ============================================================

#[test]
fn test_eval_llm_chat() {
    let mut model_fn = eval_llm_chat();

    // Basic forward pass with some token IDs
    let tokens =
        Tensor::<B, 2, Int>::from_data(TensorData::new(vec![0i32, 1, 2, 3], [1, 4]), &DEVICE);
    let full = model_fn(tokens.clone(), 0, false);

    // Should produce logits of shape (1, 4, vocab_size)
    assert_eq!(full.dims()[0], 1);
    assert_eq!(full.dims()[1], 4);
    let vocab_size = full.dims()[2];
    assert!(vocab_size > 0);

    // Outputs should be finite
    let first_logits: Vec<f32> = full
        .clone()
        .narrow(2, 0, 16.min(vocab_size))
        .reshape([4 * 16.min(vocab_size)])
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    for v in &first_logits {
        assert!(v.is_finite(), "eval_llm_chat has non-finite logits");
    }

    // KV cache consistency: full[:, 3:] should match cached tail
    let mut model_fn2 = eval_llm_chat();
    let _prefix = model_fn2(tokens.clone().narrow(1, 0, 3), 0, true);
    let tail = model_fn2(tokens.clone().narrow(1, 3, 1), 3, true);

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
        "eval_llm_chat KV cache mismatch: max_diff={max_diff}"
    );
}

#[test]
fn test_eval_llm_dpo() {
    let mut model_fn = eval_llm_dpo();

    // Basic forward pass
    let tokens =
        Tensor::<B, 2, Int>::from_data(TensorData::new(vec![0i32, 1, 2, 3], [1, 4]), &DEVICE);
    let full = model_fn(tokens.clone(), 0, false);

    assert_eq!(full.dims()[0], 1);
    assert_eq!(full.dims()[1], 4);
    let vocab_size = full.dims()[2];
    assert!(vocab_size > 0);

    // Outputs should be finite
    let first_logits: Vec<f32> = full
        .clone()
        .narrow(2, 0, 16.min(vocab_size))
        .reshape([4 * 16.min(vocab_size)])
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    for v in &first_logits {
        assert!(v.is_finite(), "eval_llm_dpo has non-finite logits");
    }

    // KV cache consistency
    let mut model_fn2 = eval_llm_dpo();
    let _prefix = model_fn2(tokens.clone().narrow(1, 0, 3), 0, true);
    let tail = model_fn2(tokens.clone().narrow(1, 3, 1), 3, true);

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
        "eval_llm_dpo KV cache mismatch: max_diff={max_diff}"
    );
}
