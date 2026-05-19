use burn::backend::ndarray::NdArrayDevice;
use burn::tensor::{Int, Tensor, TensorData};
use hw7::*;
use std::collections::HashMap;

const DEVICE: NdArrayDevice = NdArrayDevice::Cpu;

fn special_tokens() -> HashMap<String, u32> {
    let mut m = HashMap::new();
    m.insert("<QUESTION>".to_string(), 91);
    m.insert("</QUESTION>".to_string(), 92);
    m.insert("<THINK>".to_string(), 93);
    m.insert("</THINK>".to_string(), 94);
    m.insert("<TOOL>".to_string(), 95);
    m.insert("</TOOL>".to_string(), 96);
    m.insert("<RESPONSE>".to_string(), 97);
    m.insert("</RESPONSE>".to_string(), 98);
    m.insert("<ANSWER>".to_string(), 99);
    m.insert("</ANSWER>".to_string(), 100);
    m
}

fn char_encode(text: &str) -> Vec<u32> {
    text.chars().map(|c| c as u32).collect()
}

fn char_decode(tokens: &[u32]) -> String {
    tokens
        .iter()
        .filter_map(|&t| {
            if t == 0 {
                None
            } else {
                char::from_u32(t).map(|c| c.to_string())
            }
        })
        .collect()
}

// ============================================================
// Part I: Parallel Generation
// ============================================================

#[test]
fn test_generate_parallel_basic() {
    let next_tokens: Vec<Vec<i32>> = vec![
        vec![3, 4, 5], // step 0: each of 3 completions gets a different token
        vec![6, 6, 6], // step 1: all get eot_token=6
    ];
    let mut call_count = 0usize;

    let mut model_fn =
        |tokens: Tensor<B, 2, Int>, _seq_pos: usize, _use_cache: bool| -> Tensor<B, 3> {
            let batch = tokens.dims()[0];
            let seq_len = tokens.dims()[1];
            let vocab_size = 10;
            let mut data = vec![f32::NEG_INFINITY; batch * seq_len * vocab_size];
            let step_tokens = &next_tokens[call_count];
            for i in 0..batch {
                data[i * seq_len * vocab_size
                    + (seq_len - 1) * vocab_size
                    + step_tokens[i] as usize] = 0.0;
            }
            call_count += 1;
            Tensor::<B, 3>::from_data(TensorData::new(data, [batch, seq_len, vocab_size]), &DEVICE)
        };

    let tokens = generate_parallel(&mut model_fn, &[1, 2], 3, Some(6), 0.7, 5);

    // Shape: (3, total_len) where total_len includes prompt + generated
    assert_eq!(tokens.dims()[0], 3);
    let row0: Vec<i32> = tokens
        .clone()
        .narrow(0, 0, 1)
        .into_data()
        .to_vec::<i32>()
        .unwrap();
    // Row 0 should contain prompt [1,2] then generated tokens [3, 6, ...]
    assert_eq!(row0[0], 1);
    assert_eq!(row0[1], 2);
    assert_eq!(row0[2], 3);
}

#[test]
fn test_generate_parallel_max_tokens() {
    // Never hits eot_token, should stop at max_tokens
    let mut model_fn =
        |tokens: Tensor<B, 2, Int>, _seq_pos: usize, _use_cache: bool| -> Tensor<B, 3> {
            let batch = tokens.dims()[0];
            let seq_len = tokens.dims()[1];
            let vocab_size = 10;
            let mut data = vec![f32::NEG_INFINITY; batch * seq_len * vocab_size];
            for i in 0..batch {
                data[i * seq_len * vocab_size + (seq_len - 1) * vocab_size + 3] = 0.0;
            }
            Tensor::<B, 3>::from_data(TensorData::new(data, [batch, seq_len, vocab_size]), &DEVICE)
        };

    let tokens = generate_parallel(&mut model_fn, &[1], 2, None, 0.7, 4);
    // Max 4 tokens total (including prompt of 1)
    assert_eq!(tokens.dims()[0], 2);
    assert!(tokens.dims()[1] <= 4);
}

// ============================================================
// Part II: GSM8K Format and SFT
// ============================================================

#[test]
fn test_gsm8k_to_format() {
    let result = convert_gsm8k_to_format(
        "What is 6 plus 7?",
        "First compute <<6+7=13>>.\nThen compute <<13*2=26>>.\n#### 26",
    );
    let expected = concat!(
        "<QUESTION>What is 6 plus 7?</QUESTION>",
        "<THINK>First compute <TOOL>6+7</TOOL><RESPONSE>13</RESPONSE>.\n",
        "Then compute <TOOL>13*2</TOOL><RESPONSE>26</RESPONSE>.</THINK>",
        "<ANSWER>26</ANSWER>"
    );
    assert_eq!(result, expected);
}

#[test]
fn test_gsm8k_to_format_no_tool() {
    let result = convert_gsm8k_to_format("No tool call here?", "Think carefully.\n#### 11");
    let expected =
        "<QUESTION>No tool call here?</QUESTION><THINK>Think carefully.</THINK><ANSWER>11</ANSWER>";
    assert_eq!(result, expected);
}

#[test]
fn test_pretokenize_gsm8k() {
    let dir = tempfile::tempdir().unwrap();
    let in_path = dir.path().join("gsm8k.json");
    let out_path = dir.path().join("tokens.json");

    let input_json = r#"[
        {"question": "Q1", "answer": "Use <<2+3=5>>.\n#### 5"},
        {"question": "Q2", "answer": "No tool.\n#### 9"}
    ]"#;
    std::fs::write(&in_path, input_json).unwrap();

    pretokenize_gsm8k(
        &char_encode,
        in_path.to_str().unwrap(),
        out_path.to_str().unwrap(),
    );

    let output = std::fs::read_to_string(&out_path).unwrap();
    let tokens: Vec<Vec<u32>> = serde_json::from_str(&output).unwrap();
    assert_eq!(tokens.len(), 2);
    assert!(!tokens[0].is_empty());
    assert!(!tokens[1].is_empty());
}

#[test]
fn test_get_loss_mask_gsm8k() {
    let specials = special_tokens();
    // [<QUESTION>, 1, </QUESTION>, <THINK>, 2, <TOOL>, 3, </TOOL>,
    //  <RESPONSE>, 4, </RESPONSE>, 5, </THINK>, <ANSWER>, 6, </ANSWER>, 7]
    let tokens: Vec<u32> = vec![91, 1, 92, 93, 2, 95, 3, 96, 97, 4, 98, 5, 94, 99, 6, 100, 7];
    let expected = vec![
        false, false, false, false, // <QUESTION>, 1, </QUESTION>, <THINK>
        true, true, true, true, // 2, <TOOL>, 3, </TOOL>
        false, false, false, // <RESPONSE>, 4, </RESPONSE>
        true, true, true, true, true,  // 5, </THINK>, <ANSWER>, 6, </ANSWER>
        false, // 7 (after </ANSWER>)
    ];
    assert_eq!(get_loss_mask(&tokens, &specials), expected);
}

#[test]
fn test_get_loss_mask_no_tool() {
    let specials = special_tokens();
    // [<QUESTION>, 1, </QUESTION>, <THINK>, 2, <ANSWER>, 3, </ANSWER>, 4]
    let tokens: Vec<u32> = vec![91, 1, 92, 93, 2, 99, 3, 100, 4];
    let expected = vec![
        false, false, false, false, // <QUESTION>, 1, </QUESTION>, <THINK>
        true, true, true, true,  // 2, <ANSWER>, 3, </ANSWER>
        false, // 4
    ];
    assert_eq!(get_loss_mask(&tokens, &specials), expected);
}

// ============================================================
// Part III: Tool Evaluation and Answer Extraction
// ============================================================

#[test]
fn test_eval_tool_basic() {
    assert_eq!(eval_tool("6/3"), "2");
    assert_eq!(eval_tool("7/2"), "3.5");
    assert_eq!(eval_tool("3.00001"), "3");
}

#[test]
fn test_eval_tool_error() {
    assert_eq!(eval_tool("1/0"), "ERROR");
}

#[test]
fn test_eval_tool_expressions() {
    assert_eq!(eval_tool("2+3"), "5");
    assert_eq!(eval_tool("(10+5)*2"), "30");
    assert_eq!(eval_tool("48/2"), "24");
}

#[test]
fn test_extract_answer_found() {
    assert_eq!(extract_answer("blah <ANSWER>42</ANSWER> blah"), Some(42));
    assert_eq!(extract_answer("<ANSWER>3</ANSWER>"), Some(3));
    assert_eq!(
        extract_answer("<THINK>x</THINK><ANSWER>-7</ANSWER>"),
        Some(-7)
    );
}

#[test]
fn test_extract_answer_not_found() {
    assert_eq!(extract_answer("no answer here"), None);
    assert_eq!(extract_answer("<ANSWER>oops</ANSWER>"), None);
    assert_eq!(extract_answer("<THINK>no answer</THINK>"), None);
}

#[test]
fn test_grade_responses() {
    // 3 completions, ground truth answer = 9
    // Row 0: has <ANSWER>9</ANSWER> -> correct + formatted = 1.0 + 0.2 = 1.2
    // Row 1: has <ANSWER>4</ANSWER> -> wrong but formatted = 0.2
    // Row 2: no <ANSWER> tag -> 0.0
    let row0_text = "<THINK>a</THINK><ANSWER>9</ANSWER>";
    let row1_text = "<THINK>b</THINK><ANSWER>4</ANSWER>";
    let row2_text = "<THINK>c</THINK>";

    // Build token tensor
    let r0: Vec<i32> = char_encode(row0_text)
        .into_iter()
        .map(|x| x as i32)
        .collect();
    let r1: Vec<i32> = char_encode(row1_text)
        .into_iter()
        .map(|x| x as i32)
        .collect();
    let r2: Vec<i32> = char_encode(row2_text)
        .into_iter()
        .map(|x| x as i32)
        .collect();
    let max_len = r0.len().max(r1.len()).max(r2.len());
    let mut data: Vec<i32> = Vec::new();
    for row in [&r0, &r1, &r2] {
        data.extend(row);
        data.extend(vec![0i32; max_len - row.len()]);
    }
    let tokens = Tensor::<B, 2, Int>::from_data(TensorData::new(data, [3, max_len]), &DEVICE);

    let scores = grade_responses(&char_decode, tokens, 9, 1.0, 0.2);
    assert_eq!(scores.len(), 3);
    assert!((scores[0] - 1.2).abs() < 1e-6);
    assert!((scores[1] - 0.2).abs() < 1e-6);
    assert!((scores[2] - 0.0).abs() < 1e-6);
}

// ============================================================
// Part IV: RL Loss
// ============================================================

#[test]
fn test_rl_loss_shape() {
    let specials = special_tokens();
    let mask_fn = |tokens: &[u32]| -> Vec<bool> { get_loss_mask(tokens, &specials) };

    // Build tokens as if they were: <QUESTION>Q</QUESTION><THINK>a</THINK><ANSWER>5</ANSWER>
    let row: Vec<i32> = vec![91, 81, 92, 93, 65, 94, 99, 53, 100]; // Q=81, a=65, 5=53
    let tokens_data: Vec<i32> = row.iter().chain(row.iter()).cloned().collect();
    let tokens =
        Tensor::<B, 2, Int>::from_data(TensorData::new(tokens_data, [2, row.len()]), &DEVICE);
    let rewards = vec![1.5, -0.5];

    let model_fn = |t: Tensor<B, 2, Int>| -> Tensor<B, 3> {
        let dims = t.dims();
        Tensor::<B, 3>::zeros([dims[0], dims[1], 128], &DEVICE)
    };

    let loss = rl_loss(&model_fn, tokens, &rewards, &mask_fn);
    let val: f32 = loss.into_scalar();
    assert!(val.is_finite(), "RL loss should be finite, got {val}");
}

#[test]
fn test_train_llm_sft_api() {
    // Verify the function is callable and handles max_iter=0 gracefully
    let model_fn = |t: Tensor<B, 2, Int>| -> Tensor<B, 3> {
        let dims = t.dims();
        Tensor::<B, 3>::zeros([dims[0], dims[1], 10], &DEVICE)
    };
    let mut optimizer_fn = || {};
    let loader: Vec<BatchItem> = vec![];

    train_llm_sft(&model_fn, &loader, &mut optimizer_fn, Some(0));
}

// ============================================================
// End-to-end: eval_reasoning_model
// ============================================================

#[test]
fn test_eval_reasoning_model() {
    let mut model_fn = eval_reasoning_model();

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
        assert!(v.is_finite(), "eval_reasoning_model has non-finite logits");
    }

    // KV cache consistency
    let mut model_fn2 = eval_reasoning_model();
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
        "eval_reasoning_model KV cache mismatch: max_diff={max_diff}"
    );
}
