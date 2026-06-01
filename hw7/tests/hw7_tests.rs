//! Tests for hw7 — Candle version.

use candle_core::{Device, DType, Result, Tensor};
use hw7::*;
use std::collections::HashMap;

// ---------- small helpers ----------

fn to_vec_f32(t: &Tensor) -> Result<Vec<f32>> {
    t.flatten_all()?.to_vec1::<f32>()
}

fn to_vec_i32(t: &Tensor) -> Result<Vec<i32>> {
    t.flatten_all()?.to_vec1::<i32>()
}

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
fn test_generate_parallel_basic() -> Result<()> {
    let device = Device::Cpu;
    let next_tokens: Vec<Vec<i32>> = vec![
        vec![3, 4, 5], // step 0: each of 3 completions gets a different token
        vec![6, 6, 6], // step 1: all get eot_token=6
    ];
    let call_count = std::cell::RefCell::new(0usize);

    let mut model_fn = |tokens: &Tensor, _seq_pos: usize, _use_cache: bool| -> Result<Tensor> {
        let batch = tokens.dims()[0];
        let seq_len = tokens.dims()[1];
        let vocab_size = 10;
        let mut data = vec![f32::NEG_INFINITY; batch * seq_len * vocab_size];
        let count = *call_count.borrow();
        let step_tokens = &next_tokens[count];
        for i in 0..batch {
            data[i * seq_len * vocab_size
                + (seq_len - 1) * vocab_size
                + step_tokens[i] as usize] = 0.0;
        }
        call_count.replace(count + 1);
        Tensor::from_vec(data, (batch, seq_len, vocab_size), &device)
    };

    let tokens = generate_parallel(&mut model_fn, &[1, 2], 3, Some(6), 0.7, 5)?;

    assert_eq!(tokens.dims()[0], 3);
    let row0 = to_vec_i32(&tokens.narrow(0, 0, 1)?)?;
    assert_eq!(row0[0], 1);
    assert_eq!(row0[1], 2);
    assert_eq!(row0[2], 3);
    Ok(())
}

#[test]
fn test_generate_parallel_max_tokens() -> Result<()> {
    let device = Device::Cpu;
    let mut model_fn = |tokens: &Tensor, _seq_pos: usize, _use_cache: bool| -> Result<Tensor> {
        let batch = tokens.dims()[0];
        let seq_len = tokens.dims()[1];
        let vocab_size = 10;
        let mut data = vec![f32::NEG_INFINITY; batch * seq_len * vocab_size];
        for i in 0..batch {
            data[i * seq_len * vocab_size + (seq_len - 1) * vocab_size + 3] = 0.0;
        }
        Tensor::from_vec(data, (batch, seq_len, vocab_size), &device)
    };

    let tokens = generate_parallel(&mut model_fn, &[1], 2, None, 0.7, 4)?;
    assert_eq!(tokens.dims()[0], 2);
    assert!(tokens.dims()[1] <= 4);
    Ok(())
}

// ============================================================
// Part II: GSM8K Format and SFT
// ============================================================

// ---------- convert_gsm8k_to_format ----------

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
fn test_convert_gsm8k_multiple_tools() {
    let result = convert_gsm8k_to_format(
        "What is (2+3)*4?",
        "First <<2+3=5>>. Then <<5*4=20>>.\n#### 20",
    );
    assert!(result.contains("<TOOL>2+3</TOOL>"));
    assert!(result.contains("<RESPONSE>5</RESPONSE>"));
    assert!(result.contains("<TOOL>5*4</TOOL>"));
    assert!(result.contains("<RESPONSE>20</RESPONSE>"));
    assert!(result.contains("<ANSWER>20</ANSWER>"));
}

// ---------- pretokenize_gsm8k ----------

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

// ---------- get_loss_mask ----------

#[test]
fn test_get_loss_mask_gsm8k() {
    let specials = special_tokens();
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
    let tokens: Vec<u32> = vec![91, 1, 92, 93, 2, 99, 3, 100, 4];
    let expected = vec![
        false, false, false, false, // <QUESTION>, 1, </QUESTION>, <THINK>
        true, true, true, true,  // 2, <ANSWER>, 3, </ANSWER>
        false, // 4
    ];
    assert_eq!(get_loss_mask(&tokens, &specials), expected);
}

#[test]
fn test_get_loss_mask_consecutive_tools() {
    let specials = special_tokens();
    let tokens: Vec<u32> = vec![
        93, 1, 95, 2, 96, 97, 3, 98, 95, 4, 96, 97, 5, 98, 94, 99, 6, 100,
    ];
    let mask = get_loss_mask(&tokens, &specials);
    // After <THINK>: true
    assert!(mask[1]); // 1 (after THINK)
    assert!(mask[2]); // <TOOL>
    assert!(mask[3]); // 2
    assert!(mask[4]); // </TOOL>
                      // After </TOOL>: false (tool response)
    assert!(!mask[5]); // <RESPONSE>
    assert!(!mask[6]); // 3
    assert!(!mask[7]); // </RESPONSE>
                       // After </RESPONSE>: true again
    assert!(mask[8]); // <TOOL>
    assert!(mask[9]); // 4
    assert!(mask[10]); // </TOOL>
                       // After second </TOOL>: false
    assert!(!mask[11]); // <RESPONSE>
    assert!(!mask[12]); // 5
    assert!(!mask[13]); // </RESPONSE>
                        // After </RESPONSE>: true
    assert!(mask[14]); // </THINK>
    assert!(mask[15]); // <ANSWER>
    assert!(mask[16]); // 6
    assert!(mask[17]); // </ANSWER>
}

// ============================================================
// Part III: Tool Evaluation and Answer Extraction
// ============================================================

// ---------- eval_tool ----------

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
fn test_eval_tool_complex_expressions() {
    assert_eq!(eval_tool("(2+3)*4"), "20");
    assert_eq!(eval_tool("100/4"), "25");
    assert_eq!(eval_tool("3+4*2"), "11");
    assert_eq!(eval_tool("(1+1)*(2+2)"), "8");
}

#[test]
fn test_eval_tool_floating_near_integer() {
    assert_eq!(eval_tool("7.99999"), "7.99999");
    assert_eq!(eval_tool("8.00001"), "8");
}

#[test]
fn test_eval_tool_invalid_syntax() {
    assert_eq!(eval_tool(""), "ERROR");
    assert_eq!(eval_tool("abc"), "ERROR");
    assert_eq!(eval_tool("1++2"), "ERROR");
}

// ---------- extract_answer ----------

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
fn test_extract_answer_multiple_tags() {
    let text = "<ANSWER>5</ANSWER> then <ANSWER>10</ANSWER>";
    let result = extract_answer(text);
    assert!(result == Some(5) || result == Some(10));
}

#[test]
fn test_extract_answer_negative() {
    assert_eq!(extract_answer("<ANSWER>-42</ANSWER>"), Some(-42));
}

#[test]
fn test_extract_answer_whitespace() {
    assert_eq!(extract_answer("<ANSWER> 7 </ANSWER>"), Some(7));
}

// ---------- grade_responses ----------

#[test]
fn test_grade_responses() -> Result<()> {
    let device = Device::Cpu;
    // 3 completions, ground truth = 9.
    let row0_text = "<THINK>a</THINK><ANSWER>9</ANSWER>";
    let row1_text = "<THINK>b</THINK><ANSWER>4</ANSWER>";
    let row2_text = "<THINK>c</THINK>";

    let r0: Vec<i32> = char_encode(row0_text).into_iter().map(|x| x as i32).collect();
    let r1: Vec<i32> = char_encode(row1_text).into_iter().map(|x| x as i32).collect();
    let r2: Vec<i32> = char_encode(row2_text).into_iter().map(|x| x as i32).collect();
    let max_len = r0.len().max(r1.len()).max(r2.len());
    let mut data: Vec<i32> = Vec::new();
    for row in [&r0, &r1, &r2] {
        data.extend(row);
        data.extend(vec![0i32; max_len - row.len()]);
    }
    let tokens = Tensor::from_vec(data, (3, max_len), &device)?;

    let correct_weight = 1.0;
    let format_weight = 0.2;
    let scores = grade_responses(&char_decode, &tokens, 9, correct_weight, format_weight);
    assert_eq!(scores.len(), 3);
    let expected = [correct_weight + format_weight, format_weight, 0.0];
    for (actual, expected) in scores.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-6);
    }
    Ok(())
}

#[test]
fn test_grade_responses_all_correct() -> Result<()> {
    let device = Device::Cpu;
    let tokens = Tensor::from_vec(vec![0i32; 6], (2, 3), &device)?;
    let decode_fn = |_: &[u32]| -> String { "<ANSWER>42</ANSWER>".to_string() };
    let correct_weight = 1.0;
    let format_weight = 0.5;
    let scores = grade_responses(&decode_fn, &tokens, 42, correct_weight, format_weight);
    for s in &scores {
        let expected = correct_weight + format_weight;
        assert!(
            *s >= expected - 1e-6,
            "Correct + formatted should score >= {expected}, got {s}"
        );
    }
    Ok(())
}

#[test]
fn test_grade_responses_all_wrong() -> Result<()> {
    let device = Device::Cpu;
    let tokens = Tensor::from_vec(vec![0i32; 6], (2, 3), &device)?;
    let decode_fn = |_: &[u32]| -> String { "<ANSWER>99</ANSWER>".to_string() };
    let correct_weight = 1.0;
    let format_weight = 0.5;
    let scores = grade_responses(&decode_fn, &tokens, 42, correct_weight, format_weight);
    for s in &scores {
        assert!(
            (*s - format_weight).abs() < 1e-6,
            "Wrong but formatted should score {format_weight}, got {s}"
        );
    }
    Ok(())
}

#[test]
fn test_grade_responses_no_answer_tag() -> Result<()> {
    let device = Device::Cpu;
    let tokens = Tensor::from_vec(vec![0i32; 3], (1, 3), &device)?;
    let decode_fn = |_: &[u32]| -> String { "no answer here".to_string() };
    let scores = grade_responses(&decode_fn, &tokens, 42, 1.0, 0.5);
    assert!(
        (scores[0] - 0.0).abs() < 1e-6,
        "No answer tag should score 0.0, got {}",
        scores[0]
    );
    Ok(())
}

// ============================================================
// Part IV: RL Loss
// ============================================================

#[test]
fn test_rl_loss_shape() -> Result<()> {
    let device = Device::Cpu;
    let specials = special_tokens();
    let mask_fn = |tokens: &[u32]| -> Vec<bool> { get_loss_mask(tokens, &specials) };

    // <QUESTION>Q</QUESTION><THINK>a</THINK><ANSWER>5</ANSWER>
    let row: Vec<i32> = vec![91, 81, 92, 93, 65, 94, 99, 53, 100];
    let tokens_data: Vec<i32> = row.iter().chain(row.iter()).cloned().collect();
    let tokens = Tensor::from_vec(tokens_data, (2, row.len()), &device)?;
    let rewards = vec![1.5, -0.5];

    let model_fn = |t: &Tensor| -> Result<Tensor> {
        let dims = t.dims();
        Tensor::zeros((dims[0], dims[1], 128), DType::F32, &device)
    };

    let loss = rl_loss(&model_fn, &tokens, &rewards, &mask_fn)?;
    let val: f32 = loss.to_scalar::<f32>()?;
    assert!(val.is_finite(), "RL loss should be finite, got {val}");
    Ok(())
}

// ---------- train_llm_sft ----------

#[test]
fn test_train_llm_sft_api() -> Result<()> {
    let device = Device::Cpu;
    let model_fn = |t: &Tensor| -> Result<Tensor> {
        let dims = t.dims();
        Tensor::zeros((dims[0], dims[1], 10), DType::F32, &device)
    };
    let mut optimizer_fn = || {};
    let loader: Vec<BatchItem> = vec![];

    train_llm_sft(&model_fn, &loader, &mut optimizer_fn, Some(0));
    Ok(())
}

// ============================================================
// End-to-end: eval_reasoning_model
// ============================================================

#[test]
fn test_eval_reasoning_model() -> Result<()> {
    let device = Device::Cpu;
    let mut model_fn = eval_reasoning_model()?;

    let tokens = Tensor::from_vec(vec![0u32, 1, 2, 3], (1, 4), &device)?;
    let full = model_fn(&tokens, 0, false)?;

    assert_eq!(full.dims()[0], 1);
    assert_eq!(full.dims()[1], 4);
    let vocab_size = full.dims()[2];
    assert!(vocab_size > 0);

    // Outputs should be finite.
    let first_logits = to_vec_f32(&full.narrow(2, 0, 16.min(vocab_size))?)?;
    for v in &first_logits {
        assert!(v.is_finite(), "eval_reasoning_model has non-finite logits");
    }

    // KV cache consistency.
    let mut model_fn2 = eval_reasoning_model()?;
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
        "eval_reasoning_model KV cache mismatch: max_diff={max_diff}"
    );
    Ok(())
}
