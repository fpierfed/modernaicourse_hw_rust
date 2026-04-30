use hw7::*;
use std::collections::HashMap;

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

// ============================================================
// Calculator tests
// ============================================================

#[test]
fn test_calculator_basic() {
    assert!((calculator("2 + 3").unwrap() - 5.0).abs() < 1e-10);
    assert!((calculator("10 - 4").unwrap() - 6.0).abs() < 1e-10);
    assert!((calculator("3 * 7").unwrap() - 21.0).abs() < 1e-10);
    assert!((calculator("15 / 3").unwrap() - 5.0).abs() < 1e-10);
}

#[test]
fn test_calculator_precedence() {
    assert!((calculator("2 + 3 * 4").unwrap() - 14.0).abs() < 1e-10);
    assert!((calculator("(2 + 3) * 4").unwrap() - 20.0).abs() < 1e-10);
}

#[test]
fn test_calculator_nested_parens() {
    assert!((calculator("((2 + 3) * (4 - 1))").unwrap() - 15.0).abs() < 1e-10);
}

#[test]
fn test_calculator_float() {
    assert!((calculator("3.5 * 2").unwrap() - 7.0).abs() < 1e-10);
}

#[test]
fn test_calculator_division_by_zero() {
    assert!(calculator("1 / 0").is_err());
}

// ============================================================
// Tool execution tests
// ============================================================

#[test]
fn test_execute_tool_call() {
    let result = execute_tool_call("calculator(2 + 3)");
    assert_eq!(result, "5");
}

#[test]
fn test_execute_tool_call_complex() {
    let result = execute_tool_call("calculator((10 + 5) * 2)");
    assert_eq!(result, "30");
}

// ============================================================
// Answer extraction tests
// ============================================================

#[test]
fn test_extract_answer() {
    assert_eq!(extract_answer("blah <ANSWER>42</ANSWER> blah"), Some(42.0));
    assert_eq!(extract_answer("no answer here"), None);
    assert_eq!(extract_answer("<ANSWER>3.14</ANSWER>"), Some(3.14));
}

// ============================================================
// Chat formatting tests
// ============================================================

#[test]
fn test_format_chat_prompt() {
    let specials = special_tokens();
    let encode_fn = |text: &str| -> Vec<u32> {
        text.chars().map(|c| c as u32).collect()
    };

    let tokens = format_chat_prompt("What is 2+2?", &encode_fn, &specials);

    // Should start with <QUESTION> token
    assert_eq!(tokens[0], 91);
    // Should contain </QUESTION> somewhere
    assert!(tokens.contains(&92));
}

// ============================================================
// SFT example tests
// ============================================================

#[test]
fn test_create_sft_example() {
    let specials = special_tokens();
    let encode_fn = |text: &str| -> Vec<u32> {
        text.chars().map(|c| c as u32).collect()
    };

    let example = create_sft_example(
        "What is 2+2?",
        "I need to add 2 and 2.",
        "4",
        &encode_fn,
        &specials,
    );

    // input_ids should not be empty
    assert!(!example.input_ids.is_empty());
    // labels should be same length as input_ids
    assert_eq!(example.input_ids.len(), example.labels.len());
    // Question tokens should be masked (-100)
    assert!(example.labels.iter().any(|&l| l == -100));
    // Answer tokens should NOT be masked
    assert!(example.labels.iter().any(|&l| l != -100));
}
