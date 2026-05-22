use std::io::Write;
use std::process::{Command, Stdio};

pub use serde_json::{json, Value};

pub fn python_json(script: &str, input: Value) -> Value {
    let mut child = Command::new("python3")
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to start python3 for PyTorch reference test");

    {
        let stdin = child.stdin.as_mut().expect("failed to open python stdin");
        write!(stdin, "{input}").expect("failed to write JSON to python stdin");
    }

    let output = child
        .wait_with_output()
        .expect("failed to wait for python reference process");
    if !output.status.success() {
        panic!(
            "python reference failed with status {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let stdout = String::from_utf8(output.stdout).expect("python stdout was not UTF-8");
    serde_json::from_str(stdout.trim()).expect("python stdout was not valid JSON")
}

pub fn as_f32_vec(value: Value) -> Vec<f32> {
    serde_json::from_value(value).expect("expected a JSON f32 array")
}

pub fn as_f64_vec(value: Value) -> Vec<f64> {
    serde_json::from_value(value).expect("expected a JSON f64 array")
}

pub fn as_i32_vec(value: Value) -> Vec<i32> {
    serde_json::from_value(value).expect("expected a JSON i32 array")
}

pub fn assert_f32_close(actual: f32, expected: f32, tol: f32) {
    assert!(
        (actual - expected).abs() <= tol,
        "actual {actual}, expected {expected}, tolerance {tol}"
    );
}

pub fn assert_f64_close(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() <= tol,
        "actual {actual}, expected {expected}, tolerance {tol}"
    );
}

pub fn assert_f32_slice_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "actual and expected lengths differ"
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "actual[{i}] = {a}, expected {e}, tolerance {tol}"
        );
    }
}

pub fn assert_f64_slice_close(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "actual and expected lengths differ"
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "actual[{i}] = {a}, expected {e}, tolerance {tol}"
        );
    }
}
