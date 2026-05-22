use hw2::*;

use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::backend::Autodiff;
use burn::tensor::{Int, Tensor, TensorData};
use test_support::{as_f64_vec, assert_f64_slice_close, json, python_json};

type B = Autodiff<NdArray<f64>>;
const DEVICE: NdArrayDevice = NdArrayDevice::Cpu;

const EPS: f64 = 1e-6;

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < EPS,
        "actual {actual}, expected {expected}"
    );
}

fn torch_cross_entropy(logits: Vec<f64>, shape: [usize; 2], targets: Vec<i64>) -> f64 {
    as_f64_vec(python_json(
        r#"
import json
import sys
import torch
import torch.nn.functional as F

d = json.load(sys.stdin)
logits = torch.tensor(d["logits"], dtype=torch.float64).reshape(d["shape"])
targets = torch.tensor(d["targets"], dtype=torch.long)
loss = F.cross_entropy(logits, targets)
json.dump([float(loss)], sys.stdout)
"#,
        json!({ "logits": logits, "shape": shape, "targets": targets }),
    ))[0]
}

fn torch_error_rate(logits: Vec<f64>, shape: [usize; 2], targets: Vec<i64>) -> f64 {
    as_f64_vec(python_json(
        r#"
import json
import sys
import torch

d = json.load(sys.stdin)
logits = torch.tensor(d["logits"], dtype=torch.float64).reshape(d["shape"])
targets = torch.tensor(d["targets"], dtype=torch.long)
err = (logits.argmax(dim=1) != targets).double().mean()
json.dump([float(err)], sys.stdout)
"#,
        json!({ "logits": logits, "shape": shape, "targets": targets }),
    ))[0]
}

fn torch_train_sgd(
    x: Vec<f64>,
    x_shape: [usize; 2],
    y: Vec<i64>,
    n_classes: usize,
    epochs: usize,
    step_size: f64,
    batch_size: usize,
) -> Vec<f64> {
    as_f64_vec(python_json(
        r#"
import json
import sys
import torch
import torch.nn.functional as F

d = json.load(sys.stdin)
X = torch.tensor(d["x"], dtype=torch.float64).reshape(d["x_shape"])
y = torch.tensor(d["y"], dtype=torch.long)
W = torch.zeros((d["n_classes"], X.shape[1]), dtype=torch.float64, requires_grad=True)

for _ in range(d["epochs"]):
    for start in range(0, X.shape[0], d["batch_size"]):
        end = start + d["batch_size"]
        loss = F.cross_entropy(X[start:end] @ W.T, y[start:end])
        loss.backward()
        with torch.no_grad():
            W -= d["step_size"] * W.grad
            W.grad.zero_()

json.dump(W.detach().flatten().tolist(), sys.stdout)
"#,
        json!({
            "x": x,
            "x_shape": x_shape,
            "y": y,
            "n_classes": n_classes,
            "epochs": epochs,
            "step_size": step_size,
            "batch_size": batch_size,
        }),
    ))
}

fn torch_autograd_expression(script_expression: &str, x: f64, y: f64) -> Vec<f64> {
    as_f64_vec(python_json(
        &format!(
            r#"
import json
import sys
import torch

d = json.load(sys.stdin)
x = torch.tensor(float(d["x"]), dtype=torch.float64, requires_grad=True)
y = torch.tensor(float(d["y"]), dtype=torch.float64, requires_grad=True)
z = {script_expression}
z.backward()
json.dump([float(z.detach()), float(x.grad), float(y.grad)], sys.stdout)
"#
        ),
        json!({ "x": x, "y": y }),
    ))
}

// --- Function forward/backward tests ---

#[test]
fn test_add_forward_backward() {
    let f = Add;
    let inputs = [2.5, -0.5];
    let grad = 3.0;
    assert_close(f.forward(&inputs), inputs.iter().sum());
    let g = f.backward(grad, &inputs);
    assert_f64_slice_close(&g, &[grad; 2], EPS);
}

#[test]
fn test_add_alternate_values() {
    let f = Add;
    let inputs = [-1.5, 3.0];
    let grad = 2.5;
    assert_close(f.forward(&inputs), inputs.iter().sum());
    let g = f.backward(grad, &inputs);
    assert_f64_slice_close(&g, &[grad; 2], EPS);
}

#[test]
fn test_subtract_forward_backward() {
    let f = Subtract;
    let inputs = [2.5, -0.5];
    let grad = 3.0;
    assert_close(f.forward(&inputs), inputs[0] - inputs[1]);
    let g = f.backward(grad, &inputs);
    assert_f64_slice_close(&g, &[grad, -grad], EPS);
}

#[test]
fn test_subtract_alternate_values() {
    let f = Subtract;
    let inputs = [-1.5, 3.0];
    let grad = 2.5;
    assert_close(f.forward(&inputs), inputs[0] - inputs[1]);
    let g = f.backward(grad, &inputs);
    assert_f64_slice_close(&g, &[grad, -grad], EPS);
}

#[test]
fn test_divide_forward_backward() {
    let f = Divide;
    let inputs = [9.0, 3.0];
    let grad = 4.0;
    assert_close(f.forward(&inputs), inputs[0] / inputs[1]);
    let g = f.backward(grad, &inputs);
    assert_f64_slice_close(
        &g,
        &[grad / inputs[1], -inputs[0] * grad / inputs[1].powi(2)],
        EPS,
    );
}

#[test]
fn test_divide_alternate_values() {
    let f = Divide;
    let inputs = [-8.0, 2.0];
    let grad = 1.5;
    assert_close(f.forward(&inputs), inputs[0] / inputs[1]);
    let g = f.backward(grad, &inputs);
    assert_f64_slice_close(
        &g,
        &[grad / inputs[1], -inputs[0] * grad / inputs[1].powi(2)],
        EPS,
    );
}

#[test]
fn test_power_forward_backward() {
    let degree = 3.0;
    let inputs = [2.0];
    let grad = 2.0;
    let f = Power { degree };
    assert_close(f.forward(&inputs), inputs[0].powf(degree));
    let g = f.backward(grad, &inputs);
    assert_f64_slice_close(&g, &[degree * grad * inputs[0].powf(degree - 1.0)], EPS);

    let f0 = Power { degree: 0.0 };
    let inputs = [5.0];
    assert_close(f0.forward(&inputs), inputs[0].powf(0.0));
    let g0 = f0.backward(7.0, &inputs);
    assert_f64_slice_close(&g0, &[0.0], EPS);
}

#[test]
fn test_power_alternate_values() {
    let degree = 3.0;
    let inputs = [-2.0];
    let grad = 1.5;
    let f = Power { degree };
    assert_close(f.forward(&inputs), inputs[0].powf(degree));
    let g = f.backward(grad, &inputs);
    assert_f64_slice_close(&g, &[degree * grad * inputs[0].powf(degree - 1.0)], EPS);

    let f0 = Power { degree: 0.0 };
    let g0 = f0.backward(7.0, &[5.0]);
    assert_f64_slice_close(&g0, &[0.0], EPS);
}

#[test]
fn test_log_forward_backward() {
    let f = Log;
    let inputs = [std::f64::consts::E];
    assert_close(f.forward(&inputs), inputs[0].ln());
    let grad = 2.0;
    let backward_inputs = [4.0];
    let g = f.backward(grad, &backward_inputs);
    assert_f64_slice_close(&g, &[grad / backward_inputs[0]], EPS);
}

#[test]
fn test_log_alternate_values() {
    let f = Log;
    let inputs = [3.5];
    let grad = 2.0;
    assert_close(f.forward(&inputs), inputs[0].ln());
    let g = f.backward(grad, &inputs);
    assert_f64_slice_close(&g, &[grad / inputs[0]], EPS);
}

#[test]
fn test_exp_forward_backward() {
    let f = Exp;
    let inputs = [1.0];
    let grad = 2.0;
    assert_close(f.forward(&inputs), inputs[0].exp());
    let g = f.backward(grad, &inputs);
    assert_f64_slice_close(&g, &[grad * inputs[0].exp()], EPS);
}

#[test]
fn test_exp_alternate_values() {
    let f = Exp;
    let inputs = [-0.5];
    let grad = 2.0;
    assert_close(f.forward(&inputs), inputs[0].exp());
    let g = f.backward(grad, &inputs);
    assert_f64_slice_close(&g, &[grad * inputs[0].exp()], EPS);
}

// --- More complex arithmetic tests ---

#[test]
fn test_arithmetic_ops1() {
    let x = Variable::new(3.0);
    let y = Variable::new(5.0);
    let xy = &x * &y;
    let xx = &x * &x;
    let sum = &xy + &xx;
    let d = &sum / &y;
    assert_eq!(d.borrow().value, 4.8);
    assert_eq!(d.borrow().grad, None);
}

// --- Gradient computation test ---

#[test]
fn test_compute_gradients() {
    let x = Variable::new(3.0);
    let y = Variable::new(4.0);

    let xy = &x * &y;
    let neg_xy = -&xy;
    let neg_xy_x = &neg_xy * &x;
    let neg_xy_xx = &neg_xy_x * &x;
    let neg_y = -&y;
    let z = &neg_xy_xx * &neg_y;

    let expected = torch_autograd_expression("((-(x * y) * x * x) * (-y))", 3.0, 4.0);
    assert_close(z.borrow().value, expected[0]);

    z.compute_gradients();

    assert!((z.borrow().grad.unwrap() - 1.0).abs() < EPS);
    assert!(
        (x.borrow().grad.unwrap() - expected[1]).abs() < EPS,
        "x.grad = {:?}, expected {}",
        x.borrow().grad,
        expected[1]
    );
    assert!(
        (y.borrow().grad.unwrap() - expected[2]).abs() < EPS,
        "y.grad = {:?}, expected {}",
        y.borrow().grad,
        expected[2]
    );
}

#[test]
fn test_compute_gradients_leaf() {
    // Calling compute_gradients on a leaf variable should set its grad to 1.0
    let w = Variable::new(-2.0);
    w.compute_gradients();
    assert!((w.borrow().grad.unwrap() - 1.0).abs() < EPS);
}

#[test]
fn test_compute_gradients_reused_intermediate() {
    let x = Variable::new(1.5);
    let y = Variable::new(-2.0);
    let xy = &x * &y;
    let neg_x = -&x;
    let xy_neg_x = &xy * &neg_x;
    let neg_y = -&y;
    let a = &xy_neg_x * &neg_y;
    let a_squared = &a * &a;
    let z = -&a_squared;

    let expected = torch_autograd_expression("-(x * y * (-x) * (-y)) ** 2", 1.5, -2.0);
    assert_close(z.borrow().value, expected[0]);

    z.compute_gradients();

    let expected_a_grad = as_f64_vec(python_json(
        r#"
import json
import sys
import torch

a = torch.tensor(9.0, dtype=torch.float64, requires_grad=True)
z = -(a * a)
z.backward()
json.dump([float(a.grad)], sys.stdout)
"#,
        json!({}),
    ))[0];
    assert_close(a.borrow().grad.unwrap(), expected_a_grad);
    assert_close(x.borrow().grad.unwrap(), expected[1]);
    assert_close(y.borrow().grad.unwrap(), expected[2]);
}

// --- Cross-entropy loss ---

#[test]
fn test_cross_entropy_loss() {
    let logits = [[2.0f64, 1.0, 0.0], [0.0, 2.0, 1.0]];
    let targets = [0i64, 2];
    let y_pred: Tensor<B, 2> = Tensor::from_data(TensorData::from(logits), &DEVICE);
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from(targets), &DEVICE);
    let loss: f64 = cross_entropy_loss(y_pred, y).into_scalar();
    let expected = torch_cross_entropy(
        logits.concat(),
        [logits.len(), logits[0].len()],
        targets.to_vec(),
    );
    assert_close(loss, expected);
}

#[test]
fn test_cross_entropy_loss_three_class_batch() {
    let logits = [[1.0f64, 0.0, -1.0], [2.0, 1.0, 0.0], [-1.0, 2.0, 1.0]];
    let targets = [0i64, 2, 1];
    let y_pred: Tensor<B, 2> = Tensor::from_data(TensorData::from(logits), &DEVICE);
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from(targets), &DEVICE);
    let loss: f64 = cross_entropy_loss(y_pred, y).into_scalar();
    let expected = torch_cross_entropy(
        logits.concat(),
        [logits.len(), logits[0].len()],
        targets.to_vec(),
    );
    assert_close(loss, expected);
}

// --- Error rate ---

#[test]
fn test_error() {
    let logits = [[3.0f64, 1.0], [0.0, 2.0], [1.0, 1.0], [-1.0, 0.0]];
    let targets = [0i64, 1, 1, 1];
    let y_pred: Tensor<B, 2> = Tensor::from_data(TensorData::from(logits), &DEVICE);
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from(targets), &DEVICE);
    let err = error_rate(y_pred, y);
    let expected = torch_error_rate(
        logits.concat(),
        [logits.len(), logits[0].len()],
        targets.to_vec(),
    );
    assert_close(err, expected);
}

// --- SGD training ---

#[test]
fn test_train_sgd_one_epoch() {
    let x_data = [[2.0f64, 1.0], [1.0, 2.0], [-2.0, -1.0], [-1.0, -2.0]];
    let y_data = [1i64, 1, 0, 0];
    let x: Tensor<B, 2> = Tensor::from_data(TensorData::from(x_data), &DEVICE);
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from(y_data), &DEVICE);

    let n_classes = 2;
    let epochs = 1;
    let step_size = 0.1;
    let batch_size = 2;
    let w = train_sgd(x, y, n_classes, epochs, step_size, batch_size);
    assert_eq!(w.dims(), [n_classes, x_data[0].len()]);

    let expected = torch_train_sgd(
        x_data.concat(),
        [x_data.len(), x_data[0].len()],
        y_data.to_vec(),
        n_classes,
        epochs,
        step_size,
        batch_size,
    );
    let w_data: Vec<f64> = w.into_data().to_vec().unwrap();
    assert_f64_slice_close(&w_data, &expected, 1e-5);
}

#[test]
fn test_train_sgd() {
    let x_data = [[2.0f64, 1.0], [1.0, 2.0], [-2.0, -1.0], [-1.0, -2.0]];
    let y_data = [1i64, 1, 0, 0];
    let x: Tensor<B, 2> = Tensor::from_data(TensorData::from(x_data), &DEVICE);
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from(y_data), &DEVICE);

    let n_classes = 2;
    let epochs = 20;
    let step_size = 0.1;
    let batch_size = 2;
    let w = train_sgd(
        x.clone(),
        y.clone(),
        n_classes,
        epochs,
        step_size,
        batch_size,
    );
    assert_eq!(w.dims(), [n_classes, x_data[0].len()]);

    // Verify predictions are correct after training
    let w_data: Vec<f64> = w.clone().into_data().to_vec().unwrap();
    let expected = torch_train_sgd(
        x_data.concat(),
        [x_data.len(), x_data[0].len()],
        y_data.to_vec(),
        n_classes,
        epochs,
        step_size,
        batch_size,
    );
    assert_f64_slice_close(&w_data, &expected, 1e-5);

    let x_data: Vec<f64> = x.into_data().to_vec().unwrap();
    let y_data: Vec<i64> = y.into_data().to_vec().unwrap();
    for i in 0..4 {
        let yi = y_data[i] as usize;
        let scores: Vec<f64> = (0..2)
            .map(|k| (0..2).map(|j| w_data[k * 2 + j] * x_data[i * 2 + j]).sum())
            .collect();
        let pred = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(pred, yi);
    }
    assert!(w_data.iter().any(|&wij| wij.abs() > EPS as f64));
}

#[test]
fn test_train_sgd_fifteen_epochs() {
    let x_data = [[2.0f64, 1.0], [1.0, 2.0], [-2.0, -1.0], [-1.0, -2.0]];
    let y_data = [1i64, 1, 0, 0];
    let x: Tensor<B, 2> = Tensor::from_data(TensorData::from(x_data), &DEVICE);
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from(y_data), &DEVICE);

    let n_classes = 2;
    let epochs = 15;
    let step_size = 0.1;
    let batch_size = 2;
    let w = train_sgd(
        x.clone(),
        y.clone(),
        n_classes,
        epochs,
        step_size,
        batch_size,
    );
    assert_eq!(w.dims(), [n_classes, x_data[0].len()]);

    let w_data: Vec<f64> = w.clone().into_data().to_vec().unwrap();
    let expected = torch_train_sgd(
        x_data.concat(),
        [x_data.len(), x_data[0].len()],
        y_data.to_vec(),
        n_classes,
        epochs,
        step_size,
        batch_size,
    );
    assert_f64_slice_close(&w_data, &expected, 1e-5);

    let x_data: Vec<f64> = x.into_data().to_vec().unwrap();
    let y_data: Vec<i64> = y.into_data().to_vec().unwrap();
    for i in 0..4 {
        let yi = y_data[i] as usize;
        let scores: Vec<f64> = (0..2)
            .map(|k| (0..2).map(|j| w_data[k * 2 + j] * x_data[i * 2 + j]).sum())
            .collect();
        let pred = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(pred, yi);
    }

    let norm: f64 = w_data.iter().map(|&wij| wij * wij).sum::<f64>().sqrt();
    assert!(norm > EPS as f64, "trained weights should not stay at zero");
}

// --- Additional edge case and quality tests ---

#[test]
fn test_cross_entropy_loss_single_sample() {
    let logits = [[1.0f64, 2.0, 3.0]];
    let targets = [2i64];
    let y_pred: Tensor<B, 2> = Tensor::from_data(TensorData::from(logits), &DEVICE);
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from(targets), &DEVICE);
    let loss: f64 = cross_entropy_loss(y_pred, y).into_scalar();
    let expected = torch_cross_entropy(
        logits.concat(),
        [logits.len(), logits[0].len()],
        targets.to_vec(),
    );
    assert!(loss.is_finite());
    assert_close(loss, expected);
}

#[test]
fn test_cross_entropy_loss_perfect_prediction() {
    // When logits strongly favor the correct class, loss should be near zero
    let y_pred: Tensor<B, 2> = Tensor::from_data(
        TensorData::from([[100.0f64, 0.0, 0.0], [0.0, 100.0, 0.0]]),
        &DEVICE,
    );
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from([0i64, 1]), &DEVICE);
    let loss: f64 = cross_entropy_loss(y_pred, y).into_scalar();
    assert!(loss.is_finite());
    assert!(
        loss < 1e-5,
        "Loss should be near zero for perfect predictions, got {loss}"
    );
}

#[test]
fn test_cross_entropy_loss_numerically_stable() {
    // Very large logits should not produce NaN or Inf
    let y_pred: Tensor<B, 2> =
        Tensor::from_data(TensorData::from([[1000.0f64, 1001.0, 999.0]]), &DEVICE);
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from([1i64]), &DEVICE);
    let loss: f64 = cross_entropy_loss(y_pred, y).into_scalar();
    assert!(loss.is_finite(), "Loss must be finite for large logits");
    assert!(loss >= 0.0, "Cross-entropy loss must be non-negative");
}

#[test]
fn test_error_rate_perfect() {
    let logits = [[10.0f64, 0.0], [0.0, 10.0], [0.0, 10.0]];
    let targets = [0i64, 1, 1];
    let y_pred: Tensor<B, 2> = Tensor::from_data(TensorData::from(logits), &DEVICE);
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from(targets), &DEVICE);
    let err = error_rate(y_pred, y);
    let expected = torch_error_rate(
        logits.concat(),
        [logits.len(), logits[0].len()],
        targets.to_vec(),
    );
    assert_close(err, expected);
}

#[test]
fn test_error_rate_all_wrong() {
    let logits = [[0.0f64, 10.0], [10.0, 0.0]];
    let targets = [0i64, 1];
    let y_pred: Tensor<B, 2> = Tensor::from_data(TensorData::from(logits), &DEVICE);
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from(targets), &DEVICE);
    let err = error_rate(y_pred, y);
    let expected = torch_error_rate(
        logits.concat(),
        [logits.len(), logits[0].len()],
        targets.to_vec(),
    );
    assert_close(err, expected);
}

#[test]
fn test_train_sgd_zero_epochs() {
    let x_data = [[1.0f64, 2.0], [3.0, 4.0]];
    let y_data = [0i64, 1];
    let x: Tensor<B, 2> = Tensor::from_data(TensorData::from(x_data), &DEVICE);
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from(y_data), &DEVICE);
    let n_classes = 2;
    let epochs = 0;
    let step_size = 0.1;
    let batch_size = 2;
    let w = train_sgd(x, y, n_classes, epochs, step_size, batch_size);
    let expected = torch_train_sgd(
        x_data.concat(),
        [x_data.len(), x_data[0].len()],
        y_data.to_vec(),
        n_classes,
        epochs,
        step_size,
        batch_size,
    );
    let w_data: Vec<f64> = w.into_data().to_vec().unwrap();
    assert_f64_slice_close(&w_data, &expected, 1e-7);
}

#[test]
fn test_train_sgd_batch_size_one() {
    let x: Tensor<B, 2> = Tensor::from_data(
        TensorData::from([[2.0f64, 1.0], [1.0, 2.0], [-2.0, -1.0], [-1.0, -2.0]]),
        &DEVICE,
    );
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from([1i64, 1, 0, 0]), &DEVICE);
    let w = train_sgd(x, y, 2, 10, 0.1, 1);
    assert_eq!(w.dims(), [2, 2]);
    let w_data: Vec<f64> = w.into_data().to_vec().unwrap();
    assert!(
        w_data.iter().any(|&v| v.abs() > 1e-6),
        "Weights should be non-zero after training"
    );
}

#[test]
fn test_train_sgd_loss_decreases() {
    let x: Tensor<B, 2> = Tensor::from_data(
        TensorData::from([[2.0f64, 1.0], [1.0, 2.0], [-2.0, -1.0], [-1.0, -2.0]]),
        &DEVICE,
    );
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from([1i64, 1, 0, 0]), &DEVICE);

    let w1 = train_sgd(x.clone(), y.clone(), 2, 1, 0.1, 2);
    let w20 = train_sgd(x.clone(), y.clone(), 2, 20, 0.1, 2);

    // Compute predictions and loss for both
    let w1_data: Vec<f64> = w1.into_data().to_vec().unwrap();
    let w20_data: Vec<f64> = w20.into_data().to_vec().unwrap();
    let x_data: Vec<f64> = x.into_data().to_vec().unwrap();

    // Compute sum of correct predictions for w20 (should be better than w1)
    let mut correct_w1 = 0;
    let mut correct_w20 = 0;
    let y_data: Vec<i64> = y.into_data().to_vec().unwrap();
    for i in 0..4 {
        let s1: Vec<f64> = (0..2)
            .map(|k| (0..2).map(|j| w1_data[k * 2 + j] * x_data[i * 2 + j]).sum())
            .collect();
        let s20: Vec<f64> = (0..2)
            .map(|k| {
                (0..2)
                    .map(|j| w20_data[k * 2 + j] * x_data[i * 2 + j])
                    .sum()
            })
            .collect();
        let p1 = s1
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let p20 = s20
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        if p1 == y_data[i] as usize {
            correct_w1 += 1;
        }
        if p20 == y_data[i] as usize {
            correct_w20 += 1;
        }
    }
    assert!(
        correct_w20 >= correct_w1,
        "More epochs should improve accuracy: w1={correct_w1}, w20={correct_w20}"
    );
}
