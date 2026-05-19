use hw2::*;

use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::backend::Autodiff;
use burn::tensor::{Int, Tensor, TensorData};

type B = Autodiff<NdArray<f64>>;
const DEVICE: NdArrayDevice = NdArrayDevice::Cpu;

const EPS: f64 = 1e-6;

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < EPS,
        "actual {actual}, expected {expected}"
    );
}

// --- Function forward/backward tests ---

#[test]
fn test_add_forward_backward() {
    let f = Add;
    assert_eq!(f.forward(&[2.5, -0.5]), 2.0);
    let g = f.backward(3.0, &[2.5, -0.5]);
    assert_eq!(g.len(), 2);
    assert_eq!(g[0], 3.0);
    assert_eq!(g[1], 3.0);
}

#[test]
fn test_add_alternate_values() {
    let f = Add;
    assert_close(f.forward(&[-1.5, 3.0]), 1.5);
    let g = f.backward(2.5, &[-1.5, 3.0]);
    assert_eq!(g.len(), 2);
    assert_close(g[0], 2.5);
    assert_close(g[1], 2.5);
}

#[test]
fn test_subtract_forward_backward() {
    let f = Subtract;
    assert_eq!(f.forward(&[2.5, -0.5]), 3.0);
    let g = f.backward(3.0, &[2.5, -0.5]);
    assert_eq!(g.len(), 2);
    assert_eq!(g[0], 3.0);
    assert_eq!(g[1], -3.0);
}

#[test]
fn test_subtract_alternate_values() {
    let f = Subtract;
    assert_close(f.forward(&[-1.5, 3.0]), -4.5);
    let g = f.backward(2.5, &[-1.5, 3.0]);
    assert_eq!(g.len(), 2);
    assert_close(g[0], 2.5);
    assert_close(g[1], -2.5);
}

#[test]
fn test_divide_forward_backward() {
    let f = Divide;
    assert_eq!(f.forward(&[9.0, 3.0]), 3.0);
    let g = f.backward(4.0, &[9.0, 3.0]);
    assert_eq!(g.len(), 2);
    assert!((g[0] - 4.0 / 3.0).abs() < EPS);
    assert!((g[1] - (-4.0)).abs() < EPS);
}

#[test]
fn test_divide_alternate_values() {
    let f = Divide;
    assert_close(f.forward(&[-8.0, 2.0]), -4.0);
    let g = f.backward(1.5, &[-8.0, 2.0]);
    assert_eq!(g.len(), 2);
    assert_close(g[0], 0.75);
    assert_close(g[1], 3.0);
}

#[test]
fn test_power_forward_backward() {
    let f = Power { degree: 3.0 };
    assert_eq!(f.forward(&[2.0]), 8.0);
    let g = f.backward(2.0, &[2.0]);
    assert_eq!(g.len(), 1);
    assert_eq!(g[0], 24.0);

    let f0 = Power { degree: 0.0 };
    assert_eq!(f0.forward(&[5.0]), 1.0);
    let g0 = f0.backward(7.0, &[5.0]);
    assert_eq!(g0.len(), 1);
    assert_eq!(g0[0], 0.0);
}

#[test]
fn test_power_alternate_values() {
    let f = Power { degree: 3.0 };
    assert_close(f.forward(&[-2.0]), -8.0);
    let g = f.backward(1.5, &[-2.0]);
    assert_eq!(g.len(), 1);
    assert_close(g[0], 18.0);

    let f0 = Power { degree: 0.0 };
    let g0 = f0.backward(7.0, &[5.0]);
    assert_eq!(g0.len(), 1);
    assert_close(g0[0], 0.0);
}

#[test]
fn test_log_forward_backward() {
    let f = Log;
    assert!((f.forward(&[std::f64::consts::E]) - 1.0).abs() < EPS);
    let g = f.backward(2.0, &[4.0]);
    assert_eq!(g.len(), 1);
    assert!((g[0] - 0.5).abs() < EPS);
}

#[test]
fn test_log_alternate_values() {
    let f = Log;
    assert_close(f.forward(&[3.5]), 1.252762968495368);
    let g = f.backward(2.0, &[3.5]);
    assert_eq!(g.len(), 1);
    assert_close(g[0], 0.5714285714285714);
}

#[test]
fn test_exp_forward_backward() {
    let f = Exp;
    assert!((f.forward(&[1.0]) - std::f64::consts::E).abs() < EPS);
    let g = f.backward(2.0, &[1.0]);
    assert_eq!(g.len(), 1);
    assert!((g[0] - 2.0 * std::f64::consts::E).abs() < EPS);
}

#[test]
fn test_exp_alternate_values() {
    let f = Exp;
    assert_close(f.forward(&[-0.5]), 0.6065306597126334);
    let g = f.backward(2.0, &[-0.5]);
    assert_eq!(g.len(), 1);
    assert_close(g[0], 1.2130613194252668);
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
    // z = ((-(x * y) * x * x) * (-y)), x=3, y=4
    // z = (-(3*4) * 3 * 3) * (-4) = (-108) * (-4) = 432
    // dz/dx = 432, dz/dy = 216
    let x = Variable::new(3.0);
    let y = Variable::new(4.0);

    let xy = &x * &y;
    let neg_xy = -&xy;
    let neg_xy_x = &neg_xy * &x;
    let neg_xy_xx = &neg_xy_x * &x;
    let neg_y = -&y;
    let z = &neg_xy_xx * &neg_y;

    assert!((z.borrow().value - 432.0).abs() < EPS);

    z.compute_gradients();

    assert!((z.borrow().grad.unwrap() - 1.0).abs() < EPS);
    assert!(
        (x.borrow().grad.unwrap() - 432.0).abs() < EPS,
        "x.grad = {:?}, expected 432.0",
        x.borrow().grad
    );
    assert!(
        (y.borrow().grad.unwrap() - 216.0).abs() < EPS,
        "y.grad = {:?}, expected 216.0",
        y.borrow().grad
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

    assert_close(a.borrow().value, 9.0);
    assert_close(z.borrow().value, -81.0);

    z.compute_gradients();

    assert_close(a.borrow().grad.unwrap(), -18.0);
    assert_close(x.borrow().grad.unwrap(), -216.0);
    assert_close(y.borrow().grad.unwrap(), 162.0);
}

// --- Cross-entropy loss ---

#[test]
fn test_cross_entropy_loss() {
    let y_pred: Tensor<B, 2> = Tensor::from_data(
        TensorData::from([[2.0f64, 1.0, 0.0], [0.0, 2.0, 1.0]]),
        &DEVICE,
    );
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from([0i32, 2]), &DEVICE);
    let loss: f64 = cross_entropy_loss(y_pred, y).into_scalar();
    assert!((loss - 0.907_606).abs() < 1e-5);
}

#[test]
fn test_cross_entropy_loss_three_class_batch() {
    let y_pred: Tensor<B, 2> = Tensor::from_data(
        TensorData::from([[1.0f64, 0.0, -1.0], [2.0, 1.0, 0.0], [-1.0, 2.0, 1.0]]),
        &DEVICE,
    );
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from([0i32, 2, 1]), &DEVICE);
    let loss: f64 = cross_entropy_loss(y_pred, y).into_scalar();
    assert!((loss - 1.054741).abs() < 1e-4);
}

// --- Error rate ---

#[test]
fn test_error() {
    let y_pred: Tensor<B, 2> = Tensor::from_data(
        TensorData::from([[3.0f64, 1.0], [0.0, 2.0], [1.0, 1.0], [-1.0, 0.0]]),
        &DEVICE,
    );
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from([0i32, 1, 1, 1]), &DEVICE);
    let err = error_rate(y_pred, y);
    assert!((err - 0.25).abs() < 1e-6);
}

// --- SGD training ---

#[test]
fn test_train_sgd_one_epoch() {
    let x: Tensor<B, 2> = Tensor::from_data(
        TensorData::from([[2.0f64, 1.0], [1.0, 2.0], [-2.0, -1.0], [-1.0, -2.0]]),
        &DEVICE,
    );
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from([1i32, 1, 0, 0]), &DEVICE);

    let w = train_sgd(x, y, 2, 1, 0.1, 2);
    assert_eq!(w.dims(), [2, 2]);

    let expected = [[-0.13340412f64, -0.13340412], [0.13340412, 0.13340412]];
    let w_data: Vec<f64> = w.into_data().to_vec().unwrap();
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (w_data[i * 2 + j] - expected[i][j]).abs() < 1e-5,
                "w[[{i}, {j}]] = {}, expected {}",
                w_data[i * 2 + j],
                expected[i][j]
            );
        }
    }
}

#[test]
fn test_train_sgd() {
    let x: Tensor<B, 2> = Tensor::from_data(
        TensorData::from([[2.0f64, 1.0], [1.0, 2.0], [-2.0, -1.0], [-1.0, -2.0]]),
        &DEVICE,
    );
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from([1i32, 1, 0, 0]), &DEVICE);

    let w = train_sgd(x.clone(), y.clone(), 2, 20, 0.1, 2);
    assert_eq!(w.dims(), [2, 2]);

    // Verify predictions are correct after training
    let w_data: Vec<f64> = w.clone().into_data().to_vec().unwrap();
    let x_data: Vec<f64> = x.into_data().to_vec().unwrap();
    let y_data: Vec<i32> = y.into_data().to_vec().unwrap();
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
    let x: Tensor<B, 2> = Tensor::from_data(
        TensorData::from([[2.0f64, 1.0], [1.0, 2.0], [-2.0, -1.0], [-1.0, -2.0]]),
        &DEVICE,
    );
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from([1i32, 1, 0, 0]), &DEVICE);

    let w = train_sgd(x.clone(), y.clone(), 2, 15, 0.1, 2);
    assert_eq!(w.dims(), [2, 2]);

    let w_data: Vec<f64> = w.clone().into_data().to_vec().unwrap();
    let x_data: Vec<f64> = x.into_data().to_vec().unwrap();
    let y_data: Vec<i32> = y.into_data().to_vec().unwrap();
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
    let y_pred: Tensor<B, 2> = Tensor::from_data(TensorData::from([[1.0f64, 2.0, 3.0]]), &DEVICE);
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from([2i32]), &DEVICE);
    let loss: f64 = cross_entropy_loss(y_pred, y).into_scalar();
    // -3.0 + log(exp(1)+exp(2)+exp(3)) = -3 + log(e + e^2 + e^3) ≈ -3 + 3.4076 = 0.4076
    assert!(loss.is_finite());
    assert!((loss - 0.4076).abs() < 1e-3);
}

#[test]
fn test_cross_entropy_loss_perfect_prediction() {
    // When logits strongly favor the correct class, loss should be near zero
    let y_pred: Tensor<B, 2> = Tensor::from_data(
        TensorData::from([[100.0f64, 0.0, 0.0], [0.0, 100.0, 0.0]]),
        &DEVICE,
    );
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from([0i32, 1]), &DEVICE);
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
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from([1i32]), &DEVICE);
    let loss: f64 = cross_entropy_loss(y_pred, y).into_scalar();
    assert!(loss.is_finite(), "Loss must be finite for large logits");
    assert!(loss >= 0.0, "Cross-entropy loss must be non-negative");
}

#[test]
fn test_error_rate_perfect() {
    let y_pred: Tensor<B, 2> = Tensor::from_data(
        TensorData::from([[10.0f64, 0.0], [0.0, 10.0], [0.0, 10.0]]),
        &DEVICE,
    );
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from([0i32, 1, 1]), &DEVICE);
    let err = error_rate(y_pred, y);
    assert!(
        (err - 0.0).abs() < 1e-6,
        "Perfect predictions should give 0 error"
    );
}

#[test]
fn test_error_rate_all_wrong() {
    let y_pred: Tensor<B, 2> =
        Tensor::from_data(TensorData::from([[0.0f64, 10.0], [10.0, 0.0]]), &DEVICE);
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from([0i32, 1]), &DEVICE);
    let err = error_rate(y_pred, y);
    assert!(
        (err - 1.0).abs() < 1e-6,
        "All wrong predictions should give 1.0 error"
    );
}

#[test]
fn test_train_sgd_zero_epochs() {
    let x: Tensor<B, 2> = Tensor::from_data(TensorData::from([[1.0f64, 2.0], [3.0, 4.0]]), &DEVICE);
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from([0i32, 1]), &DEVICE);
    let w = train_sgd(x, y, 2, 0, 0.1, 2);
    // With 0 epochs, weights should remain at initialization (zeros)
    let w_data: Vec<f64> = w.into_data().to_vec().unwrap();
    for &val in &w_data {
        assert!(
            (val - 0.0).abs() < 1e-7,
            "0 epochs should return zero-initialized weights"
        );
    }
}

#[test]
fn test_train_sgd_batch_size_one() {
    let x: Tensor<B, 2> = Tensor::from_data(
        TensorData::from([[2.0f64, 1.0], [1.0, 2.0], [-2.0, -1.0], [-1.0, -2.0]]),
        &DEVICE,
    );
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from([1i32, 1, 0, 0]), &DEVICE);
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
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from([1i32, 1, 0, 0]), &DEVICE);

    let w1 = train_sgd(x.clone(), y.clone(), 2, 1, 0.1, 2);
    let w20 = train_sgd(x.clone(), y.clone(), 2, 20, 0.1, 2);

    // Compute predictions and loss for both
    let w1_data: Vec<f64> = w1.into_data().to_vec().unwrap();
    let w20_data: Vec<f64> = w20.into_data().to_vec().unwrap();
    let x_data: Vec<f64> = x.into_data().to_vec().unwrap();

    // Compute sum of correct predictions for w20 (should be better than w1)
    let mut correct_w1 = 0;
    let mut correct_w20 = 0;
    let y_data: Vec<i32> = y.into_data().to_vec().unwrap();
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
