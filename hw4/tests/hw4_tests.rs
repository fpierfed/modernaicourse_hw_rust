use candle_core::{Device, Tensor};
use candle_nn::VarMap;
use hw4::*;

const DEVICE: Device = Device::Cpu;

#[test]
fn test_linear_shape() {
    let varmap = VarMap::new();
    let layer = Linear::new(10, 20, &varmap, "linear").unwrap();
    let x = Tensor::randn(0.0f32, 1.0, &[50, 10], &DEVICE).unwrap();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.dims(), &[50, 20]);
}

#[test]
fn test_cross_entropy_loss() {
    let logits = Tensor::new(&[[2.0f32, 1.0, 0.0], [0.0, 2.0, 1.0]], &DEVICE).unwrap();
    let y = Tensor::new(&[0u32, 2], &DEVICE).unwrap();
    let loss = cross_entropy_loss(&logits, &y).unwrap();
    let val: f32 = loss.to_scalar().unwrap();
    assert!((val - 0.9076060).abs() < 1e-5);
}

#[test]
fn test_two_layer_nn_shape() {
    let varmap = VarMap::new();
    let model = TwoLayerNN::new(784, 128, 10, &varmap).unwrap();
    let x = Tensor::randn(0.0f32, 1.0, &[32, 784], &DEVICE).unwrap();
    let out = model.forward(&x).unwrap();
    assert_eq!(out.dims(), &[32, 10]);
}

// NOTE: eval_linear_model and eval_two_layer_nn require MNIST data.
// These are integration tests that should only run when data is available.
#[test]
#[ignore]
fn test_eval_linear_model() {
    // Requires MNIST data download
    let _model = eval_linear_model().unwrap();
    // Would test: model predicts on test set with <10% error
}

#[test]
#[ignore]
fn test_eval_two_layer_nn() {
    // Requires MNIST data download
    let _model = eval_two_layer_nn().unwrap();
    // Would test: model predicts on test set with <3% error
}
