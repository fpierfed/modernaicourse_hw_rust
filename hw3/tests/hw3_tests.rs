use candle_core::{Device, Tensor};
use candle_nn::VarMap;
use hw3::*;

const DEVICE: Device = Device::Cpu;

#[test]
fn test_linear_shape() {
    let varmap = VarMap::new();
    let layer = Linear::new(10, 20, &varmap, "test_linear").unwrap();

    let x = Tensor::randn(0.0f32, 1.0, &[50, 10], &DEVICE).unwrap();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.dims(), &[50, 20]);
}

#[test]
fn test_linear_batch_dims() {
    let varmap = VarMap::new();
    let layer = Linear::new(10, 20, &varmap, "test_linear_batch").unwrap();

    let x = Tensor::randn(0.0f32, 1.0, &[7, 9, 10], &DEVICE).unwrap();
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.dims(), &[7, 9, 20]);
}

#[test]
fn test_cross_entropy_loss() {
    let logits = Tensor::new(&[[2.0f32, 1.0, 0.0], [0.0, 2.0, 1.0]], &DEVICE).unwrap();
    let y = Tensor::new(&[0u32, 2], &DEVICE).unwrap();
    let loss = cross_entropy_loss(&logits, &y).unwrap();
    let loss_val: f32 = loss.to_scalar().unwrap();
    assert!((loss_val - 0.9076060).abs() < 1e-5);
}

#[test]
fn test_cross_entropy_numerically_stable() {
    let logits = Tensor::new(&[[1000.0f32, 1001.0, 999.5], [1.0, -2.0, 0.5]], &DEVICE).unwrap();
    let y = Tensor::new(&[1u32, 2], &DEVICE).unwrap();
    let loss = cross_entropy_loss(&logits, &y).unwrap();
    let loss_val: f32 = loss.to_scalar().unwrap();
    assert!(loss_val.is_finite());
}

#[test]
fn test_dataloader_batches() {
    let x = Tensor::arange(0.0f32, 24.0, &DEVICE)
        .unwrap()
        .reshape(&[12, 2])
        .unwrap();
    let y = Tensor::arange(0u32, 12, &DEVICE).unwrap();

    let loader = DataLoader::new(x, y, 5);
    let batches: Vec<_> = loader.collect();

    assert_eq!(batches.len(), 3);
    // First batch should have 5 samples
    assert_eq!(batches[0].0.dims()[0], 5);
    // Last batch should have 2 samples (12 - 5 - 5 = 2)
    assert_eq!(batches[2].0.dims()[0], 2);
}

#[test]
fn test_two_layer_nn() {
    let varmap = VarMap::new();
    let model = TwoLayerNN::new(8, 16, 5, &varmap).unwrap();

    let x = Tensor::randn(0.0f32, 1.0, &[13, 8], &DEVICE).unwrap();
    let out = model.forward(&x).unwrap();
    assert_eq!(out.dims(), &[13, 5]);
}

#[test]
fn test_multi_layer_nn() {
    let varmap = VarMap::new();
    let hidden_dims = vec![7, 5, 4];
    let model = MultiLayerNN::new(6, 3, &hidden_dims, &varmap).unwrap();

    let x = Tensor::randn(0.0f32, 1.0, &[9, 6], &DEVICE).unwrap();
    let out = model.forward(&x).unwrap();
    assert_eq!(out.dims(), &[9, 3]);
}
