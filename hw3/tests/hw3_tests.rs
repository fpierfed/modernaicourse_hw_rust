use candle_core::{Device, Tensor};
use candle_nn::VarMap;
use hw3::*;

const DEVICE: Device = Device::Cpu;

// --- Linear layer ---

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
fn test_linear_correctness() {
    let varmap = VarMap::new();
    let layer = Linear::new(4, 3, &varmap, "test_linear_correct").unwrap();

    // Manually compute X @ W^T and compare
    let x = Tensor::new(&[[1.0f32, 2.0, -1.0, 0.5], [0.0, -1.0, 2.0, 3.0]], &DEVICE).unwrap();
    let out = layer.forward(&x).unwrap();

    // Reference: x @ weight.T
    let w = layer.weight();
    let w_t = w.t().unwrap();
    let expected = x.matmul(&w_t).unwrap();

    let diff = out.sub(&expected).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
    assert!(diff < 1e-5, "Linear output doesn't match X @ W^T, diff={diff}");
}

#[test]
fn test_linear_kaiming_init() {
    let varmap = VarMap::new();
    let layer = Linear::new(100, 1000, &varmap, "test_kaiming").unwrap();

    let w = layer.weight();
    // Kaiming init: std ≈ sqrt(2 / in_features) = sqrt(2/100) ≈ 0.1414
    let expected_std = (2.0f64 / 100.0).sqrt();
    let mean = w.mean_all().unwrap().to_scalar::<f32>().unwrap() as f64;
    let var = w.broadcast_sub(&w.mean_all().unwrap()).unwrap()
        .sqr().unwrap()
        .mean_all().unwrap()
        .to_scalar::<f32>().unwrap() as f64;
    let std = var.sqrt();

    assert!(
        (std - expected_std).abs() < 3e-3,
        "Weight std {std} not close to expected {expected_std}"
    );
    assert!(mean.abs() < 0.02, "Weight mean {mean} not close to 0");
}

// --- Cross-entropy loss ---

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
    // Also verify the value is correct (from PyTorch nn.CrossEntropyLoss reference)
    // For [1000, 1001, 999.5] with target=1: -1001 + log(e^1000 + e^1001 + e^999.5) ≈ -1001 + 1001.31 ≈ 0.31
    // For [1.0, -2.0, 0.5] with target=2: -0.5 + log(e^1 + e^-2 + e^0.5) ≈ -0.5 + 1.486 ≈ 0.986
    // Average ≈ 0.648
    assert!(
        (loss_val - 0.6483).abs() < 0.01,
        "Expected ≈0.6483, got {loss_val}"
    );
}

// --- SGD optimizer ---

#[test]
fn test_sgd_step() {
    // Create a model, do a forward/backward, step, verify weights changed
    let varmap = VarMap::new();
    let layer = Linear::new(4, 3, &varmap, "sgd_step").unwrap();

    let w_before: Vec<f32> = layer.weight().flatten_all().unwrap().to_vec1().unwrap();

    let params: Vec<Tensor> = varmap.all_vars().iter().map(|v| v.as_tensor().clone()).collect();
    let mut opt = SGD::new(params, 0.05);

    let x = Tensor::randn(0.0f32, 1.0, &[12, 4], &DEVICE).unwrap();
    let y = Tensor::new(&[0u32, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], &DEVICE).unwrap();

    // Training loop: 3 steps
    for _ in 0..3 {
        opt.zero_grad().unwrap();
        let logits = layer.forward(&x).unwrap();
        let loss = cross_entropy_loss(&logits, &y).unwrap();
        loss.backward().unwrap();
        opt.step().unwrap();
    }

    let w_after: Vec<f32> = layer.weight().flatten_all().unwrap().to_vec1().unwrap();
    let changed = w_after.iter().zip(w_before.iter()).any(|(a, b)| (a - b).abs() > 1e-7);
    assert!(changed, "SGD step did not modify the weights after 3 training steps");
}

#[test]
fn test_sgd_zero_grad() {
    let varmap = VarMap::new();
    let layer = Linear::new(4, 3, &varmap, "sgd_zerograd").unwrap();

    let params: Vec<Tensor> = varmap.all_vars().iter().map(|v| v.as_tensor().clone()).collect();
    let mut opt = SGD::new(params, 0.1);

    // Do a forward/backward to create gradients
    let x = Tensor::randn(0.0f32, 1.0, &[4, 4], &DEVICE).unwrap();
    let y = Tensor::new(&[0u32, 1, 2, 0], &DEVICE).unwrap();
    let logits = layer.forward(&x).unwrap();
    let loss = cross_entropy_loss(&logits, &y).unwrap();
    loss.backward().unwrap();

    // zero_grad then step should not move weights (grad is zero)
    opt.zero_grad().unwrap();
    let w_before: Vec<f32> = layer.weight().flatten_all().unwrap().to_vec1().unwrap();
    opt.step().unwrap();
    let w_after: Vec<f32> = layer.weight().flatten_all().unwrap().to_vec1().unwrap();

    for (a, b) in w_after.iter().zip(w_before.iter()) {
        assert!(
            (a - b).abs() < 1e-7,
            "zero_grad did not prevent weight update: {a} vs {b}"
        );
    }
}

// --- DataLoader ---

#[test]
fn test_dataloader_batches() {
    let x = Tensor::arange(0.0f32, 24.0, &DEVICE)
        .unwrap()
        .reshape(&[12, 2])
        .unwrap();
    let y = Tensor::arange(0u32, 12, &DEVICE).unwrap();

    let loader = DataLoader::new(x.clone(), y.clone(), 5);
    let batches: Vec<_> = loader.collect();

    assert_eq!(batches.len(), 3);
    assert_eq!(batches[0].0.dims()[0], 5);
    assert_eq!(batches[1].0.dims()[0], 5);
    assert_eq!(batches[2].0.dims()[0], 2);
}

#[test]
fn test_dataloader_content() {
    let x = Tensor::arange(0.0f32, 24.0, &DEVICE)
        .unwrap()
        .reshape(&[12, 2])
        .unwrap();
    let y = Tensor::arange(0u32, 12, &DEVICE).unwrap();

    let loader = DataLoader::new(x.clone(), y.clone(), 5);
    let batches: Vec<_> = loader.collect();

    // First batch: rows 0..5
    let xb0: Vec<f32> = batches[0].0.flatten_all().unwrap().to_vec1().unwrap();
    let x_expected: Vec<f32> = x.narrow(0, 0, 5).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(xb0, x_expected);

    let yb0: Vec<u32> = batches[0].1.to_vec1().unwrap();
    let y_expected: Vec<u32> = y.narrow(0, 0, 5).unwrap().to_vec1().unwrap();
    assert_eq!(yb0, y_expected);

    // Last batch: rows 10..12
    let xb2: Vec<f32> = batches[2].0.flatten_all().unwrap().to_vec1().unwrap();
    let x_expected2: Vec<f32> = x.narrow(0, 10, 2).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(xb2, x_expected2);
}

#[test]
fn test_dataloader_reiterable() {
    let x = Tensor::arange(0.0f32, 24.0, &DEVICE)
        .unwrap()
        .reshape(&[12, 2])
        .unwrap();
    let y = Tensor::arange(0u32, 12, &DEVICE).unwrap();

    let loader1 = DataLoader::new(x.clone(), y.clone(), 5);
    let batches1: Vec<_> = loader1.collect();

    let loader2 = DataLoader::new(x, y, 5);
    let batches2: Vec<_> = loader2.collect();

    assert_eq!(batches1.len(), batches2.len());
    for ((xb1, yb1), (xb2, yb2)) in batches1.iter().zip(batches2.iter()) {
        let v1: Vec<f32> = xb1.flatten_all().unwrap().to_vec1().unwrap();
        let v2: Vec<f32> = xb2.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v1, v2);
        let l1: Vec<u32> = yb1.to_vec1().unwrap();
        let l2: Vec<u32> = yb2.to_vec1().unwrap();
        assert_eq!(l1, l2);
    }
}

// --- epoch ---

#[test]
fn test_epoch_eval() {
    // In eval mode (no optimizer), epoch should not modify model params
    // and should return correct loss and error
    let varmap = VarMap::new();
    let layer = Linear::new(4, 3, &varmap, "epoch_eval").unwrap();

    let x = Tensor::randn(0.0f32, 1.0, &[11, 4], &DEVICE).unwrap();
    let y = Tensor::new(&[0u32, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1], &DEVICE).unwrap();

    let w_before: Vec<f32> = layer.weight().flatten_all().unwrap().to_vec1().unwrap();

    let loader: Vec<(Tensor, Tensor)> = vec![
        (x.narrow(0, 0, 4).unwrap(), y.narrow(0, 0, 4).unwrap()),
        (x.narrow(0, 4, 4).unwrap(), y.narrow(0, 4, 4).unwrap()),
        (x.narrow(0, 8, 3).unwrap(), y.narrow(0, 8, 3).unwrap()),
    ];

    let model_fn = |input: &Tensor| -> candle_core::Result<Tensor> { layer.forward(input) };

    let (eval_loss, eval_err) = epoch(&model_fn, &loader, &cross_entropy_loss, None).unwrap();

    // Params should not change in eval mode
    let w_after: Vec<f32> = layer.weight().flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(w_before, w_after, "Eval mode should not modify parameters");

    // Loss should be finite and positive
    assert!(eval_loss > 0.0, "Loss should be positive, got {eval_loss}");
    assert!(eval_loss.is_finite());

    // Error should be between 0 and 1
    assert!(eval_err >= 0.0 && eval_err <= 1.0, "Error rate should be in [0,1], got {eval_err}");
}

#[test]
fn test_epoch_train() {
    // In train mode (with optimizer), epoch should modify model params
    let varmap = VarMap::new();
    let layer = Linear::new(4, 3, &varmap, "epoch_train").unwrap();

    let x = Tensor::randn(0.0f32, 1.0, &[11, 4], &DEVICE).unwrap();
    let y = Tensor::new(&[0u32, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1], &DEVICE).unwrap();

    let w_before: Vec<f32> = layer.weight().flatten_all().unwrap().to_vec1().unwrap();

    let loader: Vec<(Tensor, Tensor)> = vec![
        (x.narrow(0, 0, 4).unwrap(), y.narrow(0, 0, 4).unwrap()),
        (x.narrow(0, 4, 4).unwrap(), y.narrow(0, 4, 4).unwrap()),
        (x.narrow(0, 8, 3).unwrap(), y.narrow(0, 8, 3).unwrap()),
    ];

    let params: Vec<Tensor> = varmap.all_vars().iter().map(|v| v.as_tensor().clone()).collect();
    let mut opt = SGD::new(params, 0.1);

    let model_fn = |input: &Tensor| -> candle_core::Result<Tensor> { layer.forward(input) };

    let (train_loss, train_err) = epoch(&model_fn, &loader, &cross_entropy_loss, Some(&mut opt)).unwrap();

    // Params should change in train mode
    let w_after: Vec<f32> = layer.weight().flatten_all().unwrap().to_vec1().unwrap();
    let changed = w_after.iter().zip(w_before.iter()).any(|(a, b)| (a - b).abs() > 1e-7);
    assert!(changed, "Train mode should modify parameters");

    assert!(train_loss > 0.0);
    assert!(train_loss.is_finite());
    assert!(train_err >= 0.0 && train_err <= 1.0);
}

// --- TwoLayerNN ---

#[test]
fn test_two_layer_nn_shape() {
    let varmap = VarMap::new();
    let model = TwoLayerNN::new(8, 16, 5, &varmap).unwrap();

    let x = Tensor::randn(0.0f32, 1.0, &[13, 8], &DEVICE).unwrap();
    let out = model.forward(&x).unwrap();
    assert_eq!(out.dims(), &[13, 5]);
}

#[test]
fn test_two_layer_nn_relu() {
    // Verify that ReLU is applied between layers (output should differ from
    // a simple two-matrix multiply without nonlinearity)
    let varmap = VarMap::new();
    let model = TwoLayerNN::new(4, 8, 3, &varmap).unwrap();

    // Use input with some negative values to ensure ReLU has an effect
    let x = Tensor::new(&[[-1.0f32, 2.0, -3.0, 4.0], [5.0, -6.0, 7.0, -8.0]], &DEVICE).unwrap();
    let out = model.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, 3]);

    // All outputs should be finite
    let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    for v in &vals {
        assert!(v.is_finite());
    }
}

#[test]
fn test_two_layer_nn_batch_dims() {
    let varmap = VarMap::new();
    let model = TwoLayerNN::new(8, 16, 5, &varmap).unwrap();

    let x = Tensor::randn(0.0f32, 1.0, &[2, 7, 8], &DEVICE).unwrap();
    let out = model.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, 7, 5]);
}

// --- MultiLayerNN ---

#[test]
fn test_multi_layer_nn_shape() {
    let varmap = VarMap::new();
    let hidden_dims = vec![7, 5, 4];
    let model = MultiLayerNN::new(6, 3, &hidden_dims, &varmap).unwrap();

    let x = Tensor::randn(0.0f32, 1.0, &[9, 6], &DEVICE).unwrap();
    let out = model.forward(&x).unwrap();
    assert_eq!(out.dims(), &[9, 3]);
}

#[test]
fn test_multi_layer_nn_batch_dims() {
    let varmap = VarMap::new();
    let hidden_dims = vec![7, 5, 4];
    let model = MultiLayerNN::new(6, 3, &hidden_dims, &varmap).unwrap();

    let x = Tensor::randn(0.0f32, 1.0, &[2, 4, 6], &DEVICE).unwrap();
    let out = model.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, 4, 3]);
}

#[test]
fn test_multi_layer_nn_single_hidden() {
    // With one hidden layer, should behave like TwoLayerNN
    let varmap = VarMap::new();
    let hidden_dims = vec![16];
    let model = MultiLayerNN::new(8, 5, &hidden_dims, &varmap).unwrap();

    let x = Tensor::randn(0.0f32, 1.0, &[4, 8], &DEVICE).unwrap();
    let out = model.forward(&x).unwrap();
    assert_eq!(out.dims(), &[4, 5]);

    let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    for v in &vals {
        assert!(v.is_finite());
    }
}

// --- MNIST end-to-end tests ---

mod mnist {
    use candle_core::{Device, Tensor};
    use flate2::read::GzDecoder;
    use hf_hub::api::sync::Api;
    use std::io::Read;

    const DEVICE: Device = Device::Cpu;

    pub struct MnistDataset {
        pub x: Tensor,
        pub y: Tensor,
    }

    fn download_and_decompress(repo: &hf_hub::api::sync::ApiRepo, filename: &str) -> Vec<u8> {
        let path = repo.get(filename).unwrap_or_else(|e| {
            panic!("Failed to download {filename}: {e}");
        });
        let compressed = std::fs::read(&path).unwrap();
        let mut decoder = GzDecoder::new(&compressed[..]);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed).unwrap();
        decompressed
    }

    fn parse_idx_images_flat(data: &[u8]) -> Tensor {
        let magic = u32::from_be_bytes(data[0..4].try_into().unwrap());
        assert_eq!(magic, 2051);
        let n_images = u32::from_be_bytes(data[4..8].try_into().unwrap()) as usize;
        let rows = u32::from_be_bytes(data[8..12].try_into().unwrap()) as usize;
        let cols = u32::from_be_bytes(data[12..16].try_into().unwrap()) as usize;
        let pixels = &data[16..];
        let flat: Vec<f32> = pixels[..n_images * rows * cols]
            .iter()
            .map(|&p| p as f32 / 255.0)
            .collect();
        Tensor::from_vec(flat, &[n_images, rows * cols], &DEVICE).unwrap()
    }

    fn parse_idx_labels(data: &[u8]) -> Tensor {
        let magic = u32::from_be_bytes(data[0..4].try_into().unwrap());
        assert_eq!(magic, 2049);
        let n_labels = u32::from_be_bytes(data[4..8].try_into().unwrap()) as usize;
        let labels: Vec<u32> = data[8..8 + n_labels].iter().map(|&l| l as u32).collect();
        Tensor::from_vec(labels, &[n_labels], &DEVICE).unwrap()
    }

    pub fn load_mnist_train() -> MnistDataset {
        let api = Api::new().unwrap();
        let repo = api.dataset("ylecun/mnist".to_string());
        let image_data = download_and_decompress(&repo, "train-images-idx3-ubyte.gz");
        let label_data = download_and_decompress(&repo, "train-labels-idx1-ubyte.gz");
        MnistDataset {
            x: parse_idx_images_flat(&image_data),
            y: parse_idx_labels(&label_data),
        }
    }

    pub fn load_mnist_test() -> MnistDataset {
        let api = Api::new().unwrap();
        let repo = api.dataset("ylecun/mnist".to_string());
        let image_data = download_and_decompress(&repo, "t10k-images-idx3-ubyte.gz");
        let label_data = download_and_decompress(&repo, "t10k-labels-idx1-ubyte.gz");
        MnistDataset {
            x: parse_idx_images_flat(&image_data),
            y: parse_idx_labels(&label_data),
        }
    }
}

#[test]
fn test_eval_linear_model() {
    let train = mnist::load_mnist_train();
    let test = mnist::load_mnist_test();

    let model = eval_linear_model(&train.x, &train.y).unwrap();

    let logits = model.forward(&test.x).unwrap();
    assert_eq!(logits.dims(), &[10000, 10]);

    let preds = logits.argmax(1).unwrap();
    let correct = preds
        .eq(&test.y)
        .unwrap()
        .to_dtype(candle_core::DType::F32)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let err = 1.0 - correct / 10000.0;
    assert!(
        err < 0.1,
        "Linear model error {err:.4} is not < 0.1 on MNIST test set"
    );
}

#[test]
fn test_eval_two_layer_nn() {
    let train = mnist::load_mnist_train();
    let test = mnist::load_mnist_test();

    let model = eval_two_layer_nn(&train.x, &train.y).unwrap();

    let x_sub = test.x.narrow(0, 0, 2000).unwrap();
    let y_sub = test.y.narrow(0, 0, 2000).unwrap();

    let logits = model.forward(&x_sub).unwrap();
    assert_eq!(logits.dims(), &[2000, 10]);

    let preds = logits.argmax(1).unwrap();
    let correct = preds
        .eq(&y_sub)
        .unwrap()
        .to_dtype(candle_core::DType::F32)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let err = 1.0 - correct / 2000.0;
    assert!(
        err < 0.03,
        "Two-layer NN error {err:.4} is not < 0.03 on first 2000 MNIST test samples"
    );
}
