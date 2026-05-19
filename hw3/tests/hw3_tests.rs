use burn::backend::ndarray::NdArrayDevice;
use burn::tensor::{Distribution, Int, Tensor, TensorData};
use hw3::*;

const DEVICE: NdArrayDevice = NdArrayDevice::Cpu;

// --- Linear layer ---

#[test]
fn test_linear_shape() {
    let layer = Linear::new(10, 20, &DEVICE);

    let x: Tensor<B, 2> = Tensor::random([50, 10], Distribution::Normal(0.0, 1.0), &DEVICE);
    let out = layer.forward(x);
    assert_eq!(out.dims(), [50, 20]);
}

#[test]
fn test_linear_batch_dims() {
    let layer = Linear::new(10, 20, &DEVICE);

    let x: Tensor<B, 3> = Tensor::random([7, 9, 10], Distribution::Normal(0.0, 1.0), &DEVICE);
    let out = layer.forward(x);
    assert_eq!(out.dims(), [7, 9, 20]);
}

#[test]
fn test_linear_correctness() {
    let layer = Linear::new(4, 3, &DEVICE);

    let x: Tensor<B, 2> = Tensor::from_data(
        TensorData::from([[1.0f32, 2.0, -1.0, 0.5], [0.0, -1.0, 2.0, 3.0]]),
        &DEVICE,
    );
    let out = layer.forward(x.clone());

    // Reference: x @ weight.T
    let w = layer.weight().clone();
    let expected = x.matmul(w.transpose());

    let diff: f32 = (out - expected).abs().sum().into_scalar();
    assert!(
        diff < 1e-5,
        "Linear output doesn't match X @ W^T, diff={diff}"
    );
}

#[test]
fn test_linear_kaiming_init() {
    let layer = Linear::new(100, 1000, &DEVICE);

    let w = layer.weight().clone();
    // Kaiming init: std ≈ sqrt(2 / in_features) = sqrt(2/100) ≈ 0.1414
    let expected_std = (2.0f64 / 100.0).sqrt();
    let mean: f32 = w.clone().mean().into_scalar();
    let variance: f32 = (w.clone() - mean).powf_scalar(2.0).mean().into_scalar();
    let std = (variance as f64).sqrt();

    assert!(
        (std - expected_std).abs() < 3e-3,
        "Weight std {std} not close to expected {expected_std}"
    );
    assert!(
        (mean as f64).abs() < 0.02,
        "Weight mean {mean} not close to 0"
    );
}

// --- Cross-entropy loss ---

#[test]
fn test_cross_entropy_loss() {
    let logits: Tensor<B, 2> = Tensor::from_data(
        TensorData::from([[2.0f32, 1.0, 0.0], [0.0, 2.0, 1.0]]),
        &DEVICE,
    );
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from([0i32, 2]), &DEVICE);
    let loss = cross_entropy_loss(logits, y);
    let loss_val: f32 = loss.into_scalar();
    assert!((loss_val - 0.907_606).abs() < 1e-5);
}

#[test]
fn test_cross_entropy_numerically_stable() {
    let logits: Tensor<B, 2> = Tensor::from_data(
        TensorData::from([[1000.0f32, 1001.0, 999.5], [1.0, -2.0, 0.5]]),
        &DEVICE,
    );
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from([1i32, 2]), &DEVICE);
    let loss = cross_entropy_loss(logits, y);
    let loss_val: f32 = loss.into_scalar();
    assert!(loss_val.is_finite());
    assert!(
        (loss_val - 0.6483).abs() < 0.01,
        "Expected ≈0.6483, got {loss_val}"
    );
}

// --- SGD optimizer ---

#[test]
fn test_sgd_step() {
    let layer = Linear::new(4, 3, &DEVICE);

    let w_before: Vec<f32> = layer.weight().clone().into_data().to_vec().unwrap();

    let params: Vec<Tensor<B, 2>> = vec![layer.weight().clone()];
    let mut opt = SGD::new(params, 0.05);

    let x: Tensor<B, 2> = Tensor::random([12, 4], Distribution::Normal(0.0, 1.0), &DEVICE);
    let y: Tensor<B, 1, Int> = Tensor::from_data(
        TensorData::from([0i32, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]),
        &DEVICE,
    );

    // Training loop: 3 steps
    for _ in 0..3 {
        let logits = layer.forward(x.clone());
        let loss = cross_entropy_loss(logits, y.clone());
        let grads = loss.backward();
        opt.step(&grads);
    }

    let w_after: Vec<f32> = layer.weight().clone().into_data().to_vec().unwrap();
    let changed = w_after
        .iter()
        .zip(w_before.iter())
        .any(|(a, b)| (a - b).abs() > 1e-7);
    assert!(
        changed,
        "SGD step did not modify the weights after 3 training steps"
    );
}

#[test]
fn test_sgd_zero_grad() {
    let layer = Linear::new(4, 3, &DEVICE);

    let params: Vec<Tensor<B, 2>> = vec![layer.weight().clone()];
    let _opt = SGD::new(params, 0.1);

    // Do a forward/backward to create gradients
    let x: Tensor<B, 2> = Tensor::random([4, 4], Distribution::Normal(0.0, 1.0), &DEVICE);
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from([0i32, 1, 2, 0]), &DEVICE);
    let logits = layer.forward(x);
    let loss = cross_entropy_loss(logits, y);
    let _grads = loss.backward();

    // step with zero gradients should not move weights
    // (In burn, gradients are computed per backward call, no accumulation)
    let w_before: Vec<f32> = layer.weight().clone().into_data().to_vec().unwrap();

    // Don't call backward again, so no valid grads to step with
    let w_after: Vec<f32> = layer.weight().clone().into_data().to_vec().unwrap();

    for (a, b) in w_after.iter().zip(w_before.iter()) {
        assert!(
            (a - b).abs() < 1e-7,
            "Weights should not change without a step: {a} vs {b}"
        );
    }
}

// --- DataLoader ---

#[test]
fn test_dataloader_batches() {
    let x: Tensor<B, 2> = Tensor::arange(0..24, &DEVICE).float().reshape([12, 2]);
    let y: Tensor<B, 1, Int> = Tensor::arange(0..12, &DEVICE);

    let loader = DataLoader::new(x.clone(), y.clone(), 5);
    let batches: Vec<_> = loader.collect();

    assert_eq!(batches.len(), 3);
    assert_eq!(batches[0].0.dims()[0], 5);
    assert_eq!(batches[1].0.dims()[0], 5);
    assert_eq!(batches[2].0.dims()[0], 2);
}

#[test]
fn test_dataloader_content() {
    let x: Tensor<B, 2> = Tensor::arange(0..24, &DEVICE).float().reshape([12, 2]);
    let y: Tensor<B, 1, Int> = Tensor::arange(0..12, &DEVICE);

    let loader = DataLoader::new(x.clone(), y.clone(), 5);
    let batches: Vec<_> = loader.collect();

    // First batch: rows 0..5
    let xb0: Vec<f32> = batches[0]
        .0
        .clone()
        .reshape([10])
        .into_data()
        .to_vec()
        .unwrap();
    let x_expected: Vec<f32> = x
        .clone()
        .narrow(0, 0, 5)
        .reshape([10])
        .into_data()
        .to_vec()
        .unwrap();
    assert_eq!(xb0, x_expected);

    let yb0: Vec<i32> = batches[0].1.clone().into_data().to_vec().unwrap();
    let y_expected: Vec<i32> = y.clone().narrow(0, 0, 5).into_data().to_vec().unwrap();
    assert_eq!(yb0, y_expected);

    // Last batch: rows 10..12
    let xb2: Vec<f32> = batches[2]
        .0
        .clone()
        .reshape([4])
        .into_data()
        .to_vec()
        .unwrap();
    let x_expected2: Vec<f32> = x
        .narrow(0, 10, 2)
        .reshape([4])
        .into_data()
        .to_vec()
        .unwrap();
    assert_eq!(xb2, x_expected2);
}

#[test]
fn test_dataloader_reiterable() {
    let x: Tensor<B, 2> = Tensor::arange(0..24, &DEVICE).float().reshape([12, 2]);
    let y: Tensor<B, 1, Int> = Tensor::arange(0..12, &DEVICE);

    let loader1 = DataLoader::new(x.clone(), y.clone(), 5);
    let batches1: Vec<_> = loader1.collect();

    let loader2 = DataLoader::new(x, y, 5);
    let batches2: Vec<_> = loader2.collect();

    assert_eq!(batches1.len(), batches2.len());
    for ((xb1, yb1), (xb2, yb2)) in batches1.iter().zip(batches2.iter()) {
        let v1: Vec<f32> = xb1
            .clone()
            .reshape([xb1.dims()[0] * xb1.dims()[1]])
            .into_data()
            .to_vec()
            .unwrap();
        let v2: Vec<f32> = xb2
            .clone()
            .reshape([xb2.dims()[0] * xb2.dims()[1]])
            .into_data()
            .to_vec()
            .unwrap();
        assert_eq!(v1, v2);
        let l1: Vec<i32> = yb1.clone().into_data().to_vec().unwrap();
        let l2: Vec<i32> = yb2.clone().into_data().to_vec().unwrap();
        assert_eq!(l1, l2);
    }
}

// --- epoch ---

#[test]
fn test_epoch_eval() {
    let layer = Linear::new(4, 3, &DEVICE);

    let x: Tensor<B, 2> = Tensor::random([11, 4], Distribution::Normal(0.0, 1.0), &DEVICE);
    let y: Tensor<B, 1, Int> = Tensor::from_data(
        TensorData::from([0i32, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]),
        &DEVICE,
    );

    let w_before: Vec<f32> = layer.weight().clone().into_data().to_vec().unwrap();

    let loader: Vec<(Tensor<B, 2>, Tensor<B, 1, Int>)> = vec![
        (x.clone().narrow(0, 0, 4), y.clone().narrow(0, 0, 4)),
        (x.clone().narrow(0, 4, 4), y.clone().narrow(0, 4, 4)),
        (x.narrow(0, 8, 3), y.narrow(0, 8, 3)),
    ];

    let model_fn = |input: Tensor<B, 2>| -> Tensor<B, 2> { layer.forward(input) };

    let (eval_loss, eval_err) = epoch(&model_fn, &loader, None);

    // Params should not change in eval mode
    let w_after: Vec<f32> = layer.weight().clone().into_data().to_vec().unwrap();
    assert_eq!(w_before, w_after, "Eval mode should not modify parameters");

    // Loss should be finite and positive
    assert!(eval_loss > 0.0, "Loss should be positive, got {eval_loss}");
    assert!(eval_loss.is_finite());

    // Error should be between 0 and 1
    assert!(
        (0.0..=1.0).contains(&eval_err),
        "Error rate should be in [0,1], got {eval_err}"
    );
}

#[test]
fn test_epoch_train() {
    let layer = Linear::new(4, 3, &DEVICE);

    let x: Tensor<B, 2> = Tensor::random([11, 4], Distribution::Normal(0.0, 1.0), &DEVICE);
    let y: Tensor<B, 1, Int> = Tensor::from_data(
        TensorData::from([0i32, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]),
        &DEVICE,
    );

    let w_before: Vec<f32> = layer.weight().clone().into_data().to_vec().unwrap();

    let loader: Vec<(Tensor<B, 2>, Tensor<B, 1, Int>)> = vec![
        (x.clone().narrow(0, 0, 4), y.clone().narrow(0, 0, 4)),
        (x.clone().narrow(0, 4, 4), y.clone().narrow(0, 4, 4)),
        (x.narrow(0, 8, 3), y.narrow(0, 8, 3)),
    ];

    let params: Vec<Tensor<B, 2>> = vec![layer.weight().clone()];
    let mut opt = SGD::new(params, 0.1);

    let model_fn = |input: Tensor<B, 2>| -> Tensor<B, 2> { layer.forward(input) };

    let (train_loss, train_err) = epoch(&model_fn, &loader, Some(&mut opt));

    // Params should change in train mode
    let w_after: Vec<f32> = layer.weight().clone().into_data().to_vec().unwrap();
    let changed = w_after
        .iter()
        .zip(w_before.iter())
        .any(|(a, b)| (a - b).abs() > 1e-7);
    assert!(changed, "Train mode should modify parameters");

    assert!(train_loss > 0.0);
    assert!(train_loss.is_finite());
    assert!((0.0..=1.0).contains(&train_err));
}

// --- TwoLayerNN ---

#[test]
fn test_two_layer_nn_shape() {
    let model = TwoLayerNN::new(8, 16, 5, &DEVICE);

    let x: Tensor<B, 2> = Tensor::random([13, 8], Distribution::Normal(0.0, 1.0), &DEVICE);
    let out = model.forward(x);
    assert_eq!(out.dims(), [13, 5]);
}

#[test]
fn test_two_layer_nn_relu() {
    let model = TwoLayerNN::new(4, 8, 3, &DEVICE);

    let x: Tensor<B, 2> = Tensor::from_data(
        TensorData::from([[-1.0f32, 2.0, -3.0, 4.0], [5.0, -6.0, 7.0, -8.0]]),
        &DEVICE,
    );
    let out = model.forward(x);
    assert_eq!(out.dims(), [2, 3]);

    let vals: Vec<f32> = out.into_data().to_vec().unwrap();
    for v in &vals {
        assert!(v.is_finite());
    }
}

#[test]
fn test_two_layer_nn_batch_dims() {
    let model = TwoLayerNN::new(8, 16, 5, &DEVICE);

    let x: Tensor<B, 3> = Tensor::random([2, 7, 8], Distribution::Normal(0.0, 1.0), &DEVICE);
    let out = model.forward(x);
    assert_eq!(out.dims(), [2, 7, 5]);
}

// --- MultiLayerNN ---

#[test]
fn test_multi_layer_nn_shape() {
    let hidden_dims = vec![7, 5, 4];
    let model = MultiLayerNN::new(6, 3, &hidden_dims, &DEVICE);

    let x: Tensor<B, 2> = Tensor::random([9, 6], Distribution::Normal(0.0, 1.0), &DEVICE);
    let out = model.forward(x);
    assert_eq!(out.dims(), [9, 3]);
}

#[test]
fn test_multi_layer_nn_batch_dims() {
    let hidden_dims = vec![7, 5, 4];
    let model = MultiLayerNN::new(6, 3, &hidden_dims, &DEVICE);

    let x: Tensor<B, 3> = Tensor::random([2, 4, 6], Distribution::Normal(0.0, 1.0), &DEVICE);
    let out = model.forward(x);
    assert_eq!(out.dims(), [2, 4, 3]);
}

#[test]
fn test_multi_layer_nn_single_hidden() {
    let hidden_dims = vec![16];
    let model = MultiLayerNN::new(8, 5, &hidden_dims, &DEVICE);

    let x: Tensor<B, 2> = Tensor::random([4, 8], Distribution::Normal(0.0, 1.0), &DEVICE);
    let out = model.forward(x);
    assert_eq!(out.dims(), [4, 5]);

    let vals: Vec<f32> = out.into_data().to_vec().unwrap();
    for v in &vals {
        assert!(v.is_finite());
    }
}

// --- MNIST end-to-end tests ---

mod mnist {
    use burn::backend::ndarray::NdArrayDevice;
    use burn::tensor::{Int, Tensor, TensorData};
    use flate2::read::GzDecoder;
    use hf_hub::api::sync::Api;
    use hw3::B;
    use std::io::Read;

    const DEVICE: NdArrayDevice = NdArrayDevice::Cpu;

    pub struct MnistDataset {
        pub x: Tensor<B, 2>,
        pub y: Tensor<B, 1, Int>,
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

    fn parse_idx_images_flat(data: &[u8]) -> Tensor<B, 2> {
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
        Tensor::from_data(TensorData::new(flat, [n_images, rows * cols]), &DEVICE)
    }

    fn parse_idx_labels(data: &[u8]) -> Tensor<B, 1, Int> {
        let magic = u32::from_be_bytes(data[0..4].try_into().unwrap());
        assert_eq!(magic, 2049);
        let n_labels = u32::from_be_bytes(data[4..8].try_into().unwrap()) as usize;
        let labels: Vec<i32> = data[8..8 + n_labels].iter().map(|&l| l as i32).collect();
        Tensor::from_data(TensorData::new(labels, [n_labels]), &DEVICE)
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

    let model = eval_linear_model(train.x, train.y);

    let logits = model.forward(test.x.clone());
    assert_eq!(logits.dims(), [10000, 10]);

    let n = logits.dims()[0];
    let preds: Tensor<B, 1, Int> = logits.argmax(1).reshape([n]);
    let correct: f32 = preds.equal(test.y).float().sum().into_scalar();
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

    let model = eval_two_layer_nn(train.x, train.y);

    let x_sub = test.x.narrow(0, 0, 2000);
    let y_sub = test.y.narrow(0, 0, 2000);

    let logits = model.forward(x_sub);
    assert_eq!(logits.dims(), [2000, 10]);

    let n = logits.dims()[0];
    let preds: Tensor<B, 1, Int> = logits.argmax(1).reshape([n]);
    let correct: f32 = preds.equal(y_sub).float().sum().into_scalar();
    let err = 1.0 - correct / 2000.0;
    assert!(
        err < 0.03,
        "Two-layer NN error {err:.4} is not < 0.03 on first 2000 MNIST test samples"
    );
}

// --- Additional edge case and value-verification tests ---

#[test]
fn test_cross_entropy_loss_single_sample() {
    let logits: Tensor<B, 2> = Tensor::from_data(
        TensorData::from([[1.0f32, 2.0, 3.0]]),
        &DEVICE,
    );
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from([2i32]), &DEVICE);
    let loss: f32 = cross_entropy_loss(logits, y).into_scalar();
    assert!(loss.is_finite());
    assert!((loss - 0.4076).abs() < 1e-3, "Single sample loss incorrect: {loss}");
}

#[test]
fn test_cross_entropy_loss_perfect_prediction() {
    let logits: Tensor<B, 2> = Tensor::from_data(
        TensorData::from([[100.0f32, 0.0, 0.0], [0.0, 100.0, 0.0]]),
        &DEVICE,
    );
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from([0i32, 1]), &DEVICE);
    let loss: f32 = cross_entropy_loss(logits, y).into_scalar();
    assert!(loss.is_finite());
    assert!(loss < 1e-5, "Perfect prediction should give near-zero loss, got {loss}");
}

#[test]
fn test_cross_entropy_loss_large_logits_stable() {
    let logits: Tensor<B, 2> = Tensor::from_data(
        TensorData::from([[1e30f32, 0.0], [0.0, 1e30]]),
        &DEVICE,
    );
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from([0i32, 1]), &DEVICE);
    let loss: f32 = cross_entropy_loss(logits, y).into_scalar();
    assert!(loss.is_finite(), "Should handle very large logits without overflow");
    assert!(loss < 1e-5);
}

#[test]
fn test_linear_output_nonzero() {
    let layer = Linear::new(4, 3, &DEVICE);
    let x: Tensor<B, 2> = Tensor::from_data(
        TensorData::from([[1.0f32, 0.0, 0.0, 0.0]]),
        &DEVICE,
    );
    let out = layer.forward(x);
    let vals: Vec<f32> = out.into_data().to_vec().unwrap();
    assert!(vals.iter().any(|&v| v != 0.0), "Linear output should not be all zeros for nonzero input");
}

#[test]
fn test_sgd_moves_in_gradient_direction() {
    let layer = Linear::new(4, 3, &DEVICE);
    let _w_before: Vec<f32> = layer.weight().clone().into_data().to_vec().unwrap();

    let params: Vec<Tensor<B, 2>> = vec![layer.weight().clone()];
    let mut opt = SGD::new(params, 0.1);

    let x: Tensor<B, 2> = Tensor::from_data(
        TensorData::from([[1.0f32, 0.0, 0.0, 0.0]]),
        &DEVICE,
    );
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from([0i32]), &DEVICE);

    let logits = layer.forward(x);
    let loss = cross_entropy_loss(logits, y);
    let loss_val: f32 = loss.clone().into_scalar();
    let grads = loss.backward();
    opt.step(&grads);

    // After one step, loss should decrease on the same input
    let x2: Tensor<B, 2> = Tensor::from_data(
        TensorData::from([[1.0f32, 0.0, 0.0, 0.0]]),
        &DEVICE,
    );
    let y2: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from([0i32]), &DEVICE);
    let logits2 = layer.forward(x2);
    let loss2: f32 = cross_entropy_loss(logits2, y2).into_scalar();
    assert!(loss2 < loss_val, "Loss should decrease after SGD step: before={loss_val}, after={loss2}");
}

#[test]
fn test_dataloader_single_sample() {
    let x: Tensor<B, 2> = Tensor::from_data(TensorData::from([[1.0f32, 2.0]]), &DEVICE);
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from([0i32]), &DEVICE);

    let loader = DataLoader::new(x, y, 10);
    let batches: Vec<_> = loader.collect();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].0.dims()[0], 1);
}

#[test]
fn test_dataloader_batch_equals_dataset() {
    let x: Tensor<B, 2> = Tensor::arange(0..8, &DEVICE).float().reshape([4, 2]);
    let y: Tensor<B, 1, Int> = Tensor::arange(0..4, &DEVICE);

    let loader = DataLoader::new(x, y, 4);
    let batches: Vec<_> = loader.collect();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].0.dims()[0], 4);
}

#[test]
fn test_epoch_loss_decreases_with_training() {
    let layer = Linear::new(4, 3, &DEVICE);

    let x: Tensor<B, 2> = Tensor::random([20, 4], Distribution::Normal(0.0, 1.0), &DEVICE);
    let y: Tensor<B, 1, Int> = Tensor::from_data(
        TensorData::from([0i32, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]),
        &DEVICE,
    );

    let loader: Vec<(Tensor<B, 2>, Tensor<B, 1, Int>)> = vec![
        (x.clone().narrow(0, 0, 10), y.clone().narrow(0, 0, 10)),
        (x.narrow(0, 10, 10), y.narrow(0, 10, 10)),
    ];

    let model_fn = |input: Tensor<B, 2>| -> Tensor<B, 2> { layer.forward(input) };

    let (loss_before, _) = epoch(&model_fn, &loader, None);

    let params: Vec<Tensor<B, 2>> = vec![layer.weight().clone()];
    let mut opt = SGD::new(params, 0.1);

    // Run several training epochs
    for _ in 0..5 {
        epoch(&model_fn, &loader, Some(&mut opt));
    }

    let (loss_after, _) = epoch(&model_fn, &loader, None);
    assert!(loss_after < loss_before, "Loss should decrease after training: before={loss_before}, after={loss_after}");
}

#[test]
fn test_two_layer_nn_output_nonzero() {
    let model = TwoLayerNN::new(4, 8, 3, &DEVICE);
    let x: Tensor<B, 2> = Tensor::from_data(
        TensorData::from([[1.0f32, 0.5, -0.5, 1.0]]),
        &DEVICE,
    );
    let out = model.forward(x);
    let vals: Vec<f32> = out.into_data().to_vec().unwrap();
    assert!(vals.iter().any(|&v| v != 0.0), "TwoLayerNN output should not be all zeros");
}

#[test]
fn test_multi_layer_nn_output_nonzero() {
    let model = MultiLayerNN::new(4, 3, &[8, 6], &DEVICE);
    let x: Tensor<B, 2> = Tensor::from_data(
        TensorData::from([[1.0f32, -1.0, 0.5, -0.5]]),
        &DEVICE,
    );
    let out = model.forward(x);
    let vals: Vec<f32> = out.into_data().to_vec().unwrap();
    assert!(vals.iter().any(|&v| v != 0.0), "MultiLayerNN output should not be all zeros");
}
