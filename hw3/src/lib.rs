/*
 * Homework 3 - Training models in PyTorch
 *
 * In this homework, we'll start to build and train machine learning models (both a
 * linear model and a neural network). While a lot of the code you will develop here
 * corresponds to existing implementations in deep learning frameworks, you will implement
 * almost everything from scratch in these assignments, rather than use pre-built layers.
 *
 * ## Part I - Training a linear model
 *
 * To begin, we'll implement a linear model trained via (stochastic) gradient descent,
 * and then use it to train a classifier for the MNIST digit prediction task.
 *
 * ### Question 1 - Linear layer
 *
 * Key points about implementing a linear layer:
 * - Store the weights as a Tensor of shape (out_dim, in_dim)
 * - Initialize with sqrt(2/in_dim) scaling of random Gaussian weights (Kaiming init)
 * - The forward call takes a batch of examples (batch_size x in_dim) and returns
 *   (batch_size x out_dim)
 *
 * ### Question 2 - Cross entropy loss
 *
 * Given a (batch_size x k) real-valued tensor of logits and a (batch_size) tensor of
 * integer labels, return the average cross entropy loss.
 * Use log-sum-exp for numerical stability.
 *
 * ### Question 3 - Stochastic Gradient Descent
 *
 * In the standard optimizer paradigm:
 *   let grads = loss.backward();
 *   opt.step(&grads);
 *
 * ### Question 4 - Data Loader
 *
 * A DataLoader is an iterator that yields minibatches (X_batch, y_batch) from a dataset.
 * Implement it using the Iterator trait:
 * - Reset on each new iteration (yields same batches if iterated twice)
 * - Returns None when exhausted
 *
 * ### Question 5 - Optimization epoch
 *
 * Run one pass over all minibatches in the data loader. For each minibatch:
 * - Compute predictions and loss
 * - If optimizer is provided, update parameters
 * - Track running total of loss and error
 * Return (average_loss, average_error) as floats.
 *
 * ## Part II - Training Neural Networks
 *
 * ### Question 6 - Two-layer neural network
 *
 * Implement the model: h(x) = W2 * relu(W1 * x)
 * Two linear layers with a ReLU nonlinearity between them.
 * Store as .linear1 and .linear2.
 *
 * ### Question 7 - Multi-layer neural network
 *
 * Implement an arbitrary multi-layer deep ReLU network:
 *   h(x) = W_L * relu(W_{L-1} * relu(... W_2 * relu(W_1 * x) ...))
 *
 * Initialized with input dim, output dim, and a list of hidden dimensions.
 * Store all Linear layers in a single Vec called .linears.
 */

use burn::backend::Autodiff;
use burn::module::{Module, Param};
use burn::prelude::*;
use burn::tensor::activation::relu;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::Distribution;

#[cfg(feature = "cuda")]
pub type MyBackend = burn::backend::Cuda<f32, i32>;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
pub type MyBackend = burn::backend::Wgpu<f32, i32>;

#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
pub type MyBackend = burn::backend::NdArray<f32, i32>;

#[cfg(feature = "cuda")]
pub const DEVICE: Device<MyAutodiffBackend> = burn::backend::cuda::CudaDevice { index: 0 };

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
pub const DEVICE: Device<MyAutodiffBackend> = burn::backend::wgpu::WgpuDevice::DefaultDevice;

#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
pub const DEVICE: Device<MyAutodiffBackend> = burn::backend::ndarray::NdArrayDevice::Cpu;

pub type MyAutodiffBackend = Autodiff<MyBackend>;

/// Linear layer (no bias) with Kaiming initialization.
///
/// Initialize with Gaussian weights scaled by sqrt(2/in_dim).
///
/// Inputs:
///     in_dim: input feature dimension
///     out_dim: output feature dimension
///
/// forward() takes (batch_size x in_dim) and returns (batch_size x out_dim).

#[derive(Module, Debug)]
pub struct Linear<B: Backend> {
    pub weight: Param<Tensor<B, 2>>,
}

impl<B> Linear<B>
where
    B: Backend,
{
    pub fn new(in_dim: usize, out_dim: usize, device: &B::Device) -> Self {
        let std = (2.0 / in_dim as f64).sqrt();
        Linear {
            weight: Param::from_tensor(Tensor::<B, 2>::random(
                [out_dim, in_dim],
                Distribution::Normal(0.0, std),
                device,
            )),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        x.matmul(self.weight.val().transpose())
    }
}

// the following two finctions are copied over from hw2/src/lib.rs
// with just input variable renaming and, for the cross entropy loss,
// wrapping into a Module
fn logsumexp(x: Tensor<MyAutodiffBackend, 2>, dim: usize) -> Tensor<MyAutodiffBackend, 2> {
    assert!(dim == 0 || dim == 1, "Incompatible dimension requested");

    let max_x = x.clone().max_dim(dim);
    (x - max_x.clone()).exp().sum_dim(dim).log() + max_x
}

/// Cross-entropy loss (numerically stable via log-sum-exp).
///
/// Inputs:
///     logits: (N x k) predicted logits for each example
///     targets: (N) desired class for each example
/// Output:
///     scalar tensor - average cross entropy loss
pub fn cross_entropy_loss(
    logits: Tensor<MyAutodiffBackend, 2>,
    targets: Tensor<MyAutodiffBackend, 1, Int>,
) -> Tensor<MyAutodiffBackend, 1> {
    let batch_size = targets.dims()[0];
    let true_class_indexes = targets.reshape([batch_size, 1]);
    // prediction at what should be the true class
    let predictions = logits.clone().gather(1, true_class_indexes);
    (logsumexp(logits, 1) - predictions).mean()
}

/// SGD optimizer.
///
/// Initialize over a set of model parameters with a given learning rate.
/// step() applies: w = w - lr * w.grad
pub struct SGD {
    pub learning_rate: f64,
}

impl SGD {
    pub fn new(lr: f64) -> Self {
        SGD { learning_rate: lr }
    }

    pub fn step(
        &mut self,
        params: &mut [Param<Tensor<MyAutodiffBackend, 2>>],
        grads: &<MyAutodiffBackend as AutodiffBackend>::Gradients,
    ) {
        for param in params.iter_mut() {
            let param_tensor = param.val();
            if let Some(grad) = param_tensor.grad(grads) {
                let inner = param_tensor.inner();
                let updated_inner = inner - grad.mul_scalar(self.learning_rate);
                *param = Param::from_tensor(Tensor::from_inner(updated_inner));
            }
        }
    }
}

/// DataLoader: iterates over (X, y) in sequential minibatches.
///
/// Initialize with full dataset X (N x n), labels y (N), and batch_size.
/// Iterating twice should produce the same batches.
/// Last batch may be smaller than batch_size.
pub struct DataLoader {
    pub x_batches: Vec<Tensor<MyAutodiffBackend, 2>>,
    pub y_batches: Vec<Tensor<MyAutodiffBackend, 1, Int>>,
    index: usize,
    n: usize,
}

impl DataLoader {
    pub fn new(
        x: Tensor<MyAutodiffBackend, 2>,
        y: Tensor<MyAutodiffBackend, 1, Int>,
        batch_size: usize,
    ) -> Self {
        let n = x.clone().shape()[0].div_ceil(batch_size);

        DataLoader {
            x_batches: x.split(batch_size, 0),
            y_batches: y.split(batch_size, 0),
            index: 0,
            n,
        }
    }
}

impl Iterator for DataLoader {
    type Item = (
        Tensor<MyAutodiffBackend, 2>,
        Tensor<MyAutodiffBackend, 1, Int>,
    );

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.n {
            None
        } else {
            self.index += 1;
            Some((
                self.x_batches[self.index - 1].clone(),
                self.y_batches[self.index - 1].clone(),
            ))
        }
    }
}

/// Two-layer neural network: Linear -> ReLU -> Linear.
///
/// Implements h(x) = W2 * relu(W1 * x).
/// Store layers as .linear1 and .linear2.
#[derive(Module, Debug)]
pub struct TwoLayerNN<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
}

impl<B> TwoLayerNN<B>
where
    B: Backend,
{
    pub fn new(
        in_features: usize,
        hidden_features: usize,
        out_features: usize,
        device: &B::Device,
    ) -> Self {
        TwoLayerNN {
            linear1: Linear::new(in_features, hidden_features, device),
            linear2: Linear::new(hidden_features, out_features, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear2.forward(relu(self.linear1.forward(x)))
    }
}

/// Multi-layer neural network: [Linear -> ReLU] x N -> Linear.
///
/// Implements h(x) = W_L * relu(W_{L-1} * relu(... W_2 * relu(W_1 * x) ...))
/// Store all layers in a single .linears Vec.
#[derive(Module, Debug)]
pub struct MultiLayerNN<B: Backend> {
    linears: Vec<Linear<B>>,
}

impl<B> MultiLayerNN<B>
where
    B: Backend,
{
    pub fn new(
        in_features: usize,
        out_features: usize,
        hidden_dims: &[usize],
        device: &B::Device,
    ) -> Self {
        assert!(
            !hidden_dims.is_empty(),
            "Expecting at leat 1 hidden dimension!"
        );

        let mut d1: usize;
        let mut d0: usize = in_features;
        let mut linears: Vec<Linear<B>> = vec![];

        for &hidden_dim in hidden_dims {
            d1 = hidden_dim;
            linears.push(Linear::new(d0, d1, device));
            d0 = d1;
        }
        linears.push(Linear::new(
            // we are covered by the assert above.
            *hidden_dims.last().unwrap(),
            out_features,
            device,
        ));

        MultiLayerNN { linears }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut result = x.clone();

        // apply relu(model.forward) on all bu the last one...
        for i in 0..(self.linears.len() - 1) {
            result = relu(self.linears[i].forward(result));
        }
        self.linears.last().unwrap().forward(result)
    }
}

/// Train a linear model on MNIST and return it.
///
/// Given the full MNIST training data (X_train: N x 784, y_train: N),
/// train a Linear layer to classify all 10 digits.
/// The returned model should achieve < 10% error on the test set.
///
/// Use your Linear, CrossEntropyLoss, SGD, and DataLoader implementations.
pub fn eval_linear_model(
    x_train: Tensor<MyAutodiffBackend, 2>,
    y_train: Tensor<MyAutodiffBackend, 1, Int>,
) -> Linear<MyAutodiffBackend> {
    let device = x_train.device();
    let mut model: Linear<MyAutodiffBackend> = Linear::new(784, 10, &device);
    let mut opt = SGD::new(0.2);

    let epochs = 20;
    let batch_size = 100;

    for _ in 0..epochs {
        let loader = DataLoader::new(x_train.clone(), y_train.clone(), batch_size);
        for (x_batch, y_batch) in loader {
            let logits = model.forward(x_batch);
            let loss = cross_entropy_loss(logits, y_batch);
            let grads = loss.backward();
            opt.step(std::slice::from_mut(&mut model.weight), &grads);
        }
    }
    model
}

/// Train a two-layer neural network on MNIST and return it.
///
/// Given the full MNIST training data (X_train: N x 784, y_train: N),
/// train a TwoLayerNN to classify all 10 digits.
/// The returned model should achieve < 3% error on the first 2000 test samples.
///
/// Use your TwoLayerNN, CrossEntropyLoss, SGD, and DataLoader implementations.
pub fn eval_two_layer_nn(
    x_train: Tensor<MyAutodiffBackend, 2>,
    y_train: Tensor<MyAutodiffBackend, 1, Int>,
) -> TwoLayerNN<MyAutodiffBackend> {
    let device = x_train.device();
    let mut model: TwoLayerNN<MyAutodiffBackend> = TwoLayerNN::new(784, 300, 10, &device);
    let mut opt = SGD::new(0.2);

    let epochs = 20;
    let batch_size = 100;

    for _ in 0..epochs {
        let loader = DataLoader::new(x_train.clone(), y_train.clone(), batch_size);
        for (x_batch, y_batch) in loader {
            let logits = model.forward(x_batch);
            let loss = cross_entropy_loss(logits, y_batch);
            let grads = loss.backward();
            opt.step(std::slice::from_mut(&mut model.linear1.weight), &grads);
            opt.step(std::slice::from_mut(&mut model.linear2.weight), &grads);
        }
    }
    model
}
