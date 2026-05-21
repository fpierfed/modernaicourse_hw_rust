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

use burn::backend::ndarray::NdArray;
use burn::backend::Autodiff;
use burn::module::{Module, Param};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Distribution;

pub type MyBackend = NdArray<f32>;
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
    pub fn new(in_features: usize, out_features: usize, device: &B::Device) -> Self {
        let std = (2.0 / in_features as f64).sqrt();
        Linear {
            weight: Param::from_tensor(Tensor::<B, 2>::random(
                [out_features, in_features],
                Distribution::Normal(0.0, std),
                device,
            )),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        x.matmul(self.weight.val().transpose())
    }
}

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
    // TODO: learning rate, parameter references
}

impl SGD {
    pub fn new(_params: Vec<Tensor<MyAutodiffBackend, 2>>, _lr: f64) -> Self {
        todo!()
    }

    pub fn step(&mut self, _grads: &<MyAutodiffBackend as AutodiffBackend>::Gradients) {
        todo!()
    }
}

/// DataLoader: iterates over (X, y) in sequential minibatches.
///
/// Initialize with full dataset X (N x n), labels y (N), and batch_size.
/// Iterating twice should produce the same batches.
/// Last batch may be smaller than batch_size.
pub struct DataLoader {
    // TODO
}

impl DataLoader {
    pub fn new(
        _x: Tensor<MyAutodiffBackend, 2>,
        _y: Tensor<MyAutodiffBackend, 1, Int>,
        _batch_size: usize,
    ) -> Self {
        todo!()
    }
}

impl Iterator for DataLoader {
    type Item = (
        Tensor<MyAutodiffBackend, 2>,
        Tensor<MyAutodiffBackend, 1, Int>,
    );

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

/// Two-layer neural network: Linear -> ReLU -> Linear.
///
/// Implements h(x) = W2 * relu(W1 * x).
/// Store layers as .linear1 and .linear2.
pub struct TwoLayerNN {
    // TODO
}

impl TwoLayerNN {
    pub fn new(
        _in_features: usize,
        _hidden_features: usize,
        _out_features: usize,
        _device: &Device<MyAutodiffBackend>,
    ) -> Self {
        todo!()
    }

    pub fn forward(&self, _x: Tensor<MyAutodiffBackend, 2>) -> Tensor<MyAutodiffBackend, 2> {
        todo!()
    }
}

/// Multi-layer neural network: [Linear -> ReLU] x N -> Linear.
///
/// Implements h(x) = W_L * relu(W_{L-1} * relu(... W_2 * relu(W_1 * x) ...))
/// Store all layers in a single .linears Vec.
pub struct MultiLayerNN {
    // TODO
}

impl MultiLayerNN {
    pub fn new(
        _in_features: usize,
        _out_features: usize,
        _hidden_dims: &[usize],
        _device: &Device<MyAutodiffBackend>,
    ) -> Self {
        todo!()
    }

    pub fn forward(&self, _x: Tensor<MyAutodiffBackend, 2>) -> Tensor<MyAutodiffBackend, 2> {
        todo!()
    }
}

/// Run one epoch of training or evaluation.
///
/// If optimizer is Some, runs training (forward + backward + step).
/// Returns (average_loss, error_rate) as floats.
pub fn epoch<M>(
    _model: &M,
    _loader: &[(
        Tensor<MyAutodiffBackend, 2>,
        Tensor<MyAutodiffBackend, 1, Int>,
    )],
    _optimizer: Option<&mut SGD>,
) -> (f64, f64)
where
    M: Fn(Tensor<MyAutodiffBackend, 2>) -> Tensor<MyAutodiffBackend, 2>,
{
    todo!()
}

/// Train a linear model on MNIST and return it.
///
/// Given the full MNIST training data (X_train: N x 784, y_train: N),
/// train a Linear layer to classify all 10 digits.
/// The returned model should achieve < 10% error on the test set.
///
/// Use your Linear, CrossEntropyLoss, SGD, DataLoader, and epoch implementations.
pub fn eval_linear_model<B>(
    _x_train: Tensor<MyAutodiffBackend, 2>,
    _y_train: Tensor<MyAutodiffBackend, 1, Int>,
) -> Linear<B>
where
    B: Backend,
{
    todo!()
}

/// Train a two-layer neural network on MNIST and return it.
///
/// Given the full MNIST training data (X_train: N x 784, y_train: N),
/// train a TwoLayerNN to classify all 10 digits.
/// The returned model should achieve < 3% error on the first 2000 test samples.
///
/// Use your TwoLayerNN, CrossEntropyLoss, SGD, DataLoader, and epoch implementations.
pub fn eval_two_layer_nn(
    _x_train: Tensor<MyAutodiffBackend, 2>,
    _y_train: Tensor<MyAutodiffBackend, 1, Int>,
) -> TwoLayerNN {
    todo!()
}
