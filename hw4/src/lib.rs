/*
 * Homework 4 - Transformers
 *
 * In this homework, you will build all the components of a Transformer LLM, which can
 * load weights from a (slight variant of) the Llama 3.2 1B model, and perform inference
 * on it. While you will NOT be training the model from scratch in this homework (that
 * will happen in the next homework), this model will contain all the elements to run the
 * LLM, including the basic layers (linear, embedding, RMS norm, multihead attention, etc)
 * and the KV cache mechanism to make inference more efficient.
 *
 * ### Question 1 - Linear Layer
 * Linear layer with no bias term. Store weights in a Parameter called `weight`.
 * PyTorch stores the transpose of the weight matrices, so weight shape is (out_dim, in_dim).
 * No special initialization needed (weights will be loaded from file).
 *
 * ### Question 2 - Embedding layer
 * Converts integer token IDs to embedding vectors by table lookup.
 * Weight shape: (num_tokens, dim). Input is integer tensor of any shape.
 * Output has one additional trailing dimension of size `dim`.
 *
 * ### Question 3 - SiLU nonlinearity
 * silu(x) = x * sigmoid(x)
 *
 * ### Question 4 - RMS Norm
 * RMSNorm(x) = w * x / sqrt(||x||^2/dim + eps)
 * Applied along the last dimension. Weight initialized to all ones.
 *
 * ### Question 5 - Masked Self Attention
 * Y = softmax(QK^T / sqrt(d) + M) V
 * Works for both 2D (seq_len x d) and higher-dimensional inputs.
 *
 * ### Question 6 - Multi-head Attention (with KV Cache)
 * 1. Project X to Q, K, V via linear layers wq, wk, wv
 * 2. Split into n_heads along the last dimension
 * 3. Apply self_attention to each head
 * 4. Concatenate heads and project via wp
 *
 * With KV cache: store K, V at seq_pos in buffers of shape (1, max_cache_size, dim).
 * When use_kv_cache=true, attend against full cache up to current position.
 *
 * ### Question 7 - Gated MLP
 * GatedMLP(X) = (silu(X @ W1^T) * X @ W3^T) @ W2^T
 * W1, W3: (ffn_dim x dim), W2: (dim x ffn_dim)
 *
 * ### Question 8 - Transformer Block
 * Z = X + MHA(RMSNorm_1(X))
 * Y = Z + GatedMLP(RMSNorm_2(Z))
 *
 * ### Question 9 - Llama3 Model
 * Components: embedding, pos_embeddings, layers (ModuleList of TransformerBlocks),
 * norm (RMSNorm), output (Linear), mask (causal upper-triangular of -inf).
 * Forward: embed + pos -> layers -> norm -> output
 *
 * ### Question 10 - Generation
 * Autoregressively generate tokens using KV cache:
 * 1. Run model on full prompt to get next-token distribution
 * 2. Sample from temperature-scaled softmax
 * 3. Repeat, using KV cache for efficiency
 * 4. Stop at stop_tokens or max_tokens
 */

use burn::backend::Autodiff;
use burn::module::{Module, Param};
use burn::prelude::*;
use burn::tensor::activation::{relu, sigmoid};
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

#[derive(Module, Debug)]
pub struct Linear<B: Backend> {
    pub weight: Param<Tensor<B, 2>>,
}

impl<B> Linear<B>
where
    B: Backend,
{
    pub fn new(in_dim: usize, out_dim: usize, device: &B::Device) -> Self {
        Linear {
            weight: Param::from_tensor(Tensor::<B, 2>::zeros([out_dim, in_dim], device)),
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        // burn matmul expects both matrices to have compatibel dimensions
        // and does not reshape/unsqueeze as needed, so we meed to be
        // explicit.
        //
        // Unsqueeze or reshape? They are the same but the reshape is
        // usually somewhat faster but unsqueeze is more readable...
        x.matmul(self.weight.val().transpose().unsqueeze::<D>())
    }
}

#[derive(Module, Debug)]
pub struct Embedding<B: Backend> {
    pub weight: Param<Tensor<B, 2>>,
}

impl<B> Embedding<B>
where
    B: Backend,
{
    pub fn new(num_tokens: usize, dim: usize, device: &B::Device) -> Self {
        Embedding {
            weight: Param::from_tensor(Tensor::<B, 2>::zeros([num_tokens, dim], device)),
        }
    }

    pub fn forward(&self, indices: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        // So, in PyTorch we would just return self.weight[indices] and
        // be done with it. In burn it is not that simple: we need to
        // 1. flatten indices (so that we can use select)
        // 2. self.weight.gather(0, flattened indices)
        // 3. reshape the result to the desired size
        let [batch_size, sequence_length] = indices.dims();
        let [_num_tokens, dim] = self.weight.val().dims();

        let flattened_indices = indices.reshape([batch_size * sequence_length]);
        self.weight
            .val()
            .clone()
            .select(0, flattened_indices)
            .reshape([batch_size, sequence_length, dim])
    }
}

pub fn silu<B, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D>
where
    B: Backend,
{
    x.clone() * sigmoid(x.clone())
}

fn normalize<B, const D: usize>(x: Tensor<B, D>, eps: f64) -> Tensor<B, D>
where
    B: Backend,
{
    let last_dim_index = D - 1;
    let dims = [last_dim_index];
    x.clone() / (x.clone().powi_scalar(2).mean_dims(&dims) + eps).sqrt()
}

#[derive(Module, Debug)]
pub struct RMSNorm<B: Backend> {
    pub eps: f64,
    pub weight: Param<Tensor<B, 1>>,
}

impl<B> RMSNorm<B>
where
    B: Backend,
{
    pub fn new(dim: usize, eps: f64, device: &B::Device) -> Self {
        RMSNorm {
            eps,
            weight: Param::from_tensor(Tensor::<B, 1>::ones([dim], device)),
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let normalized_x = normalize(x, self.eps);
        self.weight.val().unsqueeze::<D>().mul(normalized_x)
    }
}

pub fn self_attention<B>(
    q: Tensor<B, 2>,
    k: Tensor<B, 2>,
    v: Tensor<B, 2>,
    mask: Option<Tensor<B, 2>>,
) -> Tensor<B, 2>
where
    B: Backend,
{
    todo!()
}

pub fn self_attention_batched<B>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    mask: Option<Tensor<B, 2>>,
) -> Tensor<B, 4>
where
    B: Backend,
{
    todo!()
}

#[derive(Module, Debug)]
pub struct MultiHeadAttentionKVCache<B: Backend> {
    pub wq: Linear<B>,
    // ...
    pub n_heads: usize,
}

impl<B> MultiHeadAttentionKVCache<B>
where
    B: Backend,
{
    pub fn new(dim: usize, n_heads: usize, max_cache: usize, device: &B::Device) -> Self {
        todo!()
    }
    pub fn forward(
        &mut self,
        x: Tensor<B, 3>,
        mask: Option<Tensor<B, 2>>,
        seq_pos: usize,
        use_cache: bool,
    ) -> Tensor<B, 3> {
        todo!()
    }
}

#[derive(Module, Debug)]
pub struct GatedMLP<B: Backend> {
    pub w1: Linear<B>,
    // ...
}

impl<B> GatedMLP<B>
where
    B: Backend,
{
    pub fn new(dim: usize, ffn_dim: usize, device: &B::Device) -> Self {
        todo!()
    }
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        todo!()
    }
}

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    pub attn: MultiHeadAttentionKVCache<B>,
    // ...
}

impl<B> TransformerBlock<B>
where
    B: Backend,
{
    pub fn new(
        dim: usize,
        n_heads: usize,
        ffn_dim: usize,
        max_seq: usize,
        device: &B::Device,
    ) -> Self {
        todo!()
    }
    pub fn forward(
        &mut self,
        x: Tensor<B, 3>,
        mask: Option<Tensor<B, 2>>,
        seq_pos: usize,
        use_cache: bool,
    ) -> Tensor<B, 3> {
        todo!()
    }
}

#[derive(Module, Debug)]
pub struct Llama3Simplified<B: Backend> {
    pub embedding: Embedding<B>,
    // ...
}

impl<B> Llama3Simplified<B>
where
    B: Backend,
{
    pub fn new(
        num_tokens: usize,
        dim: usize,
        n_heads: usize,
        max_seq: usize,
        ffn_dim: usize,
        num_layers: usize,
        device: &B::Device,
    ) -> Self {
        todo!()
    }
    pub fn forward(
        &mut self,
        tokens: Tensor<B, 2, Int>,
        seq_pos: usize,
        use_cache: bool,
    ) -> Tensor<B, 3> {
        todo!()
    }
}

#[allow(clippy::too_many_arguments)]
pub fn generate<B: Backend>(
    model: &mut dyn FnMut(Tensor<B, 2, Int>, usize, bool) -> Tensor<B, 3>,
    prompt_tokens: &[i32],
    decode_fn: &dyn Fn(&[i32]) -> String,
    stop_tokens: &[i32],
    temp: f64,
    max_tokens: usize,
    verbose: bool,
) -> Vec<i32> {
    todo!()
}

/// Load the Llama 3.2 simplified model with pretrained weights.
pub fn eval_llama3<B: Backend>() -> Llama3Simplified<B> {
    todo!()
}
