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

use core::f32;
use std::path::{Path, PathBuf};

use burn::backend::Autodiff;
use burn::module::{Module, Param};
use burn::prelude::*;
use burn::tensor::activation::{sigmoid, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::Distribution;

use burn::tensor::FloatDType;
use burn_store::pytorch::PytorchReader;
use hf_hub::api::sync::Api;
use serde::Deserialize;

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
pub type AnyError = Box<dyn std::error::Error>;

const EPSILON: f64 = 1.0e-5;
const LLAMA3_REPO: &str = "zkolter/Llama-3.2-1B-Instruct-Simplified";

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

// pub fn self_attention<B>(
//     q: Tensor<B, 2>,
//     k: Tensor<B, 2>,
//     v: Tensor<B, 2>,
//     mask: Option<Tensor<B, 2>>,
// ) -> Tensor<B, 2>
// where
//     B: Backend,
// {
//     let [_, d] = q.dims();
//     let sqrt_d = (d as f32).sqrt();
//
//     if let Some(m) = mask {
//         softmax(q.matmul(k.transpose()).div_scalar(sqrt_d).add(m), 1).matmul(v)
//     } else {
//         softmax(q.matmul(k.transpose()).div_scalar(sqrt_d), 1).matmul(v)
//     }
// }
//
// pub fn self_attention_batched<B>(
//     q: Tensor<B, 4>,
//     k: Tensor<B, 4>,
//     v: Tensor<B, 4>,
//     mask: Option<Tensor<B, 2>>,
// ) -> Tensor<B, 4>
// where
//     B: Backend,
// {
//     let dims = q.dims();
//     let d = dims[dims.len() - 1];
//     let sqrt_d = (d as f32).sqrt();
//
//     let last_index = dims.len() - 1;
//     let second_to_last_index = dims.len() - 2;
//
//     // Here we do not have  the handy-dandy PyTorch k.mT that only
//     // transposes the last two dimensions, so we have to make do...
//     if let Some(m) = mask {
//         softmax(
//             q.matmul(k.swap_dims(second_to_last_index, last_index))
//                 .div_scalar(sqrt_d)
//                 .add(m.unsqueeze::<4>()),
//             last_index,
//         )
//         .matmul(v)
//     } else {
//         softmax(
//             q.matmul(k.swap_dims(second_to_last_index, last_index))
//                 .div_scalar(sqrt_d),
//             last_index,
//         )
//         .matmul(v)
//     }
// }

pub fn self_attention<B, const D: usize>(
    q: Tensor<B, D>,
    k: Tensor<B, D>,
    v: Tensor<B, D>,
    mask: Option<Tensor<B, 2>>,
) -> Tensor<B, D>
where
    B: Backend,
{
    const { assert!(D >= 2, "Expecting at least 2D tensors!") };

    let dims = q.dims();
    let d = dims[dims.len() - 1];
    let sqrt_d = (d as f32).sqrt();

    let last_index = dims.len() - 1;
    let second_to_last_index = dims.len() - 2;

    // Here we do not have  the handy-dandy PyTorch k.mT that only
    // transposes the last two dimensions, so we have to make do...
    if let Some(m) = mask {
        softmax(
            q.matmul(k.swap_dims(second_to_last_index, last_index))
                .div_scalar(sqrt_d)
                .add(m.unsqueeze::<D>()),
            last_index,
        )
        .matmul(v)
    } else {
        softmax(
            q.matmul(k.swap_dims(second_to_last_index, last_index))
                .div_scalar(sqrt_d),
            last_index,
        )
        .matmul(v)
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    pub wq: Linear<B>,
    pub wk: Linear<B>,
    pub wv: Linear<B>,
    pub wp: Linear<B>,
    pub n_heads: usize,
}

impl<B> MultiHeadAttention<B>
where
    B: Backend,
{
    pub fn new(dim: usize, n_heads: usize, device: &B::Device) -> Self {
        MultiHeadAttention {
            wq: Linear::new(dim, dim, device),
            wk: Linear::new(dim, dim, device),
            wv: Linear::new(dim, dim, device),
            wp: Linear::new(dim, dim, device),
            n_heads,
        }
    }

    // Note that `seq_pos` and `use_cache` are not used here, just like
    // in the Python version.
    #[allow(unused_variables)]
    pub fn forward(
        &mut self,
        x: Tensor<B, 3>,
        mask: Option<Tensor<B, 2>>,
        seq_pos: usize,
        use_cache: bool,
    ) -> Tensor<B, 3> {
        let q = self.wq.forward(x.clone());
        let k = self.wk.forward(x.clone());
        let v = self.wv.forward(x.clone());

        let [batch_size, seq_len, dim] = q.dims();
        let head_dim = dim / self.n_heads;
        let new_dims = [batch_size, seq_len, self.n_heads, head_dim];

        // Need to use transpose on seq_len, self.n_heads to be able to
        // iterate over the head blocks.
        // To be pedantic:
        // X = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24], [25, 26, 27, 28, 29, 30]])
        // print(X)
        // tensor([[ 1,  2,  3,  4,  5,  6],
        //         [ 7,  8,  9, 10, 11, 12],
        //         [13, 14, 15, 16, 17, 18],
        //         [19, 20, 21, 22, 23, 24],
        //         [25, 26, 27, 28, 29, 30]])
        // print(X.shape)
        // torch.Size([5, 6])
        // X = X.reshape(5, 3, 2).transpose(0, 1)
        // print(X)
        // tensor([[[ 1,  2],
        //          [ 7,  8],
        //          [13, 14],
        //          [19, 20],
        //          [25, 26]],
        //
        //         [[ 3,  4],
        //          [ 9, 10],
        //          [15, 16],
        //          [21, 22],
        //          [27, 28]],
        //
        //         [[ 5,  6],
        //          [11, 12],
        //          [17, 18],
        //          [23, 24],
        //          [29, 30]]])
        //
        // In our case, we just have an extra leading dimension, batch_size
        let reshaped_q = q.reshape(new_dims).swap_dims(1, 2);
        let reshaped_k = k.reshape(new_dims).swap_dims(1, 2);
        let reshaped_v = v.reshape(new_dims).swap_dims(1, 2);

        // Process all batches
        let y = self_attention(reshaped_q, reshaped_k, reshaped_v, mask);

        // Go back to old dimensions, undo all operations in reverse.
        self.wp
            .forward(y.swap_dims(1, 2).reshape([batch_size, seq_len, dim]))
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadAttentionKVCache<B: Backend> {
    pub wq: Linear<B>,
    pub wk: Linear<B>,
    pub wv: Linear<B>,
    pub wp: Linear<B>,

    pub n_heads: usize,
    pub max_cache_size: usize,

    pub k_cache: Tensor<B, 3>,
    pub v_cache: Tensor<B, 3>,
}

impl<B> MultiHeadAttentionKVCache<B>
where
    B: Backend,
{
    pub fn new(dim: usize, n_heads: usize, max_cache: usize, device: &B::Device) -> Self {
        MultiHeadAttentionKVCache {
            wq: Linear::new(dim, dim, device),
            wk: Linear::new(dim, dim, device),
            wv: Linear::new(dim, dim, device),
            wp: Linear::new(dim, dim, device),
            n_heads,
            max_cache_size: max_cache,
            k_cache: Tensor::zeros([1, max_cache, dim], device),
            v_cache: Tensor::zeros([1, max_cache, dim], device),
        }
    }

    pub fn forward(
        &mut self,
        x: Tensor<B, 3>,
        mask: Option<Tensor<B, 2>>,
        seq_pos: usize,
        use_cache: bool,
    ) -> Tensor<B, 3> {
        let q = self.wq.forward(x.clone());
        let k = self.wk.forward(x.clone());
        let v = self.wv.forward(x.clone());

        let working_k: Tensor<B, 3>;
        let working_v: Tensor<B, 3>;

        let [_batch_size, seq_len, dim] = x.dims();

        if use_cache {
            let (start, end) = (seq_pos, seq_pos + seq_len);
            self.k_cache = self
                .k_cache
                .clone()
                .slice_assign([0..1, start..end, 0..dim], k);
            self.v_cache = self
                .v_cache
                .clone()
                .slice_assign([0..1, start..end, 0..dim], v);
            working_k = self.k_cache.clone().slice([0..1, 0..end, 0..dim]);
            working_v = self.v_cache.clone().slice([0..1, 0..end, 0..dim]);
        } else {
            working_k = k;
            working_v = v;
        }

        let [kv_batch_size, kv_seq_len, kv_dim] = working_k.dims();
        let kv_head_dim = kv_dim / self.n_heads;
        let new_kv_dims = [kv_batch_size, kv_seq_len, self.n_heads, kv_head_dim];

        let [batch_size, seq_len, dim] = q.dims();
        let head_dim = dim / self.n_heads;
        let new_q_dims = [batch_size, seq_len, self.n_heads, head_dim];

        // Need to use transpose on seq_len, self.n_heads to be able to
        // iterate over the head blocks.
        // Process all batches
        let y = self_attention(
            q.reshape(new_q_dims).swap_dims(1, 2),
            working_k.reshape(new_kv_dims).swap_dims(1, 2),
            working_v.reshape(new_kv_dims).swap_dims(1, 2),
            mask,
        );

        // Go back to old dimensions, undo all operations in reverse.
        self.wp
            .forward(y.swap_dims(1, 2).reshape([batch_size, seq_len, dim]))
    }
}

#[derive(Module, Debug)]
pub struct GatedMLP<B: Backend> {
    pub w1: Linear<B>,
    pub w2: Linear<B>,
    pub w3: Linear<B>,
}

impl<B> GatedMLP<B>
where
    B: Backend,
{
    pub fn new(dim: usize, ffn_dim: usize, device: &B::Device) -> Self {
        GatedMLP {
            w1: Linear::new(dim, ffn_dim, device),
            w2: Linear::new(ffn_dim, dim, device),
            w3: Linear::new(dim, ffn_dim, device),
        }
    }
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        self.w2
            .forward(silu(self.w1.forward(x.clone())) * self.w3.forward(x.clone()))
    }
}

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    pub attn: MultiHeadAttentionKVCache<B>,
    pub norm1: RMSNorm<B>,
    pub norm2: RMSNorm<B>,
    pub mlp: GatedMLP<B>,
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
        TransformerBlock {
            attn: MultiHeadAttentionKVCache::new(dim, n_heads, max_seq, device),
            norm1: RMSNorm::new(dim, EPSILON, device),
            norm2: RMSNorm::new(dim, EPSILON, device),
            mlp: GatedMLP::new(dim, ffn_dim, device),
        }
    }

    pub fn forward(
        &mut self,
        x: Tensor<B, 3>,
        mask: Option<Tensor<B, 2>>,
        seq_pos: usize,
        use_cache: bool,
    ) -> Tensor<B, 3> {
        let z = x.clone()
            + self
                .attn
                .forward(self.norm1.forward(x.clone()), mask, seq_pos, use_cache);
        z.clone() + self.mlp.forward(self.norm2.forward(z.clone()))
    }
}

// Checkpoint loading code
#[derive(Deserialize)]
struct LlamaParams {
    vocab_size: usize,
    dim: usize,
    n_heads: usize,
    max_seq_len: usize,
    ffn_dim_multiplier: f64,
    n_layers: usize,
}

fn load_param<B: Backend, const D: usize>(
    reader: &PytorchReader,
    key: &str,
    device: &B::Device,
) -> Result<Param<Tensor<B, D>>, AnyError> {
    let snapshot = reader.get(key).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("missing checkpoint tensor {key}"),
        )
    })?;

    let data = snapshot.to_data()?;

    Ok(Param::from_tensor(
        Tensor::<B, D>::from_data(data, device).cast(FloatDType::F32),
    ))
}

#[derive(Module, Debug)]
pub struct Llama3Simplified<B: Backend> {
    pub embedding: Embedding<B>,
    pub pos_embeddings: Param<Tensor<B, 2>>,
    pub layers: Vec<TransformerBlock<B>>,
    pub norm: RMSNorm<B>,
    pub output: Linear<B>,
    pub mask: Tensor<B, 2>,
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
        Llama3Simplified {
            embedding: Embedding::new(num_tokens, dim, device),
            pos_embeddings: Param::from_tensor(Tensor::<B, 2>::zeros([max_seq, dim], device)),
            layers: vec![TransformerBlock::new(dim, n_heads, ffn_dim, max_seq, device); num_layers],
            norm: RMSNorm::new(dim, EPSILON, device),
            output: Linear::new(dim, num_tokens, device),
            mask: Tensor::<B, 2>::full([max_seq, max_seq], f32::NEG_INFINITY, device).triu(1),
        }
    }

    pub fn forward(
        &mut self,
        tokens: Tensor<B, 2, Int>,
        seq_pos: usize,
        use_cache: bool,
    ) -> Tensor<B, 3> {
        let mut res = self.embedding.forward(tokens);
        let dim = self.pos_embeddings.dims()[1];
        let rdim = res.dims()[1];

        // Unsqueeze(0) is needed to ensure broadcast along the batch dim.
        res = res
            + self
                .pos_embeddings
                .val()
                .slice([seq_pos..seq_pos + rdim, 0..dim])
                .unsqueeze();

        let mstart = seq_pos;
        let mend = seq_pos + rdim;
        let mask_slice = self.mask.clone().slice([mstart..mend, 0..mend]);
        for layer in self.layers.iter_mut() {
            res = layer.forward(res.clone(), Some(mask_slice.clone()), seq_pos, use_cache);
        }
        self.output.forward(self.norm.forward(res))
    }

    pub fn load_llama_weights(
        &mut self,
        reader: &PytorchReader,
        device: &B::Device,
    ) -> Result<(), AnyError> {
        self.embedding.weight = load_param(reader, "tok_embeddings.weight", device)?;
        self.pos_embeddings = load_param(reader, "pos_embeddings.weight", device)?;
        self.norm.weight = load_param(reader, "norm.weight", device)?;
        self.output.weight = load_param(reader, "output.weight", device)?;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.attn.wq.weight =
                load_param(reader, &format!("layers.{i}.attention.wq.weight"), device)?;
            layer.attn.wk.weight =
                load_param(reader, &format!("layers.{i}.attention.wk.weight"), device)?;
            layer.attn.wv.weight =
                load_param(reader, &format!("layers.{i}.attention.wv.weight"), device)?;
            layer.attn.wp.weight =
                load_param(reader, &format!("layers.{i}.attention.wo.weight"), device)?;

            layer.mlp.w1.weight = load_param(
                reader,
                &format!("layers.{i}.feed_forward.w1.weight"),
                device,
            )?;
            layer.mlp.w2.weight = load_param(
                reader,
                &format!("layers.{i}.feed_forward.w2.weight"),
                device,
            )?;
            layer.mlp.w3.weight = load_param(
                reader,
                &format!("layers.{i}.feed_forward.w3.weight"),
                device,
            )?;

            layer.norm1.weight =
                load_param(reader, &format!("layers.{i}.attention_norm.weight"), device)?;
            layer.norm2.weight =
                load_param(reader, &format!("layers.{i}.ffn_norm.weight"), device)?;
        }

        Ok(())
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
    let device = B::Device::default();

    let mut out_tokens: Vec<i32> = vec![];
    let prompt = Tensor::<B, 1, Int>::from_data(prompt_tokens, &device).unsqueeze::<2>();

    let mut logits = model(prompt, 0, true);

    for _ in 0..max_tokens {
        let [_batch_size, seq_len, vocab_size] = logits.dims();

        let last_logits = logits
            .clone()
            .narrow(1, seq_len - 1, 1)
            .reshape([1, vocab_size]);
        let next_token = if temp == 0.0 {
            last_logits.argmax(1)
        } else {
            let probs = softmax(last_logits.div_scalar(temp), 1);
            probs.categorical(1)
        }
        .into_data()
        .to_vec::<i32>()
        .unwrap()[0];

        out_tokens.push(next_token);

        if verbose {
            print!("{}", decode_fn(&[next_token]));
        }

        if stop_tokens.contains(&next_token) {
            break;
        }

        let next_input = Tensor::<B, 1, Int>::from_data([next_token], &device).unsqueeze::<2>();
        let seq_pos = prompt_tokens.len() + out_tokens.len() - 1;
        logits = model(next_input, seq_pos, true);
    }
    out_tokens
}

/// Load the Llama 3.2 simplified model with pretrained weights.
pub fn download_llama3_paths() -> Result<(PathBuf, PathBuf), AnyError> {
    let repo = Api::new()?.model(LLAMA3_REPO.to_string());

    let checkpoint = repo.get("consolidated.00.pth")?;
    let params = repo.get("params.json")?;

    Ok((checkpoint, params))
}

pub fn eval_llama3_from_paths<B: Backend>(
    checkpoint_path: impl AsRef<Path>,
    params_path: impl AsRef<Path>,
    device: &B::Device,
) -> Result<Llama3Simplified<B>, AnyError> {
    let params_file = std::fs::File::open(params_path)?;
    let params: LlamaParams = serde_json::from_reader(params_file)?;

    let ffn_dim = (params.dim as f64 * params.ffn_dim_multiplier).round() as usize;

    let mut model = Llama3Simplified::new(
        params.vocab_size,
        params.dim,
        params.n_heads,
        params.max_seq_len,
        ffn_dim,
        params.n_layers,
        device,
    );

    let reader = PytorchReader::new(checkpoint_path)?;
    model.load_llama_weights(&reader, device)?;

    Ok(model)
}

pub fn eval_llama3<B: Backend>() -> Llama3Simplified<B> {
    let device = B::Device::default();
    let (checkpoint, params) = download_llama3_paths().expect("failed to download Llama3 files");

    eval_llama3_from_paths::<B>(checkpoint, params, &device).expect("failed to load Llama3 weights")
}
