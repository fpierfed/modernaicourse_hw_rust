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

use candle_core::{Result, Tensor};
use candle_nn::VarMap;

pub struct Linear {}
impl Linear {
    pub fn new(_in_f: usize, _out_f: usize, _vm: &VarMap, _name: &str) -> Result<Self> {
        todo!()
    }
    pub fn forward(&self, _x: &Tensor) -> Result<Tensor> {
        todo!()
    }
    pub fn weight(&self) -> &Tensor {
        todo!()
    }
}

pub struct Embedding {}
impl Embedding {
    pub fn new(_num_tokens: usize, _dim: usize, _vm: &VarMap, _name: &str) -> Result<Self> {
        todo!()
    }
    pub fn forward(&self, _indices: &Tensor) -> Result<Tensor> {
        todo!()
    }
    pub fn weight(&self) -> &Tensor {
        todo!()
    }
}

pub fn silu(_x: &Tensor) -> Result<Tensor> {
    todo!()
}

pub struct RMSNorm {}
impl RMSNorm {
    pub fn new(_dim: usize, _eps: f64, _vm: &VarMap, _name: &str) -> Result<Self> {
        todo!()
    }
    pub fn forward(&self, _x: &Tensor) -> Result<Tensor> {
        todo!()
    }
}

pub fn self_attention(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _mask: Option<&Tensor>,
) -> Result<Tensor> {
    todo!()
}

pub struct MultiHeadAttentionKVCache {
    pub n_heads: usize,
}
impl MultiHeadAttentionKVCache {
    pub fn new(
        _dim: usize,
        _n_heads: usize,
        _max_cache: usize,
        _vm: &VarMap,
        _name: &str,
    ) -> Result<Self> {
        todo!()
    }
    pub fn forward(
        &mut self,
        _x: &Tensor,
        _mask: Option<&Tensor>,
        _seq_pos: usize,
        _use_cache: bool,
    ) -> Result<Tensor> {
        todo!()
    }
}

pub struct GatedMLP {}
impl GatedMLP {
    pub fn new(_dim: usize, _ffn_dim: usize, _vm: &VarMap, _name: &str) -> Result<Self> {
        todo!()
    }
    pub fn forward(&self, _x: &Tensor) -> Result<Tensor> {
        todo!()
    }
}

pub struct TransformerBlock {}
impl TransformerBlock {
    pub fn new(
        _dim: usize,
        _n_heads: usize,
        _ffn_dim: usize,
        _max_seq: usize,
        _vm: &VarMap,
        _name: &str,
    ) -> Result<Self> {
        todo!()
    }
    pub fn forward(
        &mut self,
        _x: &Tensor,
        _mask: Option<&Tensor>,
        _seq_pos: usize,
        _use_cache: bool,
    ) -> Result<Tensor> {
        todo!()
    }
}

pub struct Llama3Simplified {}
impl Llama3Simplified {
    pub fn new(
        _num_tokens: usize,
        _dim: usize,
        _n_heads: usize,
        _max_seq: usize,
        _ffn_dim: usize,
        _num_layers: usize,
        _vm: &VarMap,
    ) -> Result<Self> {
        todo!()
    }
    pub fn forward(
        &mut self,
        _tokens: &Tensor,
        _seq_pos: usize,
        _use_cache: bool,
    ) -> Result<Tensor> {
        todo!()
    }
}

pub fn generate(
    _model: &mut dyn FnMut(&Tensor, usize, bool) -> Result<Tensor>,
    _prompt_tokens: &[u32],
    _decode_fn: &dyn Fn(&[u32]) -> String,
    _stop_tokens: &[u32],
    _temp: f64,
    _max_tokens: usize,
    _verbose: bool,
) -> Result<Vec<u32>> {
    todo!()
}
