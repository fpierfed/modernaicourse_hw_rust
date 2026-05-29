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
use std::io::Write;
use std::path::{Path, PathBuf};

use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::ops::{sigmoid, softmax};
use candle_nn::VarBuilder;

use hf_hub::api::sync::Api;
use serde::Deserialize;

pub fn default_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        return Device::new_cuda(0);
    }
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return Device::new_metal(0);
    }
    #[allow(unreachable_code)]
    Ok(Device::Cpu)
}

const EPSILON: Float = 1.0e-5;
const LLAMA3_REPO: &str = "zkolter/Llama-3.2-1B-Instruct-Simplified";
type Float = f32;

//
// Candle Note
//
// Candle always returns (and expects you to return) Result instances.
// Tensors are passed as references
//

#[derive(Clone, Debug)]
pub struct Linear {
    // Stored pre-transposed (matmul-ready) as [in_dim, out_dim] so that
    // forward is a single contiguous matmul with no per-call transpose.
    // The PyTorch checkpoint stores weights as [out_dim, in_dim]; we
    // transpose once at load time in `load_pretrained`.
    pub weight: Tensor,
}

impl Linear {
    pub fn new(in_dim: usize, out_dim: usize, device: &Device) -> Result<Self> {
        let std = (2.0 / in_dim as Float).sqrt();
        Ok(Self {
            weight: Tensor::randn(0f32, std, (in_dim, out_dim), device)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.broadcast_matmul(&self.weight)
    }
}

#[derive(Clone, Debug)]
pub struct Embedding {
    pub weight: Tensor,
}

impl Embedding {
    pub fn new(num_tokens: usize, dim: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            weight: Tensor::zeros((num_tokens, dim), DType::F32, device)?,
        })
    }

    pub fn forward(&self, indices: &Tensor) -> Result<Tensor> {
        // Candle's Tensor::embedding requires a 1-D index tensor, so we
        // flatten, look up, and reshape back to the original shape with an
        // extra trailing `dim` axis.
        let mut out_dims = indices.dims().to_vec();
        out_dims.push(self.weight.dim(1)?);
        self.weight
            .embedding(&indices.flatten_all()?)?
            .reshape(out_dims)
    }
}

pub fn silu(x: &Tensor) -> Result<Tensor> {
    x * sigmoid(x)?
}

#[derive(Clone, Debug)]
pub struct RMSNorm {
    pub eps: Float,
    pub weight: Tensor,
}

impl RMSNorm {
    pub fn new(dim: usize, eps: Float, device: &Device) -> Result<Self> {
        Ok(Self {
            eps,
            weight: Tensor::ones(dim, DType::F32, device)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // this is w * x / sqrt(x*x + eps) == w * normalized(x) :-)
        //
        // In PyTorch
        //  self.weight * x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        //
        let rms = (x.sqr()?.mean_keepdim(D::Minus1)? + self.eps as f64)?.sqrt()?;
        self.weight.broadcast_mul(x)?.broadcast_div(&rms)
    }
}

pub fn self_attention(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    let d = *q.dims().last().unwrap();
    let sqrt_d = (d as f64).sqrt();

    let k_t = k.transpose(D::Minus2, D::Minus1)?;
    let scaled = (q.matmul(&k_t)? / sqrt_d)?;

    let scores = match mask {
        Some(m) => scaled.broadcast_add(m)?,
        None => scaled,
    };
    softmax(&scores, D::Minus1)?.matmul(v)
}

#[derive(Clone, Debug)]
pub struct MultiHeadAttention {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wp: Linear,
    pub n_heads: usize,
}

impl MultiHeadAttention {
    pub fn new(dim: usize, n_heads: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            wq: Linear::new(dim, dim, device)?,
            wk: Linear::new(dim, dim, device)?,
            wv: Linear::new(dim, dim, device)?,
            wp: Linear::new(dim, dim, device)?,
            n_heads,
        })
    }

    // Note that `seq_pos` and `use_cache` are not used here, just like
    // in the Python version.
    #[allow(unused_variables)]
    pub fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        seq_pos: usize,
        use_cache: bool,
    ) -> Result<Tensor> {
        let q = self.wq.forward(x)?;
        let k = self.wk.forward(x)?;
        let v = self.wv.forward(x)?;

        let (batch_size, seq_len, dim) = x.dims3()?;
        let head_dim = dim / self.n_heads;

        // Need to use transpose on seq_len, self.n_heads to be able to
        // iterate over the head blocks.
        // Process all batches

        // Since we pass this to matmul we need to *copy* the data to a
        // contiguous block of memory.

        let q = q
            .reshape((batch_size, seq_len, self.n_heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch_size, seq_len, self.n_heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch_size, seq_len, self.n_heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let y = self_attention(&q, &k, &v, mask)?;

        // Go back to old dimensions, undo all operations in reverse.
        self.wp.forward(
            &y.transpose(1, 2)?
                .contiguous()?
                .reshape((batch_size, seq_len, dim))?,
        )
    }
}

#[derive(Clone, Debug)]
pub struct MultiHeadAttentionKVCache {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wp: Linear,

    pub n_heads: usize,
    pub max_cache_size: usize,

    pub k_cache: Tensor,
    pub v_cache: Tensor,
}

impl MultiHeadAttentionKVCache {
    pub fn new(dim: usize, n_heads: usize, max_cache: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            wq: Linear::new(dim, dim, device)?,
            wk: Linear::new(dim, dim, device)?,
            wv: Linear::new(dim, dim, device)?,
            wp: Linear::new(dim, dim, device)?,
            n_heads,
            max_cache_size: max_cache,
            k_cache: Tensor::zeros((1, max_cache, dim), DType::F32, device)?,
            v_cache: Tensor::zeros((1, max_cache, dim), DType::F32, device)?,
        })
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        seq_pos: usize,
        use_cache: bool,
    ) -> Result<Tensor> {
        let q = self.wq.forward(x)?;
        let k = self.wk.forward(x)?;
        let v = self.wv.forward(x)?;

        let (batch_size, seq_len, dim) = x.dims3()?;
        let head_dim = dim / self.n_heads;

        let (working_k, working_v) = if use_cache {
            // In-place update, no copying or allocations.
            self.k_cache.slice_set(&k, 1, seq_pos)?;
            self.v_cache.slice_set(&v, 1, seq_pos)?;

            let end = seq_pos + seq_len;
            (
                self.k_cache.narrow(1, 0, end)?,
                self.v_cache.narrow(1, 0, end)?,
            )
        } else {
            (k.clone(), v.clone())
        };

        // Need to use transpose on seq_len, self.n_heads to be able to
        // iterate over the head blocks.
        // Process all batches
        let kv_seq_len = working_k.dim(1)?;

        // Since we pass this to matmul we need to *copy* the data to a
        // contiguous block of memory.

        let q = q
            .reshape((batch_size, seq_len, self.n_heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let working_k = working_k
            .reshape((batch_size, kv_seq_len, self.n_heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let working_v = working_v
            .reshape((batch_size, kv_seq_len, self.n_heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let y = self_attention(&q, &working_k, &working_v, mask)?;

        // Go back to old dimensions, undo all operations in reverse.
        self.wp.forward(
            &y.transpose(1, 2)?
                .contiguous()?
                .reshape((batch_size, seq_len, dim))?,
        )
    }
}

#[derive(Clone, Debug)]
pub struct GatedMLP {
    pub w1: Linear,
    pub w2: Linear,
    pub w3: Linear,
}

impl GatedMLP {
    pub fn new(dim: usize, ffn_dim: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            w1: Linear::new(dim, ffn_dim, device)?,
            w2: Linear::new(ffn_dim, dim, device)?,
            w3: Linear::new(dim, ffn_dim, device)?,
        })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = silu(&self.w1.forward(x)?)?;
        let up = self.w3.forward(x)?;
        let h = (gate * up)?;
        self.w2.forward(&h)
    }
}

#[derive(Clone, Debug)]
pub struct TransformerBlock {
    pub attn: MultiHeadAttentionKVCache,
    pub norm1: RMSNorm,
    pub norm2: RMSNorm,
    pub mlp: GatedMLP,
}

impl TransformerBlock {
    pub fn new(
        dim: usize,
        n_heads: usize,
        ffn_dim: usize,
        max_seq: usize,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            attn: MultiHeadAttentionKVCache::new(dim, n_heads, max_seq, device)?,
            norm1: RMSNorm::new(dim, EPSILON, device)?,
            norm2: RMSNorm::new(dim, EPSILON, device)?,
            mlp: GatedMLP::new(dim, ffn_dim, device)?,
        })
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        seq_pos: usize,
        use_cache: bool,
    ) -> Result<Tensor> {
        let normalized_x = self.norm1.forward(x)?;
        let attn_res = self.attn.forward(&normalized_x, mask, seq_pos, use_cache)?;
        let z = (x + attn_res)?;
        let normalized_z = self.norm2.forward(&z)?;
        let mlp_res = self.mlp.forward(&normalized_z)?;
        z + mlp_res
    }
}

fn build_causal_mask(n: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            data[i * n + j] = Float::NEG_INFINITY;
        }
    }
    Tensor::from_vec(data, (n, n), device)
}

#[derive(Deserialize)]
pub struct LlamaParams {
    vocab_size: usize,
    dim: usize,
    n_heads: usize,
    max_seq_len: usize,
    ffn_dim_multiplier: f64,
    n_layers: usize,
}

#[derive(Clone, Debug)]
pub struct Llama3Simplified {
    pub embedding: Embedding,
    pub pos_embeddings: Tensor,
    pub layers: Vec<TransformerBlock>,
    pub norm: RMSNorm,
    pub output: Linear,
    pub mask: Tensor,
}

impl Llama3Simplified {
    pub fn new(
        num_tokens: usize,
        dim: usize,
        n_heads: usize,
        max_seq: usize,
        ffn_dim: usize,
        num_layers: usize,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            embedding: Embedding::new(num_tokens, dim, device)?,
            pos_embeddings: Tensor::zeros((max_seq, dim), DType::F32, device)?,
            layers: vec![
                TransformerBlock::new(dim, n_heads, ffn_dim, max_seq, device)?;
                num_layers
            ],
            norm: RMSNorm::new(dim, EPSILON, device)?,
            output: Linear::new(dim, num_tokens, device)?,
            mask: build_causal_mask(max_seq, device)?,
        })
    }

    pub fn forward(&mut self, tokens: &Tensor, seq_pos: usize, use_cache: bool) -> Result<Tensor> {
        let (_batch, seq_len) = tokens.dims2()?;

        let mut res = self.embedding.forward(tokens)?;
        let pos = self
            .pos_embeddings
            .narrow(0, seq_pos, seq_len)?
            .unsqueeze(0)?;

        res = res.broadcast_add(&pos)?;

        let mend = seq_pos + seq_len;
        let mask_slice = self.mask.narrow(0, seq_pos, seq_len)?.narrow(1, 0, mend)?;

        for layer in self.layers.iter_mut() {
            res = layer.forward(&res, Some(&mask_slice), seq_pos, use_cache)?;
        }
        let normalized_res = self.norm.forward(&res)?;
        self.output.forward(&normalized_res)
    }

    pub fn load_pretrained(
        params: &LlamaParams,
        checkpoint_path: &Path,
        device: &Device,
    ) -> Result<Self> {
        let vb = VarBuilder::from_pth(checkpoint_path, DType::F32, device)?;

        let dim = params.dim;
        let n_heads = params.n_heads;
        let n_layers = params.n_layers;
        let max_seq = params.max_seq_len;
        let vocab = params.vocab_size;
        let ffn_dim = (params.dim as f64 * params.ffn_dim_multiplier).round() as usize;

        // Causal mask is computed locally — not stored in the checkpoint.
        let mask = build_causal_mask(max_seq, device)?;

        let embedding = Embedding {
            weight: vb.get((vocab, dim), "tok_embeddings.weight")?,
        };
        let pos_embeddings = vb.get((max_seq, dim), "pos_embeddings.weight")?;
        let norm = RMSNorm {
            eps: 1e-5,
            weight: vb.get(dim, "norm.weight")?,
        };
        let output = Linear {
            weight: vb.get((vocab, dim), "output.weight")?.t()?.contiguous()?,
        };

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let lvb = vb.pp(format!("layers.{i}"));
            let avb = lvb.pp("attention");
            let fvb = lvb.pp("feed_forward");

            let attn = MultiHeadAttentionKVCache {
                wq: Linear {
                    weight: avb.get((dim, dim), "wq.weight")?.t()?.contiguous()?,
                },
                wk: Linear {
                    weight: avb.get((dim, dim), "wk.weight")?.t()?.contiguous()?,
                },
                wv: Linear {
                    weight: avb.get((dim, dim), "wv.weight")?.t()?.contiguous()?,
                },
                wp: Linear {
                    weight: avb.get((dim, dim), "wo.weight")?.t()?.contiguous()?,
                },
                n_heads,
                max_cache_size: max_seq,
                k_cache: Tensor::zeros((1, max_seq, dim), DType::F32, device)?,
                v_cache: Tensor::zeros((1, max_seq, dim), DType::F32, device)?,
            };

            let mlp = GatedMLP {
                w1: Linear {
                    weight: fvb.get((ffn_dim, dim), "w1.weight")?.t()?.contiguous()?,
                },
                w2: Linear {
                    weight: fvb.get((dim, ffn_dim), "w2.weight")?.t()?.contiguous()?,
                },
                w3: Linear {
                    weight: fvb.get((ffn_dim, dim), "w3.weight")?.t()?.contiguous()?,
                },
            };

            layers.push(TransformerBlock {
                attn,
                norm1: RMSNorm {
                    eps: 1e-5,
                    weight: lvb.get(dim, "attention_norm.weight")?,
                },
                norm2: RMSNorm {
                    eps: 1e-5,
                    weight: lvb.get(dim, "ffn_norm.weight")?,
                },
                mlp,
            });
        }

        Ok(Self {
            embedding,
            pos_embeddings,
            layers,
            norm,
            output,
            mask,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub fn generate(
    model: &mut dyn FnMut(&Tensor, usize, bool) -> Result<Tensor>,
    prompt_tokens: &[u32],
    decode_fn: &dyn Fn(&[u32]) -> String,
    stop_tokens: &[u32],
    temperature: f64,
    max_tokens: usize,
    verbose: bool,
    device: &Device,
) -> Result<Vec<u32>> {
    let mut rng = rand::rng();
    let mut out_tokens: Vec<u32> = Vec::new();

    // Prompt pass — populates the KV cache for positions [0, prompt_len).
    let prompt = Tensor::from_vec(prompt_tokens.to_vec(), (1, prompt_tokens.len()), device)?;
    let mut logits = model(&prompt, 0, true)?;

    for _ in 0..max_tokens {
        // logits shape: [1, seq_len, vocab]. Take the last position.
        let last_seq = logits.dim(1)?;
        let last = logits.narrow(1, last_seq - 1, 1)?.squeeze(1)?; // [1, vocab]

        let next_token: u32 = if temperature == 0.0 {
            // Greedy
            last.argmax(D::Minus1)?.to_vec1::<u32>()?[0]
        } else {
            // Temperature sampling
            let scaled = (last / temperature)?;
            let probs = softmax(&scaled, D::Minus1)?
                .to_vec2::<f32>()?
                .pop()
                .unwrap();
            let dist = WeightedIndex::new(&probs).unwrap();
            dist.sample(&mut rng) as u32
        };

        out_tokens.push(next_token);
        if verbose {
            print!("{}", decode_fn(&[next_token]));
            std::io::stdout().flush().ok();
        }
        if stop_tokens.contains(&next_token) {
            break;
        }

        let seq_pos = prompt_tokens.len() + out_tokens.len() - 1;
        let next_in = Tensor::from_vec(vec![next_token], (1, 1), device)?;
        logits = model(&next_in, seq_pos, true)?;
    }

    Ok(out_tokens)
}

/// Load the Llama 3.2 simplified model with pretrained weights.
fn err_to_candle<E: std::fmt::Display>(e: E) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

pub fn download_llama3_paths() -> Result<(PathBuf, PathBuf)> {
    let repo = Api::new()
        .map_err(err_to_candle)?
        .model(LLAMA3_REPO.to_string());

    let checkpoint = repo.get("consolidated.00.pth").map_err(err_to_candle)?;
    let params = repo.get("params.json").map_err(err_to_candle)?;

    Ok((checkpoint, params))
}

pub fn eval_llama3_from_paths(
    checkpoint_path: impl AsRef<Path>,
    params_path: impl AsRef<Path>,
    device: &Device,
) -> Result<Llama3Simplified> {
    let params_file = std::fs::File::open(params_path)?;
    let params: LlamaParams = serde_json::from_reader(params_file).map_err(err_to_candle)?;

    Llama3Simplified::load_pretrained(&params, checkpoint_path.as_ref(), device)
}

pub fn eval_llama3(device: &Device) -> Result<Llama3Simplified> {
    let (checkpoint, params) = download_llama3_paths()?;
    eval_llama3_from_paths(checkpoint, params, device)
}
