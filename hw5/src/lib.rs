/*
 * Homework 5 - Training an LLM
 *
 * This homework will walk you through the process of training an LLM from scratch.
 * Your LLM will be built on the Tiny Stories dataset, a synthetic dataset used for
 * LLM research. If you train the full model (which will require a GPU), it will be
 * able to generate simple children's stories.
 *
 * ## Part I - Tokenization (BPE)
 *
 * In this section, you will build a simple tokenizer using Byte-Pair Encoding (BPE).
 *
 * ### Question 1 - Splitting a document into unique words
 * Split text on whitespace into a collection of "words" (keeping whitespace as prefix).
 * Returns (corpus, counts) where corpus[i] is a word as a list of character strings,
 * and counts[i] is how many times that word appeared in the text.
 *
 * ### Question 2 - Most common pair
 * Find the most common adjacent pair of tokens in the corpus, weighted by word counts.
 *
 * ### Question 3 - Merging a pair
 * Merge all occurrences of a given pair of tokens into a single token in the corpus
 * (modifies corpus in-place).
 *
 * ### Question 4 - Training a BPE Tokenizer
 * Iteratively find and merge the most common pair until reaching the target vocab size.
 * Returns (token_to_id dictionary, list of merges).
 *
 * ### Question 5 - Encoding and decoding with a BPE tokenizer
 * Encode: split text into characters, apply merges in order, convert to IDs.
 * Decode: map IDs back to token strings, concatenate.
 *
 * ## Part II - A (Slightly) Simpler Transformer
 *
 * ### Question 6 - The Transformer Architecture
 * Same components as the previous assignment, with key differences:
 * 1. Layers must be initialized with random values (not empty):
 *    - Linear: random normal scaled by sqrt(2/in_dim)
 *    - Embedding: random normal (no scaling)
 * 2. RMS Norm is just a function (no learned scaling weight)
 * 3. Instead of GatedMLP, use a normal two-layer MLP: silu(X @ W1^T) @ W2^T
 *
 * Components: Linear, Embedding, silu, rms_norm, self_attention,
 * MultiHeadAttentionKVCache, MLP, TransformerBlock, LLM
 *
 * ## Part III - Training your LLM
 *
 * ### Question 7 - Cross Entropy Loss
 * Supports any-dimensional logits. Reshape to 2D before computing.
 * logits: (... x k), y: (...) -> scalar loss.
 *
 * ### Question 8 - Pretokenizing data
 * Read text file in chunks, tokenize each chunk, write as binary u16 file.
 *
 * ### Question 9 - Data Loader
 * Read pre-tokenized binary file, yield (input, target) batches where
 * target[i] = input[i+1] (next-token prediction).
 * Uses file seeking (not loading entire file into memory).
 *
 * ### Question 10 - Adam Optimizer
 * u := beta1*u + (1-beta1)*grad
 * v := beta2*v + (1-beta2)*grad^2
 * u_hat := u / (1 - beta1^t)
 * v_hat := v / (1 - beta2^t)
 * w := w - lr * u_hat / (sqrt(v_hat) + eps)
 *
 * ### Question 11 - Training Your LLM
 * For each (x, y) from the data loader:
 * 1. Forward pass: model(x)
 * 2. Compute cross_entropy_loss(predictions, y)
 * 3. Backward pass and optimizer step
 *
 * ### Question 12 - Generation
 * Autoregressively sample tokens using KV cache. Stop at eot_token or max_tokens.
 */

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Result, Tensor, Var, D};
use candle_nn::ops::{sigmoid, softmax};
// use candle_nn::VarBuilder;

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

const EPSILON: f32 = 1.0e-5;

// ============================================================
// Part I: BPE Tokenization
// ============================================================

use std::collections::HashMap;
use std::path::Path;

/// Split text into a corpus of words (split on whitespace, keep space as prefix).
/// Returns (corpus, counts) where corpus[i] is a word as a list of strings,
/// and counts[i] is how many times that word appeared in the text.
fn split_keeping_whitespace(text: &str) -> Vec<&str> {
    let mut word_start = 0;
    let mut res: Vec<&str> = vec![];

    // We use .char_indices() instead of .chars() because we are in Unicode
    // land and we do not want to cut through half a unicode glyph!
    for (i, ch) in text.char_indices() {
        if ch.is_whitespace() && i > word_start {
            res.push(&text[word_start..i]);
            word_start = i;
        }
    }
    if word_start < text.len() {
        res.push(&text[word_start..]);
    }
    res
}

fn string_to_vec_string(s: &str) -> Vec<String> {
    s.chars().map(|c| c.to_string()).collect()
}

pub fn text_to_corpus(text: &str) -> (Vec<Vec<String>>, Vec<usize>) {
    let mut counter: HashMap<&str, usize> = HashMap::new();
    let mut order: Vec<&str> = Vec::new();

    let mut keys: Vec<Vec<String>> = vec![];
    let mut counts: Vec<usize> = vec![];

    for token in split_keeping_whitespace(text) {
        if !counter.contains_key(token) {
            order.push(token);
        }
        *counter.entry(token).or_insert(0) += 1;
    }

    for token in order {
        keys.push(string_to_vec_string(token));
        counts.push(counter[token]);
    }
    (keys, counts)
}

/// Find the most common adjacent pair in the corpus, weighted by counts.
pub fn most_common_pair(corpus: &[Vec<String>], counts: &[usize]) -> (String, String) {
    assert!(!corpus.is_empty() && !counts.is_empty(), "Empty input");
    assert_eq!(corpus.len(), counts.len(), "Mismatched input lengths!");

    let mut counter: HashMap<(&String, &String), usize> = HashMap::new();
    let mut order: Vec<(&String, &String)> = Vec::new();

    for (i, tokens) in corpus.iter().enumerate() {
        let mult_factor = counts[i];
        for pair in tokens.windows(2) {
            let pair = (&pair[0], &pair[1]);
            if !counter.contains_key(&pair) {
                order.push(pair);
            }
            *counter.entry(pair).or_insert(0) += mult_factor;
        }
    }

    let mut first_most_common: (&String, &String) = order[0];
    let mut max_count: usize = counter[&first_most_common];

    // We could skip the first one, actually!
    for pair in order {
        let c = counter[&pair];
        if c > max_count {
            max_count = c;
            first_most_common = pair;
        }
    }
    (
        first_most_common.0.to_string(),
        first_most_common.1.to_string(),
    )
}

/// Merge all occurrences of (a, b) into "ab" in the corpus (in-place).
pub fn merge_pair(corpus: &mut [Vec<String>], pair: &(String, String)) {
    let merged = pair.0.clone() + &pair.1;
    let pair = &(&pair.0, &pair.1);

    for i in 0..corpus.len() {
        // Just like in the Python version, make the algo simple by
        // appending an empty entry to corpus[i]
        corpus[i].push("".into());

        let mut read: usize = 0;
        let mut write: usize = 0;
        while read < corpus[i].len() - 1 {
            let test_pair = (&corpus[i][read], &corpus[i][read + 1]);
            if test_pair == *pair {
                corpus[i][write] = merged.clone();
                read += 2;
            } else {
                // No match: write what we have in read to write and move on
                corpus[i][write] = corpus[i][read].clone();
                read += 1;
            }
            write += 1;
        }
        // Truncate the sequence at and including write
        corpus[i].truncate(write);
    }
}

/// Train BPE tokenizer. Returns (token_to_id, merges).
/// vocab_size is the target vocabulary size (starts from 256 base characters).
pub fn train_bpe(text: &str, vocab_size: usize) -> (HashMap<String, u32>, Vec<(String, String)>) {
    // We assume that we only have to deal with the ASCII set
    let mut tokens: Vec<String> = (0..256)
        .filter_map(|i| char::from_u32(i).map(|c| c.to_string()))
        .collect();
    let mut merges: Vec<(String, String)> = Vec::new();

    let (mut corpus, counts) = text_to_corpus(text);

    while tokens.len() < vocab_size {
        let pair = most_common_pair(&corpus, &counts);
        merge_pair(&mut corpus, &pair);
        let merged = pair.0.clone() + &pair.1;
        merges.push(pair);
        tokens.push(merged);
    }

    let token_index: HashMap<String, u32> = tokens.into_iter().zip(0..).collect();
    (token_index, merges)
}

/// Encode a string using trained BPE merges and token map.
pub fn bpe_encode(
    text: &str,
    merges: &[(String, String)],
    tokens: &HashMap<String, u32>,
) -> Vec<u32> {
    let mut raw_corpus: Vec<Vec<String>> = split_keeping_whitespace(text)
        .iter()
        .map(|s| string_to_vec_string(s))
        .collect();
    for pair in merges {
        merge_pair(&mut raw_corpus, pair);
    }
    let flattened_corpus: Vec<&String> = raw_corpus.iter().flatten().collect();
    flattened_corpus.iter().map(|ch| tokens[*ch]).collect()
}

/// Decode a list of token IDs back to a string.
pub fn bpe_decode(ids: &[u32], tokens: &HashMap<String, u32>) -> String {
    let reverse_mapping: HashMap<&u32, &String> = tokens.iter().map(|(ch, i)| (i, ch)).collect();
    // collect() does the right thing here and joins the pieces together.
    // we do need to move out of &String items and use &str instead for the
    // collect (which does a join()) to work.
    ids.iter().map(|id| reverse_mapping[id].as_str()).collect()
}

// ============================================================
// Part II: Transformer Architecture
// ============================================================

// A lot of what follows is from hw4/src/lib.rs
pub fn build_causal_mask(n: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; n * n];
    for row in data.chunks_exact_mut(n).enumerate() {
        let (i, chunk) = row;
        chunk[i + 1..].fill(f32::NEG_INFINITY);
    }
    Tensor::from_vec(data, (n, n), device)
}

#[derive(Clone, Debug)]
pub struct Linear {
    // Stored pre-transposed as [in_dim, out_dim] so forward is a single
    // contiguous matmul with no per-call transpose. The PyTorch checkpoint
    // stores weights as [out_dim, in_dim]; we transpose once at load time.
    pub weight: Tensor,
}

impl Linear {
    pub fn new(in_dim: usize, out_dim: usize, device: &Device) -> Result<Self> {
        let std = (2.0 / in_dim as f32).sqrt();
        let weight = Var::from_tensor(&Tensor::randn(0f32, std, (out_dim, in_dim), device)?)?;
        Ok(Self {
            weight: weight.as_tensor().clone(),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.broadcast_matmul(&self.weight.transpose(D::Minus2, D::Minus1)?)
    }
}

#[derive(Clone, Debug)]
pub struct Embedding {
    pub weight: Tensor,
}

impl Embedding {
    pub fn new(num_tokens: usize, dim: usize, device: &Device) -> Result<Self> {
        let weight = Var::from_tensor(&Tensor::randn(0f32, 1.0f32, (num_tokens, dim), device)?)?;
        Ok(Self {
            weight: weight.as_tensor().clone(),
        })
    }

    pub fn forward(&self, indices: &Tensor) -> Result<Tensor> {
        // Candle's Tensor::embedding requires a 1-D index tensor; flatten,
        // look up, and reshape back to the original shape with an extra
        // trailing `dim` axis.
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

/// RMS normalization (no learned weight, just a function).
/// rms_norm(x) = x / sqrt(mean(x^2) + eps)
pub fn rms_norm(x: &Tensor, eps: f32) -> Result<Tensor> {
    // Here we need to convert to f64 since candle Tensor does not define
    // scalar ops for f32 types, only f64.
    let rms = (x.sqr()?.mean_keepdim(D::Minus1)? + eps as f64)?.sqrt()?;
    x.broadcast_div(&rms)
}

pub fn self_attention(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    let d = *q.dims().last().unwrap();
    // Here we need to convert to f64 since candle Tensor does not define
    // scalar ops for f32 types, only f64.
    let sqrt_d = (d as f64).sqrt();

    let k_t = k.transpose(D::Minus2, D::Minus1)?;
    let scaled = (q.matmul(&k_t)? / sqrt_d)?;

    let scores = match mask {
        Some(m) => scaled.broadcast_add(m)?,
        None => scaled,
    };
    softmax(&scores, D::Minus1)?.matmul(v)
}

/// Reshape `[batch, seq, dim]` into `[batch, n_heads, seq, head_dim]`,
/// transposing and making contiguous for the subsequent matmul.
fn split_heads(x: &Tensor, batch: usize, seq: usize, n_heads: usize) -> Result<Tensor> {
    let dim = x.dim(D::Minus1)?;
    let head_dim = dim / n_heads;
    x.reshape((batch, seq, n_heads, head_dim))?
        .transpose(1, 2)?
        .contiguous()
}

/// Reverse of `split_heads`: `[batch, n_heads, seq, head_dim]` back to
/// `[batch, seq, dim]`.
fn merge_heads(x: &Tensor, batch: usize, seq: usize, dim: usize) -> Result<Tensor> {
    x.transpose(1, 2)?.contiguous()?.reshape((batch, seq, dim))
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

        let (working_k, working_v) = if use_cache {
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

        let kv_seq_len = working_k.dim(1)?;

        let q = split_heads(&q, batch_size, seq_len, self.n_heads)?;
        let working_k = split_heads(&working_k, batch_size, kv_seq_len, self.n_heads)?;
        let working_v = split_heads(&working_v, batch_size, kv_seq_len, self.n_heads)?;

        let y = self_attention(&q, &working_k, &working_v, mask)?;

        self.wp.forward(&merge_heads(&y, batch_size, seq_len, dim)?)
    }
}

/// Simple two-layer MLP: silu(X @ W1^T) @ W2^T
#[derive(Clone, Debug)]
pub struct MLP {
    pub w1: Linear,
    pub w2: Linear,
}

impl MLP {
    pub fn new(dim: usize, ffn_dim: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            w1: Linear::new(dim, ffn_dim, device)?,
            w2: Linear::new(ffn_dim, dim, device)?,
        })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let up = self.w1.forward(x)?;
        self.w2.forward(&silu(&up)?)
    }
}

#[derive(Clone, Debug)]
pub struct TransformerBlock {
    pub attn: MultiHeadAttentionKVCache,
    pub mlp: MLP,
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
            mlp: MLP::new(dim, ffn_dim, device)?,
        })
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        seq_pos: usize,
        use_cache: bool,
    ) -> Result<Tensor> {
        let normalized_x = rms_norm(x, EPSILON)?;
        let attn_res = self.attn.forward(&normalized_x, mask, seq_pos, use_cache)?;
        let z = (x + attn_res)?;
        let normalized_z = rms_norm(&z, EPSILON)?;
        let mlp_res = self.mlp.forward(&normalized_z)?;
        z + mlp_res
    }
}

#[derive(Clone, Debug)]
pub struct LLM {
    pub embedding: Embedding,
    pub pos_embeddings: Tensor,
    pub layers: Vec<TransformerBlock>,
    pub output: Linear,
    pub mask: Tensor,
}

impl LLM {
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
            pos_embeddings: Tensor::randn(0f32, 1.0f32, (max_seq, dim), device)?,
            layers: (0..num_layers)
                .map(|_| TransformerBlock::new(dim, n_heads, ffn_dim, max_seq, device))
                .collect::<Result<Vec<_>>>()?,
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
        let normalized_res = rms_norm(&res, EPSILON)?;
        self.output.forward(&normalized_res)
    }
}

// ============================================================
// Part III: Training
// ============================================================

/// Cross-entropy loss supporting multi-dimensional logits.
/// logits: (N x k), y: (N) -> scalar.
pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    // In python, we use PyTorch magic reshape(-1) so that we do not need
    // to wrestle with dimensions etc outselves. Candle has the same, by
    // passing () instead of -1.
    //
    // logits = logits.reshape(-1, logits.shape[-1])
    // y = y.reshape(-1)
    // return (-logits[torch.arange(len(y)), y] + torch.logsumexp(logits, dim=-1)).mean()

    let d = logits.dim(D::Minus1)?;

    // Here we are following PyTorch:
    // logits = logits.reshape(-1, logits.shape[-1])
    // where for convenience we have set d=logits.shape[-1]
    // and we then flattren targets/y
    // One difference is that candle requires us to cast targets to u32
    // to use in the gather operation below.
    let logits = logits.reshape(((), d))?;
    let targets = targets.flatten_all()?.to_dtype(DType::U32)?;

    // Here PyTorch would allow us to simply do a
    // logits[torch.arange(len(y)), y]
    // In candle, we need to gather-unsqueeze-squeeze:
    // gather needs a 2D index tensor, since logits now is 2D and
    // hence the unsqueeze on what we use as indices...
    let gathered = logits
        .gather(&targets.unsqueeze(D::Minus1)?, D::Minus1)?
        .squeeze(D::Minus1)?;
    // Loss computation is the same, pretty much...
    let loss = (logits.log_sum_exp(D::Minus1)? - gathered)?;

    // Finally, compute the average across all dimension and return
    // a scalar tensor, just like the PyTorch .mean() does.
    loss.mean_all()
}

/// Pre-tokenize a text file into a binary file of u16 token IDs.
/// Reads chunk_size characters at a time, up to max_chunks chunks.
pub fn pretokenize_data(
    encode_fn: &dyn Fn(&str) -> Vec<u16>,
    input_path: &Path,
    output_path: &Path,
    chunk_size: usize,
    max_chunks: Option<usize>,
) {
    let mut left_to_read = match max_chunks {
        Some(l) => l * chunk_size,
        _ => usize::MAX,
    };

    let infd = File::open(input_path).expect("Unable to open input file for reading");
    let outfd = File::create(output_path).expect("Unable to open output file for writing");
    let mut reader = BufReader::new(infd);
    let mut writer = BufWriter::new(outfd);

    let mut text = String::new();
    while left_to_read > 0 {
        reader
            .by_ref()
            .take(chunk_size as u64)
            .read_to_string(&mut text)
            .expect("Unable to read");
        if text.is_empty() {
            break;
        }
        left_to_read -= chunk_size;

        let encoded: Vec<u16> = encode_fn(&text);
        let encoded_bytes: Vec<u8> = encoded
            .iter()
            .map(|ch| ch.to_le_bytes())
            .flatten()
            .collect::<Vec<u8>>();

        writer
            .by_ref()
            .write_all(&encoded_bytes)
            .expect("Unable to write");

        // Avoid adding to the same text...
        text.clear();
    }
}

/// DataLoader: reads pre-tokenized binary file and yields (input, target) batches.
/// Each sample is seq_len tokens; target is shifted by 1.
pub struct DataLoader {
    pub path: Box<Path>,
    pub seq_len: usize,
    pub batch_size: usize,
    pub device: Device,

    num_batches: usize,
    chunk_size_bytes: usize,
    current_batch: usize,
    reader: BufReader<File>,
}

impl DataLoader {
    pub fn new(path: &Path, seq_len: usize, batch_size: usize, device: &Device) -> Result<Self> {
        let chunk_size_tokens = batch_size * (seq_len + 1);
        let fd = File::open(path)?;

        Ok(Self {
            path: path.into(),
            seq_len,
            batch_size,
            device: device.clone(),
            num_batches: seq_len * batch_size * 2,
            chunk_size_bytes: chunk_size_tokens * 2,
            current_batch: 0,
            reader: BufReader::new(fd),
        })
    }

    fn next_impl(&mut self) -> Result<Option<(Tensor, Tensor)>> {
        let pos = self.current_batch * self.seq_len * self.batch_size * 2;
        self.reader.seek(SeekFrom::Start(pos as u64))?;

        let mut data = vec![0u8; self.chunk_size_bytes];
        match self.reader.read_exact(&mut data) {
            Ok(()) => {}
            Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(candle_core::Error::wrap(e)),
        }

        self.current_batch += 1;
        if self.current_batch > self.num_batches {
            return Ok(None);
        }

        // Interpret u16 little-endian bytes to i32 tokens
        let data_u16: Vec<u16> = data
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();

        let data_i32: Vec<i32> = data_u16.iter().map(|&x| x as i32).collect();

        let tokens = Tensor::from_vec(data_i32, (self.batch_size, self.seq_len + 1), &self.device)?;

        // Shift targets by 1: input is [:, :-1], target is [:, 1:]
        let x = tokens.narrow(1, 0, self.seq_len)?;
        let y = tokens.narrow(1, 1, self.seq_len)?;

        Ok(Some((x, y)))
    }
}

impl Iterator for DataLoader {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_impl() {
            Ok(batch) => batch,
            Err(e) => {
                panic!("DataLoader failed during iteration: {:?}", e);
            }
        }
    }
}

/// Adam optimizer.
pub struct Adam {
    params: Vec<Tensor>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: usize,
    pub u: Vec<Tensor>,
    pub v: Vec<Tensor>,
}

impl Adam {
    fn init_from_params(params: &Vec<Tensor>) -> Result<Vec<Tensor>> {
        params
            .iter()
            .map(|p| p.zeros_like())
            .collect::<Result<Vec<Tensor>>>()
    }

    pub fn new(params: Vec<Tensor>, lr: f32, betas: (f32, f32), eps: f32) -> Result<Self> {
        let u = Self::init_from_params(&params)?;
        let v = Self::init_from_params(&params)?;

        Ok(Self {
            params,
            lr,
            beta1: betas.0,
            beta2: betas.1,
            eps,
            t: 1,
            u,
            v,
        })
    }

    pub fn step(&mut self, grads: &Result<GradStore>) {
        if grads.is_err() {
            panic!("Gradiens are not correctly computed!");
        }
        let grads = grads.as_ref().unwrap();

        for (i, p) in self.params.iter_mut().enumerate() {
            let grad = match grads.get(&p) {
                Some(g) => g,
                None => continue,
            };

            self.u[i] = ((self.u[i].clone() * (self.beta1 as f64)).unwrap()
                + (1.0 - self.beta1 as f64) * grad)
                .unwrap();
            self.v[i] = ((self.v[i].clone() * (self.beta2 as f64)).unwrap()
                + (1.0 - self.beta2 as f64) * grad * grad)
                .unwrap();

            let u_hat =
                (self.u[i].clone() / (1.0 - self.beta1.powi(self.t as i32) as f64)).unwrap();
            let v_hat =
                (self.v[i].clone() / (1.0 - self.beta2.powi(self.t as i32) as f64)).unwrap();

            let updated_p = (&*p
                - ((u_hat * (self.lr as f64)).unwrap()
                    / (v_hat.sqrt().unwrap() + self.eps as f64).unwrap())
                .unwrap())
            .unwrap()
            .contiguous()
            .unwrap();

            p.slice_set(&updated_p, 0, 0).unwrap();
        }
        self.t += 1;
    }

    pub fn zero_grad(&mut self) {
        // Whipe u and v tensor
        for t in self.u.iter_mut() {
            *t = t.zeros_like().unwrap();
        }
        for t in self.v.iter_mut() {
            *t = t.zeros_like().unwrap();
        }
    }
}

/// Train the LLM for one pass over the data loader.
pub fn train_llm(
    _model: &dyn Fn(&Tensor) -> Result<Tensor>,
    _loader: &Vec<(Tensor, Tensor)>,
    _optimizer: &mut Adam,
) {
    todo!()
}

/// Generate tokens autoregressively with temperature sampling and KV cache.
pub fn generate(
    _model: &mut dyn FnMut(&Tensor, usize, bool) -> Result<Tensor>,
    _prompt_tokens: &[u32],
    _decode_fn: &dyn Fn(&[u32]) -> String,
    _stop_token: u32,
    _temp: f32,
    _max_tokens: usize,
    _verbose: bool,
) -> Result<Vec<u32>> {
    todo!()
}

/// Load (or train) an LLM on TinyStories and return the trained model.
///
/// The returned model should achieve < 7.0 cross-entropy loss on the first
/// 48 tokens of TinyStories (tokenized with GPT-2), and support KV-cached
/// inference.
pub fn eval_llm() -> Result<LLM> {
    todo!()
}
