// Small CLI: prompt the simplified Llama 3.2 1B model and print its reply.
//
// Usage:
//   cargo run -p hw4 --bin chat --release -- "What is the capital of France?"
//
// The tokenizer is downloaded via hf-hub from `unsloth/Llama-3.2-1B-Instruct`
// (a public mirror of meta-llama's gated repo) since the simplified weights
// repo only ships the tiktoken `tokenizer.model` file, while the Rust
// `tokenizers` crate needs the HF `tokenizer.json` format.

use candle_core::{Result, Tensor};
use hf_hub::api::sync::Api;
use hw4::{default_device, eval_llama3, generate, GenerateConfig};
use tokenizers::Tokenizer;

// Llama 3 special tokens (same IDs as the simplified model).
const BOS: u32 = 128000; // <|begin_of_text|>
const EOS: u32 = 128001; // <|end_of_text|>
const START_HEADER: u32 = 128006; // <|start_header_id|>
const END_HEADER: u32 = 128007; // <|end_header_id|>
const EOT: u32 = 128009; // <|eot_id|>

fn load_tokenizer() -> Tokenizer {
    let api = Api::new().expect("failed to init hf-hub api");
    let repo = api.model("unsloth/Llama-3.2-1B-Instruct".to_string());
    let path = repo
        .get("tokenizer.json")
        .expect("failed to download tokenizer.json");
    Tokenizer::from_file(path).expect("failed to parse tokenizer.json")
}

/// Encode a single user turn using the Llama 3 chat template, returning the
/// token IDs that prompt the assistant to start replying.
fn encode_chat_prompt(tokenizer: &Tokenizer, user_msg: &str) -> Vec<u32> {
    // Encode role names and content as plain text (special tokens are added
    // explicitly so we control them regardless of tokenizer config).
    let mut tokens: Vec<u32> = vec![BOS, START_HEADER];
    tokens.extend(encode_text(tokenizer, "user"));
    tokens.push(END_HEADER);
    tokens.extend(encode_text(tokenizer, "\n\n"));
    tokens.extend(encode_text(tokenizer, user_msg.trim()));
    tokens.push(EOT);
    tokens.push(START_HEADER);
    tokens.extend(encode_text(tokenizer, "assistant"));
    tokens.push(END_HEADER);
    tokens.extend(encode_text(tokenizer, "\n\n"));
    tokens
}

fn encode_text(tokenizer: &Tokenizer, s: &str) -> Vec<u32> {
    tokenizer
        .encode(s, false)
        .expect("tokenizer encode failed")
        .get_ids()
        .to_vec()
}

fn main() -> Result<()> {
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "What is the capital of France?".to_string());

    let device = default_device()?;

    eprintln!("Loading tokenizer...");
    let tokenizer = load_tokenizer();

    eprintln!("Loading model (this may download ~2 GB on first run)...");
    let mut model = eval_llama3(&device)?;

    let prompt_tokens = encode_chat_prompt(&tokenizer, &prompt);
    eprintln!(
        "Prompt encoded into {} tokens. Generating...\n",
        prompt_tokens.len()
    );
    println!("> {}\n", prompt);

    // Decoder used by `generate` to stream tokens.
    let tk = tokenizer.clone();
    let decode_fn = move |toks: &[u32]| -> String { tk.decode(toks, false).unwrap_or_default() };

    // Wrap model.forward in a FnMut so we can pass it to `generate`.
    let mut model_fn = |tokens: &Tensor, seq_pos: usize, use_cache: bool| -> Result<Tensor> {
        model.forward(tokens, seq_pos, use_cache)
    };

    let config = GenerateConfig {
        decode_fn: &decode_fn,
        stop_tokens: &[EOS, EOT],
        temperature: 0.7,
        max_tokens: 256,
        verbose: true,
    };

    let start = std::time::Instant::now();
    let generated = generate(&mut model_fn, &prompt_tokens, &config, &device)?;
    let elapsed = start.elapsed();

    let n_tokens = generated.len();
    let tps = n_tokens as f64 / elapsed.as_secs_f64();
    println!("\n\n[generated {n_tokens} tokens in {elapsed:.2?} ({tps:.1} tok/s)]");
    Ok(())
}
