// Small CLI: prompt the simplified Llama 3.2 1B model and print its reply.
//
// Usage:
//   cargo run -p hw4 --bin chat --release -- "What is the capital of France?"
//
// The tokenizer is downloaded via hf-hub from `unsloth/Llama-3.2-1B-Instruct`
// (a public mirror of meta-llama's gated repo) since the simplified weights
// repo only ships the tiktoken `tokenizer.model` file, while the Rust
// `tokenizers` crate needs the HF `tokenizer.json` format.

use burn::tensor::{Int, Tensor};
use hf_hub::api::sync::Api;
use hw4::{eval_llama3, generate, MyBackend};
use tokenizers::Tokenizer;

type B = MyBackend;

// Llama 3 special tokens (same IDs as the simplified model).
const BOS: i32 = 128000; // <|begin_of_text|>
const EOS: i32 = 128001; // <|end_of_text|>
const START_HEADER: i32 = 128006; // <|start_header_id|>
const END_HEADER: i32 = 128007; // <|end_header_id|>
const EOT: i32 = 128009; // <|eot_id|>

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
fn encode_chat_prompt(tokenizer: &Tokenizer, user_msg: &str) -> Vec<i32> {
    // Encode role names and content as plain text (special tokens are added
    // explicitly so we control them regardless of tokenizer config).
    let mut tokens: Vec<i32> = vec![BOS, START_HEADER];
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

fn encode_text(tokenizer: &Tokenizer, s: &str) -> Vec<i32> {
    tokenizer
        .encode(s, false)
        .expect("tokenizer encode failed")
        .get_ids()
        .iter()
        .map(|&t| t as i32)
        .collect()
}

fn main() {
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "What is the capital of France?".to_string());

    eprintln!("Loading tokenizer...");
    let tokenizer = load_tokenizer();

    eprintln!("Loading model (this may download ~2 GB on first run)...");
    let mut model = eval_llama3::<B>();

    let prompt_tokens = encode_chat_prompt(&tokenizer, &prompt);
    eprintln!(
        "Prompt encoded into {} tokens. Generating...\n",
        prompt_tokens.len()
    );
    println!("> {}\n", prompt);

    // Decoder used by `generate` to stream tokens.
    let tk = tokenizer.clone();
    let decode_fn = move |toks: &[i32]| -> String {
        let ids: Vec<u32> = toks.iter().map(|&t| t as u32).collect();
        tk.decode(&ids, false).unwrap_or_default()
    };

    // Wrap model.forward in a FnMut so we can pass it to `generate`.
    let mut model_fn =
        |tokens: Tensor<B, 2, Int>, seq_pos: usize, use_cache: bool| -> Tensor<B, 3> {
            model.forward(tokens, seq_pos, use_cache)
        };

    let stop_tokens: Vec<i32> = vec![EOS, EOT];

    let generated = generate::<B>(
        &mut model_fn,
        &prompt_tokens,
        &decode_fn,
        &stop_tokens,
        0.7,
        256,
        true, // verbose: stream decoded tokens to stdout
    );

    println!("\n\n[generated {} tokens]", generated.len());
}
