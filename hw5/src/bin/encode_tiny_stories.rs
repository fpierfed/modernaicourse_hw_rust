// CLI: use the BPE encoder/decoder in hw5 to dountrip the Tiny Stories dataset
//
// Usage:
//   cargo run -p hw5 --bin encode_tiny_stories --release

use hf_hub::api::sync::{Api, ApiError};
use hw5::{bpe_decode, bpe_encode, train_bpe};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;

fn download_dataset() -> Result<PathBuf, ApiError> {
    let repo_name = "roneneldan/TinyStories";
    let dataset_name = "TinyStoriesV2-GPT4-train.txt";

    let api = Api::new().expect("failed to init hf-hub api");
    let repo = api.dataset(repo_name.to_string());
    repo.get(dataset_name)
}

fn main() -> Result<(), std::io::Error> {
    eprintln!("(Downloading and) reading the dataset...");
    let path = download_dataset().expect("Unable to download dataset!");

    let f = File::open(path)?;
    let mut reader = BufReader::new(f);

    let test_size: u64 = 1000;
    let train_size: u64 = 100000;

    let mut train_text = String::new();
    reader
        .by_ref()
        .take(train_size)
        .read_to_string(&mut train_text)?;

    let (tokens, merges) = train_bpe(&train_text, 2000);

    let mut test_text = String::new();
    reader
        .by_ref()
        .take(test_size)
        .read_to_string(&mut test_text)?;

    println!("Original text: {}", test_text);
    let tokenized_text = bpe_encode(&test_text, &merges, &tokens);
    println!("\nTokens: {:?}", tokenized_text);

    assert_eq!(test_text, bpe_decode(&tokenized_text, &tokens));

    Ok(())
}
