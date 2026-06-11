use candle_core::{DType, Device, Result, Tensor};
use hw5::*;
use std::path::Path;

fn main() -> Result<()> {
    let device: Device = default_device()?;
    let token_path = Path::new("TinyStoriesV2-GPT4-train.small.bin");

    pretokenize_tinystories(token_path)?;

    let batch_size: usize = 16;
    let loader = DataLoader::new(token_path, 512, batch_size, &device)?;

    // GPT-2's vocab size is 50257, so:
    // - 50257 // 256 = 196
    // - (196 + 1) * 256 = 50432
    let mut model = LLM::new(50432, 256, 8, 512, 512, 4, &device)?.to_dtype(DType::BF16)?;

    let mut opt = Adam::new(model.parameters(), 1.0e-3, (0.9, 0.95), 1.0e-8)?;
    {
        // Set the KV Cache to false during training / to true during inference.
        let mut step = |x: &Tensor| model.forward(x, 0, false);
        train_llm(&mut step, loader, &mut opt)?;
    }
    Ok(())
}
