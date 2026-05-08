use hw6::*;
use std::collections::HashMap;
use std::io::Write;

#[test]
fn test_text_to_corpus() {
    let (corpus, counts) = text_to_corpus("a b b");
    assert_eq!(
        corpus,
        vec![
            vec!["a".to_string()],
            vec![" ".to_string(), "b".to_string()]
        ]
    );
    assert_eq!(counts, vec![1, 2]);
}

#[test]
fn test_text_to_corpus_newline() {
    let (corpus, counts) = text_to_corpus("hi there\nthere");
    assert_eq!(
        corpus,
        vec![
            vec!["h".to_string(), "i".to_string()],
            vec![
                " ".to_string(),
                "t".to_string(),
                "h".to_string(),
                "e".to_string(),
                "r".to_string(),
                "e".to_string()
            ],
            vec![
                "\n".to_string(),
                "t".to_string(),
                "h".to_string(),
                "e".to_string(),
                "r".to_string(),
                "e".to_string()
            ],
        ]
    );
    assert_eq!(counts, vec![1, 1, 1]);
}

#[test]
fn test_most_common_pair() {
    let corpus = vec![
        vec!["a".to_string(), "b".to_string(), "a".to_string()],
        vec!["a".to_string(), "b".to_string()],
        vec!["b".to_string(), "c".to_string()],
    ];
    let counts = vec![2, 1, 3];
    assert_eq!(
        most_common_pair(&corpus, &counts),
        ("a".to_string(), "b".to_string())
    );
}

#[test]
fn test_most_common_pair_space_prefix() {
    let corpus = vec![
        vec![" ".to_string(), "x".to_string()],
        vec![" ".to_string(), "x".to_string(), "y".to_string()],
        vec!["x".to_string(), "y".to_string()],
    ];
    let counts = vec![4, 1, 1];
    assert_eq!(
        most_common_pair(&corpus, &counts),
        (" ".to_string(), "x".to_string())
    );
}

#[test]
fn test_merge_pair() {
    let mut corpus = vec![
        vec!["a".to_string(), "b".to_string(), "c".to_string()],
        vec!["c".to_string(), "a".to_string(), "b".to_string()],
    ];
    merge_pair(&mut corpus, &("a".to_string(), "b".to_string()));
    assert_eq!(
        corpus,
        vec![
            vec!["ab".to_string(), "c".to_string()],
            vec!["c".to_string(), "ab".to_string()],
        ]
    );
}

#[test]
fn test_merge_pair_yz() {
    let mut corpus = vec![
        vec!["x".to_string(), "y".to_string(), "z".to_string()],
        vec!["x".to_string(), "y".to_string()],
        vec!["y".to_string(), "z".to_string()],
    ];
    merge_pair(&mut corpus, &("y".to_string(), "z".to_string()));
    assert_eq!(
        corpus,
        vec![
            vec!["x".to_string(), "yz".to_string()],
            vec!["x".to_string(), "y".to_string()],
            vec!["yz".to_string()],
        ]
    );
}

#[test]
fn test_train_bpe() {
    let (tokens, merges) = train_bpe("aa aa aa", 258);
    assert_eq!(tokens["a"], 'a' as u32);
    assert_eq!(tokens[" "], ' ' as u32);
    assert_eq!(tokens["aa"], 256);
    assert_eq!(tokens[" aa"], 257);
    assert_eq!(
        merges,
        vec![
            ("a".to_string(), "a".to_string()),
            (" ".to_string(), "aa".to_string())
        ]
    );
}

#[test]
fn test_bpe_encode() {
    let mut tokens = HashMap::new();
    tokens.insert("a".to_string(), 0);
    tokens.insert(" ".to_string(), 1);
    tokens.insert("aa".to_string(), 2);
    tokens.insert(" aa".to_string(), 3);
    let merges = vec![
        ("a".to_string(), "a".to_string()),
        (" ".to_string(), "aa".to_string()),
    ];

    assert_eq!(bpe_encode("aa aa", &merges, &tokens), vec![2, 3]);
    assert_eq!(bpe_encode("aa", &merges, &tokens), vec![2]);
}

#[test]
fn test_bpe_decode() {
    let mut tokens = HashMap::new();
    tokens.insert("a".to_string(), 0);
    tokens.insert(" ".to_string(), 1);
    tokens.insert("aa".to_string(), 2);
    tokens.insert(" aa".to_string(), 3);

    assert_eq!(bpe_decode(&[2, 3], &tokens), "aa aa");
    assert_eq!(bpe_decode(&[0, 1, 0], &tokens), "a a");
}

#[test]
fn test_pretokenize_data() {
    let dir = tempfile::tempdir().unwrap();
    let in_path = dir.path().join("sample.txt");
    let out_path = dir.path().join("sample.bin");

    std::fs::write(&in_path, "abcdefg").unwrap();

    let mut calls: Vec<String> = Vec::new();
    let encode_fn = |text: &str| -> Vec<u16> {
        // Simple: each char becomes its ASCII value as u16
        text.chars().map(|c| c as u16).collect()
    };

    pretokenize_data(&encode_fn, &in_path, &out_path, 3, Some(2));

    let bytes = std::fs::read(&out_path).unwrap();
    let tokens: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();

    // Should have processed "abc" and "def" (2 chunks of 3)
    assert_eq!(tokens, vec![97, 98, 99, 100, 101, 102]);
}

#[test]
fn test_dataloader() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokens.bin");

    // Write tokens 0..20 as u16
    let data: Vec<u8> = (0u16..20).flat_map(|t| t.to_le_bytes()).collect();
    std::fs::write(&path, &data).unwrap();

    let loader = DataLoader::new(&path, 3, 2);
    let batches: Vec<_> = loader.collect();

    assert_eq!(batches.len(), 3);

    // First batch: X=[[0,1,2],[4,5,6]], Y=[[1,2,3],[5,6,7]]
    let (x0, y0) = &batches[0];
    let x0_vals: Vec<i64> = x0.to_vec2().unwrap().into_iter().flatten().collect();
    assert_eq!(x0_vals, vec![0, 1, 2, 4, 5, 6]);
    let y0_vals: Vec<i64> = y0.to_vec2().unwrap().into_iter().flatten().collect();
    assert_eq!(y0_vals, vec![1, 2, 3, 5, 6, 7]);
}

#[test]
fn test_cross_entropy_loss() {
    use candle_core::{Device, Tensor};
    let logits = Tensor::new(&[[2.0f32, 1.0, 0.0], [0.0, 2.0, 1.0]], &Device::Cpu).unwrap();
    let y = Tensor::new(&[0u32, 2], &Device::Cpu).unwrap();
    let loss = cross_entropy_loss(&logits, &y).unwrap();
    let val: f32 = loss.to_scalar().unwrap();
    assert!((val - 0.9076060).abs() < 1e-5);
}

#[test]
fn test_cross_entropy_loss_multidim() {
    use candle_core::{Device, Tensor};
    // (batch=2, seq=2, vocab=3) logits
    let logits = Tensor::new(
        &[
            [[1.0f32, 0.0, -1.0], [0.5, -0.5, 0.0]],
            [[-1.0, 2.0, 0.0], [3.0, 1.0, 0.0]],
        ],
        &Device::Cpu,
    )
    .unwrap();
    let y = Tensor::new(&[[0u32, 1], [1, 0]], &Device::Cpu).unwrap();
    let loss = cross_entropy_loss(&logits, &y).unwrap();
    let val: f32 = loss.to_scalar().unwrap();
    assert!(val.is_finite());
}
