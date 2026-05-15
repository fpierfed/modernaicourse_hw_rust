use hw1::*;
use ndarray::{Array1, Array2, Array3, Array4, Ix4};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

mod mnist {
    use flate2::read::GzDecoder;
    use ndarray::Array2;
    use std::io::Read;

    const MIRRORS: &[&str] = &[
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "http://yann.lecun.com/exdb/mnist/",
    ];

    pub struct MnistData {
        pub images: Vec<Array2<f32>>,
        pub targets: Vec<u8>,
    }

    fn parse_idx_images(data: &[u8]) -> Vec<Array2<f32>> {
        let magic = u32::from_be_bytes(data[0..4].try_into().unwrap());
        assert_eq!(magic, 2051);
        let n_images = u32::from_be_bytes(data[4..8].try_into().unwrap()) as usize;
        let rows = u32::from_be_bytes(data[8..12].try_into().unwrap()) as usize;
        let cols = u32::from_be_bytes(data[12..16].try_into().unwrap()) as usize;
        let pixels = &data[16..];
        (0..n_images)
            .map(|i| {
                let start = i * rows * cols;
                let raw: Vec<f32> = pixels[start..start + rows * cols]
                    .iter()
                    .map(|&p| p as f32 / 255.0)
                    .collect();
                Array2::from_shape_vec((rows, cols), raw).unwrap()
            })
            .collect()
    }

    fn parse_idx_labels(data: &[u8]) -> Vec<u8> {
        let magic = u32::from_be_bytes(data[0..4].try_into().unwrap());
        assert_eq!(magic, 2049);
        let n_labels = u32::from_be_bytes(data[4..8].try_into().unwrap()) as usize;
        data[8..8 + n_labels].to_vec()
    }

    fn cache_dir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("mnist_cache");
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn download_and_decompress(filename: &str) -> Vec<u8> {
        let cached = cache_dir().join(filename);
        if cached.exists() {
            let compressed = std::fs::read(&cached).unwrap();
            let mut decoder = GzDecoder::new(&compressed[..]);
            let mut decompressed = Vec::new();
            decoder.read_to_end(&mut decompressed).unwrap();
            return decompressed;
        }

        let client = reqwest::blocking::Client::new();
        let mut last_err = None;
        for mirror in MIRRORS {
            let url = format!("{mirror}{filename}");
            match client.get(&url).send().and_then(|r| r.error_for_status()) {
                Ok(resp) => {
                    let compressed = resp.bytes().unwrap();
                    std::fs::write(&cached, &compressed).unwrap();
                    let mut decoder = GzDecoder::new(&compressed[..]);
                    let mut decompressed = Vec::new();
                    decoder.read_to_end(&mut decompressed).unwrap();
                    return decompressed;
                }
                Err(e) => last_err = Some(e),
            }
        }
        panic!("Failed to download {filename} from all mirrors: {}", last_err.unwrap());
    }

    pub fn load_mnist_zero_one() -> MnistData {
        let image_data = download_and_decompress("train-images-idx3-ubyte.gz");
        let label_data = download_and_decompress("train-labels-idx1-ubyte.gz");

        let all_images = parse_idx_images(&image_data);
        let all_labels = parse_idx_labels(&label_data);

        let mut images = Vec::new();
        let mut targets = Vec::new();
        for (img, &label) in all_images.into_iter().zip(all_labels.iter()) {
            if label == 0 || label == 1 {
                images.push(img);
                targets.push(label);
            }
        }

        MnistData { images, targets }
    }
}

#[test]
fn test_classify_zero_one() {
    let data = mnist::load_mnist_zero_one();

    // Test on the first two images
    assert_eq!(classify_zero_one(&data.images[0]), data.targets[0]);
    assert_eq!(classify_zero_one(&data.images[1]), data.targets[1]);

    // Test on the first 100 items, make sure accuracy >90%
    let n = 100;
    let correct: usize = (0..n)
        .filter(|&i| classify_zero_one(&data.images[i]) == data.targets[i])
        .count();
    let accuracy = correct as f64 / n as f64;
    assert!(
        accuracy > 0.9,
        "Accuracy {accuracy:.2} is not > 0.9 on the first {n} filtered MNIST samples"
    );
}

fn reference_matmul(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    a.dot(b)
}

fn seeded_rng() -> StdRng {
    StdRng::seed_from_u64(0x5eed_5eed)
}

fn random_f32(rng: &mut StdRng) -> f32 {
    StandardNormal.sample(rng)
}

fn random_array1(rng: &mut StdRng, len: usize) -> Array1<f32> {
    Array1::from_shape_fn(len, |_| random_f32(rng))
}

fn random_array2(rng: &mut StdRng, shape: (usize, usize)) -> Array2<f32> {
    Array2::from_shape_fn(shape, |_| random_f32(rng))
}

fn random_array4(rng: &mut StdRng, shape: (usize, usize, usize, usize)) -> Array4<f32> {
    Array4::from_shape_fn(shape, |_| random_f32(rng))
}

#[test]
fn test_vector_add() {
    let mut rng = seeded_rng();
    let a = random_array1(&mut rng, 5);
    let b = random_array1(&mut rng, 5);
    let z = vector_add(&a, &b);
    let expected = &a + &b;
    assert!(z.abs_diff_eq(&expected, 1e-6));
}

#[test]
#[should_panic]
fn test_vector_add_dimension_mismatch() {
    let a = Array1::from(vec![1.0f32; 5]);
    let b = Array1::from(vec![1.0f32; 6]);
    vector_add(&a, &b);
}

#[test]
fn test_vector_inner_product() {
    let mut rng = seeded_rng();
    let a = random_array1(&mut rng, 5);
    let b = random_array1(&mut rng, 5);
    let z = vector_inner_product(&a, &b);
    let expected = a.dot(&b);
    assert!((z - expected).abs() < 1e-6);
}

#[test]
#[should_panic]
fn test_vector_inner_product_dimension_mismatch() {
    let a = Array1::from(vec![1.0f32; 5]);
    let b = Array1::from(vec![1.0f32; 6]);
    vector_inner_product(&a, &b);
}

#[test]
fn test_matrix_vector_product_1() {
    let mut rng = seeded_rng();
    let a = random_array2(&mut rng, (5, 4));
    let b = random_array1(&mut rng, 4);
    let z = matrix_vector_product_1(&a, &b);
    let expected = a.dot(&b);
    assert!(z.abs_diff_eq(&expected, 1e-5));
}

#[test]
#[should_panic]
fn test_matrix_vector_product_1_dimension_mismatch() {
    let a = Array2::from_shape_vec((5, 4), (0..20).map(|x| x as f32).collect()).unwrap();
    let b = Array1::from(vec![1.0f32; 6]);
    matrix_vector_product_1(&a, &b);
}

#[test]
fn test_matrix_vector_product_2() {
    let mut rng = seeded_rng();
    let a = random_array2(&mut rng, (5, 4));
    let b = random_array1(&mut rng, 4);
    let z = matrix_vector_product_2(&a, &b);
    let expected = a.dot(&b);
    assert!(z.abs_diff_eq(&expected, 1e-5));
}

#[test]
fn test_vector_matrix_product_2() {
    let mut rng = seeded_rng();
    let a = random_array2(&mut rng, (4, 5));
    let b = random_array1(&mut rng, 4);
    let z = vector_matrix_product_2(&b, &a);
    let expected = b.dot(&a);
    assert!(z.abs_diff_eq(&expected, 1e-5));
}

#[test]
#[should_panic]
fn test_vector_matrix_product_2_dimension_mismatch() {
    let a = Array2::from_shape_vec((4, 5), (0..20).map(|x| x as f32).collect()).unwrap();
    let b = Array1::from(vec![1.0f32; 6]);
    vector_matrix_product_2(&b, &a);
}

#[test]
fn test_matmul_1() {
    let mut rng = seeded_rng();
    let a = random_array2(&mut rng, (4, 5));
    let b = random_array2(&mut rng, (5, 6));
    let z = matmul_1(&a, &b);
    let expected = reference_matmul(&a, &b);
    assert!(z.abs_diff_eq(&expected, 1e-4));
}

#[test]
#[should_panic]
fn test_matmul_1_dimension_mismatch() {
    let a = Array2::from_shape_vec((4, 5), (0..20).map(|x| x as f32).collect()).unwrap();
    let b = Array2::from_shape_vec((4, 5), (0..20).map(|x| x as f32).collect()).unwrap();
    matmul_1(&a, &b);
}

#[test]
fn test_matmul_2() {
    let mut rng = seeded_rng();
    let a = random_array2(&mut rng, (4, 5));
    let b = random_array2(&mut rng, (5, 6));
    let z = matmul_2(&a, &b);
    let expected = reference_matmul(&a, &b);
    assert!(z.abs_diff_eq(&expected, 1e-4));
}

#[test]
fn test_matmul_3() {
    let mut rng = seeded_rng();
    let a = random_array2(&mut rng, (4, 5));
    let b = random_array2(&mut rng, (5, 6));
    let z = matmul_3(&a, &b);
    let expected = reference_matmul(&a, &b);
    assert!(z.abs_diff_eq(&expected, 1e-4));
}

#[test]
fn test_block_matmul() {
    let mut rng = seeded_rng();
    let a = random_array2(&mut rng, (16, 12));
    let b = random_array2(&mut rng, (12, 8));
    let z = block_matmul(&a, &b);
    let expected = reference_matmul(&a, &b);
    assert!(z.abs_diff_eq(&expected, 1e-3));
}

#[test]
#[should_panic]
fn test_block_matmul_inner_dim_mismatch() {
    let a = Array2::from_shape_vec((16, 12), vec![0.0f32; 192]).unwrap();
    let b = Array2::from_shape_vec((16, 12), vec![0.0f32; 192]).unwrap();
    block_matmul(&a, &b);
}

#[test]
#[should_panic]
fn test_block_matmul_not_divisible_by_4() {
    let a = Array2::from_shape_vec((16, 12), vec![0.0f32; 192]).unwrap();
    let b = Array2::from_shape_vec((12, 7), vec![0.0f32; 84]).unwrap();
    block_matmul(&a, &b);
}

#[test]
#[should_panic]
fn test_matrix_vector_product_2_dimension_mismatch() {
    let a = Array2::from_shape_vec((5, 4), (0..20).map(|x| x as f32).collect()).unwrap();
    let b = Array1::from(vec![1.0f32; 6]);
    matrix_vector_product_2(&a, &b);
}

#[test]
#[should_panic]
fn test_matmul_2_dimension_mismatch() {
    let a = Array2::from_shape_vec((4, 5), (0..20).map(|x| x as f32).collect()).unwrap();
    let b = Array2::from_shape_vec((4, 5), (0..20).map(|x| x as f32).collect()).unwrap();
    matmul_2(&a, &b);
}

#[test]
#[should_panic]
fn test_matmul_3_dimension_mismatch() {
    let a = Array2::from_shape_vec((4, 5), (0..20).map(|x| x as f32).collect()).unwrap();
    let b = Array2::from_shape_vec((4, 5), (0..20).map(|x| x as f32).collect()).unwrap();
    matmul_3(&a, &b);
}

#[test]
fn test_batch_matmul() {
    let mut rng = seeded_rng();
    let a = random_array4(&mut rng, (2, 3, 4, 5));
    let b = random_array4(&mut rng, (2, 3, 5, 6));
    let z = batch_matmul(&a.clone().into_dyn(), &b.clone().into_dyn())
        .into_dimensionality::<Ix4>()
        .unwrap();
    // Verify each batch element against reference matmul
    for i in 0..2 {
        for j in 0..3 {
            let a_slice = a.slice(ndarray::s![i, j, .., ..]).to_owned();
            let b_slice = b.slice(ndarray::s![i, j, .., ..]).to_owned();
            let expected = reference_matmul(&a_slice, &b_slice);
            let z_slice = z.slice(ndarray::s![i, j, .., ..]).to_owned();
            assert!(z_slice.abs_diff_eq(&expected, 1e-3));
        }
    }
}

#[test]
#[should_panic]
fn test_batch_matmul_inner_dim_mismatch() {
    let a = Array4::from_shape_vec((2, 3, 4, 5), vec![0.0f32; 120])
        .unwrap()
        .into_dyn();
    let b = Array4::from_shape_vec((2, 3, 4, 5), vec![0.0f32; 120])
        .unwrap()
        .into_dyn();
    batch_matmul(&a, &b);
}

#[test]
#[should_panic]
fn test_batch_matmul_batch_dim_mismatch() {
    let a = Array4::from_shape_vec((2, 3, 4, 5), vec![0.0f32; 120])
        .unwrap()
        .into_dyn();
    let b = Array4::from_shape_vec((4, 3, 5, 6), vec![0.0f32; 360])
        .unwrap()
        .into_dyn();
    batch_matmul(&a, &b);
}

#[test]
#[should_panic]
fn test_batch_matmul_batch_dim2_mismatch() {
    let a = Array4::from_shape_vec((2, 3, 4, 5), vec![0.0f32; 120])
        .unwrap()
        .into_dyn();
    let b = Array4::from_shape_vec((2, 4, 5, 6), vec![0.0f32; 240])
        .unwrap()
        .into_dyn();
    batch_matmul(&a, &b);
}

#[test]
#[should_panic]
fn test_batch_matmul_rank_mismatch() {
    let a = Array4::from_shape_vec((2, 3, 4, 5), vec![0.0f32; 120])
        .unwrap()
        .into_dyn();
    let b = Array3::from_shape_vec((3, 5, 6), vec![0.0f32; 90])
        .unwrap()
        .into_dyn();
    batch_matmul(&a, &b);
}
