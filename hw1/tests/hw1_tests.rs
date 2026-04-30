use hw1::*;
use ndarray::{Array1, Array2};

fn reference_matmul(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    a.dot(b)
}

#[test]
fn test_vector_add() {
    let a = Array1::from(vec![1.0f32, 2.0, 3.0, 4.0, 5.0]);
    let b = Array1::from(vec![5.0f32, 4.0, 3.0, 2.0, 1.0]);
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
    let a = Array1::from(vec![1.0f32, 2.0, 3.0, 4.0, 5.0]);
    let b = Array1::from(vec![5.0f32, 4.0, 3.0, 2.0, 1.0]);
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
    let a = Array2::from_shape_vec((5, 4), (0..20).map(|x| x as f32).collect()).unwrap();
    let b = Array1::from(vec![1.0f32, 2.0, 3.0, 4.0]);
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
    let a = Array2::from_shape_vec((5, 4), (0..20).map(|x| x as f32).collect()).unwrap();
    let b = Array1::from(vec![1.0f32, 2.0, 3.0, 4.0]);
    let z = matrix_vector_product_2(&a, &b);
    let expected = a.dot(&b);
    assert!(z.abs_diff_eq(&expected, 1e-5));
}

#[test]
fn test_vector_matrix_product_2() {
    let a = Array2::from_shape_vec((4, 5), (0..20).map(|x| x as f32).collect()).unwrap();
    let b = Array1::from(vec![1.0f32, 2.0, 3.0, 4.0]);
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
    let a = Array2::from_shape_vec((4, 5), (0..20).map(|x| x as f32).collect()).unwrap();
    let b = Array2::from_shape_vec((5, 6), (0..30).map(|x| x as f32).collect()).unwrap();
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
    let a = Array2::from_shape_vec((4, 5), (0..20).map(|x| x as f32).collect()).unwrap();
    let b = Array2::from_shape_vec((5, 6), (0..30).map(|x| x as f32).collect()).unwrap();
    let z = matmul_2(&a, &b);
    let expected = reference_matmul(&a, &b);
    assert!(z.abs_diff_eq(&expected, 1e-4));
}

#[test]
fn test_matmul_3() {
    let a = Array2::from_shape_vec((4, 5), (0..20).map(|x| x as f32).collect()).unwrap();
    let b = Array2::from_shape_vec((5, 6), (0..30).map(|x| x as f32).collect()).unwrap();
    let z = matmul_3(&a, &b);
    let expected = reference_matmul(&a, &b);
    assert!(z.abs_diff_eq(&expected, 1e-4));
}

#[test]
fn test_block_matmul() {
    let a = Array2::from_shape_vec((16, 12), (0..192).map(|x| x as f32 / 100.0).collect()).unwrap();
    let b = Array2::from_shape_vec((12, 8), (0..96).map(|x| x as f32 / 100.0).collect()).unwrap();
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
