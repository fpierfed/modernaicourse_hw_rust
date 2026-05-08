/*
 * Homework 1 - Introduction to Linear Algebra
 *
 * This homework is aimed to familiarize you with some of the basic linear algebra
 * operations covered in class, as well as how to implement these functions.
 *
 * In this assignment, you're going to implement a wide variety of simple linear algebra
 * operators, WITHOUT using any built-in tensor addition or matrix multiplication
 * operators. Your code should also panic with assertion errors if any of the sizes do
 * not match as allowed for the given operation. Instead, you should use explicit for
 * loops and element-by-element assignment/operations to implement your functions.
 */

use ndarray::{Array1, Array2, ArrayD};

/*
 * Problem 1: "Classical" programming for digit classification
 *
 * This course deals primarily with machine learning approaches, but it's worth
 * emphasizing that you CAN try to approach many of the problems you'll want to solve
 * with machine learning with traditional programming approaches as well. In this
 * problem, you should experiment with developing a "manual" classifier between images
 * of digits in the MNIST dataset. Specifically, implement the function
 * `classify_zero_one` to classify between images of zeros and ones.
 * Try to think intuitively about features that might distinguish between zeros and ones.
 */

/// Classify a 28x28 grayscale image (pixel values in [0.0, 1.0]) as either
/// a zero (return 0) or a one (return 1).
///
/// Input:
///     image: 2D array (28 x 28) with f32 values normalized to [0, 1]
///
/// Output:
///     u8 - predicted digit (0 or 1)
pub fn classify_zero_one(image: &Array2<f32>) -> u8 {
    todo!()
}

/*
 * Problem 2: Vector Addition
 *
 * Implement a simple vector addition function that adds two vectors together,
 * x, y in R^n. The function should panic if the vectors are not the proper size
 * to add together.
 */

/// Add two vectors x and y, WITHOUT using built-in vectorized addition.
/// Instead, manually iterate through the elements of x and y and add them together.
/// The function should panic if the vectors are not the proper size to add together.
///
/// Input:
///     x: 1D array - first term to add
///     y: 1D array - second term to add
///
/// Output:
///     1D array - sum of x + y
pub fn vector_add(a: &Array1<f32>, b: &Array1<f32>) -> Array1<f32> {
    todo!()
}

/*
 * Problem 3: Vector inner product
 *
 * Implement the vector inner product. For two vectors x, y in R^n, return the inner
 * product:
 *     <x, y> = x^T y = sum_{i=1}^{n} x_i * y_i
 *
 * Don't use any library functions that compute a matrix multiplication or inner product
 * directly, but do it all with for loops.
 */

/// Compute the inner product between two vectors x and y, WITHOUT using built-in
/// dot product operations. The function should panic if the vectors are not the
/// proper size.
///
/// Input:
///     x: 1D array - first vector
///     y: 1D array - second vector
///
/// Output:
///     f32 - inner product <x, y>
pub fn vector_inner_product(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    todo!()
}

/*
 * Problem 4: Matrix-vector product approach #1
 *
 * Compute the matrix-vector product Ax for A in R^{m x n} and x in R^n.
 * This version should compute each entry of the resulting vector using the inner
 * product between rows of A and the vector x:
 *
 *     Ax = [ a1^T ]     [ a1^T x ]
 *          [ a2^T ] x = [ a2^T x ]
 *          [ ...  ]     [ ...    ]
 *          [ am^T ]     [ am^T x ]
 *
 * Only use the vector_inner_product() function for this routine.
 */

/// Compute the matrix vector product Ax using inner products of rows of A with x.
/// Panics if the product is not valid due to dimension mismatch.
///
/// Input:
///     A: 2D array (m x n)
///     x: 1D array (n elements)
///
/// Output:
///     1D array (m elements) - vector Ax
pub fn matrix_vector_product_1(a: &Array2<f32>, b: &Array1<f32>) -> Array1<f32> {
    todo!()
}

/*
 * Problem 5: Matrix-vector product approach #2
 *
 * Compute the matrix-vector product Ax for A in R^{m x n} and x in R^n.
 * This version should compute the result as a linear combination of the columns
 * of A with coefficients given by the entries of x:
 *
 *     Ax = [a1 | a2 | ... | an] [x1]   = a1*x1 + a2*x2 + ... + an*xn
 *                                [x2]
 *                                [..]
 *                                [xn]
 *
 * Only use the vector_add() function to implement your solution (plus scalar-vector
 * multiplication).
 */

/// Compute the matrix vector product Ax as a linear combination of the columns
/// of A with coefficients given by the entries of x. Only use vector_add.
/// Panics if sizes do not allow for a valid product.
///
/// Input:
///     A: 2D array (m x n)
///     x: 1D array (n elements)
///
/// Output:
///     1D array (m elements) - vector Ax
pub fn matrix_vector_product_2(a: &Array2<f32>, b: &Array1<f32>) -> Array1<f32> {
    todo!()
}

/*
 * Problem 6: Vector-matrix product approach #2
 *
 * Compute the vector-matrix product x^T A for A in R^{m x n} and x in R^m.
 * This version should compute the result as a linear combination of the rows of A
 * with coefficients given by the entries of x:
 *
 *     x^T A = [x1 x2 ... xm] [-- a1^T --]   = x1*a1^T + x2*a2^T + ... + xm*am^T
 *                              [-- a2^T --]
 *                              [   ...    ]
 *                              [-- am^T --]
 *
 * Only use the vector_add() function to implement your solution.
 */

/// Compute the vector-matrix product x^T A as a linear combination of the rows
/// of A with coefficients given by the entries of x. Only use vector_add.
/// Panics if sizes do not allow for a valid product.
///
/// Input:
///     v: 1D array (m elements)
///     A: 2D array (m x n)
///
/// Output:
///     1D array (n elements) - vector x^T A
pub fn vector_matrix_product_2(v: &Array1<f32>, a: &Array2<f32>) -> Array1<f32> {
    todo!()
}

/*
 * Problem 7: Matrix-matrix multiplication approach #1
 *
 * For A in R^{m x n} and B in R^{n x p}, compute each element (AB)_{ij} as the
 * inner product of the i-th row of A and the j-th column of B:
 *
 *     (AB)_{ij} = a_i^T b_j
 *
 * Only use the vector_inner_product() function.
 */

/// Compute matrix-matrix multiplication AB where each entry is the inner product
/// of a row of A and a column of B. Only use vector_inner_product.
/// Panics if sizes of the matrices do not make for a valid product.
///
/// Input:
///     A: 2D array (m x n)
///     B: 2D array (n x p)
///
/// Output:
///     2D array (m x p) - matrix AB
pub fn matmul_1(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    todo!()
}

/*
 * Problem 8: Matrix-matrix multiplication approach #2
 *
 * For A in R^{m x n} and B in R^{n x p}, compute the i-th column of AB as the
 * matrix-vector product between A and the i-th column of B:
 *
 *     AB = A [b1 | b2 | ... | bp] = [Ab1 | Ab2 | ... | Abp]
 *
 * Only use the matrix_vector_product_1() or matrix_vector_product_2() function.
 */

/// Compute matrix-matrix multiplication AB by computing each column of the
/// result as a matrix-vector product of A with a column of B.
/// Only use matrix_vector_product_1 or matrix_vector_product_2.
/// Panics if sizes of the matrices do not make for a valid product.
///
/// Input:
///     A: 2D array (m x n)
///     B: 2D array (n x p)
///
/// Output:
///     2D array (m x p) - matrix AB
pub fn matmul_2(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    todo!()
}

/*
 * Problem 9: Matrix-matrix multiplication approach #3
 *
 * For A in R^{m x n} and B in R^{n x p}, compute the i-th row of AB as the
 * vector-matrix product between the i-th row of A and B:
 *
 *     AB = [-- a1^T --]     [-- a1^T B --]
 *          [-- a2^T --] B = [-- a2^T B --]
 *          [   ...    ]     [    ...     ]
 *          [-- am^T --]     [-- am^T B --]
 *
 * Only use the vector_matrix_product_2() function.
 */

/// Compute matrix-matrix multiplication AB by computing each row of the result
/// as a vector-matrix product of a row of A with B.
/// Only use vector_matrix_product_2.
/// Panics if sizes of the matrices do not make for a valid product.
///
/// Input:
///     A: 2D array (m x n)
///     B: 2D array (n x p)
///
/// Output:
///     2D array (m x p) - matrix AB
pub fn matmul_3(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    todo!()
}

/*
 * Problem 11: Block matrix multiplication
 *
 * Implement a "blocked" form of matrix multiplication. Although we defined matrix
 * multiplication in terms of the individual scalar entries of a matrix, it can also
 * be defined by operating on subblocks of the matrices. Specifically for a matrix
 * A in R^{4m x 4n} we can define A_{ij} in R^{4x4} to be a subblock of the matrix,
 * and similarly for the matrix B in R^{4n x 4p}. Then the corresponding 4x4 subblock
 * of the matrix product AB can be computed as:
 *
 *     (AB)_{ij} = sum_{k=1}^{n} A_{ik} * B_{kj}
 *
 * analogous to the usual definition of matrix multiplication, but with A_{ik} * B_{kj}
 * now being a matrix product.
 *
 * In practice, techniques like this (with proper memory layouts) are how one writes
 * fast matrix multiplication primitives on GPUs (where e.g., so-called "tensor cores"
 * actually exactly perform 4x4 matrix multiplication).
 *
 * You should check to ensure that the matrices form a valid matrix multiplication,
 * and that their dimensions are all divisible by 4.
 */

/// Implement a block matrix multiplication to compute the matrix-matrix product AB.
/// Splits matrices into 4x4 blocks and multiplies block-by-block.
/// Panics if matrices are improper shapes or have dimensions not divisible by 4.
///
/// Input:
///     A: 2D array with dimensions divisible by 4
///     B: 2D array with dimensions divisible by 4
///
/// Output:
///     2D array - matrix AB
pub fn block_matmul(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    todo!()
}

/*
 * Problem 10: Batch matrix multiplication
 *
 * Implement a batched form of matrix multiplication. For input tensors of shape
 * (b1, b2, ..., bn, m, k) and (b1, b2, ..., bn, k, p), compute the matrix product
 * along the last two dimensions for each batch element. The batch dimensions must
 * match exactly and both inputs must have the same number of dimensions.
 *
 * Use one of the matmul_1, matmul_2, or matmul_3 functions for the inner
 * matrix multiplication.
 */

/// Compute batched matrix multiplication on N-dimensional arrays.
/// For A of shape (..., m, k) and B of shape (..., k, p), compute the matrix
/// product along the last two dimensions for each batch element.
/// Panics if batch dimensions don't match, inner dimensions don't match,
/// or the arrays don't have the same number of dimensions.
///
/// Input:
///     A: N-dimensional array (..., m, k)
///     B: N-dimensional array (..., k, p)
///
/// Output:
///     N-dimensional array (..., m, p)
pub fn batch_matmul(a: &ArrayD<f32>, b: &ArrayD<f32>) -> ArrayD<f32> {
    todo!()
}
