/*
 * Homework 0
 *
 * This homework is different from the other assignments in this course, in that we don't
 * actually build anything related to AI or chatbots. Instead, this is a basic assignment
 * meant to test two things:
 * 1. Getting set up with the development environment, and
 * 2. Evaluating some basic background in programming and math proficiency, of the kind
 *    you'll be expected to need for this course.
 *
 * If you can complete this assignment relatively easily, it is a good sign that your
 * background will be sufficient for the course. If the assignment is particularly
 * challenging, then this might be a good indication that you would benefit from additional
 * background before taking the course.
 */

/*
 * Problem 1: Simple addition
 *
 * Implement a function that adds together its two arguments.
 */
use std::cmp::max;

/// Add x and y
///
/// Input:
///     x: number (f64)
///     y: number (f64)
///
/// Output:
///     f64, addition of x and y
pub fn add(a: f64, b: f64) -> f64 {
    a + b
}

/*
 * Problem 2: Computing primes
 *
 * Write a simple routine to compute prime numbers. You will specifically do this using a
 * method called the Sieve of Eratosthenes, which computes all prime numbers up to some
 * number n by iteratively setting elements of an array to false if they cannot be a prime
 * because they are a multiple of another number.
 *
 * The pseudocode of the algorithm on the Wikipedia link
 * (https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes#Pseudocode) provides a pretty
 * reasonable description of the algorithm.
 */

/// Compute all the primes up to (but not including) n via sieve of Eratosthenes.
///
/// Input:
///     n: integer
/// Output:
///     list of primes up to (not including) n
pub fn primes(n: u64) -> Vec<u64> {
    if n < 2 {
        return vec![];
    }
    let sqrtn = (n as f64).sqrt();
    let mut temp: Vec<bool> = vec![true; (n + 1) as usize];
    temp[0] = false;
    temp[1] = false;
    temp[n as usize] = false;

    let mut i = 2;
    let mut j;
    let n = n as usize;
    while (i as f64) < sqrtn {
        if temp[i] {
            j = i * i;
            while j < n {
                temp[j] = false;
                j += i;
            }
        }
        i += 1;
    }
    temp.iter()
        .enumerate()
        .filter(|(_, &val)| val)
        .map(|(i, _)| i as u64)
        .collect()
}

/*
 * Problem 3: Operations on polynomials
 *
 * Write code to manipulate polynomials represented by a struct. This serves as a useful
 * evaluation of your familiarity with some basic mathematical concepts and how you
 * translate these into code.
 *
 * The Polynomial struct contains a coefficients member, which is a Vec where
 * coefficients[i] represents the coefficient on the i-th degree term, x^i.
 * In other words, the vec:
 *     [1, 0, 4, 3]
 * would represent the polynomial:
 *     3x^3 + 4x^2 + 1
 *
 * The vec:
 *     [4, 3, 5]
 * would represent the polynomial:
 *     5x^2 + 3x + 4
 *
 * Any term of degree beyond the length of the list implicitly has coefficient zero.
 */

/// This struct represents a polynomial as a list of coefficients. Each item in the list
/// at position i (zero-indexed) represents the coefficient corresponding to the x^i
/// term of the polynomial. For instance, the list:
///
/// [1, 0, 4, 3]
/// would represent the polynomial
/// 3x^3 + 4x^2 + 1
#[derive(Debug, Clone, PartialEq)]
pub struct Polynomial {
    pub coefficients: Vec<f64>,
}

impl Polynomial {
    pub fn new(coefficients: Vec<f64>) -> Self {
        let mut c = coefficients;
        while c.len() > 1 && c.last() == Some(&0.0) {
            c.pop();
        }
        Polynomial { coefficients: c }
    }

    pub fn degree(&self) -> usize {
        self.coefficients.len().saturating_sub(1)
    }
}

/*
 * Problem 3a: Polynomial addition
 *
 * Adding polynomials just involves adding the respective coefficients of the same degree.
 * For example, if you had two polynomials:
 *     p1(x) = 3x^3 + 4x^2 + 3
 *     p2(x) = x^2 + 5x + 5
 * then:
 *     p1(x) + p2(x) = 3x^3 + 5x^2 + 5x + 8
 *
 * Always return a new Polynomial rather than modifying either of the input objects
 * in-place.
 */

/// Add two polynomials together.
///
/// Input:
///     p1: &Polynomial
///     p2: &Polynomial
///
/// Output:
///     Polynomial corresponding to the addition of p1 and p2
pub fn poly_add(p1: &Polynomial, p2: &Polynomial) -> Polynomial {
    let max_len = max(p1.coefficients.len(), p2.coefficients.len());
    let mut coefficients = vec![0.0; max_len];

    for (i, el) in p1.coefficients.iter().enumerate() {
        coefficients[i] += el;
    }

    for (i, el) in p2.coefficients.iter().enumerate() {
        coefficients[i] += el;
    }

    Polynomial::new(coefficients)
}

/*
 * Problem 3b: Polynomial multiplication
 *
 * Multiplying polynomials involves multiplying every term in the first polynomial with
 * every term in the second, and adding together the results. For example:
 *     p1(x) = 3x^3 + 2x + 3
 *     p2(x) = 2x^2 + 5
 * Their multiplication is given by:
 *     p1(x) * p2(x) = (3x^3 + 2x + 3) * 2x^2 + (3x^3 + 2x + 3) * 5
 *                    = (6x^5 + 4x^3 + 6x^2) + (15x^3 + 10x + 15)
 *                    = 6x^5 + 19x^3 + 6x^2 + 10x + 15
 */

/// Multiply two polynomials together and return the result as a new Polynomial.
///
/// Input:
///     p1: &Polynomial
///     p2: &Polynomial
///
/// Output:
///     Polynomial corresponding to the multiplication of p1 and p2
pub fn poly_mul(p1: &Polynomial, p2: &Polynomial) -> Polynomial {
    let len_p1 = p1.coefficients.len();
    let len_p2 = p2.coefficients.len();
    let mut coefficients = vec![0.0; len_p1 * len_p2];

    for i in 0..len_p1 {
        for j in 0..len_p2 {
            coefficients[i + j] += p1.coefficients[i] * p2.coefficients[j];
        }
    }

    Polynomial::new(coefficients)
}

/*
 * Problem 3c: Polynomial differentiation
 *
 * Compute the derivative of the polynomial with respect to x.
 * For example, if we have the polynomial:
 *     p(x) = 4x^3 + 3x + 3
 * then the derivative of p(x) with respect to x is:
 *     p'(x) = 12x^2 + 3
 * where in general the derivative of any term c*x^n is given by n*c*x^(n-1)
 * and the derivative of any constant term c is zero.
 */

/// Compute the derivative of the polynomial with respect to x.
///
/// Input:
///     p: &Polynomial
///
/// Output:
///     Polynomial corresponding to the derivative of p with respect to x
pub fn poly_derivative(p: &Polynomial) -> Polynomial {
    let mut coefficients: Vec<f64> = (1..p.coefficients.len())
        .map(|i| p.coefficients[i] * i as f64)
        .collect();
    coefficients.push(0.0);
    Polynomial::new(coefficients)
}
