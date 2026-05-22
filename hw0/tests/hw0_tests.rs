use hw0::*;

fn reference_primes(n: u64) -> Vec<u64> {
    fn is_prime(x: u64) -> bool {
        x >= 2 && (2..).take_while(|d| d * d <= x).all(|d| x % d != 0)
    }

    (0..n).filter(|&x| is_prime(x)).collect()
}

fn reference_poly_add(p1: &Polynomial, p2: &Polynomial) -> Polynomial {
    let len = p1.coefficients.len().max(p2.coefficients.len());
    let coefficients = (0..len)
        .map(|i| {
            p1.coefficients.get(i).copied().unwrap_or_default()
                + p2.coefficients.get(i).copied().unwrap_or_default()
        })
        .collect();
    Polynomial::new(coefficients)
}

fn reference_poly_mul(p1: &Polynomial, p2: &Polynomial) -> Polynomial {
    let mut coefficients = vec![0.0; p1.coefficients.len() + p2.coefficients.len() - 1];
    for (i, &a) in p1.coefficients.iter().enumerate() {
        for (j, &b) in p2.coefficients.iter().enumerate() {
            coefficients[i + j] += a * b;
        }
    }
    Polynomial::new(coefficients)
}

fn reference_poly_derivative(p: &Polynomial) -> Polynomial {
    if p.coefficients.len() <= 1 {
        return Polynomial::new(vec![0.0]);
    }

    Polynomial::new(
        p.coefficients
            .iter()
            .enumerate()
            .skip(1)
            .map(|(degree, &coefficient)| degree as f64 * coefficient)
            .collect(),
    )
}

fn assert_poly_close(actual: Polynomial, expected: Polynomial) {
    assert_eq!(actual.coefficients.len(), expected.coefficients.len());
    for (a, b) in actual.coefficients.iter().zip(expected.coefficients.iter()) {
        assert!((a - b).abs() < 1e-10, "{a} != {b}");
    }
}

#[test]
fn test_add() {
    let cases = [(5.0, 6.0), (2.1, 2.3)];
    for (a, b) in cases {
        let expected = a + b;
        assert!((add(a, b) - expected).abs() < 1e-10);
    }
}

#[test]
fn test_primes() {
    let n = 10;
    let p = primes(n);
    assert_eq!(p, reference_primes(n));
}

#[test]
fn test_primes_100() {
    let n = 100;
    let p = primes(n);
    assert_eq!(p, reference_primes(n));
}

#[test]
fn test_poly_add() {
    let p1 = Polynomial::new(vec![1.0, 5.0, 0.0, 5.0]);
    let p2 = Polynomial::new(vec![0.0, 2.0]);
    let p3 = Polynomial::new(vec![-1.0, 6.0, 7.0, -5.0]);
    let p4 = Polynomial::new(vec![0.3, 0.4, 1.6, 1.9]);

    for (left, right) in [
        (&p1, &p2),
        (&p1, &p3),
        (&p1, &Polynomial::new(vec![0.0])),
        (&p2, &p4),
    ] {
        assert_poly_close(poly_add(left, right), reference_poly_add(left, right));
    }
}

#[test]
fn test_poly_mul() {
    let p1 = Polynomial::new(vec![1.0, 5.0, 0.0, 5.0]);
    let p2 = Polynomial::new(vec![0.0, 2.0]);
    let p3 = Polynomial::new(vec![-1.0, 6.0, 7.0, -5.0]);

    let p4 = Polynomial::new(vec![0.3, 0.4, 1.6, 1.9]);
    for (left, right) in [
        (&p1, &p2),
        (&p1, &p3),
        (&p1, &Polynomial::new(vec![1.0])),
        (&p1, &p4),
    ] {
        assert_poly_close(poly_mul(left, right), reference_poly_mul(left, right));
    }
}

#[test]
fn test_poly_derivative() {
    let p1 = Polynomial::new(vec![1.0, 5.0, 0.0, 5.0]);
    let p2 = Polynomial::new(vec![0.3, 0.4, 1.6]);

    for p in [&p1, &p2, &Polynomial::new(vec![0.0])] {
        assert_poly_close(poly_derivative(p), reference_poly_derivative(p));
    }
}
