use hw0::*;

#[test]
fn test_add() {
    assert_eq!(add(5.0, 6.0), 11.0);
    assert!((add(2.1, 2.3) - 4.4).abs() < 1e-10);
}

#[test]
fn test_primes() {
    let p = primes(10);
    assert_eq!(p, vec![2, 3, 5, 7]);
}

#[test]
fn test_primes_100() {
    let p = primes(100);
    assert_eq!(p.len(), 25);
    assert_eq!(*p.last().unwrap(), 97);
}

#[test]
fn test_poly_add() {
    let p1 = Polynomial::new(vec![1.0, 5.0, 0.0, 5.0]);
    let p2 = Polynomial::new(vec![0.0, 2.0]);
    let p3 = Polynomial::new(vec![-1.0, 6.0, 7.0, -5.0]);
    let p4 = Polynomial::new(vec![0.3, 0.4, 1.6, 1.9]);

    assert_eq!(
        poly_add(&p1, &p2),
        Polynomial::new(vec![1.0, 7.0, 0.0, 5.0])
    );
    assert_eq!(poly_add(&p1, &p3), Polynomial::new(vec![0.0, 11.0, 7.0]));
    assert_eq!(poly_add(&p1, &Polynomial::new(vec![0.0])), p1);

    let result = poly_add(&p2, &p4);
    let expected = Polynomial::new(vec![0.3, 2.4, 1.6, 1.9]);
    for (a, b) in result.coefficients.iter().zip(expected.coefficients.iter()) {
        assert!((a - b).abs() < 1e-10);
    }
}

#[test]
fn test_poly_mul() {
    let p1 = Polynomial::new(vec![1.0, 5.0, 0.0, 5.0]);
    let p2 = Polynomial::new(vec![0.0, 2.0]);
    let p3 = Polynomial::new(vec![-1.0, 6.0, 7.0, -5.0]);

    assert_eq!(
        poly_mul(&p1, &p2),
        Polynomial::new(vec![0.0, 2.0, 10.0, 0.0, 10.0])
    );
    assert_eq!(
        poly_mul(&p1, &p3),
        Polynomial::new(vec![-1.0, 1.0, 37.0, 25.0, 5.0, 35.0, -25.0])
    );
    assert_eq!(poly_mul(&p1, &Polynomial::new(vec![1.0])), p1);

    let p4 = Polynomial::new(vec![0.3, 0.4, 1.6, 1.9]);
    let result = poly_mul(&p1, &p4);
    let expected = vec![0.3, 1.9, 3.6, 11.4, 11.5, 8.0, 9.5];
    assert_eq!(result.coefficients.len(), expected.len());
    for (a, b) in result.coefficients.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-10, "{a} != {b}");
    }
}

#[test]
fn test_poly_derivative() {
    let p1 = Polynomial::new(vec![1.0, 5.0, 0.0, 5.0]);
    let p2 = Polynomial::new(vec![0.3, 0.4, 1.6]);

    assert_eq!(poly_derivative(&p1), Polynomial::new(vec![5.0, 0.0, 15.0]));
    assert_eq!(poly_derivative(&p2), Polynomial::new(vec![0.4, 3.2]));
    assert_eq!(
        poly_derivative(&Polynomial::new(vec![0.0])),
        Polynomial::new(vec![0.0])
    );
}
