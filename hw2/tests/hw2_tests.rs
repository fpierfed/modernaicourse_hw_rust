use hw2::*;

const EPS: f64 = 1e-6;

// --- Function forward/backward tests ---

#[test]
fn test_add_forward_backward() {
    let f = Add;
    assert_eq!(f.forward(&[2.5, -0.5]), 2.0);
    let g = f.backward(3.0, &[2.5, -0.5]);
    assert_eq!(g.len(), 2);
    assert_eq!(g[0], 3.0);
    assert_eq!(g[1], 3.0);
}

#[test]
fn test_subtract_forward_backward() {
    let f = Subtract;
    assert_eq!(f.forward(&[2.5, -0.5]), 3.0);
    let g = f.backward(3.0, &[2.5, -0.5]);
    assert_eq!(g.len(), 2);
    assert_eq!(g[0], 3.0);
    assert_eq!(g[1], -3.0);
}

#[test]
fn test_divide_forward_backward() {
    let f = Divide;
    assert_eq!(f.forward(&[9.0, 3.0]), 3.0);
    let g = f.backward(4.0, &[9.0, 3.0]);
    assert_eq!(g.len(), 2);
    assert!((g[0] - 4.0 / 3.0).abs() < EPS);
    assert!((g[1] - (-4.0)).abs() < EPS);
}

#[test]
fn test_power_forward_backward() {
    let f = Power { degree: 3 };
    assert_eq!(f.forward(&[2.0]), 8.0);
    let g = f.backward(2.0, &[2.0]);
    assert_eq!(g.len(), 1);
    assert_eq!(g[0], 24.0);

    let f0 = Power { degree: 0 };
    assert_eq!(f0.forward(&[5.0]), 1.0);
    let g0 = f0.backward(7.0, &[5.0]);
    assert_eq!(g0[0], 0.0);
}

#[test]
fn test_log_forward_backward() {
    let f = Log;
    assert!((f.forward(&[std::f64::consts::E]) - 1.0).abs() < EPS);
    let g = f.backward(2.0, &[4.0]);
    assert_eq!(g.len(), 1);
    assert!((g[0] - 0.5).abs() < EPS);
}

#[test]
fn test_exp_forward_backward() {
    let f = Exp;
    assert!((f.forward(&[1.0]) - std::f64::consts::E).abs() < EPS);
    let g = f.backward(2.0, &[1.0]);
    assert_eq!(g.len(), 1);
    assert!((g[0] - 2.0 * std::f64::consts::E).abs() < EPS);
}

// --- Gradient computation test ---

#[test]
fn test_compute_gradients() {
    // z = ((-(x * y) * x * x) * (-y)), x=3, y=4
    // z = 432, dz/dx = 432, dz/dy = 216
    // This requires the full autodiff graph to be wired up.
    // Left as integration test once Variable graph operations are implemented.
    // TODO: uncomment once Variable operator overloading is done.
    // let x = Variable::new(3.0);
    // let y = Variable::new(4.0);
    // ... build expression, call compute_gradients, check grads
}

// --- Cross-entropy loss ---

#[test]
fn test_cross_entropy_loss() {
    let y_pred = vec![vec![2.0, 1.0, 0.0], vec![0.0, 2.0, 1.0]];
    let y = vec![0, 2];
    let loss = cross_entropy_loss(&y_pred, &y);
    assert!((loss - 0.9076060056686401).abs() < 1e-6);
}

// --- Error rate ---

#[test]
fn test_error() {
    let y_pred = vec![
        vec![3.0, 1.0],
        vec![0.0, 2.0],
        vec![1.0, 1.0],
        vec![-1.0, 0.0],
    ];
    let y = vec![0, 1, 1, 1];
    let err = error(&y_pred, &y);
    assert!((err - 0.25).abs() < 1e-6);
}

// --- SGD training ---

#[test]
fn test_train_sgd() {
    let x = vec![
        vec![2.0, 1.0],
        vec![1.0, 2.0],
        vec![-2.0, -1.0],
        vec![-1.0, -2.0],
    ];
    let y = vec![1, 1, 0, 0];

    let w = train_sgd(&x, &y, 2, 20, 0.1, 2);
    assert_eq!(w.len(), 2);
    assert_eq!(w[0].len(), 2);

    // Verify predictions are correct after training
    for (xi, &yi) in x.iter().zip(y.iter()) {
        let scores: Vec<f64> = w
            .iter()
            .map(|wj| wj.iter().zip(xi.iter()).map(|(a, b)| a * b).sum())
            .collect();
        let pred = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(pred, yi);
    }
}
