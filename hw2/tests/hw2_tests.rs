use hw2::*;
use std::cell::RefCell;
use std::rc::Rc;

const EPS: f64 = 1e-6;

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < EPS,
        "actual {actual}, expected {expected}"
    );
}

fn apply_fn(func: Box<dyn Function>, args: &[&Rc<RefCell<Variable>>]) -> Rc<RefCell<Variable>> {
    let inputs: Vec<f64> = args.iter().map(|a| a.borrow().value).collect();
    let value = func.forward(&inputs);
    for a in args {
        a.borrow_mut().num_children += 1;
    }
    let parents: Vec<Rc<RefCell<Variable>>> = args.iter().map(|a| Rc::clone(a)).collect();
    Rc::new(RefCell::new(Variable {
        value,
        grad: None,
        function: Some(func),
        parents,
        num_children: 0,
    }))
}

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
fn test_add_alternate_values() {
    let f = Add;
    assert_close(f.forward(&[-1.5, 3.0]), 1.5);
    let g = f.backward(2.5, &[-1.5, 3.0]);
    assert_eq!(g.len(), 2);
    assert_close(g[0], 2.5);
    assert_close(g[1], 2.5);
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
fn test_subtract_alternate_values() {
    let f = Subtract;
    assert_close(f.forward(&[-1.5, 3.0]), -4.5);
    let g = f.backward(2.5, &[-1.5, 3.0]);
    assert_eq!(g.len(), 2);
    assert_close(g[0], 2.5);
    assert_close(g[1], -2.5);
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
fn test_divide_alternate_values() {
    let f = Divide;
    assert_close(f.forward(&[-8.0, 2.0]), -4.0);
    let g = f.backward(1.5, &[-8.0, 2.0]);
    assert_eq!(g.len(), 2);
    assert_close(g[0], 0.75);
    assert_close(g[1], 3.0);
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
    assert_eq!(g0.len(), 1);
    assert_eq!(g0[0], 0.0);
}

#[test]
fn test_power_alternate_values() {
    let f = Power { degree: 3 };
    assert_close(f.forward(&[-2.0]), -8.0);
    let g = f.backward(1.5, &[-2.0]);
    assert_eq!(g.len(), 1);
    assert_close(g[0], 18.0);

    let f0 = Power { degree: 0 };
    let g0 = f0.backward(7.0, &[5.0]);
    assert_eq!(g0.len(), 1);
    assert_close(g0[0], 0.0);
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
fn test_log_alternate_values() {
    let f = Log;
    assert_close(f.forward(&[3.5]), 1.252762968495368);
    let g = f.backward(2.0, &[3.5]);
    assert_eq!(g.len(), 1);
    assert_close(g[0], 0.5714285714285714);
}

#[test]
fn test_exp_forward_backward() {
    let f = Exp;
    assert!((f.forward(&[1.0]) - std::f64::consts::E).abs() < EPS);
    let g = f.backward(2.0, &[1.0]);
    assert_eq!(g.len(), 1);
    assert!((g[0] - 2.0 * std::f64::consts::E).abs() < EPS);
}

#[test]
fn test_exp_alternate_values() {
    let f = Exp;
    assert_close(f.forward(&[-0.5]), 0.6065306597126334);
    let g = f.backward(2.0, &[-0.5]);
    assert_eq!(g.len(), 1);
    assert_close(g[0], 1.2130613194252668);
}

// --- Gradient computation test ---

#[test]
fn test_compute_gradients() {
    // z = ((-(x * y) * x * x) * (-y)), x=3, y=4
    // z = (-(3*4) * 3 * 3) * (-4) = (-108) * (-4) = 432
    // dz/dx = 432, dz/dy = 216
    let x = Variable::new(3.0);
    let y = Variable::new(4.0);

    // Build: x * y
    let xy = apply_fn(Box::new(Multiply), &[&x, &y]);
    // -(x * y)
    let neg_xy = apply_fn(Box::new(Negate), &[&xy]);
    // -(x*y) * x
    let neg_xy_x = apply_fn(Box::new(Multiply), &[&neg_xy, &x]);
    // -(x*y) * x * x
    let neg_xy_xx = apply_fn(Box::new(Multiply), &[&neg_xy_x, &x]);
    // -y
    let neg_y = apply_fn(Box::new(Negate), &[&y]);
    // (-(x*y)*x*x) * (-y)
    let z = apply_fn(Box::new(Multiply), &[&neg_xy_xx, &neg_y]);

    assert!((z.borrow().value - 432.0).abs() < EPS);

    compute_gradients(&z);

    assert!((z.borrow().grad.unwrap() - 1.0).abs() < EPS);
    assert!(
        (x.borrow().grad.unwrap() - 432.0).abs() < EPS,
        "x.grad = {:?}, expected 432.0",
        x.borrow().grad
    );
    assert!(
        (y.borrow().grad.unwrap() - 216.0).abs() < EPS,
        "y.grad = {:?}, expected 216.0",
        y.borrow().grad
    );
}

#[test]
fn test_compute_gradients_leaf() {
    // Calling compute_gradients on a leaf variable should set its grad to 1.0
    let w = Variable::new(-2.0);
    compute_gradients(&w);
    assert!((w.borrow().grad.unwrap() - 1.0).abs() < EPS);
}

#[test]
fn test_compute_gradients_reused_intermediate() {
    let x = Variable::new(1.5);
    let y = Variable::new(-2.0);

    let xy = apply_fn(Box::new(Multiply), &[&x, &y]);
    let neg_x = apply_fn(Box::new(Negate), &[&x]);
    let xy_neg_x = apply_fn(Box::new(Multiply), &[&xy, &neg_x]);
    let neg_y = apply_fn(Box::new(Negate), &[&y]);
    let a = apply_fn(Box::new(Multiply), &[&xy_neg_x, &neg_y]);
    let a_squared = apply_fn(Box::new(Multiply), &[&a, &a]);
    let z = apply_fn(Box::new(Negate), &[&a_squared]);

    assert_close(a.borrow().value, 9.0);
    assert_close(z.borrow().value, -81.0);

    compute_gradients(&z);

    assert_close(a.borrow().grad.unwrap(), -18.0);
    assert_close(x.borrow().grad.unwrap(), -216.0);
    assert_close(y.borrow().grad.unwrap(), 162.0);
}

// --- Cross-entropy loss ---

#[test]
fn test_cross_entropy_loss() {
    let y_pred = vec![vec![2.0, 1.0, 0.0], vec![0.0, 2.0, 1.0]];
    let y = vec![0, 2];
    let loss = cross_entropy_loss(&y_pred, &y);
    assert!((loss - 0.9076060056686401).abs() < 1e-6);
}

#[test]
fn test_cross_entropy_loss_three_class_batch() {
    let y_pred = vec![
        vec![1.0, 0.0, -1.0],
        vec![2.0, 1.0, 0.0],
        vec![-1.0, 2.0, 1.0],
    ];
    let y = vec![0, 2, 1];
    let loss = cross_entropy_loss(&y_pred, &y);
    assert!((loss - 1.054741381885649).abs() < 1e-6);
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
fn test_train_sgd_one_epoch() {
    let x = vec![
        vec![2.0, 1.0],
        vec![1.0, 2.0],
        vec![-2.0, -1.0],
        vec![-1.0, -2.0],
    ];
    let y = vec![1, 1, 0, 0];

    let w = train_sgd(&x, &y, 2, 1, 0.1, 2);
    let expected = vec![vec![-0.13340412, -0.13340412], vec![0.13340412, 0.13340412]];
    assert_eq!(w.len(), 2);
    assert_eq!(w[0].len(), 2);
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (w[i][j] - expected[i][j]).abs() < 1e-5,
                "w[{i}][{j}] = {}, expected {}",
                w[i][j],
                expected[i][j]
            );
        }
    }
}

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
    assert!(w.iter().flatten().any(|wij| wij.abs() > EPS));
}

#[test]
fn test_train_sgd_fifteen_epochs() {
    let x = vec![
        vec![2.0, 1.0],
        vec![1.0, 2.0],
        vec![-2.0, -1.0],
        vec![-1.0, -2.0],
    ];
    let y = vec![1, 1, 0, 0];

    let w = train_sgd(&x, &y, 2, 15, 0.1, 2);
    assert_eq!(w.len(), 2);
    assert_eq!(w[0].len(), 2);

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

    let norm = w.iter().flatten().map(|wij| wij * wij).sum::<f64>().sqrt();
    assert!(norm > EPS, "trained weights should not stay at zero");
}
