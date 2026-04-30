/*
 * Homework 2 - Automatic differentiation and linear models
 *
 * This homework contains two main portions. In part one, you will implement an extremely
 * minimal automatic differentiation module. This is the same technique that underlies
 * PyTorch, and while you will not implement anything close to the complexity of a library
 * like PyTorch, it will give you a basic understanding of the basic principles of the
 * approach, giving you some insight into how the nuts and bolts of PyTorch do work under
 * the hood. In the second part, you will use automatic differentiation to train a simple
 * linear model.
 *
 * ## Part I - Automatic differentiation
 *
 * At the core of automatic differentiation is a technique that builds a "compute graph",
 * which constructs a graph out of a series of functions applied to variables. In our
 * setting, we will implement this functionality with two simple types: a Variable struct
 * that represents the variables we will differentiate with respect to and a Function
 * trait that contains the logic to both implement the function itself and compute its
 * gradient.
 *
 * The Variable contains the following items:
 *   - .value : a f64 value that contains the numerical value of the variable
 *   - .grad : an Option<f64> that will be populated with the variable's derivative
 *   - .parents : the parents of the node in the graph (or empty if leaf)
 *   - .function : a reference to the function used to create the node from its parents
 *   - .num_children : the number of children that each node has
 *
 * Subclasses of Function need to implement two methods:
 *   1. forward() - actually computes the function (e.g., Multiply computes x*y)
 *   2. backward() - computes the product of an "incoming derivative" (gradient from
 *      downstream) and the partial derivatives of this function. For f(x,y) and
 *      incoming grad g, backward returns: [df/dx * g, df/dy * g]
 *
 * ## Backpropagation algorithm (compute_gradients):
 *
 * 1. If grad is None (i.e., this is the output node), set grad to 1.0
 * 2. If the node has parents and a function:
 *    a. Call function.backward(self.grad, parent_values) -> list of grad*partial products
 *    b. For each parent:
 *       - Add the corresponding product to parent's .grad (or set it if None)
 *       - Decrease the parent's num_children
 *       - If parent's num_children == 0, call compute_gradients recursively on it
 *
 * ## Part II - Training a digit classifier
 *
 * In this second part, you'll train a linear classifier using automatic differentiation.
 * Given predictions y_pred and labels y, the cross entropy loss is:
 *     L_ce(y_pred, y) = -y_pred_y + log(sum_j exp(y_pred_j))
 *
 * You'll implement minibatch SGD to optimize a linear classifier W in R^{k x n}:
 *     - Iterate over the dataset `epochs` times
 *     - Split data into chunks of batch_size
 *     - For each chunk: compute predictions, compute cross-entropy loss gradient,
 *       update W by taking a step in the direction of negative gradient
 */

use std::cell::RefCell;
use std::rc::Rc;

/// A node in the computation graph.
#[derive(Debug)]
pub struct Variable {
    pub value: f64,
    pub grad: Option<f64>,
    pub function: Option<Box<dyn Function>>,
    pub parents: Vec<Rc<RefCell<Variable>>>,
    pub num_children: usize,
}

impl Variable {
    pub fn new(value: f64) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Variable {
            value,
            grad: None,
            function: None,
            parents: vec![],
            num_children: 0,
        }))
    }
}

/// Trait for differentiable operations.
pub trait Function: std::fmt::Debug {
    fn forward(&self, inputs: &[f64]) -> f64;
    fn backward(&self, grad: f64, inputs: &[f64]) -> Vec<f64>;
}

/*
 * Question 1 - Function implementations
 *
 * Implement the following Function types. Remember that backward() needs to always
 * return a list of products between the incoming derivative and each partial derivative,
 * even if there is only a single argument.
 */

/// Implements addition: f(x, y) = x + y
/// Partials: df/dx = 1, df/dy = 1
#[derive(Debug)]
pub struct Add;

/// Implements subtraction: f(x, y) = x - y
/// Partials: df/dx = 1, df/dy = -1
#[derive(Debug)]
pub struct Subtract;

/// Implements multiplication: f(x, y) = x * y
/// Partials: df/dx = y, df/dy = x
#[derive(Debug)]
pub struct Multiply;

/// Implements division: f(x, y) = x / y
/// Partials: df/dx = 1/y, df/dy = -x/y^2
#[derive(Debug)]
pub struct Divide;

/// Implements power: f(x) = x^d
/// The degree d is stored in the struct (not differentiated w.r.t. d).
/// Partial: df/dx = d * x^(d-1). Handle d=0 case (derivative is 0).
#[derive(Debug)]
pub struct Power {
    pub degree: i32,
}

/// Implements natural logarithm: f(x) = ln(x)
/// Partial: df/dx = 1/x
#[derive(Debug)]
pub struct Log;

/// Implements exponential: f(x) = e^x
/// Partial: df/dx = e^x
#[derive(Debug)]
pub struct Exp;

/// Implements negation: f(x) = -x
/// Partial: df/dx = -1
#[derive(Debug)]
pub struct Negate;

// TODO: Implement Function trait for each operation.

/*
 * Question 2 - Implementing the full backward pass (compute_gradients)
 *
 * Recursively compute derivatives in a computation graph. This method iteratively
 * computes the gradients for a node and all its parents. It has no input or output
 * arguments, but instead directly modifies the Variable objects, populating the
 * .grad fields as needed and calling the function recursively on parents as needed.
 */

/// Compute gradients via reverse-mode autodiff (backpropagation).
/// Call on the output variable; fills in .grad for all ancestors.
pub fn compute_gradients(_output: &Rc<RefCell<Variable>>) {
    todo!()
}

/*
 * Question 3 - Cross entropy loss and error
 *
 * The cross entropy loss is defined for y_pred in R^k and y in {1,...,k} as:
 *     L_ce(y_pred, y) = -y_pred_y + log(sum_j exp(y_pred_j))
 *
 * Use the log-sum-exp trick for numerical stability.
 *
 * The error is the fraction of predictions that are wrong (argmax of y_pred != y).
 */

/// Compute the average cross entropy loss between predictions and desired outputs.
///
/// Input:
///     y_pred: slice of Vec<f64> (N x k) - each row is predicted outputs for the ith example
///     y: slice of usize (N) - each element is the desired class of the ith example
///
/// Output:
///     f64 - average cross entropy loss
pub fn cross_entropy_loss(y_pred: &[Vec<f64>], y: &[usize]) -> f64 {
    todo!()
}

/// Compute the average error between predictions and desired outputs, assuming
/// we make a "hard" prediction of whichever class has the highest predicted value.
///
/// Input:
///     y_pred: slice of Vec<f64> (N x k) - each row is predicted outputs
///     y: slice of usize (N) - each element is the desired class
///
/// Output:
///     f64 - average error rate
pub fn error(y_pred: &[Vec<f64>], y: &[usize]) -> f64 {
    todo!()
}

/*
 * Question 4 - (Minibatch) Stochastic Gradient descent
 *
 * Implement minibatch SGD to optimize a linear classifier specified by W in R^{k x n}.
 * The function should:
 *   - Iterate over the dataset `epochs` times
 *   - Split data into chunks of size `batch_size`
 *   - For each chunk: compute predictions (X_batch * W^T), compute gradient of
 *     cross-entropy loss, and update W by step_size in the direction of negative gradient
 */

/// Run minibatch stochastic gradient descent to minimize cross entropy loss.
///
/// Inputs:
///     x: slice of Vec<f64> (N x n) - training inputs
///     y: slice of usize (N) - desired outputs in 0..k-1
///     n_classes: number of classes k
///     epochs: number of passes over the training set
///     step_size: gradient descent step size
///     batch_size: number of examples in a minibatch
///
/// Output:
///     Vec<Vec<f64>> (k x n) - trained linear classifier weights
pub fn train_sgd(
    x: &[Vec<f64>],
    y: &[usize],
    n_classes: usize,
    epochs: usize,
    step_size: f64,
    batch_size: usize,
) -> Vec<Vec<f64>> {
    todo!()
}
