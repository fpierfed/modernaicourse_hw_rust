# Implementation Guide: ML Course Homeworks in Rust

This document describes the architecture, design choices, and implementation
strategy for porting a series of machine learning homework assignments from
Python/PyTorch to Rust. The original course builds up from basic polynomial
arithmetic to training a full Llama3-style transformer with tool-use
capabilities.

---

## Table of Contents

1. [Project Organization](#project-organization)
2. [Dependency Strategy](#dependency-strategy)
3. [HW0: Foundations (Pure Rust)](#hw0-foundations)
4. [HW1: Manual Linear Algebra (ndarray)](#hw1-manual-linear-algebra)
5. [HW2: Automatic Differentiation (Graph-based)](#hw2-automatic-differentiation)
6. [HW3–HW4: Neural Network Modules (burn)](#hw3-hw4-neural-network-modules)
7. [HW5: Transformer Architecture](#hw5-transformer-architecture)
8. [HW6: LLM Training Pipeline](#hw6-llm-training-pipeline)
9. [HW7: Tool-Use Agent](#hw7-tool-use-agent)
10. [Key Differences from the Python Implementation](#key-differences)
11. [Testing Philosophy](#testing-philosophy)

---

## Project Organization

The workspace is structured as a Cargo workspace with eight independent crates:

```
moderaicourse_hw_rust_ws/
├── Cargo.toml          # Workspace manifest (shared dependency versions)
├── hw0/                # Pure algorithms: polynomials, primes
├── hw1/                # Manual linalg: vector/matrix ops from scratch
├── hw2/                # Autodiff: computation graph with backprop
├── hw3/                # NN modules: Linear, SGD, DataLoader, training
├── hw4/                # End-to-end: train models on MNIST
├── hw5/                # Transformers: attention, KV cache, Llama3
├── hw6/                # LLM training: BPE, Adam, data pipeline
└── hw7/                # Agents: tool-use, structured generation
```

Each crate is self-contained with its own `Cargo.toml`, `src/lib.rs` (public
API), and `tests/` directory. The workspace `Cargo.toml` declares shared
dependency versions to avoid conflicts and keep upgrades synchronized.

**Why separate crates instead of one library?**

- Each homework has distinct dependencies (hw0 needs nothing, hw1 needs ndarray,
  hw3+ need burn). Separate crates mean you don't pull in a tensor framework
  just to work on polynomial multiplication.
- Compile times: `cargo test -p hw0` compiles only hw0 and its (zero)
  dependencies. This makes the edit-compile-test cycle fast for early homeworks.
- Conceptual isolation: each crate represents a self-contained learning unit.
  You can hand off `hw5/` to someone who hasn't seen the earlier code and
  they'd have everything they need.

---

## Dependency Strategy

The workspace uses three tiers of dependencies, introduced progressively:

| Tier | Crates | Dependencies | Rationale |
|------|--------|-------------|-----------|
| 0 | hw0 | None | Pure algorithms need no external types |
| 1 | hw1, hw2 | `ndarray` | Need multi-dimensional arrays but not autograd |
| 2 | hw3–hw7 | `burn` | Need tensors with autograd, typed dimensions, nn modules |

### Why ndarray for Tier 1?

The whole point of hw1 is implementing matrix multiplication from scratch. You
need a container that can hold a 2D grid of floats and let you index into it,
but you explicitly do *not* want a built-in `matmul`. `ndarray::Array2<f32>`
gives you exactly that: a shaped container with element access, slicing, and
reshaping, without pulling in BLAS.

### Why burn for Tier 2?

[Burn](https://github.com/tracel-ai/burn) is a comprehensive deep learning
framework in Rust with a strong type system. Tensors carry their dimensionality
at compile time (`Tensor<B, D>` where D is a const generic), catching shape
errors before runtime. Compared to alternatives:

- **candle**: Simpler API but untyped dimensions — shape errors only appear at
  runtime. Burn's compile-time dimension tracking catches more bugs early.
- **tch-rs**: Rust bindings to libtorch. Faithful to PyTorch but requires
  linking against a 2GB C++ library. Burn is pure Rust.
- **dfdx**: Interesting but less ecosystem support for loading model weights.

Burn gives us typed tensors, automatic differentiation via the `Autodiff`
backend wrapper, and a clean module system — everything needed to port
PyTorch-style training code with added compile-time safety.

---

## HW0: Foundations

**Concepts**: Polynomial arithmetic, sieve of Eratosthenes.

### Design

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct Polynomial {
    pub coefficients: Vec<f64>,  // coefficients[i] = coefficient of x^i
}
```

The polynomial is stored in "ascending degree" order: index 0 is the constant
term, index `n` is the x^n coefficient. This matches the Python implementation
and makes addition/multiplication algorithms natural (the index *is* the
degree).

The constructor enforces a canonical form by stripping trailing zeros:

```rust
impl Polynomial {
    pub fn new(mut coefficients: Vec<f64>) -> Self {
        while coefficients.len() > 1 && coefficients.last() == Some(&0.0) {
            coefficients.pop();
        }
        Polynomial { coefficients }
    }
}
```

This guarantees that `degree()` returns the correct value and that two
polynomials compare equal if and only if they represent the same mathematical
object.

### Key algorithms

- **poly_add**: Zip coefficients, padding the shorter vector with zeros.
- **poly_mul**: Classic O(n*m) convolution — for each pair (i, j), accumulate
  `a[i] * b[j]` into result[i+j].
- **poly_derivative**: Shift coefficients left, multiplying by degree.
- **primes**: Sieve of Eratosthenes up to n.

### Testing approach

Tests use exact equality for integer-coefficient polynomials and epsilon
comparison for floating-point coefficients. This mirrors the Python tests
which use `==` on the `Polynomial` class (which compares coefficient lists).

---

## HW1: Manual Linear Algebra

**Concepts**: Implementing matmul from first principles, understanding how
different decompositions (row-wise, column-wise, block) produce the same result.

### Design

All functions operate on `ndarray::Array1<f32>` and `ndarray::Array2<f32>`.
The key constraint is: **you may not call `.dot()` or any BLAS routine**. All
products must be built from scalar operations.

The Python tests enforce this via `PreventTorchOps`, a dispatch mode that
intercepts and blocks any tensor multiplication. In Rust, we enforce this
architecturally: the functions simply don't have access to `.dot()` because
they're written using explicit loops.

### Implementation hierarchy

```
vector_inner_product(a, b)          → Σ a[i]*b[i]
vector_add(a, b)                    → element-wise a[i]+b[i]
    ↓                                       ↓
matrix_vector_product_1(A, b)       matrix_vector_product_2(A, b)
    (row-wise: each row dot b)          (column-wise: Σ b[j]*col_j)
    ↓                                       ↓
matmul_1(A, B)                      matmul_2(A, B)         matmul_3(A, B)
    (inner products)                    (mat-vec per col)     (vec-mat per row)
    ↓
block_matmul(A, B)
    (partition into 4×4 blocks, multiply blocks)
```

Each matmul variant produces the same result but via a different access
pattern. The pedagogical point is understanding that matrix multiplication can
be decomposed in multiple ways, each with different cache/parallelism
implications.

### Dimension checking

Every function panics on dimension mismatch with an `assert!` at the top:

```rust
pub fn vector_add(a: &Array1<f32>, b: &Array1<f32>) -> Array1<f32> {
    assert_eq!(a.len(), b.len(), "dimension mismatch");
    // ...
}
```

The Python tests use `pytest.raises(AssertionError)` to verify these checks;
in Rust we use `#[should_panic]` test attributes.

---

## HW2: Automatic Differentiation

**Concepts**: Computation graphs, reverse-mode autodiff (backpropagation),
chain rule.

### Design: The Variable Graph

This is the most architecturally interesting crate. The Python version uses
mutable objects with operator overloading. In Rust, we need to handle shared
mutable state — the classic challenge.

```rust
pub struct Variable {
    pub value: f64,
    pub grad: Option<f64>,
    pub function: Option<Box<dyn Function>>,
    pub parents: Vec<Rc<RefCell<Variable>>>,
    pub num_children: usize,
}
```

**Why `Rc<RefCell<Variable>>`?**

The computation graph has shared ownership (multiple children point to the same
parent) and requires interior mutability (gradients are written during
backprop). This is the standard Rust pattern for graph structures:

- `Rc` provides shared ownership (reference counting)
- `RefCell` provides runtime-checked mutable borrows

An alternative would be an arena allocator (like `slotmap` or `typed-arena`),
which avoids reference counting overhead but makes the API less ergonomic for
this pedagogical use case.

### The Function Trait

```rust
pub trait Function: std::fmt::Debug {
    fn forward(&self, inputs: &[f64]) -> f64;
    fn backward(&self, grad: f64, inputs: &[f64]) -> Vec<f64>;
}
```

Each operation (Add, Multiply, Power, etc.) implements this trait. The
`backward` method receives the upstream gradient and the original inputs, and
returns one gradient per input (the chain rule).

For example, `Multiply::backward(grad, [x, y])` returns `[y*grad, x*grad]`
because ∂(x*y)/∂x = y and ∂(x*y)/∂y = x.

### Backpropagation

`compute_gradients` performs a topological traversal of the graph in reverse:

1. Set output's grad to 1.0
2. For each node (in reverse topological order):
   - Call `function.backward(self.grad, parent_values)`
   - Accumulate returned gradients into each parent's `.grad`

The `num_children` counter enables a simple topological sort without a separate
data structure: each time a node propagates its gradient, it decrements the
child count of its parents. When a parent's count reaches zero, all its
children have contributed, and it's ready to propagate.

### Cross-entropy and SGD

These are implemented as standalone functions (not part of the autodiff graph)
using plain `Vec<Vec<f64>>` matrices. The SGD implementation computes gradients
of cross-entropy loss manually (softmax derivative), since the autodiff graph
operates on scalars and would be impractically slow for batch training.

---

## HW3–HW4: Neural Network Modules

**Concepts**: Parameterized layers, optimizers, data loading, training loops.

### Design: burn's Module System

Burn uses typed tensors with compile-time dimensionality and a backend
abstraction for parameter management:

```rust
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::backend::Autodiff;
use burn::tensor::{Tensor, Int};

type B = Autodiff<NdArray<f32>>;

pub struct Linear {
    weight: Tensor<B, 2>,  // (out_dim, in_dim)
}

impl Linear {
    pub fn new(in_f: usize, out_f: usize, device: &NdArrayDevice) -> Self {
        // Kaiming initialization
        todo!()
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        // x @ weight.T (generic over input dimensionality)
        todo!()
    }
}
```

**Kaiming initialization**: The Python tests verify that weight standard
deviation matches `sqrt(2/fan_in)`. This is He initialization, appropriate for
ReLU networks.

### SGD Optimizer

```rust
pub struct SGD {
    params: Vec<Tensor<B, 2>>,
    lr: f64,
}

impl SGD {
    pub fn step(&mut self, grads: &<B as AutodiffBackend>::Gradients) {
        for p in &mut self.params {
            let grad = p.grad(grads).unwrap();
            // p = p - lr * grad (detach and re-attach for next backward)
            todo!()
        }
    }
}
```

### DataLoader

The Rust DataLoader implements `Iterator`, yielding `(Tensor<B, 2>, Tensor<B, 1, Int>)` pairs.
Unlike Python's `__iter__`/`__next__`, Rust iterators are types that implement
the `Iterator` trait:

```rust
impl Iterator for DataLoader {
    type Item = (Tensor<B, 2>, Tensor<B, 1, Int>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.n_samples {
            self.offset = 0;
            return None;
        }
        let end = (self.offset + self.batch_size).min(self.n_samples);
        let batch = /* slice X and y */;
        self.offset = end;
        Some(batch)
    }
}
```

The test verifies that iterating twice produces identical batches (the loader
resets when exhausted).

### The `epoch` Function

This is a higher-order function that takes a model (as a closure/function
pointer) and optionally an optimizer:

```rust
pub fn epoch<M>(
    model: &M,
    loader: &[(Tensor<B, 2>, Tensor<B, 1, Int>)],
    optimizer: Option<&mut SGD>,
) -> (f64, f64)
where
    M: Fn(Tensor<B, 2>) -> Tensor<B, 2>,
```

If `optimizer` is `None`, it's an evaluation pass (no gradient updates). If
`Some`, it's a training pass. This dual-mode design avoids code duplication
between train and eval, matching the PyTorch pattern of `model.train()` vs
`model.eval()`.

---

## HW5: Transformer Architecture

**Concepts**: Self-attention, multi-head attention, KV caching, RMSNorm,
gated MLP, full decoder-only transformer.

### Building Blocks

The implementation builds up layer by layer:

```
silu(x)                         → x * sigmoid(x)
RMSNorm(x)                     → x / rms(x) * weight
self_attention(Q, K, V, mask)   → softmax(QK^T/√d + mask) V
MultiHeadAttention              → split heads, attend, project
GatedMLP                        → W2(silu(W1(x)) ⊙ W3(x))
TransformerBlock                → norm→attn→residual→norm→mlp→residual
Llama3Simplified                → embed + pos + N×block + norm + output
```

### KV Cache

The key optimization for autoregressive generation. Without caching, generating
N tokens requires O(N²) attention computations. With KV cache, each new token
only computes attention against cached keys/values from prior positions.

```rust
pub struct MultiHeadAttentionKVCache {
    k_cache: Tensor,  // (1, max_seq_len, dim) buffer
    v_cache: Tensor,  // (1, max_seq_len, dim) buffer
    // ...
}
```

The `forward` method has two modes:
- `use_kv_cache=false`: Standard full attention (for the prompt/prefill phase)
- `use_kv_cache=true`: Append new K,V to cache at `seq_pos`, attend against
  the full cache up to the current position

The test verifies cache consistency: `full_forward[:, 3:]` must equal
`prefix_forward + tail_forward` (where prefix fills the cache and tail reads
from it).

### Causal Mask

```rust
fn causal_mask(length: usize) -> Tensor {
    // Upper triangular matrix of -infinity
    // mask[i][j] = -inf if j > i, else 0
}
```

Adding this to attention scores before softmax ensures that position `i` can
only attend to positions `≤ i`. This is what makes the model autoregressive.

---

## HW6: LLM Training Pipeline

**Concepts**: BPE tokenization, binary data format, Adam optimizer, training
loop with loss casting.

### BPE Tokenization

Byte-Pair Encoding is implemented from scratch in four functions:

1. **text_to_corpus**: Split text into words (on space boundaries), count
   frequencies. Each word is a `Vec<String>` of characters.

2. **most_common_pair**: Find the adjacent pair `(a, b)` with highest
   weighted frequency across the corpus.

3. **merge_pair**: Replace all occurrences of `[..., a, b, ...]` with
   `[..., ab, ...]` in-place.

4. **train_bpe**: Iteratively find and merge the most common pair until
   reaching the target vocabulary size.

The encoding/decoding functions apply learned merges to new text.

### Data Pipeline

```
Text file → pretokenize_data → Binary u16 file → DataLoader → (X, Y) batches
```

The binary format stores token IDs as little-endian `u16` values. This is
memory-mappable and fast to load — no parsing overhead at training time.

The DataLoader reads this binary file and yields `(input, target)` pairs where
`target[i] = input[i+1]` (next-token prediction).

### Adam Optimizer

Adam maintains per-parameter exponential moving averages of gradients (first
moment `u`) and squared gradients (second moment `v`):

```
u_t = β₁ * u_{t-1} + (1-β₁) * g
v_t = β₂ * v_{t-1} + (1-β₂) * g²
û = u_t / (1 - β₁^t)        # bias correction
v̂ = v_t / (1 - β₂^t)
θ = θ - lr * û / (√v̂ + ε)
```

The test verifies exact numerical agreement with PyTorch's `optim.Adam` over
multiple steps.

### Training Loop

A critical detail from the Python tests: logits must be cast to `float32`
before computing cross-entropy loss, even if the model outputs `bfloat16`. This
prevents numerical issues in the log-sum-exp computation. The test harness
explicitly checks this with `_checked_cross_entropy`.

---

## HW7: Tool-Use Agent

**Concepts**: Structured generation, tool calling, SFT data preparation,
calculator parsing.

### Calculator

A recursive-descent parser for arithmetic expressions. Supports:
- Binary operators: `+`, `-`, `*`, `/`
- Parentheses for grouping
- Integer and floating-point literals
- Standard operator precedence (PEMDAS)

```rust
pub fn calculator(expr: &str) -> Result<f64, String> {
    // Lexer → Token stream → Recursive descent parser
    // expr → term ((+|-) term)*
    // term → factor ((*|/) factor)*
    // factor → NUMBER | '(' expr ')'
}
```

### Tool-Use Generation

The generation loop extends standard autoregressive generation with a state
machine:

```
GENERATING → produces <TOOL> → EXTRACTING_TOOL_CALL
EXTRACTING_TOOL_CALL → produces </TOOL> → EXECUTING
EXECUTING → runs calculator, injects <RESPONSE>result</RESPONSE> → GENERATING
GENERATING → produces </ANSWER> → DONE
```

This requires the model, tokenizer encode/decode, and special token IDs to all
coordinate. The `generate_with_tools` function orchestrates this loop.

### SFT Data Preparation

Supervised fine-tuning examples mask the question tokens (setting labels to -100
so they don't contribute to the loss) while keeping reasoning and answer tokens
as training targets. This teaches the model to *generate* reasoning, not to
*memorize* questions.

---

## Key Differences from the Python Implementation

### 1. No `dis.check_function_calls`

The Python tests for hw1 use bytecode inspection to verify that
`matrix_vector_product_1` actually calls `vector_inner_product` (not some other
approach). In Rust, there's no runtime bytecode to inspect.

**Solution**: Enforce composition architecturally. If `matmul_1` must use
`vector_inner_product`, make it a required parameter or use the type system to
distinguish implementations. Alternatively, trust the implementation and test
only correctness.

### 2. No operator overloading for Variable (hw2)

Python allows `x * y` to return a new `Variable`. Rust can implement `Mul` for
`Rc<RefCell<Variable>>`, but the ergonomics are poor due to borrowing. The
recommended approach is explicit function calls:

```rust
let z = multiply(&x, &y);  // instead of x * y
```

Or use a builder pattern / expression macro.

### 3. Error handling: Typed tensors vs runtime panics

The Python code uses assertions and lets exceptions propagate. The Rust code
uses two strategies:

- **Dimension mismatches** (hw1): `panic!` / `assert!` — these are programmer
  errors that should never happen in correct code.
- **Shape errors** (hw3+): Burn catches many shape errors at compile time via
  its `Tensor<B, D>` type system (a 2D tensor can't be passed where a 3D tensor
  is expected). Runtime shape mismatches (e.g., incompatible matrix dimensions)
  panic rather than returning Results.

### 4. Random number generation

PyTorch's `manual_seed(0)` produces specific sequences. Rust's `rand` crate
with the same seed will produce *different* numbers. For tests that depend on
specific random values, we either:

- Use deterministic fixed inputs (preferred)
- Accept that numerical values will differ and test properties instead
  (e.g., "loss decreases" rather than "loss equals 1.234")

### 5. No implicit mutation

Python freely mutates `model.parameters()` in-place during `optimizer.step()`.
Rust requires explicit `&mut` references. The optimizer must hold mutable
references or owned copies of parameter tensors, which affects API design.

---

## Testing Philosophy

### What we port from the Python tests

- **Correctness assertions**: "Does `poly_mul` produce the right coefficients?"
  These port directly.
- **Shape assertions**: "Does `model(X)` output the right dimensions?" Direct
  port.
- **Numerical agreement**: "Does our attention match PyTorch's
  `F.scaled_dot_product_attention`?" We pre-compute expected values and
  hard-code them, since we can't call PyTorch at test time.
- **Error handling**: `#[should_panic]` replaces `pytest.raises`.

### What we skip

- **`mugrade.submit()` calls**: These submit answers to a grading server.
  Irrelevant for local testing.
- **Bytecode inspection** (`dis.check_function_calls`): No Rust equivalent.
  Test correctness, not implementation strategy.
- **MNIST download tests**: Marked `#[ignore]` since they require network
  access and large downloads.

### Test tolerances

Floating-point comparisons use explicit epsilon values:

```rust
assert!((result - expected).abs() < 1e-6);       // scalar
assert!(tensor.abs_diff_eq(&expected, 1e-5));     // ndarray
```

For burn tensors, compute max absolute difference:

```rust
let diff: f32 = (a - b).abs().sum().into_scalar();
assert!(diff < 1e-5, "sum abs diff = {diff}");
```

---

## Getting Started

```bash
# Run the simplest homework first
cargo test -p hw0

# Run a specific test
cargo test -p hw1 test_vector_add

# Run all tests (many will panic until implemented)
cargo test --workspace 2>&1 | grep -E "(FAILED|ok)"

# Check that everything compiles without running
cargo check --workspace
```

Implement the functions in `src/lib.rs` for each crate, replacing `todo!()`
with working code. The tests tell you exactly what behavior is expected.

Start with hw0 (zero dependencies, pure logic), then hw1 (introduces ndarray),
then hw2 (the autodiff graph is the most architecturally challenging piece of
pure Rust), and finally hw3+ (which lean on burn's autograd so you can focus
on the ML concepts rather than fighting the borrow checker).
