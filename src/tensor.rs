//! `tensor` — the one numeric container M2 computes over: a row-major `Matrix`.
//!
//! We deliberately keep this *small and transparent* (the `ds4` ethos). A
//! [`Matrix`] is a flat `Vec<f32>` plus its `rows`/`cols` — nothing more. There
//! are **no strides**: the layout is always contiguous, row-major, stride
//! `(cols, 1)`. That's a choice (see the M2 design dialogue in `PROGRESS.md`):
//! a strided N-D tensor would give free transposes/views, but it hides the
//! indexing arithmetic we want *visible* while learning. When we later need a
//! transpose for speed, it becomes an explicit, measurable copy — a teaching
//! moment, not stride magic. How the flat blob is indexed (`row*cols + col`) is
//! itself the lesson in [`docs/learnings/08-row-major-strides.md`].
//!
//! **Everything here is f32.** The weights are bf16 on disk, but M2 widens them
//! to f32 on load (see [`crate::forward`]) and computes entirely in f32 —
//! clearest, and it lets us match an fp32 oracle to tight tolerance. bf16-compute
//! is a later memory/speed lesson, not a correctness one.
//!
//! **Shapes fail loudly.** Every op here asserts its dimension contract before
//! touching memory (the standing shape-clarity invariant), so a mis-shaped matmul
//! panics *at the call* with the offending dims — not as silent garbage in logits.

#![allow(dead_code)] // scaffold: bodies land helper-by-helper (M0/M1 cadence).

use std::fmt;

/// A row-major matrix of `f32`, `data.len() == rows * cols`.
///
/// Element `(r, c)` lives at `data[r * cols + c]`. We expose whole *rows* as
/// slices (`row`/`row_mut`) because every op we write walks a row at a time — a
/// token's activation vector, or one output neuron's weights — which is also the
/// cache-friendly access pattern.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Matrix {
    /// A `rows × cols` matrix of zeros.
    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix { data: vec![0.0; rows * cols], rows, cols }
    }

    /// Wrap an existing flat buffer, asserting it is exactly `rows * cols` long.
    /// The assert is the whole point — it turns a shape bug into a loud panic at
    /// construction rather than an out-of-bounds read later.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Matrix {
        assert_eq!(data.len(), rows * cols, "Matrix::from_vec: {rows}×{cols} needs {} elems, got {}", rows * cols, data.len());
        Matrix { data, rows, cols }
    }

    /// Row `r` as a contiguous slice of `cols` elements.
    pub fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    /// Row `r` as a mutable contiguous slice (for in-place ops like RoPE).
    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        &mut self.data[r * self.cols..(r + 1) * self.cols]
    }

    /// Textbook matmul: `self[m,k] · other[k,n] → [m,n]`.
    ///
    /// This is the M2 primitive (PLAN sub-step 1). Naive triple loop — clarity
    /// first; loop-tiling / SIMD / threads are a *later* speed lesson and don't
    /// change this signature. We use it where both operands are genuine matrices
    /// (e.g. attention's `scores · V`); weight projections go through [`linear`]
    /// instead, because weights are stored transposed.
    ///
    /// [`linear`]: Matrix::linear
    pub fn matmul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "matmul: inner dims must match: [{}×{}] · [{}×{}]", self.rows, self.cols, other.rows, other.cols);
        // for i in 0..m: for j in 0..n: out[i,j] = Σ_k self[i,k]·other[k,j]
        todo!("naive triple loop over (m, n, k), accumulating in an f32 sum")
    }

    /// A Linear layer's forward: `y = x · Wᵀ`, where `W` is stored `[out, in]`.
    ///
    /// This is the shape convention learning 05 nailed down: a weight row is one
    /// **output** neuron's `in` incoming weights, laid out contiguously. So
    /// `y[t, o] = Σ_in x[t, in] · W[o, in]` is a dot of an `x` row with a `W` row —
    /// both contiguous, no physical transpose needed. `self` is `x[seq, in]`,
    /// `w` is `[out, in]`, result is `[seq, out]`.
    ///
    /// (No bias: Qwen3's projections are bias-free.)
    pub fn linear(&self, w: &Matrix) -> Matrix {
        assert_eq!(self.cols, w.cols, "linear: x cols ({}) must equal W in-dim ({}); W is [out={}, in={}]", self.cols, w.cols, w.rows, w.cols);
        // out is [seq, out]; out[t,o] = dot(self.row(t), w.row(o))
        todo!("for each token row t, for each output neuron o: dot the two contiguous rows")
    }
}

/// Shape-first `Debug`-lite display: `Matrix[rows×cols]`. Keeps shapes visible in
/// logs/`dbg!` without dumping a million floats.
impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Matrix[{}×{}]", self.rows, self.cols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_vec_enforces_shape() {
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(m.row(0), &[1.0, 2.0, 3.0]);
        assert_eq!(m.row(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    #[should_panic]
    fn from_vec_wrong_len_panics() {
        Matrix::from_vec(2, 3, vec![1.0, 2.0]); // needs 6
    }

    #[test]
    #[ignore = "scaffold: matmul is todo!()"]
    fn matmul_small_known_answer() {
        // [1 2 3; 4 5 6] · [7 8; 9 10; 11 12] = [58 64; 139 154]
        let a = Matrix::from_vec(2, 3, vec![1., 2., 3., 4., 5., 6.]);
        let b = Matrix::from_vec(3, 2, vec![7., 8., 9., 10., 11., 12.]);
        let c = a.matmul(&b);
        assert_eq!(c.data, vec![58., 64., 139., 154.]);
    }

    #[test]
    #[ignore = "scaffold: linear is todo!()"]
    fn linear_is_matmul_against_transposed_weight() {
        // x[1×2] · Wᵀ where W=[out=2, in=2] = [[1,2],[3,4]]
        // y[0] = dot([1,1],[1,2]) = 3 ; y[1] = dot([1,1],[3,4]) = 7
        let x = Matrix::from_vec(1, 2, vec![1., 1.]);
        let w = Matrix::from_vec(2, 2, vec![1., 2., 3., 4.]);
        assert_eq!(x.linear(&w).data, vec![3., 7.]);
    }
}
