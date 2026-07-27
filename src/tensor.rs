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

use std::fmt;

/// A row-major matrix of `f32`, `data.len() == rows * cols`.
///
/// Element `(r, c)` lives at `data[r * cols + c]`. We expose whole *rows* as
/// slices (`row`/`row_mut`) because every op we write walks a row at a time — a
/// token's activation vector, or one output neuron's weights — which is also the
/// cache-friendly access pattern.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub(crate) data: Vec<f32>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

impl Matrix {
    /// A `rows × cols` matrix of zeros.
    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        let len = rows
            .checked_mul(cols)
            .expect("Matrix::zeros: rows × cols overflow");
        Matrix {
            data: vec![0.0; len],
            rows,
            cols,
        }
    }

    /// Wrap an existing flat buffer, asserting it is exactly `rows * cols` long.
    /// The assert is the whole point — it turns a shape bug into a loud panic at
    /// construction rather than an out-of-bounds read later.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Matrix {
        let len = rows
            .checked_mul(cols)
            .expect("Matrix::from_vec: rows × cols overflow");
        assert_eq!(
            data.len(),
            len,
            "Matrix::from_vec: {rows}×{cols} needs {} elems, got {}",
            len,
            data.len()
        );
        Matrix { data, rows, cols }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Row `r` as a contiguous slice of `cols` elements.
    pub fn row(&self, r: usize) -> &[f32] {
        assert!(
            r < self.rows,
            "Matrix::row: row {r} outside {} rows",
            self.rows
        );
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    /// Row `r` as a mutable contiguous slice (for in-place ops like RoPE).
    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        assert!(
            r < self.rows,
            "Matrix::row_mut: row {r} outside {} rows",
            self.rows
        );
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
        assert_eq!(
            self.cols, other.rows,
            "matmul: inner dims must match: [{}×{}] · [{}×{}]",
            self.rows, self.cols, other.rows, other.cols
        );
        let mut out = Matrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    // Row-major offsets: A[i,k] = i·K+k; B[k,j] = k·N+j.
                    sum += self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
                out.data[i * other.cols + j] = sum;
            }
        }
        out
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
        assert_eq!(
            self.cols, w.cols,
            "linear: x cols ({}) must equal W in-dim ({}); W is [out={}, in={}]",
            self.cols, w.cols, w.rows, w.cols
        );
        let mut out = Matrix::zeros(self.rows, w.rows);
        for t in 0..self.rows {
            for o in 0..w.rows {
                let x_row = self.row(t);
                let w_row = w.row(o);
                let mut sum = 0.0;
                for i in 0..self.cols {
                    sum += x_row[i] * w_row[i];
                }
                out.data[t * w.rows + o] = sum;
            }
        }
        out
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
    #[should_panic(expected = "rows × cols overflow")]
    fn matrix_dimensions_cannot_overflow() {
        Matrix::zeros(usize::MAX, 2);
    }

    #[test]
    fn matmul_small_known_answer() {
        // [1 2 3; 4 5 6] · [7 8; 9 10; 11 12] = [58 64; 139 154]
        let a = Matrix::from_vec(2, 3, vec![1., 2., 3., 4., 5., 6.]);
        let b = Matrix::from_vec(3, 2, vec![7., 8., 9., 10., 11., 12.]);
        let c = a.matmul(&b);
        assert_eq!(c.data, vec![58., 64., 139., 154.]);
    }

    #[test]
    fn linear_is_matmul_against_transposed_weight() {
        // x[seq=2,in=3] · Wᵀ where W=[out=2,in=3]. Two rows and a non-square
        // weight make both the output shape and flat row indexing observable.
        let x = Matrix::from_vec(2, 3, vec![1., 2., 3., 4., 5., 6.]);
        let w = Matrix::from_vec(2, 3, vec![1., 0., -1., 2., 1., 0.]);
        let y = x.linear(&w);
        assert_eq!((y.rows, y.cols), (2, 2));
        assert_eq!(y.data, vec![-2., 4., -2., 13.]);
    }

    #[test]
    #[should_panic(expected = "matmul: inner dims must match")]
    fn matmul_rejects_mismatched_inner_dims() {
        Matrix::zeros(2, 3).matmul(&Matrix::zeros(2, 4));
    }

    #[test]
    #[should_panic(expected = "linear: x cols")]
    fn linear_rejects_wrong_input_width() {
        Matrix::zeros(2, 3).linear(&Matrix::zeros(4, 2));
    }

    #[test]
    #[should_panic(expected = "Matrix::row: row 3 outside 3 rows")]
    fn zero_width_matrix_still_checks_row_bounds() {
        Matrix::zeros(3, 0).row(3);
    }
}
