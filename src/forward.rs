//! `forward` — M2: turn token IDs into next-token **logits** via one forward pass.
//!
//! This is the "it understands" milestone. Given a prompt's token IDs, we run the
//! full Qwen3-0.6B network on the CPU in f32 and read out the logits (one score
//! per vocab entry) for the **last** position — the model's belief about what
//! token comes next. No sampling, no KV cache, no generation loop yet (those are
//! M3/M4); this is a single, clear, deliberately-slow prefill.
//!
//! ## The pass, with shapes (the residual stream is width `H`)
//!
//! ```text
//!   ids[seq] ─gather─▶ x[seq,H] ─┬─▶ block₀ ─▶ … ─▶ block_{L-1} ─▶ RMSNorm ─▶ x[seq,H]
//!                                 │                                              │
//!                                 └────────── residual bus, width H ────────────┘
//!                                                        last row x[H] ─lm_head─▶ logits[V]
//! ```
//!
//! One pre-norm block (learning 10), everything hanging off the residual bus:
//!
//! ```text
//!   x ─RMSNorm(input_ln)─▶ h ─┬─q_proj─▶ q[seq, heads·d] ─┐
//!                             ├─k_proj─▶ k[seq, kv·d] ─────┼─ q/k-norm ─ RoPE ─ causal GQA ─▶ a[seq, heads·d]
//!                             └─v_proj─▶ v[seq, kv·d] ─────┘                          a ─o_proj─▶ [seq,H] ─(+x)─▶ x
//!   x ─RMSNorm(post_attn_ln)─▶ h ─┬─gate_proj─▶ [seq,I] ─SiLU─┐
//!                                 └─up_proj───▶ [seq,I] ──────⊙──▶ [seq,I] ─down_proj─▶ [seq,H] ─(+x)─▶ x
//! ```
//!
//! ## Verification (owed at M2 close)
//! A **layered golden vector** from the HF reference in fp32 — captured at the
//! embedding output, block-0 output, final-norm output, and the logits — so a
//! mismatch bisects to the stage that broke, not just "logits are wrong." Tight
//! tolerance (~1e-4). Oracle is `scripts/gen_forward_golden.py`; Python is only
//! ever the one-shot oracle, never a second engine.
//!
//! 📖 §2.1, §2.2.2–2.2.3 · 🔧 `ds4.c` + `metal/{norm,dsv4_rope,flash_attn,glu,dense,get_rows}.metal`.

#![allow(dead_code, unused_variables, unused_imports)] // scaffold: bodies land helper-by-helper.

use crate::config::Config;
use crate::safetensors::{SafeTensors, bf16_to_f32};
use crate::tensor::Matrix;
use crate::tokenizer::Tokenizer;

// ── Weights: the f32 working copy of the model ─────────────────────────────────
//
// M1's `SafeTensors` keeps tensors as borrowed bf16 byte slices in the mmap. M2
// needs to *compute*, so here we widen each tensor we'll touch into an owned f32
// `Matrix` (2-D projections/embeddings) or `Vec<f32>` (1-D norm scale vectors),
// mirroring the architecture. This is the one place bf16 → f32 happens; after
// this, everything downstream is pure f32.

/// The eleven weights of one transformer block (learning 10), widened to f32.
/// Comments give each tensor's `[out, in]` shape from the config.
pub struct LayerWeights {
    pub input_layernorm: Vec<f32>, // [H]        RMSNorm scale, pre-attention
    pub q_proj: Matrix,            // [heads·d, H]
    pub k_proj: Matrix,            // [kv·d,   H]
    pub v_proj: Matrix,            // [kv·d,   H]
    pub q_norm: Vec<f32>,          // [d]        per-head QK-norm scale (Qwen3)
    pub k_norm: Vec<f32>,          // [d]
    pub o_proj: Matrix,            // [H, heads·d]
    pub post_attention_layernorm: Vec<f32>, // [H]        RMSNorm scale, pre-MLP
    pub gate_proj: Matrix,         // [I, H]
    pub up_proj: Matrix,           // [I, H]
    pub down_proj: Matrix,         // [H, I]
}

/// The whole model's weights in f32: embeddings, `L` blocks, final norm, lm_head.
pub struct Weights {
    pub embed_tokens: Matrix, // [V, H] — the token table (gathered, not multiplied)
    pub layers: Vec<LayerWeights>,
    pub norm: Vec<f32>,  // [H] — final RMSNorm before the head
    pub lm_head: Matrix, // [V, H] — hidden → logits (tied: a copy of embed_tokens)
}

impl Weights {
    /// Widen every tensor we need from `st` into f32, guided by `cfg`.
    ///
    /// Steps:
    /// 1. `embed_tokens` = `matrix_from(st, "model.embed_tokens.weight", [V, H])`.
    /// 2. for `l in 0..L`: pull the eleven `model.layers.{l}.*` tensors, each via
    ///    `matrix_from` / `vector_from` at the shape the config implies.
    /// 3. `norm` = `vector_from(st, "model.norm.weight", H)`.
    /// 4. `lm_head`: load `"lm_head.weight"` if present (Qwen3 ships a tied copy);
    ///    else clone `embed_tokens` (tie_word_embeddings). Either way it's `[V, H]`.
    ///
    /// Any missing/mis-shaped tensor is a loud `ForwardError` — but M1's `inspect`
    /// already cross-checked the file, so in practice this just materializes.
    pub fn load(cfg: &Config, st: &SafeTensors) -> Result<Weights, ForwardError> {
        todo!("materialize embed_tokens, the L blocks, norm, and lm_head as f32")
    }
}

/// Load a 2-D tensor by name and widen bf16 → f32 into a `[expect_rows, expect_cols]`
/// [`Matrix`], asserting the shape matches (fail loud, per the shape invariant).
fn matrix_from(st: &SafeTensors, name: &str, expect: [usize; 2]) -> Result<Matrix, ForwardError> {
    // 1. st.tensor(name) → ForwardError::MissingTensor if absent.
    // 2. check t.shape == expect → ForwardError::ShapeMismatch.
    // 3. widen: for each 2-byte bf16 pair in st.bytes(t), bf16_to_f32 → f32 Vec.
    // 4. Matrix::from_vec(expect[0], expect[1], data).
    todo!("look up, shape-check, widen every bf16 pair to f32, wrap in a Matrix")
}

/// Load a 1-D tensor (a norm scale vector) by name and widen to `Vec<f32>`,
/// asserting length `expect`.
fn vector_from(st: &SafeTensors, name: &str, expect: usize) -> Result<Vec<f32>, ForwardError> {
    todo!("look up, shape-check [expect], widen bf16 → f32 Vec")
}

// ── The ops (PLAN sub-steps; each is independently unit-testable) ──────────────

/// Embedding gather: `ids[seq] → x[seq, H]`, one row per token copied from the
/// table. This is a *row lookup*, not a matmul — `x.row(t) = embed.row(ids[t])`.
/// (The lm_head at the end reuses the *same* table as a matmul; see learning 10.)
pub fn embedding_gather(embed: &Matrix, ids: &[u32]) -> Matrix {
    let mut out = Matrix::zeros(ids.len(), embed.cols);
    for (t, &id) in ids.iter().enumerate() {
        let row = usize::try_from(id).expect("embedding_gather: token id does not fit usize");
        assert!(
            row < embed.rows,
            "embedding_gather: token id {id} outside vocabulary size {}",
            embed.rows
        );
        out.row_mut(t).copy_from_slice(embed.row(row));
    }
    out
}

/// RMSNorm one vector: `y_i = x_i / sqrt(mean(x²) + eps) · w_i`.
///
/// No mean-subtraction (that's LayerNorm); RMSNorm only rescales by the root-mean-
/// square, then applies the learned per-element scale `w`. Used three ways: over
/// `H` (the two block norms + final norm) and over `d` (per-head QK-norm).
/// `x` and `w` must be the same length; compute the sum of squares in f32.
pub fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(
        x.len(),
        w.len(),
        "rms_norm: x len {} != scale len {}",
        x.len(),
        w.len()
    );
    assert!(!x.is_empty(), "rms_norm: vector must not be empty");
    assert!(eps >= 0.0, "rms_norm: eps must be non-negative, got {eps}");

    let mut sum_squares = 0.0;
    for &value in x {
        sum_squares += value * value;
    }
    let mean_square = sum_squares / x.len() as f32;
    let inv_rms = 1.0 / (mean_square + eps).sqrt();
    x.iter()
        .zip(w)
        .map(|(&value, &scale)| value * inv_rms * scale)
        .collect()
}

/// Apply `rms_norm` to every row of a matrix (the common case for the `H`-wide
/// residual stream). Returns a new `[seq, H]` matrix.
pub fn rms_norm_rows(x: &Matrix, w: &[f32], eps: f32) -> Matrix {
    assert_eq!(
        x.cols,
        w.len(),
        "rms_norm_rows: matrix width {} != scale len {}",
        x.cols,
        w.len()
    );
    assert!(x.cols > 0, "rms_norm_rows: matrix width must be positive");
    assert!(
        eps >= 0.0,
        "rms_norm_rows: eps must be non-negative, got {eps}"
    );
    let mut out = Matrix::zeros(x.rows, x.cols);
    for row in 0..x.rows {
        out.row_mut(row)
            .copy_from_slice(&rms_norm(x.row(row), w, eps));
    }
    out
}

/// SiLU (a.k.a. swish): `x · sigmoid(x)`. The SwiGLU gate's activation.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Precomputed RoPE rotation table for a run of positions.
///
/// RoPE rotates each query/key vector by an angle that depends on position, so
/// attention sees *relative* position. Qwen3 uses the HF "rotate-half" convention
/// over the full `head_dim = d`: frequencies `invᵢ = θ^(-2i/d)` for `i in 0..d/2`,
/// and at position `m` the angle for pair `i` is `m·invᵢ`. We store `cos`/`sin`
/// as `[seq, d]` (each half-frequency repeated to fill `d`) so applying RoPE to a
/// head is one elementwise pass. `θ` is `cfg.rope_theta` (1e6 for Qwen3).
pub struct Rope {
    pub cos: Matrix, // [seq, d]
    pub sin: Matrix, // [seq, d]
}

impl Rope {
    /// Build the table for positions `0..seq` at head width `head_dim`, base `theta`.
    pub fn new(seq: usize, head_dim: usize, theta: f64) -> Rope {
        assert!(
            head_dim > 0 && head_dim.is_multiple_of(2),
            "RoPE: head_dim must be positive and even, got {head_dim}"
        );
        let theta = theta as f32;
        assert!(
            theta.is_finite() && theta > 0.0,
            "RoPE: theta must be positive and finite f32, got {theta}"
        );

        let half = head_dim / 2;
        let mut cos = Matrix::zeros(seq, head_dim);
        let mut sin = Matrix::zeros(seq, head_dim);
        for pos in 0..seq {
            for i in 0..half {
                // Match the locked HF fp32 operation order exactly: exponent,
                // power, reciprocal, position product, then trig all stay f32.
                let exponent = (2 * i) as f32 / head_dim as f32;
                let inv_freq = 1.0_f32 / theta.powf(exponent);
                let angle = pos as f32 * inv_freq;
                let c = angle.cos();
                let s = angle.sin();
                // HF's rotate-half convention repeats each half's frequencies.
                cos.data[pos * head_dim + i] = c;
                cos.data[pos * head_dim + half + i] = c;
                sin.data[pos * head_dim + i] = s;
                sin.data[pos * head_dim + half + i] = s;
            }
        }
        Rope { cos, sin }
    }

    /// Rotate one head-vector `v[d]` at position `pos`, in place, using the
    /// rotate-half rule: `out = v·cos + rotate_half(v)·sin`, where
    /// `rotate_half([a | b]) = [-b | a]` (halves of width `d/2`).
    pub fn apply(&self, v: &mut [f32], pos: usize) {
        assert_eq!(
            v.len(),
            self.cos.cols,
            "RoPE::apply: vector width {} != table width {}",
            v.len(),
            self.cos.cols
        );
        assert!(
            pos < self.cos.rows,
            "RoPE::apply: position {pos} outside table length {}",
            self.cos.rows
        );
        let half = v.len() / 2;
        let original = v.to_vec();
        let cos = self.cos.row(pos);
        let sin = self.sin.row(pos);
        for i in 0..half {
            v[i] = original[i] * cos[i] - original[half + i] * sin[i];
            v[half + i] = original[half + i] * cos[half + i] + original[i] * sin[half + i];
        }
    }
}

/// Scaled-dot-product attention for **one** head over the whole prefill, causal.
///
/// `q`,`k`,`v` are `[seq, d]` (this head's rows). For each query position `t`:
/// scores `sⱼ = (q_t · k_j)/√d` for `j ≤ t` (causal: future masked to −∞),
/// `softmax` over `j`, then `out_t = Σⱼ softmaxⱼ · v_j`. Returns `[seq, d]`.
/// This is the "one head first" sub-step; GQA wiring is [`multi_head_attention`].
pub fn attention_one_head(q: &Matrix, k: &Matrix, v: &Matrix) -> Matrix {
    assert_eq!(
        (q.rows, q.cols),
        (k.rows, k.cols),
        "attention_one_head: q shape [{}×{}] != k shape [{}×{}]",
        q.rows,
        q.cols,
        k.rows,
        k.cols
    );
    assert_eq!(
        (q.rows, q.cols),
        (v.rows, v.cols),
        "attention_one_head: q shape [{}×{}] != v shape [{}×{}]",
        q.rows,
        q.cols,
        v.rows,
        v.cols
    );
    assert!(
        q.cols > 0,
        "attention_one_head: head width must be positive"
    );

    let seq = q.rows;
    let d = q.cols;
    let scale = 1.0 / (d as f32).sqrt();
    let mut out = Matrix::zeros(seq, d);

    for t in 0..seq {
        // Future positions never enter the score vector: this is the causal mask
        // made structural rather than represented by allocated -∞ entries.
        let mut scores = Vec::with_capacity(t + 1);
        for j in 0..=t {
            let mut dot = 0.0;
            for i in 0..d {
                dot += q.row(t)[i] * k.row(j)[i];
            }
            scores.push(dot * scale);
        }

        // Stable softmax: subtracting the maximum preserves the probabilities
        // while preventing exp(large score) from overflowing.
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut denominator = 0.0;
        for score in &mut scores {
            *score = (*score - max_score).exp();
            denominator += *score;
        }

        for (j, numerator) in scores.iter().enumerate() {
            let weight = numerator / denominator;
            for i in 0..d {
                out.data[t * d + i] += weight * v.row(j)[i];
            }
        }
    }
    out
}

/// Full grouped-query attention for one block: projections → QK-norm → RoPE →
/// per-head causal attention (GQA: `gqa_group` query heads share each kv head) →
/// concat → `o_proj`. Input/òutput both ride the residual bus at width `H`.
///
/// Steps (with `hn = num_attention_heads`, `kvn = num_key_value_heads`, `d`):
/// 1. `q = h.linear(q_proj)` → `[seq, hn·d]`; `k,v = h.linear(k/v_proj)` → `[seq, kvn·d]`.
/// 2. reshape into heads; **RMSNorm each q head by `q_norm`, each k head by `k_norm`**
///    (Qwen3's per-head QK-norm, over width `d`), then **RoPE** each q/k head.
/// 3. for each query head `hd`, its kv head is `hd / gqa_group`; run
///    [`attention_one_head`] on that (q head, shared k head, shared v head).
/// 4. concat head outputs → `[seq, hn·d]`, then `.linear(o_proj)` → `[seq, H]`.
pub fn multi_head_attention(h: &Matrix, layer: &LayerWeights, cfg: &Config, rope: &Rope) -> Matrix {
    todo!("project, qk-norm, rope, per-head causal attention with GQA sharing, o_proj")
}

/// SwiGLU feed-forward for one block: `down( SiLU(gate(h)) ⊙ up(h) )`.
/// `h` is `[seq, H]`; gate/up lift to `[seq, I]`, elementwise gated, down projects
/// back to `[seq, H]`. Returns the `[seq, H]` contribution to the residual stream.
pub fn swiglu_ffn(h: &Matrix, layer: &LayerWeights) -> Matrix {
    // gate = h.linear(gate_proj); up = h.linear(up_proj)
    // act[t,i] = silu(gate[t,i]) * up[t,i]; then act.linear(down_proj)
    todo!("gate/up projections, SiLU-gated elementwise product, down projection")
}

/// One pre-norm transformer block: `x → x + attn(norm₁(x)) → x + ffn(norm₂(x))`.
/// Both sub-layers read a normed copy of the bus and add their result *back* onto
/// the un-normed bus (the residual connection). Returns the updated `[seq, H]`.
pub fn transformer_block(x: &Matrix, layer: &LayerWeights, cfg: &Config, rope: &Rope) -> Matrix {
    // let a = multi_head_attention(rms_norm_rows(x, input_layernorm), …); x = x + a
    // let f = swiglu_ffn(rms_norm_rows(x, post_attention_layernorm)); x = x + f
    todo!("attention residual, then MLP residual, both pre-normed")
}

/// The full forward pass: `ids → logits[V]` for the **last** position.
///
/// Steps:
/// 1. `x = embedding_gather(embed_tokens, ids)` → `[seq, H]`.
/// 2. `rope = Rope::new(seq, head_dim, rope_theta)`.
/// 3. fold every block: `x = transformer_block(x, layer, …)`.
/// 4. `x = rms_norm_rows(x, norm, eps)` (final norm).
/// 5. logits = last row `x[seq-1]` `.linear(lm_head)` → `[V]`. (We only need the
///    last position; projecting the whole `[seq, V]` would be `seq×` the work.)
pub fn forward(weights: &Weights, cfg: &Config, ids: &[u32]) -> Vec<f32> {
    todo!("gather → blocks → final norm → lm_head on the last position")
}

/// Top-`k` `(token_id, logit)` pairs, highest first — for the CLI to show the
/// model's ranked next-token guesses. A partial sort is fine (k ≪ V), but M2 can
/// start with a full sort for clarity.
pub fn top_k(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    assert!(
        logits.len() <= u32::MAX as usize,
        "top_k: {} logits do not fit u32 token ids",
        logits.len()
    );
    assert!(
        logits.iter().all(|value| value.is_finite()),
        "top_k: logits must be finite"
    );
    let mut ranked: Vec<(u32, f32)> = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(id, logit)| (id as u32, logit))
        .collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .expect("top_k: finiteness checked above")
            .then_with(|| a.0.cmp(&b.0))
    });
    ranked.truncate(k.min(ranked.len()));
    ranked
}

/// `fs logits <TEXT>` end to end.
///
/// Steps:
/// 1. `Tokenizer::load(model_dir)` → `encode(text)` → `ids`.
/// 2. `Config::load` + `SafeTensors::load` + `Weights::load` (the f32 copy).
/// 3. `forward(&weights, &cfg, &ids)` → `logits[V]`.
/// 4. `top_k(&logits, k)`; print each as `id  logit  «decoded piece»` (decode the
///    single id via the tokenizer so the guess is human-readable).
pub fn run(model_dir: &str, text: &str, k: usize) -> Result<(), ForwardError> {
    todo!("tokenize → load weights → forward → print top-k next tokens")
}

/// Everything `fs logits` can fail on: loading either half of the model, a tensor
/// we expected but couldn't find/shape, or tokenization.
#[derive(Debug)]
pub enum ForwardError {
    Config(crate::config::ConfigError),
    SafeTensors(crate::safetensors::SafeTensorsError),
    Tokenizer(crate::tokenizer::TokenizerError),
    /// A tensor the forward pass needs is absent from the file.
    MissingTensor {
        name: String,
    },
    /// A tensor is present but not the shape the config implies.
    ShapeMismatch {
        name: String,
        got: Vec<usize>,
        expected: Vec<usize>,
    },
}

impl std::fmt::Display for ForwardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ForwardError::Config(e) => write!(f, "{e}"),
            ForwardError::SafeTensors(e) => write!(f, "{e}"),
            ForwardError::Tokenizer(e) => write!(f, "{e}"),
            ForwardError::MissingTensor { name } => {
                write!(f, "weight tensor '{name}' missing from file")
            }
            ForwardError::ShapeMismatch {
                name,
                got,
                expected,
            } => {
                write!(
                    f,
                    "weight tensor '{name}' has shape {got:?}, expected {expected:?}"
                )
            }
        }
    }
}

impl std::error::Error for ForwardError {}

impl From<crate::config::ConfigError> for ForwardError {
    fn from(e: crate::config::ConfigError) -> Self {
        ForwardError::Config(e)
    }
}
impl From<crate::safetensors::SafeTensorsError> for ForwardError {
    fn from(e: crate::safetensors::SafeTensorsError) -> Self {
        ForwardError::SafeTensors(e)
    }
}
impl From<crate::tokenizer::TokenizerError> for ForwardError {
    fn from(e: crate::tokenizer::TokenizerError) -> Self {
        ForwardError::Tokenizer(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(got: f32, want: f32) {
        assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
    }

    #[test]
    fn embedding_gather_copies_rows_in_token_order() {
        let embed = Matrix::from_vec(
            4,
            3,
            vec![0., 1., 2., 10., 11., 12., 20., 21., 22., 30., 31., 32.],
        );
        let got = embedding_gather(&embed, &[2, 0, 2]);
        assert_eq!(got.rows, 3);
        assert_eq!(got.cols, 3);
        assert_eq!(got.data, vec![20., 21., 22., 0., 1., 2., 20., 21., 22.]);
    }

    #[test]
    #[should_panic(expected = "outside vocabulary size")]
    fn embedding_gather_rejects_unknown_token_id() {
        embedding_gather(&Matrix::zeros(2, 3), &[2]);
    }

    #[test]
    fn rms_norm_matches_a_known_vector_and_scale() {
        // Locked Torch fp32 result. Nonzero epsilon catches omission/placement.
        let got = rms_norm(&[3.0, 4.0], &[2.0, 0.5], 1e-6);
        assert_close(got[0], 1.697_056_2);
        assert_close(got[1], 0.565_685_4);
    }

    #[test]
    fn rms_norm_rows_preserves_the_matrix_shape() {
        let x = Matrix::from_vec(2, 2, vec![3., 4., 0., 5.]);
        let got = rms_norm_rows(&x, &[1., 1.], 0.0);
        assert_eq!((got.rows, got.cols), (2, 2));
        assert_close(got.data[0], 0.848_528_15);
        assert_close(got.data[1], 1.131_370_9);
        assert_close(got.data[2], 0.0);
        assert_close(got.data[3], 2.0_f32.sqrt());
    }

    #[test]
    fn silu_known_values() {
        assert_eq!(silu(0.0), 0.0);
        assert_close(silu(1.0), 0.731_058_6);
        assert_close(silu(-1.0), -0.268_941_43);
    }

    #[test]
    fn rope_position_zero_is_identity() {
        let rope = Rope::new(2, 4, 10_000.0);
        let mut v = [1., 2., 3., 4.];
        rope.apply(&mut v, 0);
        assert_eq!(v, [1., 2., 3., 4.]);
    }

    #[test]
    fn rope_rotate_half_matches_the_formula_and_preserves_norm() {
        let rope = Rope::new(2, 4, 10_000.0);
        let original = [1., 2., 3., 4.];
        let mut got = original;
        rope.apply(&mut got, 1);

        assert_close(got[0], 1.0_f32.cos() - 3.0 * 1.0_f32.sin());
        assert_close(got[1], 2.0 * 0.01_f32.cos() - 4.0 * 0.01_f32.sin());
        assert_close(got[2], 3.0 * 1.0_f32.cos() + 1.0_f32.sin());
        assert_close(got[3], 4.0 * 0.01_f32.cos() + 2.0 * 0.01_f32.sin());
        let before: f32 = original.iter().map(|x| x * x).sum();
        let after: f32 = got.iter().map(|x| x * x).sum();
        assert_close(after, before);
    }

    #[test]
    fn rope_table_matches_hf_fp32_at_the_real_context_edge() {
        // Locked Torch 2.13 fp32 Qwen3 values: d=128, theta=1e6, last legal
        // position 40959. This catches accidentally computing the table in f64.
        let rope = Rope::new(40_960, 128, 1_000_000.0);
        let cos = rope.cos.row(40_959);
        let sin = rope.sin.row(40_959);
        assert_close(cos[0], 0.466_897_22);
        assert_close(sin[0], -0.884_311_6);
        assert_close(cos[31], 0.846_143_36);
        assert_close(sin[31], 0.532_955_3);
        assert_close(cos[63], 0.998_708_55);
        assert_close(sin[63], 0.050_805_688);
        assert_eq!(cos[0], cos[64]);
        assert_eq!(sin[63], sin[127]);
    }

    #[test]
    fn attention_one_head_is_scaled_and_causal() {
        // At t=0 only v₀ is visible. At t=1 the scores are [0, 1/√2], so this
        // simultaneously locks the causal prefix, √d scaling, softmax, and the
        // weighted value sum on a two-token non-square-valued example.
        let q = Matrix::from_vec(2, 2, vec![1., 0., 0., 1.]);
        let k = Matrix::from_vec(2, 2, vec![1., 0., 0., 1.]);
        let v = Matrix::from_vec(2, 2, vec![10., 0., 0., 20.]);

        let got = attention_one_head(&q, &k, &v);

        assert_eq!((got.rows, got.cols), (2, 2));
        assert_eq!(got.row(0), &[10., 0.]);
        assert_close(got.row(1)[0], 3.302_384_6);
        assert_close(got.row(1)[1], 13.395_231);
    }

    #[test]
    #[should_panic(expected = "q shape [2×2] != k shape [1×2]")]
    fn attention_one_head_rejects_mismatched_shapes() {
        attention_one_head(
            &Matrix::zeros(2, 2),
            &Matrix::zeros(1, 2),
            &Matrix::zeros(2, 2),
        );
    }

    #[test]
    fn top_k_sorts_descending_and_breaks_ties_by_token_id() {
        assert_eq!(
            top_k(&[0.5, 3.0, -1.0, 3.0], 3),
            vec![(1, 3.0), (3, 3.0), (0, 0.5)]
        );
        assert_eq!(top_k(&[1.0], 10), vec![(0, 1.0)]);
        assert!(top_k(&[1.0], 0).is_empty());
        assert_eq!(top_k(&[-0.0, 0.0], 2), vec![(0, -0.0), (1, 0.0)]);
    }
}
