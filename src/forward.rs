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
//! tolerance (~1e-4). Oracle is `scripts/gen_golden.py`; Python is only ever the
//! one-shot oracle, never a second engine.
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
    pub input_layernorm: Vec<f32>,          // [H]        RMSNorm scale, pre-attention
    pub q_proj: Matrix,                     // [heads·d, H]
    pub k_proj: Matrix,                     // [kv·d,   H]
    pub v_proj: Matrix,                     // [kv·d,   H]
    pub q_norm: Vec<f32>,                   // [d]        per-head QK-norm scale (Qwen3)
    pub k_norm: Vec<f32>,                   // [d]
    pub o_proj: Matrix,                     // [H, heads·d]
    pub post_attention_layernorm: Vec<f32>, // [H]        RMSNorm scale, pre-MLP
    pub gate_proj: Matrix,                  // [I, H]
    pub up_proj: Matrix,                    // [I, H]
    pub down_proj: Matrix,                  // [H, I]
}

/// The whole model's weights in f32: embeddings, `L` blocks, final norm, lm_head.
pub struct Weights {
    pub embed_tokens: Matrix, // [V, H] — the token table (gathered, not multiplied)
    pub layers: Vec<LayerWeights>,
    pub norm: Vec<f32>, // [H] — final RMSNorm before the head
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
    todo!("allocate [seq, H]; copy embed.row(id) into row t for each token")
}

/// RMSNorm one vector: `y_i = x_i / sqrt(mean(x²) + eps) · w_i`.
///
/// No mean-subtraction (that's LayerNorm); RMSNorm only rescales by the root-mean-
/// square, then applies the learned per-element scale `w`. Used three ways: over
/// `H` (the two block norms + final norm) and over `d` (per-head QK-norm).
/// `x` and `w` must be the same length; compute the sum of squares in f32.
pub fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), w.len(), "rms_norm: x len {} != scale len {}", x.len(), w.len());
    // ms = Σ x_i² / n ; inv = 1/sqrt(ms + eps) ; y_i = x_i · inv · w_i
    todo!("mean of squares → rsqrt → scale each element by inv·w_i")
}

/// Apply `rms_norm` to every row of a matrix (the common case for the `H`-wide
/// residual stream). Returns a new `[seq, H]` matrix.
pub fn rms_norm_rows(x: &Matrix, w: &[f32], eps: f32) -> Matrix {
    todo!("map rms_norm over each row")
}

/// SiLU (a.k.a. swish): `x · sigmoid(x)`. The SwiGLU gate's activation.
pub fn silu(x: f32) -> f32 {
    todo!("x / (1 + e^-x)")
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
        // invᵢ = theta^(-(2i)/head_dim), i in 0..head_dim/2
        // cos[m, i] = cos(m·invᵢ) (and repeated for the second half); sin likewise.
        todo!("fill cos/sin [seq, head_dim] from the position × inv-freq outer product")
    }

    /// Rotate one head-vector `v[d]` at position `pos`, in place, using the
    /// rotate-half rule: `out = v·cos + rotate_half(v)·sin`, where
    /// `rotate_half([a | b]) = [-b | a]` (halves of width `d/2`).
    pub fn apply(&self, v: &mut [f32], pos: usize) {
        todo!("elementwise v·cos[pos] + rotate_half(v)·sin[pos]")
    }
}

/// Scaled-dot-product attention for **one** head over the whole prefill, causal.
///
/// `q`,`k`,`v` are `[seq, d]` (this head's rows). For each query position `t`:
/// scores `sⱼ = (q_t · k_j)/√d` for `j ≤ t` (causal: future masked to −∞),
/// `softmax` over `j`, then `out_t = Σⱼ softmaxⱼ · v_j`. Returns `[seq, d]`.
/// This is the "one head first" sub-step; GQA wiring is [`multi_head_attention`].
pub fn attention_one_head(q: &Matrix, k: &Matrix, v: &Matrix) -> Matrix {
    let d = q.cols;
    // for t in 0..seq: scores over j in 0..=t, scale 1/√d, softmax, weighted sum of v
    todo!("causal scaled-dot-product for a single head")
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
    todo!("argsort logits desc, take k, pair with token id")
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
    MissingTensor { name: String },
    /// A tensor is present but not the shape the config implies.
    ShapeMismatch { name: String, got: Vec<usize>, expected: Vec<usize> },
}

impl std::fmt::Display for ForwardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ForwardError::Config(e) => write!(f, "{e}"),
            ForwardError::SafeTensors(e) => write!(f, "{e}"),
            ForwardError::Tokenizer(e) => write!(f, "{e}"),
            ForwardError::MissingTensor { name } => write!(f, "weight tensor '{name}' missing from file"),
            ForwardError::ShapeMismatch { name, got, expected } => {
                write!(f, "weight tensor '{name}' has shape {got:?}, expected {expected:?}")
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

    // Op-level unit tests land alongside each helper as we implement it (M0/M1
    // cadence): rms_norm against a hand-computed value, embedding_gather row copy,
    // Rope orthogonality, attention_one_head on a 2-token toy, silu/top_k. The
    // end-to-end correctness check is the layered golden vector (see module doc),
    // gated on the real assets like the other reality tests.
    #[test]
    #[ignore = "scaffold: forward ops are todo!()"]
    fn placeholder() {}
}
