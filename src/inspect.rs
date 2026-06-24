//! `inspect` — the `fs inspect` command: load a model, prove its tensor set lines
//! up with its config, and print a shape-first table.
//!
//! This is M1's verification *and* its teaching artifact. It ties the two halves
//! together — [`crate::config::Config`] (the architecture) and
//! [`crate::safetensors::SafeTensors`] (the weights) — and answers one question:
//! **does the file contain exactly the tensors, with exactly the shapes, that the
//! architecture implies?** Because the expected tensor set is *derivable* from the
//! config, that check is nearly free, and it's stronger than a checksum: it
//! catches a mis-shaped projection (the `[2048,1024]` vs `[1024,1024]` family of
//! bugs) the moment we load, not deep in M2's forward pass.
//!
//! The whole presentation follows [`docs/learnings/05-reading-shapes.md`]: a
//! dimension legend first, then a grouped table whose last column is the
//! `in ──▶ out` arrow, then the cross-check verdict.

#![allow(dead_code)] // scaffold: remove once `run` is wired.

use crate::config::Config;
use crate::safetensors::SafeTensors;

/// The expected name + shape of one tensor, derived from the config. `optional`
/// marks tensors that legitimately may be absent (e.g. `lm_head.weight` when
/// `tie_word_embeddings` is set).
pub struct Expected {
    pub name: String,
    pub shape: Vec<usize>,
    pub optional: bool,
}

/// The result of comparing the file against the config.
pub struct CrossCheck {
    pub expected_count: usize,
    pub found_count: usize,
    /// One human-readable line per problem, each naming the offending dimension —
    /// e.g. "layers.3.self_attn.q_proj: shape [1024,1024], expected [2048,1024]".
    pub problems: Vec<String>,
    pub total_params: usize,
}

impl CrossCheck {
    /// Clean iff nothing mismatched and the counts agree.
    pub fn ok(&self) -> bool {
        self.problems.is_empty() && self.expected_count == self.found_count
    }
}

/// Build the full set of tensors the architecture implies, with their shapes.
///
/// This function *is* the spec from learning 05, written as code. Every entry is
/// an `in ──▶ out` relationship made literal:
///
/// - global: `model.embed_tokens.weight = [V, H]`, `model.norm.weight = [H]`,
///   and `lm_head.weight = [V, H]` **only if not tied**.
/// - per layer `i` in `0..L`, prefix `model.layers.{i}.`:
///     - `input_layernorm.weight              = [H]`
///     - `self_attn.q_proj.weight             = [q_width,  H]`   (H ─▶ heads·d)
///     - `self_attn.k_proj.weight             = [kv_width, H]`   (H ─▶ kv·d)
///     - `self_attn.v_proj.weight             = [kv_width, H]`
///     - `self_attn.q_norm.weight             = [head_dim]`      (per-head scale)
///     - `self_attn.k_norm.weight             = [head_dim]`
///     - `self_attn.o_proj.weight             = [H, q_width]`    (heads·d ─▶ H)
///     - `post_attention_layernorm.weight     = [H]`
///     - `mlp.gate_proj.weight                = [I, H]`          (H ─▶ I)
///     - `mlp.up_proj.weight                  = [I, H]`
///     - `mlp.down_proj.weight                = [H, I]`          (I ─▶ H)
pub fn expected_tensors(cfg: &Config) -> Vec<Expected> {
    let _ = cfg;
    todo!("emit the global + per-layer expected (name, shape) set from cfg")
}

/// Compare the file's tensors against [`expected_tensors`].
///
/// Steps:
/// 1. `let want = expected_tensors(cfg);`
/// 2. for each `want`: find it in `st`; record a problem if missing (and not
///    `optional`) or if `tensor.shape != want.shape` (name the dims).
/// 3. record any tensor present in `st` but not in `want` (an "extra").
/// 4. sum `num_elements()` over the file's tensors for `total_params`.
pub fn cross_check(cfg: &Config, st: &SafeTensors) -> CrossCheck {
    let _ = (cfg, st);
    todo!("diff expected vs. actual; collect problems; total the params")
}

/// Render the dimension legend — the named dims every shape is built from
/// (`V/H/L/d`, q/kv head counts and their widths, `I`), plus the `[out, in]`
/// convention reminder. See learning 05 for the exact layout.
pub fn render_legend(cfg: &Config) -> String {
    let _ = cfg;
    todo!("format the V/H/L/d/heads/I legend + the [out,in] convention note")
}

/// Render the grouped tensor table: the global embedding, one labelled
/// `× L` block, the final norm, and the tied `lm_head` line — each row showing
/// `dtype`, `[out, in]` shape, param count, and the `in ──▶ out` arrow.
///
/// We print *one* representative block (they're identical across layers) and label
/// it `× {L}` rather than dumping 28 copies — readability over completeness.
pub fn render_table(cfg: &Config, st: &SafeTensors) -> String {
    let _ = (cfg, st);
    todo!("format the grouped, aligned table with the in──▶out column")
}

/// Render the cross-check verdict: counts, any problems (each naming the dim that
/// didn't line up), total params, and the embeddings' share.
pub fn render_verdict(xc: &CrossCheck) -> String {
    let _ = xc;
    todo!("format ✓/✗, the problem lines, and the param totals")
}

/// `fs inspect <model_dir>` end to end.
///
/// Steps:
/// 1. `Config::load(model_dir)?` and `SafeTensors::load(<dir>/model.safetensors)?`.
/// 2. print `render_legend`, `render_table`, then `render_verdict`.
/// 3. return whether the cross-check was clean (the CLI maps that to its exit
///    code — a shape mismatch is a *failure*, not a warning).
pub fn run(model_dir: &str) -> Result<bool, InspectError> {
    let _ = model_dir;
    todo!("load config + safetensors, cross-check, print legend/table/verdict")
}

/// Errors surfaced by `fs inspect`: either half can fail to load.
#[derive(Debug)]
pub enum InspectError {
    Config(crate::config::ConfigError),
    SafeTensors(crate::safetensors::SafeTensorsError),
}

impl std::fmt::Display for InspectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InspectError::Config(e) => write!(f, "{e}"),
            InspectError::SafeTensors(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for InspectError {}

impl From<crate::config::ConfigError> for InspectError {
    fn from(e: crate::config::ConfigError) -> Self {
        InspectError::Config(e)
    }
}

impl From<crate::safetensors::SafeTensorsError> for InspectError {
    fn from(e: crate::safetensors::SafeTensorsError) -> Self {
        InspectError::SafeTensors(e)
    }
}
