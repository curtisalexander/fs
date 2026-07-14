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

use crate::config::Config;
use crate::safetensors::SafeTensors;
use std::collections::HashSet;

/// The expected name + shape of one tensor, derived from the config. `optional`
/// marks tensors that legitimately may be **absent** from the file.
///
/// The one optional tensor is `lm_head.weight`. `tie_word_embeddings: true` means
/// the output projection *is* the embedding table (same weights, mathematically),
/// so a file is free to omit `lm_head.weight` and reuse `embed_tokens`. But "tied"
/// is a statement about the math, **not** a promise about the file: Qwen3-0.6B is
/// tied and *still ships* a byte-identical `lm_head.weight` copy. So when tied,
/// `lm_head.weight` may be either absent or a redundant duplicate — both are fine,
/// hence `optional`. When *not* tied it is required (real, distinct weights).
#[derive(Debug)]
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
    /// A problem means the file disagrees with the architecture — a *failure*.
    pub problems: Vec<String>,
    /// Non-fatal observations worth surfacing in the verdict but not failing on —
    /// e.g. "lm_head.weight present but tied: a redundant copy of embed_tokens".
    pub notes: Vec<String>,
    /// Params physically stored in the file: the naive sum over every tensor. For
    /// Qwen3-0.6B this is ~751M, because the tied `lm_head` is a stored duplicate.
    pub stored_params: usize,
    /// Params counted once (a redundant tied `lm_head` not double-counted): the
    /// "true" model size. This is the ~596M that "0.6B" refers to.
    pub logical_params: usize,
    /// Size of the embedding table (`embed_tokens`) found in the file, so the
    /// verdict can show its share of `logical_params` (~26% for Qwen3-0.6B).
    pub embed_params: usize,
}

impl CrossCheck {
    /// Clean iff nothing mismatched and the counts agree. `notes` are non-fatal
    /// (a redundant tied `lm_head` is legal), so they don't affect the verdict.
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
///   and `lm_head.weight = [V, H]` — **required if not tied; optional if tied**
///   (a tied file may omit it or ship a redundant copy of `embed_tokens`; see
///   [`Expected`]).
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
    // The named dims (learning 05), pulled out once so each row below reads as a
    // literal `[out, in]` — the shape exactly as it sits in the file.
    let v = cfg.vocab_size; // V
    let h = cfg.hidden_size; // H
    let d = cfg.head_dim; // d — width of one attention head
    let q = cfg.q_width(); // heads · d      (2048 for Qwen3-0.6B)
    let kv = cfg.kv_width(); // kv_heads · d   (1024 — smaller under GQA)
    let i = cfg.intermediate_size; // I — FFN inner width

    // A row is `name : [out, in] (optional?)`. Every projection is stored `[out, in]`
    // (learning 05); norms are 1-D scale vectors, so their single dim is the width
    // they scale. `false` = required; the one `true` is the tied lm_head, below.
    let e = |name: String, shape: Vec<usize>, optional: bool| Expected { name, shape, optional };

    let mut out = Vec::with_capacity(3 + 11 * cfg.num_hidden_layers);

    // ── global input: the token table (a gather, id ─▶ H) ──
    out.push(e("model.embed_tokens.weight".into(), vec![v, h], false));

    // ── the transformer blocks, one identical set of 11 per layer (learning 10) ──
    for l in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{l}");
        // attention half: norm → q/k/v → QK-norm → o, all leaving/returning the bus
        out.push(e(format!("{p}.input_layernorm.weight"), vec![h], false)); //         scale H
        out.push(e(format!("{p}.self_attn.q_proj.weight"), vec![q, h], false)); //  H ─▶ q
        out.push(e(format!("{p}.self_attn.k_proj.weight"), vec![kv, h], false)); // H ─▶ kv
        out.push(e(format!("{p}.self_attn.v_proj.weight"), vec![kv, h], false)); // H ─▶ kv
        out.push(e(format!("{p}.self_attn.q_norm.weight"), vec![d], false)); //       scale d
        out.push(e(format!("{p}.self_attn.k_norm.weight"), vec![d], false)); //       scale d
        out.push(e(format!("{p}.self_attn.o_proj.weight"), vec![h, q], false)); //  q ─▶ H
        // mlp half (SwiGLU): norm → gate/up → down, back onto the bus
        out.push(e(format!("{p}.post_attention_layernorm.weight"), vec![h], false)); // scale H
        out.push(e(format!("{p}.mlp.gate_proj.weight"), vec![i, h], false)); //     H ─▶ I
        out.push(e(format!("{p}.mlp.up_proj.weight"), vec![i, h], false)); //       H ─▶ I
        out.push(e(format!("{p}.mlp.down_proj.weight"), vec![h, i], false)); //     I ─▶ H
    }

    // ── global output: final norm, then the LM head ──
    out.push(e("model.norm.weight".into(), vec![h], false)); // scale H
    // lm_head: H ─▶ V logits. Required if untied; optional if tied — a tied file may
    // omit it OR ship a redundant byte-identical copy of embed_tokens (learning 10 §3).
    out.push(e("lm_head.weight".into(), vec![v, h], cfg.tie_word_embeddings));

    out
}

/// Compare the file's tensors against [`expected_tensors`].
///
/// Steps:
/// 1. `let want = expected_tensors(cfg);`
/// 2. for each `want`: find it in `st`. Missing + required → problem. Missing +
///    optional → fine (skip). Present but `tensor.shape != want.shape` → problem
///    (name the dims). A present-but-optional `lm_head` under tying is legal, but
///    add a `note` if its bytes duplicate `embed_tokens` (the redundant-copy case).
/// 3. record any tensor present in `st` but not in `want` (an "extra") as a problem.
/// 4. `stored_params` = sum `num_elements()` over every file tensor; `logical_params`
///    subtracts a redundant tied `lm_head` so the count matches the "0.6B" nominal.
pub fn cross_check(cfg: &Config, st: &SafeTensors) -> CrossCheck {
    let want = expected_tensors(cfg);
    let mut problems = Vec::new();
    let mut notes = Vec::new();

    // Names the config implies — so we can spot file tensors we didn't expect.
    let expected_names: HashSet<&str> = want.iter().map(|e| e.name.as_str()).collect();

    // 1. Walk the spec. Each expected tensor is present (check its shape), absent but
    //    optional (fine), or absent and required (a problem). `expected_present`
    //    counts the ones we expect to physically see, so in a clean file it equals
    //    `found_count` — optional-and-omitted tensors don't inflate the expectation.
    let mut expected_present = 0usize;
    for e in &want {
        match st.tensor(&e.name) {
            Some(t) => {
                expected_present += 1;
                if t.shape != e.shape {
                    // Name the dims — this is the [2048,1024]-vs-[1024,1024] class of
                    // bug we built M1 to catch (learning 05).
                    problems.push(format!("{}: shape {:?}, expected {:?}", e.name, t.shape, e.shape));
                }
            }
            None if !e.optional => {
                problems.push(format!("{}: required tensor missing from file", e.name));
            }
            None => {} // optional + absent = fine (e.g. a tied file that omits lm_head)
        }
    }

    // 2. Any tensor in the file we didn't expect is a problem — a model whose shape we
    //    can't fully account for is exactly what this check exists to surface.
    for t in st.tensors() {
        if !expected_names.contains(t.name.as_str()) {
            problems.push(format!("{}: unexpected tensor {:?} not implied by the config", t.name, t.shape));
        }
    }

    // 3. Params. `stored` sums every file tensor naively. `logical` dedups a tied
    //    lm_head that merely duplicates embed_tokens (learning 10 §3), so it matches
    //    the model's nominal size. We confirm the duplication by comparing bytes —
    //    which pages both copies in, but this is a one-shot inspect, not a hot path.
    let stored_params: usize = st.tensors().iter().map(|t| t.num_elements()).sum();
    let mut logical_params = stored_params;

    if cfg.tie_word_embeddings
        && let (Some(head), Some(embed)) = (st.tensor("lm_head.weight"), st.tensor("model.embed_tokens.weight"))
    {
        if st.bytes(head) == st.bytes(embed) {
            logical_params -= head.num_elements();
            notes.push(format!(
                "lm_head.weight present but tied — a redundant byte-identical copy of \
                 embed_tokens ({} params counted once)",
                commafy(head.num_elements())
            ));
        } else {
            notes.push(
                "lm_head.weight present under tie_word_embeddings but NOT byte-identical to \
                 embed_tokens — unusual; counting both"
                    .into(),
            );
        }
    }

    let embed_params = st.tensor("model.embed_tokens.weight").map_or(0, |t| t.num_elements());

    CrossCheck {
        expected_count: expected_present,
        found_count: st.tensors().len(),
        problems,
        notes,
        stored_params,
        logical_params,
        embed_params,
    }
}

/// The 11 tensors of one block, in forward-pass order, as name suffixes under
/// `model.layers.{i}.`. `render_table` shows layer 0 as the representative block.
const BLOCK_SUFFIXES: [&str; 11] = [
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
];

/// `1234567` → `"1,234,567"`. Group counts are big; commas make the table legible.
fn commafy(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    let len = digits.len();
    for (i, ch) in digits.char_indices() {
        if i > 0 && (len - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// The `in ──▶ out` reading of a tensor's shape (learning 05). Projections are
/// stored `[out, in]` so the arrow reverses them; 1-D norms are scale vectors; the
/// embedding is a row gather, not a matmul, so it gets a bespoke label.
fn arrow_for(full_name: &str, shape: &[usize]) -> String {
    if full_name.ends_with("embed_tokens.weight") {
        return "id ──▶ H   (row gather)".to_string();
    }
    match shape {
        [out, in_] => format!("{in_} ──▶ {out}"),
        [w] => format!("scale {w}"),
        _ => format!("{shape:?}"),
    }
}

/// Render the dimension legend — the named dims every shape is built from
/// (`V/H/L/d`, q/kv head counts and their widths, `I`), plus the `[out, in]`
/// convention reminder. See learning 05 for the exact layout.
pub fn render_legend(cfg: &Config) -> String {
    let mut s = String::new();
    s.push_str("── dimensions (from config.json) ───────────────────────────────────────────\n");
    let row = |sym: &str, name: &str, val: usize, note: &str| format!("  {sym:<3}{name:<22}{val:>7}   {note}\n");
    s.push_str(&row("V", "vocab_size", cfg.vocab_size, "distinct tokens"));
    s.push_str(&row("H", "hidden_size", cfg.hidden_size, "residual-stream width (the bus)"));
    s.push_str(&row("L", "num_hidden_layers", cfg.num_hidden_layers, "transformer blocks"));
    s.push_str(&row("d", "head_dim", cfg.head_dim, "width of one attention head"));
    s.push_str(&row(
        "",
        "num_attention_heads",
        cfg.num_attention_heads,
        &format!("query heads → q width = {}·{} = {}", cfg.num_attention_heads, cfg.head_dim, cfg.q_width()),
    ));
    s.push_str(&row(
        "",
        "num_key_value_heads",
        cfg.num_key_value_heads,
        &format!(
            "kv heads → kv width = {}·{} = {}   (GQA group {})",
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.kv_width(),
            cfg.gqa_group()
        ),
    ));
    s.push_str(&row("I", "intermediate_size", cfg.intermediate_size, "FFN inner width"));
    s.push_str("  weights are stored [out, in]; read a row as   in ──▶ out   (y = x·Wᵀ)\n");
    s
}

/// Render the grouped tensor table: the global embedding, one labelled
/// `× L` block, the final norm, and the `lm_head` line (flagged as tied/redundant
/// when it duplicates `embed_tokens`) — each row showing `dtype`, `[out, in]`
/// shape, param count, and the `in ──▶ out` arrow.
///
/// We print *one* representative block (they're identical across layers) and label
/// it `× {L}` rather than dumping 28 copies — readability over completeness.
pub fn render_table(cfg: &Config, st: &SafeTensors) -> String {
    let mut s = String::new();
    s.push_str("── tensors ─────────────────────────────────────────────────────────────────\n");
    s.push_str(&format!("  {:<40}{:<6}{:<16}{:>15}   {}\n", "TENSOR", "DTYPE", "SHAPE", "PARAMS", "in ──▶ out"));

    // Look a tensor up by its full name and format one aligned row. `display` is the
    // (possibly shortened) name shown; absent tensors are marked, not skipped.
    let row = |display: &str, full: &str| -> String {
        match st.tensor(full) {
            Some(t) => format!(
                "    {:<38}{:<6}{:<16}{:>15}   {}\n",
                display,
                format!("{:?}", t.dtype),
                format!("{:?}", t.shape),
                commafy(t.num_elements()),
                arrow_for(full, &t.shape),
            ),
            None => format!("    {display:<38}(absent)\n"),
        }
    };

    s.push_str("  global\n");
    s.push_str(&row("model.embed_tokens.weight", "model.embed_tokens.weight"));

    // One representative block; the other L−1 are identical, so we label rather than
    // dump 28 copies. Suffixes are shown; the real names carry the `model.layers.0.`
    // prefix (which is what we look up).
    s.push_str(&format!("  each block  × {}   (shown: layer 0)\n", cfg.num_hidden_layers));
    for suffix in BLOCK_SUFFIXES {
        s.push_str(&row(suffix, &format!("model.layers.0.{suffix}")));
    }

    s.push_str("  final\n");
    s.push_str(&row("model.norm.weight", "model.norm.weight"));
    let head_display = if cfg.tie_word_embeddings { "lm_head.weight   (tied)" } else { "lm_head.weight" };
    s.push_str(&row(head_display, "lm_head.weight"));
    s
}

/// Render the cross-check verdict: counts, any problems (each naming the dim that
/// didn't line up), any notes (e.g. the redundant tied `lm_head`), the stored vs.
/// logical param totals, and the embeddings' share of the logical count.
pub fn render_verdict(xc: &CrossCheck) -> String {
    let mut s = String::new();
    s.push_str("── verdict ─────────────────────────────────────────────────────────────────\n");
    if xc.ok() {
        s.push_str(&format!("  ✓ all {} expected tensors present, shapes match the config\n", xc.expected_count));
    } else {
        s.push_str(&format!(
            "  ✗ {} problem(s)   (expected {}, found {} in file):\n",
            xc.problems.len(),
            xc.expected_count,
            xc.found_count
        ));
        for p in &xc.problems {
            s.push_str(&format!("      • {p}\n"));
        }
    }
    for n in &xc.notes {
        s.push_str(&format!("  note: {n}\n"));
    }
    s.push_str(&format!("  params: {} stored · {} logical (the \"0.6B\")\n", commafy(xc.stored_params), commafy(xc.logical_params)));
    if xc.logical_params > 0 {
        let share = xc.embed_params as f64 / xc.logical_params as f64 * 100.0;
        s.push_str(&format!("  embeddings: {} = {share:.1}% of logical\n", commafy(xc.embed_params)));
    }
    s
}

/// `fs inspect <model_dir>` end to end.
///
/// Steps:
/// 1. `Config::load(model_dir)?` and `SafeTensors::load(<dir>/model.safetensors)?`.
/// 2. print `render_legend`, `render_table`, then `render_verdict`.
/// 3. return whether the cross-check was clean (the CLI maps that to its exit
///    code — a shape mismatch is a *failure*, not a warning).
pub fn run(model_dir: &str) -> Result<bool, InspectError> {
    // Load the two halves of "what a model is" (learning 09): the architecture
    // description and the weights it arranges.
    let cfg = Config::load(model_dir)?;
    let st = SafeTensors::load(&format!("{model_dir}/model.safetensors"))?;
    let xc = cross_check(&cfg, &st);

    print!("{}", render_legend(&cfg));
    println!();
    print!("{}", render_table(&cfg, &st));
    println!();
    print!("{}", render_verdict(&xc));

    // A shape mismatch is a *failure*, not a warning — the CLI turns this into an
    // exit code so `fs inspect` can gate a build/script.
    Ok(xc.ok())
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

#[cfg(test)]
mod tests {
    use super::*;

    // A tiny architecture with dims chosen so every derived width is easy to eyeball:
    // q = 4·4 = 16, kv = 2·4 = 8, d = 4, I = 16, V = 100, H = 8, L = 2, tied.
    // (Mirrors the MINI config in `config.rs`, built directly to avoid a temp file.)
    fn mini_cfg() -> Config {
        Config {
            vocab_size: 100,
            hidden_size: 8,
            num_hidden_layers: 2,
            head_dim: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 16,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 1,
            eos_token_id: 2,
            max_position_embeddings: 32,
        }
    }

    fn find<'a>(want: &'a [Expected], name: &str) -> &'a Expected {
        want.iter().find(|e| e.name == name).unwrap_or_else(|| panic!("missing expected tensor {name}"))
    }

    #[test]
    fn emits_global_count_and_the_three_globals() {
        let want = expected_tensors(&mini_cfg());

        // 3 global + 11 per layer × 2 layers = 25.
        assert_eq!(want.len(), 3 + 11 * 2);

        let embed = find(&want, "model.embed_tokens.weight");
        assert_eq!(embed.shape, vec![100, 8]); // [V, H]
        assert!(!embed.optional);

        let norm = find(&want, "model.norm.weight");
        assert_eq!(norm.shape, vec![8]); // [H] scale vector

        // Tied here, so lm_head is [V, H] but OPTIONAL (may be absent or a copy).
        let head = find(&want, "lm_head.weight");
        assert_eq!(head.shape, vec![100, 8]);
        assert!(head.optional);
    }

    #[test]
    fn emits_eleven_per_layer_with_gqa_shapes() {
        let want = expected_tensors(&mini_cfg());

        // Exactly 11 tensors carry each layer's prefix.
        let layer1: Vec<_> = want.iter().filter(|e| e.name.starts_with("model.layers.1.")).collect();
        assert_eq!(layer1.len(), 11);

        // The GQA asymmetry is visible in the shapes: q wider than k/v (16 vs 8),
        // o_proj mirrors q, the MLP fans H→I and back. All required.
        let cases = [
            ("model.layers.0.input_layernorm.weight", vec![8]),
            ("model.layers.0.self_attn.q_proj.weight", vec![16, 8]), // H ─▶ q=16
            ("model.layers.0.self_attn.k_proj.weight", vec![8, 8]),  // H ─▶ kv=8
            ("model.layers.0.self_attn.v_proj.weight", vec![8, 8]),
            ("model.layers.0.self_attn.q_norm.weight", vec![4]), // scale d=4
            ("model.layers.0.self_attn.k_norm.weight", vec![4]),
            ("model.layers.0.self_attn.o_proj.weight", vec![8, 16]), // q ─▶ H
            ("model.layers.0.post_attention_layernorm.weight", vec![8]),
            ("model.layers.0.mlp.gate_proj.weight", vec![16, 8]), // H ─▶ I=16
            ("model.layers.0.mlp.up_proj.weight", vec![16, 8]),
            ("model.layers.0.mlp.down_proj.weight", vec![8, 16]), // I ─▶ H
        ];
        for (name, shape) in cases {
            let t = find(&want, name);
            assert_eq!(t.shape, shape, "shape for {name}");
            assert!(!t.optional, "{name} should be required");
        }
    }

    #[test]
    fn lm_head_is_required_when_not_tied() {
        let mut cfg = mini_cfg();
        cfg.tie_word_embeddings = false;
        let want = expected_tensors(&cfg);
        let head = find(&want, "lm_head.weight");
        assert!(!head.optional, "untied lm_head is real, distinct weights → required");
    }

    #[test]
    fn every_expected_name_is_unique() {
        let want = expected_tensors(&mini_cfg());
        let mut names: Vec<&str> = want.iter().map(|e| e.name.as_str()).collect();
        names.sort_unstable();
        let count = names.len();
        names.dedup();
        assert_eq!(names.len(), count, "expected_tensors emitted a duplicate name");
    }

    #[test]
    fn real_qwen_config_implies_311_tensors_if_present() {
        // Reality anchor: the real config must imply exactly the 311 tensors we found
        // in the header (3 global + 11 × 28). Skipped on a fresh checkout.
        let dir = "models/qwen3-0.6b";
        if !std::path::Path::new(dir).join("config.json").exists() {
            eprintln!("skipping: {dir}/config.json not fetched");
            return;
        }
        let cfg = Config::load(dir).expect("real Qwen config loads");
        let want = expected_tensors(&cfg);
        assert_eq!(want.len(), 311); // 3 + 11 × 28
        assert!(find(&want, "lm_head.weight").optional); // Qwen3-0.6B is tied
    }

    // ── cross_check ───────────────────────────────────────────────────────────

    /// One tensor to write into a synthetic file: `(name, shape, fill_byte)`. All
    /// tensors are BF16 (2 B/elem), filled with a constant byte so we can control
    /// whether two tensors (e.g. lm_head vs embed) come out byte-identical.
    type Row = (String, Vec<usize>, u8);

    /// The full, correct tensor set for `cfg` as writable rows, all zero-filled — so
    /// `lm_head` and `embed_tokens` are byte-identical (the redundant-copy case). Tests
    /// mutate the returned Vec to inject a specific defect.
    fn rows_for(cfg: &Config) -> Vec<Row> {
        expected_tensors(cfg).iter().map(|e| (e.name.clone(), e.shape.clone(), 0u8)).collect()
    }

    /// Write a valid `[u64 LE header len][JSON header][blob]` safetensors file at
    /// `path`. All tensors BF16, `fill`-filled so byte-equality is controllable.
    fn write_model_file(path: &std::path::Path, rows: &[Row]) {
        use std::io::Write;
        let mut entries = Vec::new();
        let mut blob = Vec::new();
        for (name, shape, fill) in rows {
            let bytes = shape.iter().product::<usize>() * 2; // BF16
            let start = blob.len();
            blob.resize(start + bytes, *fill);
            let end = blob.len();
            entries.push(format!(r#""{name}":{{"dtype":"BF16","shape":{shape:?},"data_offsets":[{start},{end}]}}"#));
        }
        let header = format!("{{{}}}", entries.join(","));
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(header.as_bytes()).unwrap();
        f.write_all(&blob).unwrap();
    }

    /// Write a temp model file and load it back through the real `SafeTensors::load`,
    /// so cross_check sees a genuinely parsed file.
    fn load_model(tag: &str, rows: &[Row]) -> SafeTensors {
        let path = std::env::temp_dir().join(format!("fs_inspect_{tag}.safetensors"));
        write_model_file(&path, rows);
        SafeTensors::load(path.to_str().unwrap()).expect("synthetic model loads")
    }

    /// The MINI config as JSON, matching `mini_cfg()` — for `run` end-to-end tests.
    const MINI_JSON: &str = r#"{
        "vocab_size": 100, "hidden_size": 8, "num_hidden_layers": 2,
        "head_dim": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
        "intermediate_size": 16, "rms_norm_eps": 1e-6, "rope_theta": 10000,
        "tie_word_embeddings": true, "bos_token_id": 1, "eos_token_id": 2,
        "max_position_embeddings": 32
    }"#;

    #[test]
    fn clean_model_cross_checks_ok_and_dedups_the_tied_head() {
        let cfg = mini_cfg();
        let st = load_model("clean", &rows_for(&cfg));
        let xc = cross_check(&cfg, &st);

        assert!(xc.ok(), "clean model should pass; problems: {:?}", xc.problems);
        assert!(xc.problems.is_empty());
        assert_eq!(xc.expected_count, 25);
        assert_eq!(xc.found_count, 25);

        // The redundant tied lm_head is a *note*, not a failure, and it's deduped:
        // stored counts the V·H table twice, logical once.
        assert_eq!(xc.notes.len(), 1);
        assert!(xc.notes[0].contains("redundant"), "note was: {}", xc.notes[0]);
        assert_eq!(xc.stored_params - xc.logical_params, 100 * 8, "one V·H table deduped");
    }

    #[test]
    fn shape_mismatch_is_a_named_problem() {
        let cfg = mini_cfg();
        let mut rows = rows_for(&cfg);
        // Break q_proj: store it square [8,8] instead of the GQA-correct [16,8].
        for r in &mut rows {
            if r.0 == "model.layers.0.self_attn.q_proj.weight" {
                r.1 = vec![8, 8];
            }
        }
        let st = load_model("badshape", &rows);
        let xc = cross_check(&cfg, &st);

        assert!(!xc.ok());
        assert!(
            xc.problems.iter().any(|p| p.contains("q_proj") && p.contains("[8, 8]") && p.contains("[16, 8]")),
            "problems: {:?}",
            xc.problems
        );
    }

    #[test]
    fn missing_required_tensor_is_a_problem() {
        let cfg = mini_cfg();
        let mut rows = rows_for(&cfg);
        rows.retain(|r| r.0 != "model.layers.0.mlp.down_proj.weight");
        let st = load_model("missing", &rows);
        let xc = cross_check(&cfg, &st);

        assert!(!xc.ok());
        assert!(
            xc.problems.iter().any(|p| p.contains("down_proj") && p.contains("missing")),
            "problems: {:?}",
            xc.problems
        );
    }

    #[test]
    fn unexpected_extra_tensor_is_a_problem() {
        let cfg = mini_cfg();
        let mut rows = rows_for(&cfg);
        rows.push(("model.layers.0.self_attn.rotary_emb.inv_freq".into(), vec![2], 0));
        let st = load_model("extra", &rows);
        let xc = cross_check(&cfg, &st);

        assert!(!xc.ok());
        assert!(
            xc.problems.iter().any(|p| p.contains("rotary_emb") && p.contains("unexpected")),
            "problems: {:?}",
            xc.problems
        );
    }

    #[test]
    fn tied_lm_head_may_be_omitted() {
        let cfg = mini_cfg(); // tied
        let mut rows = rows_for(&cfg);
        rows.retain(|r| r.0 != "lm_head.weight"); // a tied file is free to drop it
        let st = load_model("omitted", &rows);
        let xc = cross_check(&cfg, &st);

        assert!(xc.ok(), "omitting a tied lm_head is legal; problems: {:?}", xc.problems);
        assert!(xc.notes.is_empty(), "nothing to note when there's no duplicate");
        assert_eq!(xc.stored_params, xc.logical_params, "no dedup when nothing is duplicated");
        assert_eq!(xc.found_count, 24);
        assert_eq!(xc.expected_count, 24);
    }

    #[test]
    fn tied_lm_head_that_differs_is_noted_not_deduped() {
        let cfg = mini_cfg(); // tied
        let mut rows = rows_for(&cfg);
        // Present but NOT a copy of embed (which is zero-filled): fill lm_head with 0x11.
        for r in &mut rows {
            if r.0 == "lm_head.weight" {
                r.2 = 0x11;
            }
        }
        let st = load_model("differs", &rows);
        let xc = cross_check(&cfg, &st);

        assert!(xc.ok(), "a differing tied head is a note, not a failure");
        assert!(xc.notes.iter().any(|n| n.contains("NOT byte-identical")), "notes: {:?}", xc.notes);
        assert_eq!(xc.stored_params, xc.logical_params, "differing bytes → count both, no dedup");
    }

    #[test]
    fn real_model_cross_checks_clean_if_present() {
        // The M1 payoff: load the real config + the 1.4 GB weights and prove the whole
        // tensor set lines up. Skipped on a fresh checkout (assets git-ignored).
        let dir = "models/qwen3-0.6b";
        let weights = format!("{dir}/model.safetensors");
        if !std::path::Path::new(&weights).exists() {
            eprintln!("skipping: {weights} not fetched");
            return;
        }
        let cfg = Config::load(dir).expect("real config loads");
        let st = SafeTensors::load(&weights).expect("real weights load");
        let xc = cross_check(&cfg, &st);

        assert!(xc.ok(), "real Qwen3-0.6B should cross-check clean; problems: {:?}", xc.problems);
        assert_eq!(xc.found_count, 311);
        assert_eq!(xc.expected_count, 311);
        assert_eq!(xc.stored_params, 751_632_384, "751M physically stored");
        assert_eq!(xc.logical_params, 596_049_920, "596M logical = the '0.6B'");
        assert_eq!(xc.embed_params, 155_582_464);
        assert_eq!(xc.notes.len(), 1);
        assert!(xc.notes[0].contains("redundant"));
    }

    // ── commafy / arrow_for ─────────────────────────────────────────────────────

    #[test]
    fn commafy_groups_by_threes() {
        assert_eq!(commafy(0), "0");
        assert_eq!(commafy(999), "999");
        assert_eq!(commafy(1_000), "1,000");
        assert_eq!(commafy(155_582_464), "155,582,464");
    }

    #[test]
    fn arrow_for_reads_shapes_as_in_to_out() {
        // A projection is stored [out, in]; the arrow reverses it.
        assert_eq!(arrow_for("model.layers.0.self_attn.q_proj.weight", &[2048, 1024]), "1024 ──▶ 2048");
        // A 1-D norm is a scale vector.
        assert_eq!(arrow_for("model.layers.0.input_layernorm.weight", &[1024]), "scale 1024");
        // The embedding is a gather, not a matmul.
        assert!(arrow_for("model.embed_tokens.weight", &[151936, 1024]).contains("gather"));
    }

    // ── render_* / run ──────────────────────────────────────────────────────────

    #[test]
    fn legend_shows_dims_and_the_gqa_derivation() {
        let s = render_legend(&mini_cfg());
        assert!(s.contains("vocab_size"));
        assert!(s.contains("hidden_size"));
        // Derived widths + GQA group are computed, not hard-coded: q = 4·4 = 16.
        assert!(s.contains("q width = 4·4 = 16"), "legend was:\n{s}");
        assert!(s.contains("GQA group 2"), "legend was:\n{s}");
        assert!(s.contains("in ──▶ out"));
    }

    #[test]
    fn table_groups_the_block_and_reads_shapes() {
        let cfg = mini_cfg();
        let st = load_model("render_table", &rows_for(&cfg));
        let s = render_table(&cfg, &st);

        assert!(s.contains("each block  × 2"), "table was:\n{s}");
        assert!(s.contains("self_attn.q_proj.weight"));
        assert!(s.contains("[16, 8]")); // q_proj shape for MINI
        assert!(s.contains("8 ──▶ 16")); // its in ──▶ out
        assert!(s.contains("row gather")); // embed_tokens
        assert!(s.contains("lm_head.weight   (tied)")); // MINI is tied
    }

    #[test]
    fn verdict_reports_clean_with_note_and_shares() {
        let cfg = mini_cfg();
        let st = load_model("render_verdict_ok", &rows_for(&cfg));
        let xc = cross_check(&cfg, &st);
        let s = render_verdict(&xc);

        assert!(s.contains('✓'), "verdict was:\n{s}");
        assert!(s.contains("note:"));
        assert!(s.contains("stored"));
        assert!(s.contains("logical"));
        assert!(s.contains("% of logical"));
    }

    #[test]
    fn verdict_lists_problems_when_dirty() {
        let cfg = mini_cfg();
        let mut rows = rows_for(&cfg);
        rows.retain(|r| r.0 != "model.norm.weight"); // drop a required tensor
        let st = load_model("render_verdict_bad", &rows);
        let xc = cross_check(&cfg, &st);
        let s = render_verdict(&xc);

        assert!(s.contains('✗'), "verdict was:\n{s}");
        assert!(s.contains("model.norm.weight"));
        assert!(s.contains("missing"));
    }

    #[test]
    fn run_end_to_end_on_a_synthetic_model() {
        // Full path — Config::load + SafeTensors::load + cross_check + the three
        // renders — with no real assets, so this runs on a fresh checkout.
        let cfg = mini_cfg();
        let dir = std::env::temp_dir().join("fs_inspect_run_ok");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), MINI_JSON).unwrap();
        write_model_file(&dir.join("model.safetensors"), &rows_for(&cfg));

        let clean = run(dir.to_str().unwrap()).expect("inspect runs end to end");
        assert!(clean, "a well-formed synthetic model should cross-check clean");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn run_returns_false_on_a_shape_mismatch() {
        let cfg = mini_cfg();
        let mut rows = rows_for(&cfg);
        for r in &mut rows {
            if r.0 == "model.layers.0.mlp.gate_proj.weight" {
                r.1 = vec![8, 8]; // wrong: should be [16, 8]
            }
        }
        let dir = std::env::temp_dir().join("fs_inspect_run_bad");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), MINI_JSON).unwrap();
        write_model_file(&dir.join("model.safetensors"), &rows);

        let clean = run(dir.to_str().unwrap()).expect("inspect runs even when the model is malformed");
        assert!(!clean, "a shape mismatch must make run() report not-clean (→ exit 1)");

        std::fs::remove_dir_all(&dir).ok();
    }
}
