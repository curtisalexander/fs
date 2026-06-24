//! `Config` — the model's architecture hyperparameters, read from `config.json`.
//!
//! This is one of the *two things* a model is (see
//! [`docs/learnings/01-safetensors-vs-gguf.md`]): the **description** of how the
//! weights are arranged. The other thing — the weights themselves — lives in
//! `model.safetensors` and is handled by [`crate::safetensors`].
//!
//! Every shape in the engine is built from the seven named dimensions below; the
//! map for how they line up is [`docs/learnings/05-reading-shapes.md`]. We keep a
//! field per dimension we actually use (Qwen3-0.6B's `config.json` carries more;
//! we ignore the rest) and expose the *derived* widths (`q_width`, `kv_width`,
//! `gqa_group`) as methods so the relationships are executable, not just prose.
//!
//! JSON parsing is conceded to `serde_json` (deciding bytes-are-JSON is not the
//! lesson; see the M0 dependency note). We parse a `serde_json::Value` by hand
//! rather than `#[derive(Deserialize)]` so the field extraction stays visible and
//! we add no new dependency — graduating to a `serde` derive is an open option we
//! can take later if this gets tedious.

/// The architecture hyperparameters we need to load and run Qwen3-0.6B.
///
/// Symbols in comments match `docs/learnings/05-reading-shapes.md`.
#[derive(Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,          // V — number of distinct tokens (151936)
    pub hidden_size: usize,         // H — residual-stream width (1024)
    pub num_hidden_layers: usize,   // L — transformer blocks (28)
    pub head_dim: usize,            // d — width of one attention head (128)
    pub num_attention_heads: usize, // query heads (16) → q width = heads · d
    pub num_key_value_heads: usize, // kv heads    (8)  → kv width = kv_heads · d
    pub intermediate_size: usize,   // I — FFN inner width (3072)
    pub rms_norm_eps: f32,          // RMSNorm epsilon (1e-6) — used at M2
    pub rope_theta: f64,            // RoPE base frequency (1e6) — used at M2
    pub tie_word_embeddings: bool,  // true → no separate lm_head; reuse embeddings
    pub bos_token_id: u32,          // begin-of-sequence id — used at M3
    pub eos_token_id: u32,          // end-of-sequence id   — used at M3
    pub max_position_embeddings: usize, // context length (40960)
}

impl Config {
    /// Load and parse `<model_dir>/config.json`.
    ///
    /// Steps:
    /// 1. read `<model_dir>/config.json` to a string (→ `ConfigError::NotFound`
    ///    on a missing/unreadable file, naming the path).
    /// 2. `serde_json::from_str::<serde_json::Value>` (→ `ConfigError::Parse`).
    /// 3. pull each field by key, erroring with the *field name* if it's missing
    ///    or the wrong JSON type (→ `ConfigError::MissingField` /
    ///    `ConfigError::BadField`). No silent defaults — a model we don't
    ///    understand should fail loudly here, not produce wrong shapes later.
    pub fn load(model_dir: &str) -> Result<Self, ConfigError> {
        let _ = model_dir;
        todo!("read config.json → serde_json::Value → extract the fields above")
    }

    /// Query-projection output width: `num_attention_heads · head_dim`.
    ///
    /// NOTE the deliberate decoupling — for Qwen3-0.6B this is `16 · 128 = 2048`,
    /// which is **not** `hidden_size` (1024). `q_proj` is therefore `[2048, 1024]`,
    /// not square. See learning 05 §"head_dim is decoupled from hidden_size".
    pub fn q_width(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// Key/Value-projection output width: `num_key_value_heads · head_dim`.
    ///
    /// Smaller than `q_width` under GQA — here `8 · 128 = 1024`. That asymmetry is
    /// grouped-query attention, visible in the shapes before we write attention.
    pub fn kv_width(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// GQA group size: how many query heads share one key/value head
    /// (`num_attention_heads / num_key_value_heads`, here `16 / 8 = 2`).
    pub fn gqa_group(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

/// Everything that can go wrong reading `config.json`. Hand-written (like M0's
/// `TokenizerError`) so the failure modes stay visible and testable.
#[derive(Debug)]
pub enum ConfigError {
    /// `config.json` was missing or unreadable.
    NotFound { path: String, message: String },
    /// The file was not valid JSON.
    Parse { path: String, message: String },
    /// A required key was absent.
    MissingField { field: &'static str },
    /// A key was present but the wrong JSON type / range.
    BadField { field: &'static str, message: String },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::NotFound { path, message } => {
                write!(f, "could not read {path}: {message}")
            }
            ConfigError::Parse { path, message } => {
                write!(f, "{path} is not valid JSON: {message}")
            }
            ConfigError::MissingField { field } => {
                write!(f, "config.json is missing required field '{field}'")
            }
            ConfigError::BadField { field, message } => {
                write!(f, "config.json field '{field}' is invalid: {message}")
            }
        }
    }
}

impl std::error::Error for ConfigError {}
