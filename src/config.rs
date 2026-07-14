//! `Config` — the model's architecture hyperparameters, read from `config.json`.
//!
//! This is one of the *two things* a model is (see
//! [`docs/learnings/01-safetensors-vs-gguf.md`]): the **description** of how the
//! weights are arranged. The other thing — the weights themselves — lives in
//! `model.safetensors` and is handled by [`crate::safetensors`]. You need *both* to
//! rebuild a network; what `config.json` is (and the surprising fact that it
//! *parameterizes* an architecture rather than describing one) is
//! [`docs/learnings/09-config.md`].
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
        let path = format!("{model_dir}/config.json");

        // 1. Read the file to a string. A missing/unreadable file is NotFound.
        let text = std::fs::read_to_string(&path)
            .map_err(|e| ConfigError::NotFound { path: path.clone(), message: e.to_string() })?;

        // 2. Parse the JSON once. serde_json owns "are these bytes JSON?" (not the
        //    lesson — see the M0 dependency note); we pull fields out by hand below.
        let v: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| ConfigError::Parse { path: path.clone(), message: e.to_string() })?;

        // 3. One typed extractor per JSON scalar kind. Each names the field it
        //    couldn't satisfy: absent → MissingField, wrong type/range → BadField.
        //    No silent defaults — a model we don't understand fails loudly here.
        let uint = |field: &'static str| -> Result<usize, ConfigError> {
            v.get(field)
                .ok_or(ConfigError::MissingField { field })?
                .as_u64()
                .map(|n| n as usize)
                .ok_or_else(|| ConfigError::BadField { field, message: "expected a non-negative integer".into() })
        };
        let token_id = |field: &'static str| -> Result<u32, ConfigError> {
            let n = v
                .get(field)
                .ok_or(ConfigError::MissingField { field })?
                .as_u64()
                .ok_or_else(|| ConfigError::BadField { field, message: "expected a non-negative integer".into() })?;
            u32::try_from(n).map_err(|_| ConfigError::BadField { field, message: format!("{n} does not fit in u32") })
        };
        let float = |field: &'static str| -> Result<f64, ConfigError> {
            v.get(field)
                .ok_or(ConfigError::MissingField { field })?
                .as_f64()
                .ok_or_else(|| ConfigError::BadField { field, message: "expected a number".into() })
        };
        let boolean = |field: &'static str| -> Result<bool, ConfigError> {
            v.get(field)
                .ok_or(ConfigError::MissingField { field })?
                .as_bool()
                .ok_or_else(|| ConfigError::BadField { field, message: "expected a boolean".into() })
        };

        Ok(Config {
            vocab_size: uint("vocab_size")?,
            hidden_size: uint("hidden_size")?,
            num_hidden_layers: uint("num_hidden_layers")?,
            head_dim: uint("head_dim")?,
            num_attention_heads: uint("num_attention_heads")?,
            num_key_value_heads: uint("num_key_value_heads")?,
            intermediate_size: uint("intermediate_size")?,
            // eps/theta are floats in JSON (theta may be written as an int like
            // 1000000 — as_f64 handles both); eps narrows to f32, our compute width.
            rms_norm_eps: float("rms_norm_eps")? as f32,
            rope_theta: float("rope_theta")?,
            tie_word_embeddings: boolean("tie_word_embeddings")?,
            bos_token_id: token_id("bos_token_id")?,
            eos_token_id: token_id("eos_token_id")?,
            max_position_embeddings: uint("max_position_embeddings")?,
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    // A minimal but complete config, with dims chosen so the derived widths are
    // easy to eyeball: q = 4·4 = 16, kv = 2·4 = 8, gqa group = 4/2 = 2.
    const MINI: &str = r#"{
        "vocab_size": 100, "hidden_size": 8, "num_hidden_layers": 2,
        "head_dim": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
        "intermediate_size": 16, "rms_norm_eps": 1e-6, "rope_theta": 10000,
        "tie_word_embeddings": true, "bos_token_id": 1, "eos_token_id": 2,
        "max_position_embeddings": 32
    }"#;

    /// Write `<dir>/config.json` in a fresh temp dir and return the dir path.
    fn write_config(tag: &str, json: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("fs_config_{tag}"));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), json).unwrap();
        dir
    }

    #[test]
    fn parses_fields_and_derives_widths() {
        let dir = write_config("mini", MINI);
        let cfg = Config::load(dir.to_str().unwrap()).expect("valid config loads");

        assert_eq!(cfg.vocab_size, 100);
        assert_eq!(cfg.head_dim, 4);
        assert_eq!(cfg.num_attention_heads, 4);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert!(cfg.tie_word_embeddings);
        assert_eq!(cfg.bos_token_id, 1);
        assert_eq!(cfg.rms_norm_eps, 1e-6);
        assert_eq!(cfg.rope_theta, 10000.0);

        // The relationships that matter for the shape table (learning 05).
        assert_eq!(cfg.q_width(), 16);
        assert_eq!(cfg.kv_width(), 8);
        assert_eq!(cfg.gqa_group(), 2);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_field_names_the_field() {
        // Drop hidden_size from an otherwise-valid config.
        let json = MINI.replace("\"hidden_size\": 8,", "");
        let dir = write_config("missing", &json);
        let err = Config::load(dir.to_str().unwrap()).unwrap_err();
        assert!(matches!(err, ConfigError::MissingField { field: "hidden_size" }));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn wrong_type_is_a_bad_field() {
        // hidden_size present but a string, not a number.
        let json = MINI.replace("\"hidden_size\": 8", "\"hidden_size\": \"lots\"");
        let dir = write_config("badtype", &json);
        let err = Config::load(dir.to_str().unwrap()).unwrap_err();
        assert!(matches!(err, ConfigError::BadField { field: "hidden_size", .. }));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_file_is_not_found() {
        let err = Config::load("/no/such/failed-star/model").unwrap_err();
        assert!(matches!(err, ConfigError::NotFound { .. }));
    }

    #[test]
    fn malformed_json_is_a_parse_error() {
        let dir = write_config("badjson", "{ not valid json");
        let err = Config::load(dir.to_str().unwrap()).unwrap_err();
        assert!(matches!(err, ConfigError::Parse { .. }));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn reads_the_real_qwen_config_if_present() {
        // Reality check against the shipped config.json — skipped on a fresh
        // checkout (assets git-ignored), like the golden tokenizer test.
        let dir = "models/qwen3-0.6b";
        if !std::path::Path::new(dir).join("config.json").exists() {
            eprintln!("skipping: {dir}/config.json not fetched");
            return;
        }
        let cfg = Config::load(dir).expect("real Qwen config loads");

        // The seven named dims + the M2/M3 scalars, straight from the model card.
        assert_eq!(cfg.vocab_size, 151936);
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_hidden_layers, 28);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.intermediate_size, 3072);
        assert_eq!(cfg.rms_norm_eps, 1e-6);
        assert_eq!(cfg.rope_theta, 1_000_000.0);
        assert!(cfg.tie_word_embeddings);
        assert_eq!(cfg.bos_token_id, 151643);
        assert_eq!(cfg.eos_token_id, 151645);
        assert_eq!(cfg.max_position_embeddings, 40960);

        // Derived widths that drive the M1 shape table: q=16·128, kv=8·128, group=2.
        assert_eq!(cfg.q_width(), 2048);
        assert_eq!(cfg.kv_width(), 1024);
        assert_eq!(cfg.gqa_group(), 2);
    }
}
