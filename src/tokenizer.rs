//! Byte-level BPE tokenizer for Qwen3-0.6B.  (Milestone M0)
//!
//! ──────────────────────────────────────────────────────────────────────────
//! THE BIG PICTURE — what "byte-level BPE" means (GPT-2 → Qwen lineage)
//! ──────────────────────────────────────────────────────────────────────────
//! Encoding text → token IDs happens in four stages (with special-token carving
//! sitting in front — see `encode`):
//!
//!   1. PRE-TOKENIZE.  A regex chops the text into chunks ("words"), so a merge
//!      can never cross, say, a word/punctuation boundary. Crucially a leading
//!      space stays *attached* to the word after it — that's why our golden has
//!      "hello world" -> [14990, 1879] and " hello world" -> [23811, 1879]
//!      (note the different FIRST token).
//!
//!   2. BYTES → "BYTE-LEVEL UNICODE".  Each chunk is raw UTF-8 bytes. We remap
//!      all 256 byte values onto printable Unicode chars (GPT-2's trick), so a
//!      token string never contains a real space/newline/control byte. Under
//!      this map a space (0x20) becomes 'Ġ', a newline (0x0A) becomes 'Ċ'. The
//!      vocab on disk is written in THIS alphabet — it literally has keys like
//!      "Ġworld".
//!
//!   3. MERGE.  Starting from single chars, repeatedly glue the adjacent pair
//!      with the best (lowest-numbered) rank in the merge list, until no
//!      adjacent pair is mergeable. The survivors are the final token pieces.
//!
//!   4. LOOK UP.  Each final piece is a key in the vocab → its integer ID.
//!
//! Decoding reverses 4→2: IDs → pieces → concatenate → undo the byte map →
//! interpret the bytes as UTF-8 (special-token IDs decode to their literal text).
//!
//! SINGLE SOURCE OF TRUTH: `models/qwen3-0.6b/tokenizer.json` — the official
//! Hugging Face tokenizer file. One file carries everything we need:
//!   - `model.vocab`   — { "<piece in byte-level-unicode>": id }
//!   - `model.merges`  — ["<left>","<right>"] pairs, in PRIORITY order (rank = index)
//!   - `pre_tokenizer` — the stage-1 split regex
//!   - `added_tokens`  — special tokens (content ↔ id) that bypass BPE
//!
//! We used to read GPT-2's split `vocab.json` + `merges.txt`; `tokenizer.json`
//! supersedes both and is what newer models ship, so we parse it directly.
//! (See docs/01-tokenizer.md.)
//!
//! Verify (M0 "done"): reproduce `tests/golden/tokenizer.json` exactly, and
//! round-trip decode(encode(s)) == s.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

/// Tokenizer-specific result type.
pub type Result<T> = std::result::Result<T, TokenizerError>;

/// Errors that can happen while loading or running the tokenizer.
///
/// We keep this hand-written instead of pulling in `thiserror`: the point is to
/// make the failure modes visible without adding another abstraction layer.
#[derive(Debug)]
pub enum TokenizerError {
    /// Could not read a tokenizer asset from disk.
    Io {
        path: PathBuf,
        source: std::io::Error,
    },

    /// Could not parse a JSON tokenizer asset.
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },

    /// `tokenizer.json` parsed as JSON, but its structure/content is not what we
    /// need: a missing `model.vocab`, a malformed merge entry, a non-contiguous
    /// vocab, no pre-tokenizer regex, a bad `added_tokens` entry, and so on.
    BadTokenizer { path: PathBuf, message: String },

    /// Qwen's pre-tokenization regex failed to compile or run.
    Regex(fancy_regex::Error),

    /// BPE produced a piece that was not present in the vocabulary.
    UnknownToken(String),

    /// Decode was asked to use a token id outside `id_to_token`.
    InvalidTokenId(u32),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(f, "could not read {}: {source}", path.display())
            }
            Self::Json { path, source } => {
                write!(f, "could not parse JSON in {}: {source}", path.display())
            }
            Self::BadTokenizer { path, message } => {
                write!(f, "bad tokenizer file {}: {message}", path.display())
            }
            Self::Regex(source) => write!(f, "tokenizer regex error: {source}"),
            Self::UnknownToken(token) => write!(f, "token not found in vocab: {token:?}"),
            Self::InvalidTokenId(id) => write!(f, "invalid token id: {id}"),
        }
    }
}

impl Error for TokenizerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json { source, .. } => Some(source),
            Self::Regex(source) => Some(source),
            Self::BadTokenizer { .. } | Self::UnknownToken(_) | Self::InvalidTokenId(_) => None,
        }
    }
}

/// One slice of input on the way into `encode`: either a special token matched
/// verbatim (already resolved to its id), or a run of ordinary text still to be
/// pre-tokenized + BPE'd.
enum Segment<'a> {
    Special(u32),
    Normal(&'a str),
}

/// Owns the loaded vocabulary + merge rules and maps text <-> token IDs.
///
/// All the BPE string keys here live in the *byte-level-unicode* alphabet (stage
/// 2 above), NOT raw UTF-8 — e.g. the piece for " world" is stored as "Ġworld".
pub struct Tokenizer {
    /// piece (byte-level-unicode) → token id.  Built from `model.vocab`.
    token_to_id: HashMap<String, u32>,

    /// token id → piece.  The inverse of `token_to_id`, for decoding.
    /// Dense: indexed directly by id (the vocab ids are 0..vocab_size).
    id_to_token: Vec<String>,

    /// A merge rule `(left, right)` → its rank (priority). LOWER = applied
    /// first. Built from `model.merges` (array index = rank). During stage 3 we
    /// pick, among a chunk's adjacent pairs, the one with the smallest rank.
    merge_ranks: HashMap<(String, String), u32>,

    /// byte value (0..=255) → its byte-level-unicode char. The GPT-2 map of
    /// stage 2. Indexed by the byte itself: `byte_encoder[0x20]` == 'Ġ'.
    byte_encoder: [char; 256],

    /// The inverse map: byte-level-unicode char → original byte. Used by
    /// `decode` to turn ordinary pieces back into raw bytes.
    byte_decoder: HashMap<char, u8>,

    /// Special/added tokens that match VERBATIM and bypass BPE, e.g.
    /// "<|im_start|>", "<|endoftext|>".  content → id.  From `added_tokens`.
    /// Their ids sit just past the BPE vocab (151643+), so they are NOT in
    /// `token_to_id`/`id_to_token`; `encode` carves them out before stage 1.
    special_tokens: HashMap<String, u32>,

    /// The reverse of `special_tokens`: id → content, so `decode` can turn a
    /// special id straight back into its literal text.
    special_ids: HashMap<u32, String>,

    /// Compiled stage-1 pre-tokenization regex, read from `tokenizer.json`'s
    /// `pre_tokenizer`. Qwen's pattern needs BOTH Unicode classes (\p{L}, \p{N})
    /// and a negative look-ahead (\s+(?!\S)), which is why we depend on
    /// `fancy-regex` instead of the `regex` crate.
    regex: fancy_regex::Regex,
}

impl Tokenizer {
    /// Load the tokenizer from a model directory (e.g. `models/qwen3-0.6b`),
    /// reading everything from `tokenizer.json`.
    ///
    /// Uses the custom `TokenizerError` enum so a missing file, malformed JSON,
    /// a malformed structure, or a regex problem stay distinguishable in tests
    /// and CLI output.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let path = model_dir.as_ref().join("tokenizer.json");
        let doc = Self::read_json(&path)?;

        let vocab = doc["model"]["vocab"]
            .as_object()
            .ok_or_else(|| TokenizerError::BadTokenizer {
                path: path.clone(),
                message: "missing `model.vocab` object".into(),
            })?;
        let (token_to_id, id_to_token) = Self::build_vocab(vocab, &path)?;

        let merges = doc["model"]["merges"]
            .as_array()
            .ok_or_else(|| TokenizerError::BadTokenizer {
                path: path.clone(),
                message: "missing `model.merges` array".into(),
            })?;
        let merge_ranks = Self::build_merges(merges, &path)?;

        // added_tokens is optional; absent → no special tokens.
        let (special_tokens, special_ids) = match doc["added_tokens"].as_array() {
            Some(added) => Self::build_special_tokens(added, &path)?,
            None => (HashMap::new(), HashMap::new()),
        };

        let byte_encoder = Self::build_byte_encoder();
        // byte_decoder is just the inverse map. build_byte_encoder is a bijection
        // over all 256 byte values, so this covers every char a piece can hold.
        let byte_decoder = byte_encoder
            .iter()
            .enumerate()
            .map(|(b, &ch)| (ch, b as u8))
            .collect();

        let pattern = Self::extract_pattern(&doc, &path)?;
        let regex = fancy_regex::Regex::new(&pattern).map_err(TokenizerError::Regex)?;

        Ok(Self {
            token_to_id,
            id_to_token,
            merge_ranks,
            byte_encoder,
            byte_decoder,
            special_tokens,
            special_ids,
            regex,
        })
    }

    /// Encode text into token IDs.  Special-token literals are carved out first
    /// (each → its id, bypassing BPE); every other run goes through the four
    /// stages: pre-tokenize → bytes→unicode → merge → look up.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        for segment in self.split_on_special_tokens(text) {
            match segment {
                Segment::Special(id) => ids.push(id),
                Segment::Normal(run) => {
                    for chunk in self.pretokenize(run)? {
                        // Stage 2: remap each raw byte to its byte-level-unicode
                        // char, so the chunk is in the vocab/merge alphabet.
                        let mapped: String =
                            chunk.bytes().map(|b| self.byte_encoder[b as usize]).collect();
                        // Stages 3+4: merge within this chunk, then look up.
                        ids.extend(self.bpe(&mapped)?);
                    }
                }
            }
        }
        Ok(ids)
    }

    /// Decode token IDs back into a UTF-8 string. A special id decodes to its
    /// literal text; an ordinary id is the inverse of stages 4→2 (piece → undo
    /// the byte map → raw bytes).
    ///
    /// (`_lossy` because an arbitrary ID slice can split a multi-byte char; for
    /// a faithful round-trip the bytes are always valid UTF-8.)
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            // Special tokens carry their literal text (raw UTF-8), not a
            // byte-level-unicode piece, so emit their bytes directly.
            if let Some(content) = self.special_ids.get(&id) {
                bytes.extend_from_slice(content.as_bytes());
                continue;
            }
            // Stage 4 inverse: id → piece. An id past the vocab that isn't a
            // known special is a typed error, not a panic.
            let piece = self
                .id_to_token
                .get(id as usize)
                .ok_or(TokenizerError::InvalidTokenId(id))?;
            // Stage 2 inverse: each char maps back to one byte. Every char of a
            // vocab piece is in the byte-level alphabet, so this never misses.
            for ch in piece.chars() {
                let byte = self
                    .byte_decoder
                    .get(&ch)
                    .expect("vocab piece char must be in the byte-level alphabet");
                bytes.push(*byte);
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    // ───────────────────────── private helpers ─────────────────────────

    /// Split text into special-token literals (resolved to ids) and the ordinary
    /// runs between them. Special tokens match VERBATIM and take priority over
    /// BPE; at each position we take the LONGEST matching special so a token that
    /// is a prefix of another can't shadow it.
    fn split_on_special_tokens<'a>(&self, text: &'a str) -> Vec<Segment<'a>> {
        if self.special_tokens.is_empty() {
            return vec![Segment::Normal(text)];
        }

        let mut out = Vec::new();
        let mut normal_start = 0; // start of the current run of ordinary text
        let mut i = 0;
        while i < text.len() {
            // Longest special token whose literal starts at byte position i.
            let mut best: Option<(&str, u32)> = None;
            for (content, &id) in &self.special_tokens {
                if text[i..].starts_with(content.as_str())
                    && best.is_none_or(|(b, _)| content.len() > b.len())
                {
                    best = Some((content.as_str(), id));
                }
            }
            if let Some((content, id)) = best {
                if normal_start < i {
                    out.push(Segment::Normal(&text[normal_start..i]));
                }
                out.push(Segment::Special(id));
                i += content.len();
                normal_start = i;
            } else {
                // No special here; advance one whole char (keep byte indices on
                // char boundaries so the `text[i..]` slicing above stays valid).
                i += text[i..].chars().next().map_or(1, char::len_utf8);
            }
        }
        if normal_start < text.len() {
            out.push(Segment::Normal(&text[normal_start..]));
        }
        out
    }

    /// Stage 1. Split an ordinary run into pre-token chunks per the Qwen regex.
    /// Returns borrowed slices into `text` (no copying).
    ///
    /// Example: " hello world" -> [" hello", " world"]   (spaces lead the word)
    fn pretokenize<'a>(&self, text: &'a str) -> Result<Vec<&'a str>> {
        // "Isolated" Split behavior means each regex match IS one chunk, and the
        // matches tile the whole input in order — so we just collect them, as
        // borrowed slices (no copying). fancy-regex backtracks, so a match can
        // fail mid-scan; that's why each item is a Result we propagate.
        let mut chunks = Vec::new();
        for found in self.regex.find_iter(text) {
            chunks.push(found.map_err(TokenizerError::Regex)?.as_str());
        }
        Ok(chunks)
    }

    /// Stages 3+4 for ONE chunk already mapped into byte-level-unicode.
    /// Replays the training-time merges, in priority (rank) order, on this one
    /// chunk: start from single chars and keep applying the highest-priority
    /// merge currently present until none apply; the survivors are the tokens.
    /// Then look each survivor up as an id.
    ///
    /// PSEUDO-CODE (the classic GPT-2 `bpe()` loop):
    ///   let mut symbols: Vec<String> = piece.chars().map(to_string).collect()
    ///   loop:
    ///       // the adjacent pair with the smallest rank ACROSS THE WHOLE chunk
    ///       // — NOT the first mergeable pair left-to-right (see trace below).
    ///       let best = Self::adjacent_pairs(&symbols)
    ///           .into_iter()
    ///           .filter_map(|p| self.merge_ranks.get(&p).map(|r| (*r, p)))
    ///           .min_by_key(|(r, _)| *r);
    ///       let Some((_, (l, r))) = best else { break };  // none mergeable → done
    ///       symbols = Self::merge_pair(symbols, &l, &r);  // glue every occurrence
    ///   // ranks are array indices, hence unique — no tie-break logic needed.
    ///   symbols.iter()
    ///       .map(|s| self.token_to_id.get(s).copied()
    ///                    .ok_or_else(|| TokenizerError::UnknownToken(s.clone())))
    ///       .collect()
    ///
    /// Why `.get().ok_or()` not `token_to_id[s]`: every char and every merge
    /// output is *supposed* to be in vocab, so a miss is "impossible" — but we
    /// surface it as a typed `UnknownToken` rather than panicking on the index.
    ///
    /// PERF — DEFERRED OPTIMIZATION (also logged in PROGRESS, Session 5): this
    /// rescans every pair each pass, so it's O(n²) per chunk. Fine for M0
    /// because `pretokenize` bounds n to a single "word." GPT-2's real code
    /// memoizes `bpe(word)` in a HashMap; add that cache in a later pass, NOT now.
    ///
    /// WORKED TRACE ("hello" -> 14990) — the canonical M0 example (also the
    /// worked example in docs/01-tokenizer.md). `#N` is the merge rank:
    ///   [h,e,l,l,o] --(e,l)#45--> [h,el,l,o] --(l,o)#129--> [h,el,lo]
    ///   --(el,lo)#4535--> [h,ello] --(h,ello)#14734--> [hello] -> id 14990
    /// Note (h,e)#127 was available at step 1 but (e,l)#45 outranked it; once
    /// 'e' was absorbed into 'el', (h,e) could never fire again — 'h' waited and
    /// eventually merged with the whole 'ello'. That starvation is BPE behaving
    /// correctly, and it's why we pick the global-min rank, not left-to-right.
    fn bpe(&self, piece: &str) -> Result<Vec<u32>> {
        // Stage 3: start from single chars, then repeatedly apply the single
        // highest-priority (lowest-rank) merge present anywhere in the chunk.
        let mut symbols: Vec<String> = piece.chars().map(|c| c.to_string()).collect();

        loop {
            let best = Self::adjacent_pairs(&symbols)
                .into_iter()
                .filter_map(|pair| self.merge_ranks.get(&pair).map(|&rank| (rank, pair)))
                .min_by_key(|(rank, _)| *rank);

            // No adjacent pair is a known merge → the chunk is fully tokenized.
            let Some((_, (left, right))) = best else { break };
            symbols = Self::merge_pair(symbols, &left, &right);
        }

        // Stage 4: each surviving piece is a vocab key. A miss is "impossible"
        // (every char and every merge output is in vocab) — but we surface it as
        // a typed error rather than panic on a bad index. See the doc above.
        symbols
            .iter()
            .map(|s| {
                self.token_to_id
                    .get(s)
                    .copied()
                    .ok_or_else(|| TokenizerError::UnknownToken(s.clone()))
            })
            .collect()
    }

    /// Enumerate adjacent `(left, right)` symbol pairs, in left-to-right order.
    /// Returns owned clones so the caller can look each up in `merge_ranks` and
    /// hand the winning pair to `merge_pair` without fighting the borrow checker.
    fn adjacent_pairs(symbols: &[String]) -> Vec<(String, String)> {
        symbols
            .windows(2)
            .map(|w| (w[0].clone(), w[1].clone()))
            .collect()
    }

    /// Rewrite `symbols`, gluing EVERY non-overlapping occurrence of `(l, r)` in
    /// a single left-to-right pass (advance by 2 on a hit, by 1 otherwise). So
    /// `[a, a, a]` merging `(a, a)` yields `[aa, a]` — never a reused middle 'a'.
    fn merge_pair(symbols: Vec<String>, l: &str, r: &str) -> Vec<String> {
        let mut out = Vec::with_capacity(symbols.len());
        let mut iter = symbols.into_iter().peekable();
        while let Some(sym) = iter.next() {
            // On a hit, consume the right half too (the second `next`), so the
            // next loop step starts past it — that's what makes it non-overlapping.
            if sym == l && iter.peek().map(String::as_str) == Some(r) {
                let right = iter.next().expect("peek just confirmed a next element");
                out.push(sym + &right);
            } else {
                out.push(sym);
            }
        }
        out
    }

    /// Build GPT-2's byte→unicode table (stage 2). The 188 "printable" bytes
    /// (ranges !..~, ¡..¬, ®..ÿ) map to themselves; the other 68 map to
    /// codepoints 256, 257, … in byte order. That's why 0x20 → U+0120 ('Ġ').
    fn build_byte_encoder() -> [char; 256] {
        // `next` hands out the spare codepoints (U+0100, U+0101, …) to the bytes
        // that aren't already a safe printable char, in increasing byte order.
        let mut encoder = ['\0'; 256];
        let mut next: u32 = 0x100;
        for b in 0u32..256 {
            let is_printable = (0x21..=0x7E).contains(&b)   // ! .. ~
                || (0xA1..=0xAC).contains(&b)               // ¡ .. ¬
                || (0xAE..=0xFF).contains(&b); // ® .. ÿ
            let code = if is_printable {
                b // map to the same codepoint
            } else {
                let c = next; // borrow the next spare, then advance
                next += 1;
                c
            };
            // Every `code` here is 0x21..=0xFF or 0x100..=0x143 — all valid
            // (non-surrogate) scalar values, so `from_u32` can never be None.
            encoder[b as usize] = char::from_u32(code).expect("valid scalar value");
        }
        encoder
    }

    /// Read + parse a JSON file into a `serde_json::Value`. Parsing JSON is not
    /// the tokenizer lesson, so we lean on serde_json.
    fn read_json(path: &Path) -> Result<Value> {
        let bytes = std::fs::read(path).map_err(|source| TokenizerError::Io {
            path: path.into(),
            source,
        })?;
        serde_json::from_slice(&bytes).map_err(|source| TokenizerError::Json {
            path: path.into(),
            source,
        })
    }

    /// Build the forward map and the dense reverse vector from `model.vocab`
    /// (a JSON object of piece → id). Returns (token_to_id, id_to_token).
    fn build_vocab(
        vocab: &Map<String, Value>,
        path: &Path,
    ) -> Result<(HashMap<String, u32>, Vec<String>)> {
        let mut token_to_id = HashMap::with_capacity(vocab.len());
        for (token, id) in vocab {
            let id = id
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| TokenizerError::BadTokenizer {
                    path: path.into(),
                    message: format!("vocab id for {token:?} is not a u32: {id}"),
                })?;
            token_to_id.insert(token.clone(), id);
        }

        // Build the dense reverse map. Qwen's ids are exactly 0..vocab_size, so
        // a Vec indexed by id is the smallest and fastest inverse. We fill an
        // Option slot per id so we can name the precise defect (id out of range,
        // duplicate, or a hole) instead of silently producing a wrong table.
        let vocab_size = token_to_id.len();
        let mut slots: Vec<Option<String>> = vec![None; vocab_size];
        for (token, &id) in &token_to_id {
            let slot = slots
                .get_mut(id as usize)
                .ok_or_else(|| TokenizerError::BadTokenizer {
                    path: path.into(),
                    message: format!("token {token:?} has id {id} >= vocab size {vocab_size}"),
                })?;
            if slot.is_some() {
                return Err(TokenizerError::BadTokenizer {
                    path: path.into(),
                    message: format!("id {id} is claimed by more than one token"),
                });
            }
            *slot = Some(token.clone());
        }
        // Any leftover hole means the ids weren't a contiguous 0..vocab_size run.
        let id_to_token = slots
            .into_iter()
            .enumerate()
            .map(|(id, slot)| {
                slot.ok_or_else(|| TokenizerError::BadTokenizer {
                    path: path.into(),
                    message: format!("no token maps to id {id}; ids are not contiguous"),
                })
            })
            .collect::<Result<Vec<String>>>()?;

        Ok((token_to_id, id_to_token))
    }

    /// Build `(left, right) -> rank` from `model.merges`. The array is already in
    /// priority order, so the rank is simply the index — earliest = highest
    /// priority. No header to skip (unlike GPT-2's `merges.txt`).
    fn build_merges(merges: &[Value], path: &Path) -> Result<HashMap<(String, String), u32>> {
        let mut merge_ranks = HashMap::with_capacity(merges.len());
        for (rank, entry) in merges.iter().enumerate() {
            let (left, right) =
                Self::merge_entry(entry).ok_or_else(|| TokenizerError::BadTokenizer {
                    path: path.into(),
                    message: format!("merge #{rank} is not a [left, right] pair: {entry}"),
                })?;
            merge_ranks.insert((left.to_string(), right.to_string()), rank as u32);
        }
        Ok(merge_ranks)
    }

    /// A merges entry is `["left", "right"]` (modern `tokenizer.json`) or the
    /// legacy space-joined `"left right"` string. Accept both, for robustness
    /// across model formats. A piece never contains a space (the space byte maps
    /// to 'Ġ'), so the legacy `split_once(' ')` is exact, not a guess.
    fn merge_entry(entry: &Value) -> Option<(&str, &str)> {
        match entry {
            Value::Array(pair) if pair.len() == 2 => Some((pair[0].as_str()?, pair[1].as_str()?)),
            Value::String(line) => line.split_once(' '),
            _ => None,
        }
    }

    /// Build the special-token maps (content → id and id → content) from
    /// `added_tokens`. These match verbatim and bypass BPE.
    fn build_special_tokens(
        added: &[Value],
        path: &Path,
    ) -> Result<(HashMap<String, u32>, HashMap<u32, String>)> {
        let mut by_content = HashMap::with_capacity(added.len());
        let mut by_id = HashMap::with_capacity(added.len());
        for entry in added {
            let content =
                entry["content"]
                    .as_str()
                    .ok_or_else(|| TokenizerError::BadTokenizer {
                        path: path.into(),
                        message: format!("added token has no string `content`: {entry}"),
                    })?;
            let id = entry["id"]
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| TokenizerError::BadTokenizer {
                    path: path.into(),
                    message: format!("added token {content:?} has a non-u32 `id`"),
                })?;
            by_content.insert(content.to_string(), id);
            by_id.insert(id, content.to_string());
        }
        Ok((by_content, by_id))
    }

    /// Pull the stage-1 split pattern out of `pre_tokenizer`. It's normally a
    /// Sequence of sub-tokenizers (a Split with the Regex, then ByteLevel), but
    /// we also handle a single node — we just take the first Regex we find.
    fn extract_pattern(doc: &Value, path: &Path) -> Result<String> {
        let pt = &doc["pre_tokenizer"];
        let nodes: &[Value] = match pt["pretokenizers"].as_array() {
            Some(arr) => arr.as_slice(),
            None => std::slice::from_ref(pt),
        };
        for node in nodes {
            if let Some(pattern) = node["pattern"]["Regex"].as_str() {
                return Ok(pattern.to_string());
            }
        }
        Err(TokenizerError::BadTokenizer {
            path: path.into(),
            message: "no pre_tokenizer Regex pattern found".into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashSet;

    /// Qwen3's exact pre-tokenization pattern — a verbatim copy of what
    /// `tokenizer.json` carries, used to drive the unit tests below. (`load`
    /// reads the live pattern from the file; this is just a fixed reference so
    /// the `pretokenize` tests don't need the model assets.)
    const PRETOKENIZE_PATTERN: &str =
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    /// Build a Tokenizer from tiny in-memory tables, for exercising `bpe` /
    /// `pretokenize` / special-token carving without loading the real assets.
    fn mini_tokenizer(vocab: &[(&str, u32)], merges: &[((&str, &str), u32)]) -> Tokenizer {
        Tokenizer {
            token_to_id: vocab.iter().map(|(t, id)| (t.to_string(), *id)).collect(),
            id_to_token: Vec::new(),
            merge_ranks: merges
                .iter()
                .map(|((l, r), rank)| ((l.to_string(), r.to_string()), *rank))
                .collect(),
            byte_encoder: Tokenizer::build_byte_encoder(),
            byte_decoder: HashMap::new(),
            special_tokens: HashMap::new(),
            special_ids: HashMap::new(),
            regex: fancy_regex::Regex::new(PRETOKENIZE_PATTERN).expect("valid pattern"),
        }
    }

    #[test]
    fn byte_encoder_landmarks() {
        let enc = Tokenizer::build_byte_encoder();

        // Printable bytes map to themselves.
        assert_eq!(enc[b'!' as usize], '!');
        assert_eq!(enc[b'A' as usize], 'A');
        assert_eq!(enc[b'~' as usize], '~');
        assert_eq!(enc[0xFF], '\u{00FF}'); // 'ÿ', top of the high printable range

        // The famous remaps that show up in vocab keys.
        assert_eq!(enc[0x20], '\u{0120}'); // space   -> 'Ġ'
        assert_eq!(enc[0x0A], '\u{010A}'); // newline -> 'Ċ'

        // The first non-printable byte (0x00) takes the first spare codepoint.
        assert_eq!(enc[0x00], '\u{0100}');
        // The three single-byte gaps inside the printable ranges are non-printable.
        assert_eq!(enc[0x7F], '\u{0121}'); // right after space's 0x120
    }

    #[test]
    fn byte_encoder_is_a_bijection() {
        let enc = Tokenizer::build_byte_encoder();
        let distinct: HashSet<char> = enc.iter().copied().collect();
        assert_eq!(distinct.len(), 256, "every byte must map to a distinct char");
    }

    #[test]
    fn build_vocab_builds_dense_reverse_map() {
        let vocab = json!({ "a": 0, "b": 1, "Ġc": 2 });
        let (fwd, rev) =
            Tokenizer::build_vocab(vocab.as_object().unwrap(), Path::new("x")).unwrap();
        assert_eq!(fwd["a"], 0);
        assert_eq!(fwd["Ġc"], 2);
        // reverse map is indexed by id, so order == id order
        assert_eq!(rev, vec!["a".to_string(), "b".to_string(), "Ġc".to_string()]);
    }

    #[test]
    fn build_vocab_rejects_an_out_of_range_id() {
        let vocab = json!({ "a": 0, "b": 2 }); // len 2, but id 2
        assert!(matches!(
            Tokenizer::build_vocab(vocab.as_object().unwrap(), Path::new("x")),
            Err(TokenizerError::BadTokenizer { .. })
        ));
    }

    #[test]
    fn build_merges_ranks_from_zero() {
        // Modern array form, in priority order: rank is the index.
        let merges = json!([["Ġ", "Ġ"], ["i", "n"], ["Ġ", "t"]]);
        let m = Tokenizer::build_merges(merges.as_array().unwrap(), Path::new("x")).unwrap();
        assert_eq!(m[&("Ġ".to_string(), "Ġ".to_string())], 0);
        assert_eq!(m[&("i".to_string(), "n".to_string())], 1);
        assert_eq!(m[&("Ġ".to_string(), "t".to_string())], 2);
        assert_eq!(m.len(), 3);
    }

    #[test]
    fn build_merges_accepts_legacy_string_form() {
        // Older tokenizer.json files store each merge as "left right".
        let merges = json!(["a b", "c d"]);
        let m = Tokenizer::build_merges(merges.as_array().unwrap(), Path::new("x")).unwrap();
        assert_eq!(m[&("a".to_string(), "b".to_string())], 0);
        assert_eq!(m[&("c".to_string(), "d".to_string())], 1);
    }

    #[test]
    fn build_merges_rejects_a_malformed_entry() {
        let merges = json!([["only-one-element"]]);
        assert!(matches!(
            Tokenizer::build_merges(merges.as_array().unwrap(), Path::new("x")),
            Err(TokenizerError::BadTokenizer { .. })
        ));
    }

    #[test]
    fn extract_pattern_finds_regex_in_sequence() {
        let doc = json!({
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    { "type": "Split", "pattern": { "Regex": "ABC" } },
                    { "type": "ByteLevel" }
                ]
            }
        });
        assert_eq!(Tokenizer::extract_pattern(&doc, Path::new("x")).unwrap(), "ABC");
    }

    #[test]
    fn adjacent_pairs_are_consecutive() {
        let s = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        assert_eq!(
            Tokenizer::adjacent_pairs(&s),
            vec![("x".into(), "y".into()), ("y".into(), "z".into())]
        );
        // Fewer than two symbols → no pairs → `bpe` breaks immediately.
        let empty: Vec<String> = vec![];
        assert!(Tokenizer::adjacent_pairs(&empty).is_empty());
        assert!(Tokenizer::adjacent_pairs(&["solo".to_string()]).is_empty());
    }

    #[test]
    fn merge_pair_is_non_overlapping() {
        let s = vec!["a".to_string(), "a".to_string(), "a".to_string()];
        // The middle 'a' is consumed by the first merge, not reused by a second.
        assert_eq!(
            Tokenizer::merge_pair(s, "a", "a"),
            vec!["aa".to_string(), "a".to_string()]
        );
    }

    #[test]
    fn bpe_reproduces_the_hello_trace() {
        // A synthetic table mirroring the real merge ORDER for "hello":
        //   (e,l) < (l,o) < (el,lo) < (h,ello).  Ranks just need that ordering.
        let tok = mini_tokenizer(
            &[
                ("h", 0),
                ("e", 1),
                ("l", 2),
                ("o", 3),
                ("el", 4),
                ("lo", 5),
                ("ello", 6),
                ("hello", 7),
            ],
            &[
                (("e", "l"), 0),
                (("l", "o"), 1),
                (("el", "lo"), 2),
                (("h", "ello"), 3),
            ],
        );
        // "hello" is pure printable ASCII, so its byte-level-unicode form is itself.
        // [h,e,l,l,o] -> [h,el,l,o] -> [h,el,lo] -> [h,ello] -> [hello] -> id 7.
        assert_eq!(tok.bpe("hello").unwrap(), vec![7]);
    }

    #[test]
    fn bpe_errors_on_a_piece_outside_the_vocab() {
        // 'z' never merges and isn't in the vocab → typed UnknownToken, no panic.
        let tok = mini_tokenizer(&[("a", 0)], &[]);
        assert!(matches!(tok.bpe("z"), Err(TokenizerError::UnknownToken(_))));
    }

    #[test]
    fn pretokenize_splits_into_words_keeping_leading_spaces() {
        let tok = mini_tokenizer(&[], &[]);
        // Ground truth from the official tokenizer (with Ġ mapped back to space).
        assert_eq!(tok.pretokenize("hello world").unwrap(), ["hello", " world"]);
        assert_eq!(tok.pretokenize(" hello world").unwrap(), [" hello", " world"]);
        assert_eq!(
            tok.pretokenize("The capital of France is").unwrap(),
            ["The", " capital", " of", " France", " is"]
        );
    }

    #[test]
    fn pretokenize_handles_contractions_digits_and_symbols() {
        let tok = mini_tokenizer(&[], &[]);
        assert_eq!(tok.pretokenize("don't").unwrap(), ["don", "'t"]);
        // each digit is its own chunk: the pattern is \p{N}, not \p{N}+
        assert_eq!(tok.pretokenize("abc123").unwrap(), ["abc", "1", "2", "3"]);
        assert_eq!(tok.pretokenize("a + b").unwrap(), ["a", " +", " b"]);
    }

    #[test]
    fn pretokenize_chunks_tile_the_whole_input() {
        let tok = mini_tokenizer(&[], &[]);
        // Every byte must belong to exactly one chunk — concatenation recovers
        // the original string (the property `encode` relies on for round-trips).
        for s in ["hello world", " leading", "trailing ", "a\n\nb", "café 🚀"] {
            assert_eq!(tok.pretokenize(s).unwrap().concat(), s);
        }
    }

    #[test]
    fn special_tokens_are_carved_out_and_decoded() {
        // A mini tokenizer that can BPE "hello" (-> 7), plus a special token.
        let mut tok = mini_tokenizer(
            &[
                ("h", 0),
                ("e", 1),
                ("l", 2),
                ("o", 3),
                ("el", 4),
                ("lo", 5),
                ("ello", 6),
                ("hello", 7),
            ],
            &[
                (("e", "l"), 0),
                (("l", "o"), 1),
                (("el", "lo"), 2),
                (("h", "ello"), 3),
            ],
        );
        tok.special_tokens.insert("<|s|>".to_string(), 100);
        tok.special_ids.insert(100, "<|s|>".to_string());

        // encode: the special literal becomes its id, bypassing BPE; the rest
        // ("hello") is BPE'd normally to [7].
        assert_eq!(tok.encode("<|s|>hello").unwrap(), vec![100, 7]);
        assert_eq!(tok.encode("hello<|s|>").unwrap(), vec![7, 100]);
        // decode: a special id returns its literal text.
        assert_eq!(tok.decode(&[100]).unwrap(), "<|s|>");
    }

    #[test]
    fn no_special_tokens_means_one_normal_segment() {
        // With an empty special map, encode just runs the normal pipeline.
        let tok = mini_tokenizer(&[("a", 0)], &[]);
        assert_eq!(tok.encode("a").unwrap(), vec![0]);
    }
}
