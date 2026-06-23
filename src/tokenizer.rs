//! Byte-level BPE tokenizer for Qwen3-0.6B.  (Milestone M0)
//!
//! STATUS: **annotated sketch.** This is the *shape* of the whole thing — real
//! fields, real helper signatures, and step-by-step pseudo-code in comments —
//! but every body is still `todo!()`. We read this top-to-bottom, then fill in
//! one method at a time, together. Nothing here runs yet; it only compiles.
//!
//! ──────────────────────────────────────────────────────────────────────────
//! THE BIG PICTURE — what "byte-level BPE" means (GPT-2 → Qwen lineage)
//! ──────────────────────────────────────────────────────────────────────────
//! Encoding text → token IDs happens in four stages:
//!
//!   1. PRE-TOKENIZE.  A regex chops the text into chunks ("words"), so a merge
//!      can never cross, say, a word/punctuation boundary. Crucially a leading
//!      space stays *attached* to the word after it — that's why our golden has
//!          "hello world"  -> [14990, 1879]
//!          " hello world" -> [23811, 1879]   (different FIRST token)
//!
//!   2. BYTES → "BYTE-LEVEL UNICODE".  Each chunk is raw UTF-8 bytes. We remap
//!      all 256 byte values onto printable Unicode chars (GPT-2's trick), so a
//!      token string never contains a real space/newline/control byte. Under
//!      this map a space (0x20) becomes 'Ġ', a newline (0x0A) becomes 'Ċ'. The
//!      vocab on disk is written in THIS alphabet — `vocab.json` literally has
//!      keys like "Ġworld".
//!
//!   3. MERGE.  Starting from single chars, repeatedly glue the adjacent pair
//!      with the best (lowest-numbered) rank in `merges.txt`, until no adjacent
//!      pair is mergeable. The survivors are the final token pieces.
//!
//!   4. LOOK UP.  Each final piece is a key in the vocab → its integer ID.
//!
//! Decoding reverses 4→2: IDs → pieces → concatenate → undo the byte map →
//! interpret the bytes as UTF-8.
//!
//! Inputs on disk (from `scripts/fetch_model.py`, in `models/qwen3-0.6b/`):
//!   - `vocab.json`   — { "<piece in byte-level-unicode>": id, … }
//!   - `merges.txt`   — one "<left> <right>" per line, in PRIORITY order
//!                      (line 0 is a "#version:" header; earlier line = better).
//!   - `tokenizer_config.json` / `tokenizer.json` — special tokens + the regex.
//!
//! Verify (M0 "done"): reproduce `tests/golden/tokenizer.json` exactly, and
//! round-trip decode(encode(s)) == s. We generated that file from the official
//! tokenizer with `add_special_tokens=False` on plain text — so for the first
//! pass we can ignore special-token *insertion* and focus on stages 1–4.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};

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

    /// `vocab.json` parsed as JSON, but its shape/content is not what we need.
    BadVocab { path: PathBuf, message: String },

    /// `merges.txt` has a malformed line or inconsistent merge rule.
    BadMerges {
        path: PathBuf,
        line: usize,
        message: String,
    },

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
            Self::BadVocab { path, message } => {
                write!(f, "bad vocab file {}: {message}", path.display())
            }
            Self::BadMerges {
                path,
                line,
                message,
            } => write!(
                f,
                "bad merges file {} at line {line}: {message}",
                path.display()
            ),
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
            Self::BadVocab { .. }
            | Self::BadMerges { .. }
            | Self::UnknownToken(_)
            | Self::InvalidTokenId(_) => None,
        }
    }
}

/// Owns the loaded vocabulary + merge rules and maps text <-> token IDs.
///
/// All the string keys here live in the *byte-level-unicode* alphabet (stage 2
/// above), NOT raw UTF-8 — e.g. the piece for " world" is stored as "Ġworld".
pub struct Tokenizer {
    /// piece (byte-level-unicode) → token id.  Built from `vocab.json`.
    token_to_id: HashMap<String, u32>,

    /// token id → piece.  The inverse of `token_to_id`, for decoding.
    /// Dense: indexed directly by id (the vocab ids are 0..vocab_size).
    id_to_token: Vec<String>,

    /// A merge rule `(left, right)` → its rank (priority). LOWER = applied
    /// first. Built from `merges.txt` (line number = rank). During stage 3 we
    /// pick, among a chunk's adjacent pairs, the one with the smallest rank.
    merge_ranks: HashMap<(String, String), u32>,

    /// byte value (0..=255) → its byte-level-unicode char. The GPT-2 map of
    /// stage 2. Indexed by the byte itself: `byte_encoder[0x20]` == 'Ġ'.
    byte_encoder: [char; 256],

    /// The inverse map: byte-level-unicode char → original byte. Used by
    /// `decode` to turn pieces back into raw bytes.
    byte_decoder: HashMap<char, u8>,

    /// Special/added tokens that must match VERBATIM and bypass BPE, e.g.
    /// "<|im_start|>", "<|endoftext|>". From `tokenizer_config.json`. These
    /// account for the gap between `vocab.json`'s size and config's
    /// `vocab_size` (151936). Not exercised by our first golden pass.
    #[allow(dead_code)] // wired in phase 2 (special-token carving); stubbed empty for M0
    special_tokens: HashMap<String, u32>,

    /// Compiled stage-1 pre-tokenization regex (see `PRETOKENIZE_PATTERN`).
    /// Qwen's pattern needs BOTH Unicode classes (\p{L}, \p{N}) and a negative
    /// look-ahead (\s+(?!\S)), which is why we depend on `fancy-regex` instead of
    /// the `regex` crate. Compiled once in `load`.
    regex: fancy_regex::Regex,
}

/// Qwen3's exact pre-tokenization pattern, copied VERBATIM from
/// `tokenizer.json` (`pre_tokenizer` → Split → Regex) — this is the GPT-4 /
/// cl100k style. Each match is one chunk, and the matches tile the whole input.
/// The branches, in order: contractions (`'s`, `'t`, …, case-insensitive); an
/// optional leading symbol then a run of letters (this is what keeps a leading
/// space attached, e.g. " world"); a SINGLE digit; an optional space then a run
/// of symbols; newline runs; trailing whitespace via the `(?!\S)` look-ahead;
/// and any remaining whitespace. The look-ahead is the reason for fancy-regex.
const PRETOKENIZE_PATTERN: &str =
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

impl Tokenizer {
    /// Load vocab + merges from a model directory (e.g. `models/qwen3-0.6b`).
    ///
    /// PSEUDO-CODE:
    ///   1. let (token_to_id, id_to_token) = load_vocab(dir/"vocab.json")?
    ///   2. let merge_ranks               = load_merges(dir/"merges.txt")?
    ///   3. let byte_encoder              = build_byte_encoder()
    ///   4. let byte_decoder              = invert(byte_encoder)
    ///   5. let special_tokens            = load_special_tokens(dir/"tokenizer_config.json")?  // later
    ///   6. Ok(Tokenizer { … })
    ///
    /// Uses the custom `TokenizerError` enum so missing files, malformed JSON,
    /// malformed merges, and regex problems stay distinguishable in tests and
    /// CLI output.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let dir = model_dir.as_ref();
        let (token_to_id, id_to_token) = Self::load_vocab(&dir.join("vocab.json"))?;
        let merge_ranks = Self::load_merges(&dir.join("merges.txt"))?;

        let byte_encoder = Self::build_byte_encoder();
        // byte_decoder is just the inverse map. build_byte_encoder is a bijection
        // over all 256 byte values, so this covers every char a piece can hold.
        let byte_decoder = byte_encoder
            .iter()
            .enumerate()
            .map(|(b, &ch)| (ch, b as u8))
            .collect();

        let regex = fancy_regex::Regex::new(PRETOKENIZE_PATTERN).map_err(TokenizerError::Regex)?;

        Ok(Self {
            token_to_id,
            id_to_token,
            merge_ranks,
            byte_encoder,
            byte_decoder,
            // Special tokens bypass BPE; our golden uses add_special_tokens=False
            // with no special-token literals, so an empty map is correct for M0.
            special_tokens: HashMap::new(),
            regex,
        })
    }

    /// Encode text into token IDs via byte-level BPE (stages 1→4).
    ///
    /// PSEUDO-CODE:
    ///   let mut ids = vec![]
    ///   // (phase 2, later) first carve out any special tokens verbatim.
    ///   for chunk in self.pretokenize(text):                    // stage 1
    ///       let mapped: String =
    ///           chunk.bytes().map(|b| self.byte_encoder[b]).collect()  // stage 2
    ///       ids.extend(self.bpe(&mapped))                        // stages 3+4
    ///   ids
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        for chunk in self.pretokenize(text)? {
            // Stage 2: remap each raw byte to its byte-level-unicode char, so the
            // chunk is in the same alphabet as the vocab/merge keys.
            let mapped: String = chunk.bytes().map(|b| self.byte_encoder[b as usize]).collect();
            // Stages 3+4: merge within this chunk, then look the pieces up.
            ids.extend(self.bpe(&mapped)?);
        }
        Ok(ids)
    }

    /// Decode token IDs back into a UTF-8 string (inverse of stages 4→2).
    ///
    /// PSEUDO-CODE:
    ///   let mut bytes = vec![]
    ///   for id in ids:
    ///       let piece = &self.id_to_token[id]      // stage 4 inverse
    ///       for ch in piece.chars():
    ///           bytes.push(self.byte_decoder[ch])  // stage 2 inverse
    ///   String::from_utf8_lossy(&bytes).into_owned()
    ///
    /// (`_lossy` because an arbitrary ID slice can split a multi-byte char;
    /// for a faithful round-trip the bytes are always valid UTF-8.)
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            // Stage 4 inverse: id → piece. Out-of-range ids (e.g. special tokens,
            // which our id_to_token doesn't cover) are a typed error, not a panic.
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
        // `_lossy` guards an arbitrary id slice that splits a multi-byte char; a
        // faithful round-trip always yields valid UTF-8.
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    // ───────────────────────── private helpers ─────────────────────────
    // These are the pieces we'll build and test one-by-one. Splitting them out
    // means each gets its own small unit test against the golden data.

    /// Stage 1. Split raw text into pre-token chunks per the Qwen regex.
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
    ///   // ranks are line numbers, hence unique — no tie-break logic needed.
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

    /// Parse `vocab.json` into the forward map and the dense reverse vector.
    /// Returns (token_to_id, id_to_token).
    fn load_vocab(path: &Path) -> Result<(HashMap<String, u32>, Vec<String>)> {
        // `vocab.json` is one flat JSON object: { "<piece>": <id>, … }. We let
        // serde_json do the parsing (not the tokenizer lesson) and deserialize
        // straight into the forward map.
        let bytes = std::fs::read(path).map_err(|source| TokenizerError::Io {
            path: path.into(),
            source,
        })?;
        let token_to_id: HashMap<String, u32> =
            serde_json::from_slice(&bytes).map_err(|source| TokenizerError::Json {
                path: path.into(),
                source,
            })?;

        // Build the dense reverse map. Qwen's ids are exactly 0..vocab_size, so
        // a Vec indexed by id is the smallest and fastest inverse. We fill an
        // Option slot per id so we can name the precise defect (id out of range,
        // duplicate, or a hole) instead of silently producing a wrong table.
        let vocab_size = token_to_id.len();
        let mut slots: Vec<Option<String>> = vec![None; vocab_size];
        for (token, &id) in &token_to_id {
            let slot = slots
                .get_mut(id as usize)
                .ok_or_else(|| TokenizerError::BadVocab {
                    path: path.into(),
                    message: format!("token {token:?} has id {id} >= vocab size {vocab_size}"),
                })?;
            if slot.is_some() {
                return Err(TokenizerError::BadVocab {
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
                slot.ok_or_else(|| TokenizerError::BadVocab {
                    path: path.into(),
                    message: format!("no token maps to id {id}; ids are not contiguous"),
                })
            })
            .collect::<Result<Vec<String>>>()?;

        Ok((token_to_id, id_to_token))
    }

    /// Parse `merges.txt` into `(left, right) -> rank`. Skip the `#version`
    /// header; the rank is the (post-header) line index — earlier = higher
    /// priority. Each line is exactly two space-separated pieces.
    fn load_merges(path: &Path) -> Result<HashMap<(String, String), u32>> {
        let text = std::fs::read_to_string(path).map_err(|source| TokenizerError::Io {
            path: path.into(),
            source,
        })?;

        // `rank` counts only real merge lines, so the first one is 0 regardless
        // of the header/blank lines we skip. A piece can never contain a space
        // (the space byte maps to 'Ġ'), so `split(' ')` is exact, not a guess.
        let mut merge_ranks = HashMap::new();
        let mut rank: u32 = 0;
        for (lineno, line) in text.lines().enumerate() {
            if line.is_empty() || line.starts_with("#version") {
                continue;
            }
            let mut parts = line.split(' ');
            let (Some(left), Some(right), None) = (parts.next(), parts.next(), parts.next()) else {
                return Err(TokenizerError::BadMerges {
                    path: path.into(),
                    line: lineno + 1, // 1-based, to match an editor's gutter
                    message: format!("expected exactly two space-separated pieces, got {line:?}"),
                });
            };
            merge_ranks.insert((left.to_string(), right.to_string()), rank);
            rank += 1;
        }
        Ok(merge_ranks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};
    use std::path::PathBuf;

    /// Build a Tokenizer from tiny in-memory tables, for exercising `bpe` without
    /// loading the real assets. Only the fields `bpe` touches need real content.
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
            regex: fancy_regex::Regex::new(PRETOKENIZE_PATTERN).expect("valid pattern"),
        }
    }

    /// Write a small fixture to a uniquely-named temp file (the real assets live
    /// in the git-ignored `models/`, so unit tests use their own tiny inputs).
    fn write_temp(name: &str, contents: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("fs-tok-test-{name}"));
        std::fs::write(&path, contents).expect("write temp fixture");
        path
    }

    #[test]
    fn byte_encoder_landmarks() {
        let enc = Tokenizer::build_byte_encoder();

        // Printable bytes map to themselves.
        assert_eq!(enc[b'!' as usize], '!');
        assert_eq!(enc[b'A' as usize], 'A');
        assert_eq!(enc[b'~' as usize], '~');
        assert_eq!(enc[0xFF], '\u{00FF}'); // 'ÿ', top of the high printable range

        // The famous remaps that show up in vocab.json keys.
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
    fn load_vocab_builds_dense_reverse_map() {
        let path = write_temp("vocab.json", r#"{"a":0,"b":1,"Ġc":2}"#);
        let (fwd, rev) = Tokenizer::load_vocab(&path).unwrap();
        assert_eq!(fwd["a"], 0);
        assert_eq!(fwd["Ġc"], 2);
        // reverse map is indexed by id, so order == id order
        assert_eq!(rev, vec!["a".to_string(), "b".to_string(), "Ġc".to_string()]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_vocab_rejects_an_out_of_range_id() {
        let path = write_temp("vocab-bad.json", r#"{"a":0,"b":2}"#); // len 2, but id 2
        assert!(matches!(
            Tokenizer::load_vocab(&path),
            Err(TokenizerError::BadVocab { .. })
        ));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_merges_ranks_from_zero_skipping_header() {
        let path = write_temp("merges.txt", "#version: 0.2\nĠ Ġ\ni n\nĠ t\n");
        let merges = Tokenizer::load_merges(&path).unwrap();
        // first real merge is rank 0, not rank 1 — the off-by-one we care about
        assert_eq!(merges[&("Ġ".to_string(), "Ġ".to_string())], 0);
        assert_eq!(merges[&("i".to_string(), "n".to_string())], 1);
        assert_eq!(merges[&("Ġ".to_string(), "t".to_string())], 2);
        assert_eq!(merges.len(), 3); // header + trailing blank both skipped
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_merges_rejects_a_malformed_line() {
        let path = write_temp("merges-bad.txt", "#version: 0.2\na b c\n");
        assert!(matches!(
            Tokenizer::load_merges(&path),
            Err(TokenizerError::BadMerges { .. })
        ));
        std::fs::remove_file(&path).ok();
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
        assert!(matches!(
            tok.bpe("z"),
            Err(TokenizerError::UnknownToken(_))
        ));
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
}
