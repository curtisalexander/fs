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

// TEMPORARY: this whole module is a sketch, so fields/helpers are not used yet.
// Delete this once the methods are implemented and everything is wired up.
#![allow(dead_code)]

use std::collections::HashMap;
use std::path::Path;

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
    special_tokens: HashMap<String, u32>,
    // TODO(decision): we also need the PRE-TOKENIZATION regex (stage 1). Rust's
    // std has no regex engine, and the Qwen pattern uses BOTH Unicode classes
    // (\p{L}, \p{N}) and a negative look-ahead (\s+(?!\S)). The `regex` crate
    // does Unicode classes but NOT look-ahead; `fancy-regex` does both. So the
    // options are: (a) add `fancy-regex`, or (b) hand-roll a splitter for just
    // this pattern. We'll choose together when we implement `pretokenize`. Until
    // then there's no regex field here.
}

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
    /// NOTE on the return type: `Result<Self, String>` is a deliberately cheap
    /// placeholder so the CLI compiles. We may graduate this to a real error
    /// enum (missing file vs. bad JSON vs. malformed merge line) when we wire it.
    pub fn load(_model_dir: impl AsRef<Path>) -> Result<Self, String> {
        todo!("Tokenizer::load — implement after we read load_vocab/load_merges")
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
    pub fn encode(&self, _text: &str) -> Vec<u32> {
        todo!("Tokenizer::encode — implement after pretokenize + bpe")
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
    pub fn decode(&self, _ids: &[u32]) -> String {
        todo!("Tokenizer::decode — implement after id_to_token + byte_decoder exist")
    }

    // ───────────────────────── private helpers ─────────────────────────
    // These are the pieces we'll build and test one-by-one. Splitting them out
    // means each gets its own small unit test against the golden data.

    /// Stage 1. Split raw text into pre-token chunks per the Qwen regex.
    /// Returns borrowed slices into `text` (no copying).
    ///
    /// Example: " hello world" -> [" hello", " world"]   (spaces lead the word)
    fn pretokenize<'a>(&self, _text: &'a str) -> Vec<&'a str> {
        todo!("pretokenize — depends on the regex DECISION noted on the struct")
    }

    /// Stages 3+4 for ONE chunk already mapped into byte-level-unicode.
    /// Greedily applies merges, then looks each surviving piece up as an id.
    ///
    /// PSEUDO-CODE (the classic GPT-2 `bpe()` loop):
    ///   let mut symbols: Vec<String> = piece.chars().map(to_string).collect()
    ///   loop:
    ///       // find the adjacent pair with the smallest merge rank
    ///       let best = adjacent_pairs(&symbols)
    ///           .filter_map(|p| self.merge_ranks.get(&p).map(|r| (r, p)))
    ///           .min_by_key(|(r, _)| *r)
    ///       let Some((_, (l, r))) = best else { break }   // none mergeable → done
    ///       // merge EVERY occurrence of exactly that (l, r) pair, left to right
    ///       symbols = merge_pair(symbols, &l, &r)
    ///   symbols.iter().map(|s| self.token_to_id[s]).collect()
    ///
    /// Every single byte-level-unicode char is itself in the vocab, and every
    /// merge result is too, so the final `token_to_id[s]` lookups never miss.
    fn bpe(&self, _piece: &str) -> Vec<u32> {
        todo!("bpe — the merge loop; the heart of M0")
    }

    /// Build GPT-2's byte→unicode table (stage 2). The 188 "printable" bytes
    /// (ranges !..~, ¡..¬, ®..ÿ) map to themselves; the other 68 map to
    /// codepoints 256, 257, … in byte order. That's why 0x20 → U+0120 ('Ġ').
    fn build_byte_encoder() -> [char; 256] {
        todo!("build_byte_encoder — a fixed, well-known construction")
    }

    /// Parse `vocab.json` into the forward map and the dense reverse vector.
    /// Returns (token_to_id, id_to_token).
    fn load_vocab(_path: &Path) -> Result<(HashMap<String, u32>, Vec<String>), String> {
        todo!("load_vocab — read JSON object of piece -> id")
    }

    /// Parse `merges.txt` into `(left, right) -> rank`. Skip the `#version`
    /// header; the rank is the (post-header) line index — earlier = higher
    /// priority. Each line is exactly two space-separated pieces.
    fn load_merges(_path: &Path) -> Result<HashMap<(String, String), u32>, String> {
        todo!("load_merges — line number becomes the merge rank")
    }
}
