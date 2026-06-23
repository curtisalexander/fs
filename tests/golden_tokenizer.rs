//! End-to-end M0 check: our tokenizer must reproduce the official Qwen3-0.6B
//! token IDs — and decode them back — for every case in
//! [`tests/golden/tokenizer.json`], the fixture emitted by the official oracle
//! (`scripts/gen_golden.py`). This is the milestone's "done" gate.
//!
//! Needs the real assets in `models/qwen3-0.6b/` (git-ignored; fetch with
//! `uv run --directory scripts fetch_model.py`). If they're absent we skip with
//! a notice instead of failing, so `cargo test` stays green on a fresh checkout.

use std::path::Path;

use fs::tokenizer::Tokenizer;

const MODEL_DIR: &str = "models/qwen3-0.6b";
const GOLDEN: &str = include_str!("golden/tokenizer.json");

#[test]
fn reproduces_official_ids_on_golden_cases() {
    if !Path::new(MODEL_DIR).join("vocab.json").exists() {
        eprintln!("SKIP: {MODEL_DIR} not found — run `uv run --directory scripts fetch_model.py`");
        return;
    }

    let tok = Tokenizer::load(MODEL_DIR).expect("load tokenizer from model dir");
    let doc: serde_json::Value = serde_json::from_str(GOLDEN).expect("parse golden json");
    let cases = doc["cases"].as_array().expect("golden 'cases' array");

    // Collect every mismatch instead of failing on the first, so one run shows
    // the full picture of what's wrong.
    let mut failures: Vec<String> = Vec::new();
    for case in cases {
        let text = case["text"].as_str().unwrap();
        let want_ids: Vec<u32> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let want_decoded = case["decoded"].as_str().unwrap();

        // 1. encode matches the official IDs.
        let got_ids = tok.encode(text).expect("encode");
        if got_ids != want_ids {
            failures.push(format!("encode({text:?})\n   got  {got_ids:?}\n   want {want_ids:?}"));
            continue; // decode checks below assume encode is correct
        }
        // 2. decode of the official IDs reproduces the official text.
        let got_decoded = tok.decode(&want_ids).expect("decode");
        if got_decoded != want_decoded {
            failures.push(format!(
                "decode({want_ids:?})\n   got  {got_decoded:?}\n   want {want_decoded:?}"
            ));
        }
        // 3. round-trip: decode(encode(text)) == text.
        let round = tok.decode(&got_ids).expect("decode round-trip");
        if round != text {
            failures.push(format!("round-trip {text:?} -> {round:?}"));
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} golden cases failed:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}
