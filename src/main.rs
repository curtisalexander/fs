//! `fs` — command-line entry point for the Failed Star inference engine.
//!
//! M0 scope: a `tokenize` / `detokenize` pair that turns text <-> token IDs
//! using Qwen3-0.6B's real byte-level BPE vocabulary.
//!
//! This file is deliberately a *thin dispatcher*: parse argv by hand (no clap —
//! see Cargo.toml on why), pick a subcommand, hand off to the library. The
//! tokenizer itself still `todo!()`s — we sketch and fill it in together next
//! (see `src/tokenizer.rs`). So `cargo build` succeeds today; `fs tokenize ...`
//! compiles and runs but panics with a "not built yet" message until then.

use std::process::ExitCode;

use fs::tokenizer::Tokenizer;

/// Where the tokenizer assets live. Populated by `scripts/fetch_model.py`.
const MODEL_DIR: &str = "models/qwen3-0.6b";

const USAGE: &str = "\
fs — Failed Star inference engine

USAGE:
    fs tokenize <TEXT>       Encode TEXT into token IDs
    fs detokenize <IDS...>   Decode token IDs (space-separated) back into text
    fs help                  Show this message

EXAMPLES:
    fs tokenize \"hello world\"
    fs detokenize 14990 1879

SETUP:
    Tokenizer assets load from ./models/qwen3-0.6b/.
    Fetch them first:  uv run --directory scripts fetch_model.py
";

fn main() -> ExitCode {
    // args()[0] is the program name; skip it and dispatch on the first word.
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.split_first() {
        Some((cmd, rest)) if cmd == "tokenize" => cmd_tokenize(rest),
        Some((cmd, rest)) if cmd == "detokenize" => cmd_detokenize(rest),
        Some((cmd, _)) if matches!(cmd.as_str(), "help" | "--help" | "-h") => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Some((cmd, _)) => {
            eprintln!("fs: unknown command '{cmd}'\n");
            eprint!("{USAGE}");
            ExitCode::FAILURE
        }
        None => {
            eprint!("{USAGE}");
            ExitCode::FAILURE
        }
    }
}

/// `fs tokenize <TEXT>` — encode one string argument into token IDs.
fn cmd_tokenize(args: &[String]) -> ExitCode {
    let Some(text) = args.first() else {
        eprintln!("fs tokenize: missing TEXT argument\n");
        eprint!("{USAGE}");
        return ExitCode::FAILURE;
    };

    let tokenizer = match Tokenizer::load(MODEL_DIR) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("fs: could not load tokenizer from {MODEL_DIR}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let ids = tokenizer.encode(text);
    let rendered: Vec<String> = ids.iter().map(u32::to_string).collect();
    println!("{}", rendered.join(" "));
    ExitCode::SUCCESS
}

/// `fs detokenize <IDS...>` — decode token IDs back into text. IDs may be passed
/// as separate args or as a single space-separated string (or a mix).
fn cmd_detokenize(args: &[String]) -> ExitCode {
    let mut ids: Vec<u32> = Vec::new();
    for arg in args {
        for field in arg.split_whitespace() {
            match field.parse::<u32>() {
                Ok(id) => ids.push(id),
                Err(_) => {
                    eprintln!("fs detokenize: '{field}' is not a valid token id");
                    return ExitCode::FAILURE;
                }
            }
        }
    }
    if ids.is_empty() {
        eprintln!("fs detokenize: missing IDS argument\n");
        eprint!("{USAGE}");
        return ExitCode::FAILURE;
    }

    let tokenizer = match Tokenizer::load(MODEL_DIR) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("fs: could not load tokenizer from {MODEL_DIR}: {e}");
            return ExitCode::FAILURE;
        }
    };

    println!("{}", tokenizer.decode(&ids));
    ExitCode::SUCCESS
}
