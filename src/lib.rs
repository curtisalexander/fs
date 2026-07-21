//! Failed Star (`fs`) — a tiny, from-scratch LLM inference engine built to learn.
//!
//! This is the library crate root. Each milestone from [`PLAN.md`] lands here as
//! a module; the [`crate::tokenizer`] module is M0. The `fs` binary
//! (`src/main.rs`) is just a thin CLI over what lives here.
//!
//! [`PLAN.md`]: ../PLAN.md

pub mod tokenizer;

// M1 — load the weights: read `config.json` (the architecture) and
// `model.safetensors` (the weights), and verify they line up via `fs inspect`.
pub mod config;
pub mod inspect;
pub mod safetensors;

// M2 — forward pass → logits: widen the weights to f32 and run the network on the
// CPU (`fs logits`). `tensor` is the row-major `Matrix` everything computes over.
pub mod forward;
pub mod tensor;
