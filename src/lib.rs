//! Failed Star (`fs`) — a tiny, from-scratch LLM inference engine built to learn.
//!
//! This is the library crate root. Each milestone from [`PLAN.md`] lands here as
//! a module; the [`crate::tokenizer`] module is M0. The `fs` binary
//! (`src/main.rs`) is just a thin CLI over what lives here.
//!
//! [`PLAN.md`]: ../PLAN.md

pub mod tokenizer;
