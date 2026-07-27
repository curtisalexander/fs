//! Minimal Qwen tensor schema shared by inspection and weight loading.

use crate::config::Config;
use crate::safetensors::Dtype;

/// Expected dtype, shape, and presence rule for one Qwen weight.
#[derive(Debug)]
pub struct Expected {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub optional: bool,
}

/// Build the complete tensor set implied by a Qwen config.
pub fn expected_tensors(cfg: &Config) -> Vec<Expected> {
    let v = cfg.vocab_size;
    let h = cfg.hidden_size;
    let d = cfg.head_dim;
    let q = cfg.q_width();
    let kv = cfg.kv_width();
    let i = cfg.intermediate_size;
    let e = |name: String, shape: Vec<usize>, optional: bool| Expected {
        name,
        dtype: Dtype::BF16,
        shape,
        optional,
    };

    let block_count = 11usize
        .checked_mul(cfg.num_hidden_layers)
        .and_then(|n| n.checked_add(3))
        .expect("Qwen schema tensor count overflow");
    let mut out = Vec::with_capacity(block_count);
    out.push(e("model.embed_tokens.weight".into(), vec![v, h], false));
    for l in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{l}");
        out.push(e(format!("{p}.input_layernorm.weight"), vec![h], false));
        out.push(e(format!("{p}.self_attn.q_proj.weight"), vec![q, h], false));
        out.push(e(
            format!("{p}.self_attn.k_proj.weight"),
            vec![kv, h],
            false,
        ));
        out.push(e(
            format!("{p}.self_attn.v_proj.weight"),
            vec![kv, h],
            false,
        ));
        out.push(e(format!("{p}.self_attn.q_norm.weight"), vec![d], false));
        out.push(e(format!("{p}.self_attn.k_norm.weight"), vec![d], false));
        out.push(e(format!("{p}.self_attn.o_proj.weight"), vec![h, q], false));
        out.push(e(
            format!("{p}.post_attention_layernorm.weight"),
            vec![h],
            false,
        ));
        out.push(e(format!("{p}.mlp.gate_proj.weight"), vec![i, h], false));
        out.push(e(format!("{p}.mlp.up_proj.weight"), vec![i, h], false));
        out.push(e(format!("{p}.mlp.down_proj.weight"), vec![h, i], false));
    }
    out.push(e("model.norm.weight".into(), vec![h], false));
    out.push(e(
        "lm_head.weight".into(),
        vec![v, h],
        cfg.tie_word_embeddings,
    ));
    out
}
