# Learning 04 — "Embedding model" ≠ the token-embedding table

> **Date:** 2026-06-14 · **Context:** untangling what "embeddings" means · **Status:** background (not on our inference path, but a common confusion worth pinning down)
>
> 📖 *Inference Engineering* §2.1 — token embeddings as the model's input layer
> 🔧 `ds4`: token-embedding table is `token_embd.weight` (`ds4.c:3033`), gathered by `metal/get_rows.metal`
> 🧭 Lineage: word2vec/GloVe (static tables) → sentence-transformers / BGE / E5 / OpenAI `text-embedding-3` (forward-pass models)
> 🔗 see also [`03-bpe.md`](03-bpe.md) (tokens, the thing being embedded) and Stage B in [`../00-map.md`](../00-map.md)

The word **"embedding"** gets attached to at least three different things, and
conflating them causes real confusion — especially the question *"when I use an
embedding model for search/RAG, am I just reading rows out of the learned table
inside an LLM?"* Short answer: **no.** This note pins down why.

> This is *background*. Our engine only ever uses sense #1 below (the input lookup
> table). Embedding models (sense #3) are a separate kind of model we are **not**
> building — but knowing the difference keeps the vocabulary straight.

---

## Three things called "embedding"

**1. Token embeddings — the lookup table.**
- Shape `[vocab, d_model]`, one vector per *token*.
- A **static lookup**: token `15339` → row `15339`, the *same vector every time*,
  regardless of surrounding words. **Context-free.**
- It is the network's **input layer**. *This is the only sense our engine uses.*

**2. Contextual embeddings — the hidden states inside the network.**
- The vectors *between* layers, after attention has mixed in surrounding context.
- **Context-dependent:** "bank" gets a different vector in "river bank" vs "bank
  account."
- They are **activations** computed by a forward pass — not stored anywhere.

**3. Text / sentence embeddings — what an "embedding model" returns.**
- **One** fixed-size vector for an *entire* sentence or document, used for semantic
  search, RAG, clustering, dedup.
- It is the **output** of running text through a whole network and then **pooling**.

When you call an "embedding model" (`text-embedding-3`, BGE, E5,
`sentence-transformers`, …) you get **#3** — not the table from #1.

---

## What an embedding model actually does

It is **not** a table lookup. For each input text it runs a **full forward pass**:

```
text → tokenize → gather token-embedding rows (#1)   ← the lookup, as the INPUT layer
     → transformer layers (attention mixes in context) → contextual vectors (#2)
     → POOL into one vector  (mean over tokens, or the [CLS] / last-token vector)
     → (usually) L2-normalize → the sentence embedding (#3)
```

The learned token table **is** in there — it's the front door — but the vector you
receive is the product of the *entire network*, squashed into one. The meaning comes
from the whole forward pass, not from the raw input rows.

Two more things distinguish a real embedding model:

- **Different training objective.** A generative LLM trains on next-token
  prediction. An embedding model is trained/fine-tuned with a **contrastive**
  objective (InfoNCE / cosine-similarity loss): show it (query, relevant doc,
  irrelevant doc) and push relevant texts' vectors *together*, irrelevant ones
  *apart*. That is what makes its output geometry good for cosine comparison; a raw
  LLM's token rows are **not** good for that off the shelf.
- **Often a different architecture.** Many embedding models are smaller,
  **encoder-only / bidirectional** (BERT-style), specialized for this — not decoder
  LLMs.

---

## Where the "pull the table out of a trained net" intuition *is* right

That intuition perfectly describes the **previous generation**: **word2vec and
GloVe** (~2013). Their *entire output was a static word→vector table* — train a
shallow net on a context-prediction task, throw the net away, keep the table. A
lookup was a pure table read, context-free (exactly sense #1).

Modern sentence-embedding models replaced that approach because the static table has
hard limits:

| | word2vec / GloVe (old) | embedding model (modern) |
|---|---|---|
| What you use | the **table** itself (a lookup) | the **output** of a forward pass |
| Context-aware? | No — one vector per word, forever | Yes — same word, different vector by context |
| Word order | ignored | captured (attention) |
| Granularity | per word | per sentence / document |
| Trained for | predicting nearby words | **similarity** (contrastive) |

---

## Practical upshot

You *could* build a cheap text embedding by **averaging the token-embedding rows**
of a sentence (a "bag of embeddings") — people did exactly that with word2vec. It's
a weak baseline: it discards word order and context ("dog bites man" = "man bites
dog"). Modern embedding models win because they run the full contextual network and
are trained for similarity.

## Mental model to keep

> **Token embedding** = a *lookup* in the model's input table; context-free; one per
> token; it's the **input**. *(The only sense our inference engine touches.)*
> **Embedding model** = a *forward pass + pooling* over a separate, similarity-trained
> network; context-aware; one per text; it's the **output**.
>
> Using an embedding model means **running a neural network over your text and
> pooling its output** — not reading rows from the token table we study at M2. That
> table is merely the first layer of the machine doing the work.
