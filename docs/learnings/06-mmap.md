# Learning 06 — `mmap`: turning a file into memory (raw POSIX FFI)

> **Date:** 2026-06-24 · **Context:** M1, reading `model.safetensors` · **Status:** living
>
> 📖 *Inference Engineering* §4.2.2 (model file formats / loading, p.103)
> 🔧 `ds4`: *"Loading is mmap based"* (`ds4.c:11`) — the same idea, in C
> 🧭 POSIX `mmap(2)` / `munmap(2)` man pages

To load the weights we had a choice (M1 design dialogue): read the 1.4 GB
`model.safetensors` into a `Vec<u8>`, or **`mmap` it**. We chose `mmap` — and
because the project's rule is *no hidden abstraction*, we call it with **raw POSIX
FFI, no `libc` crate**. This note is what that means and why.

See [`learning 01`](01-safetensors-vs-gguf.md) for the file *format* and
[`learning 05`](05-reading-shapes.md) for the tensor *shapes*; this note is purely
about *getting the bytes into our address space*.

---

## What `mmap` actually does

`std::fs::read` **copies**: the kernel reads the file into a buffer you own, all
1.4 GB, up front, into your heap. `mmap` does something different — it **maps the
file into your virtual address space** and hands you a pointer. No bytes are
copied yet. The pages are pulled in **lazily, on first touch** (a page fault), and
the OS can drop clean pages under memory pressure and re-read them from the file
later. For read-only weights this is ideal:

```
 std::fs::read                          mmap
 ┌──────────┐  copy all   ┌─────────┐   ┌──────────┐  map (no copy)   ┌─────────┐
 │  file    │ ──────────▶ │  heap   │   │  file    │ ──────────────▶  │ address │
 │  1.4 GB  │             │ 1.4 GB  │   │  1.4 GB  │   pages fault    │  space  │
 └──────────┘             └─────────┘   └──────────┘   in on touch    └─────────┘
   eager, owns the bytes                  lazy, the file *is* the backing store
```

Why it fits us:

- **Zero-copy.** Our `Tensor`s are just `&[u8]` slices into the mapping (see
  `SafeTensors::bytes`). We never duplicate a weight. This is the whole reason the
  "lazy bf16" decision works — nothing is materialized until M2 reads it.
- **Lazy + reclaimable.** Touch only the tensors you use; the OS handles paging.
- **It's what `ds4` does**, so the mental model transfers to the bigger engine.

The cost: a mapping is *unsafe* to hold (a raw pointer with a lifetime the
compiler can't see), and the file shouldn't change underneath us. We contain that
unsafety in one small `Mmap` type.

## The POSIX call, argument by argument

```c
void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t offset);
int   munmap(void *addr, size_t len);
```

For mapping a whole file read-only we pass:

| arg      | we pass                | why |
|----------|------------------------|-----|
| `addr`   | `NULL`                 | let the kernel choose the address |
| `len`    | file size              | map the entire file (from `file.metadata().len()`) |
| `prot`   | `PROT_READ` (`0x1`)    | pages may be read, not written |
| `flags`  | `MAP_PRIVATE` (`0x2`)  | copy-on-write, private to us — we only read |
| `fd`     | the file's descriptor  | which file to map (`File::as_raw_fd()`) |
| `offset` | `0`                    | start at the beginning of the file |

Two gotchas worth burning in:

- **Failure is `MAP_FAILED`, not null.** `mmap` returns `(void *) -1` on error
  (`usize::MAX as *mut c_void`). Checking for null would miss every failure.
- **A zero-length file can't be mapped** (`len` must be > 0) — we error out before
  calling.

After a successful map, the **file descriptor can be closed**: the mapping keeps
its own reference to the file, so the bytes stay valid until we `munmap`.

## Calling it from Rust without `libc`

`libc` is just declarations for calls the kernel already provides, so we write
those declarations ourselves. In **edition 2024** an FFI block is `unsafe extern`:

```rust
use std::ffi::{c_int, c_void};

unsafe extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: c_int,
            flags: c_int, fd: c_int, offset: i64) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> c_int;
}

const PROT_READ:   c_int = 0x1;
const MAP_PRIVATE: c_int = 0x2;
const MAP_FAILED: *mut c_void = usize::MAX as *mut c_void;
```

(`offset` is `off_t`, which is 64-bit on macOS/arm64, so `i64`.)

## Containing the unsafety: one RAII wrapper

All the danger lives in a small owner that maps on construction and **unmaps on
`Drop`** — so cleanup is automatic and can't be forgotten:

```rust
struct Mmap { ptr: *const u8, len: usize }

impl Mmap {
    fn as_bytes(&self) -> &[u8] {
        // SAFETY: ptr points at len valid, read-only bytes for self's lifetime.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        // SAFETY: ptr/len came from a successful mmap; unmapped exactly once.
        unsafe { munmap(self.ptr as *mut c_void, self.len); }
    }
}
```

Why this is sound to expose safely:

- the mapping is **read-only**, so `&[u8]` can't be used to mutate it;
- the slice **borrows `self`**, so it can't outlive the mapping (no use-after-unmap);
- `munmap` runs **exactly once**, when the `Mmap` is dropped — RAII, same shape as
  C's "map then `munmap`," but the compiler enforces the cleanup.

`SafeTensors` then owns one `Mmap` and hands out tensor slices into it. The raw
pointer never escapes; everything above this type is safe Rust.

---

## Cross-links

- ⬅ [`learning 01 · safetensors vs GGUF`](01-safetensors-vs-gguf.md) — the byte
  layout we're mapping (`[u64 len][JSON header][blob]`).
- 🔗 [`learning 05 · reading shapes`](05-reading-shapes.md) — what the mapped bytes
  *mean* once we slice them into tensors.
- 🔧 `src/safetensors.rs` — the `Mmap` wrapper + `SafeTensors` reader this note
  documents.
- 🔧 `ds4.c` — the C version of the same mmap-based loading strategy.
