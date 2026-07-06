//! `safetensors` — the model-agnostic reader for the safetensors file format.
//!
//! This is the *other* of a model's two things (see [`crate::config`]): the raw
//! **weights**. The format is tiny (see
//! [`docs/learnings/01-safetensors-vs-gguf.md`]):
//!
//! ```text
//! [ 8 bytes: u64 LE = header length N ][ N bytes: JSON header ][ raw tensor blob ]
//! ```
//!
//! The JSON header maps each tensor name → `{dtype, shape, data_offsets:[s,e]}`,
//! where `[s, e)` indexes into the trailing blob. So "reading" the file is: read
//! 8 bytes, parse N bytes of JSON, and now every tensor is a `[s, e)` slice of the
//! rest. We never copy a weight — tensors stay as borrowed byte slices into the
//! mapping, and we convert `bf16 → f32` only when M2 actually computes.
//!
//! **We `mmap` the file rather than read it into a `Vec`.** That keeps the 1.4 GB
//! of weights out of our heap — the OS maps the file into our address space and
//! pages it in lazily as we touch it (zero-copy). The mapping is done with **raw
//! POSIX FFI, no `libc` crate**, matching the project's no-hidden-abstraction
//! ethos. How `mmap`/`munmap` actually work — virtual memory, lazy paging, RAII
//! cleanup on `Drop` — is its own lesson: [`docs/learnings/06-mmap.md`].

#![allow(dead_code)] // scaffold: remove once `load` constructs these and `inspect` reads them.

use std::collections::HashMap;
use std::ffi::{c_int, c_void};

// ── Raw POSIX mmap FFI (no `libc` crate) ────────────────────────────────────
//
// In edition 2024 an FFI block is `unsafe extern`. These are declarations only;
// the kernel provides the bodies. See learning 06 for what each argument means.
unsafe extern "C" {
    /// `void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t offset)`
    fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        offset: i64, // off_t on macOS/arm64
    ) -> *mut c_void;

    /// `int munmap(void *addr, size_t len)` — releases the mapping.
    fn munmap(addr: *mut c_void, len: usize) -> c_int;
}

const PROT_READ: c_int = 0x1; // pages may be read
const MAP_PRIVATE: c_int = 0x2; // copy-on-write, private to us (we only read)
/// `mmap` returns `(void *) -1` on failure, not null.
const MAP_FAILED: *mut c_void = usize::MAX as *mut c_void;

/// A read-only memory mapping of a whole file. Owns the mapping; unmaps on drop.
#[derive(Debug)] // so `Result<Mmap, _>::unwrap_err()` can print the Ok side in tests
struct Mmap {
    ptr: *const u8,
    len: usize,
}

impl Mmap {
    /// `mmap` the entire file at `path`, read-only.
    ///
    /// Steps (see learning 06):
    /// 1. open the file (`std::fs::File::open`) — keep it only long enough to map.
    /// 2. its byte length comes from `file.metadata()?.len()`.
    /// 3. `mmap(null, len, PROT_READ, MAP_PRIVATE, file.as_raw_fd(), 0)`.
    ///    A zero-length file can't be mapped — error out (`MapFailed`) first.
    /// 4. compare the result against `MAP_FAILED`; on success keep `(ptr, len)`.
    ///    The fd may be closed after mapping — the mapping keeps the file alive.
    fn open(path: &str) -> Result<Self, SafeTensorsError> {
        // `File::as_raw_fd()` lives behind this trait: a fd is just a small int the
        // kernel uses to name our open file, and it's all `mmap` needs to find it.
        use std::os::fd::AsRawFd;

        // Every failure below should name the file it was mapping.
        let fail = |message: String| SafeTensorsError::MapFailed { path: path.to_string(), message };

        // 1. Open the file → an OS file descriptor (the kernel's handle to it).
        let file = std::fs::File::open(path).map_err(|e| fail(e.to_string()))?;

        // 2. The map length is the file's size: `mmap` maps a byte *range*, so to map
        //    the whole file we must first tell it how many bytes that is.
        let len = file.metadata().map_err(|e| fail(e.to_string()))?.len() as usize;

        // 3. `mmap` rejects a zero-length mapping (EINVAL). Catch it here so the error
        //    reads "empty file" instead of a bare errno from step 5.
        if len == 0 {
            return Err(fail("file is empty — nothing to map".into()));
        }

        // 4. The syscall. This one FFI call is where all the kernel work happens, so it
        //    is `unsafe`: we promise the arguments uphold mmap's contract and that we'll
        //    honor the returned pointer's rules. Args (see learning 06's table):
        //      addr=null → kernel picks the address · len → map the whole file
        //      PROT_READ → read-only pages · MAP_PRIVATE → copy-on-write, private to us
        //      fd → which file · offset=0 → from the first byte
        let ptr = unsafe {
            mmap(std::ptr::null_mut(), len, PROT_READ, MAP_PRIVATE, file.as_raw_fd(), 0)
        };

        // 5. Failure is MAP_FAILED = (void*)-1, *not* null — a null check would miss
        //    every error. `last_os_error()` reads the errno the syscall just set.
        if ptr == MAP_FAILED {
            return Err(fail(std::io::Error::last_os_error().to_string()));
        }

        // The fd has done its job: a live mapping keeps its own reference to the file,
        // so `file` can drop here (closing the fd) without tearing the mapping down.
        Ok(Mmap { ptr: ptr as *const u8, len })
    }

    /// The mapped bytes as a slice. Safe to expose: the mapping is read-only and
    /// lives exactly as long as `self`.
    fn as_bytes(&self) -> &[u8] {
        // SAFETY: `ptr` points at `len` valid, read-only bytes for `self`'s life.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        // SAFETY: `ptr`/`len` came from a successful `mmap`; unmapped exactly once.
        unsafe {
            munmap(self.ptr as *mut c_void, self.len);
        }
    }
}

// ── Dtypes ──────────────────────────────────────────────────────────────────

/// The element types we recognize in a safetensors header. Qwen3-0.6B is **all
/// `BF16`**; the others are here so the reader degrades with a clear error rather
/// than a silent misread, and so M5's quantized path has somewhere to grow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    BF16,
    F16,
    F32,
}

impl Dtype {
    /// Parse the header's dtype string (e.g. `"BF16"`).
    pub fn parse(s: &str) -> Result<Self, SafeTensorsError> {
        match s {
            "BF16" => Ok(Dtype::BF16),
            "F16" => Ok(Dtype::F16),
            "F32" => Ok(Dtype::F32),
            other => Err(SafeTensorsError::UnknownDtype { dtype: other.to_string() }),
        }
    }

    /// Bytes per element.
    pub fn size(self) -> usize {
        match self {
            Dtype::BF16 | Dtype::F16 => 2,
            Dtype::F32 => 4,
        }
    }
}

/// Decode one `bf16` value to `f32`.
///
/// `bf16` is *literally the top 16 bits of an `f32`* (same exponent, truncated
/// mantissa), so widening is just a 16-bit left shift — no table, no branch. This
/// helper exists now but is **first used at M2**, when we actually compute; M1
/// checksums raw bytes and never materializes floats. See the "lazy bf16"
/// decision in `PROGRESS.md`.
pub fn bf16_to_f32(bytes: [u8; 2]) -> f32 {
    f32::from_bits((u16::from_le_bytes(bytes) as u32) << 16)
}

// ── Tensors + the file ────────────────────────────────────────────────────────

/// One tensor's entry from the header: its name, type, shape, and where its bytes
/// live in the blob (`[start, end)` relative to the start of the data section).
#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub start: usize, // offset of first byte within the data blob
    pub end: usize,   // one-past-last byte
}

impl Tensor {
    /// Number of elements = product of the shape (1 for a scalar / empty shape).
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Number of raw bytes this tensor occupies in the blob.
    pub fn num_bytes(&self) -> usize {
        self.end - self.start
    }
}

/// A parsed, memory-mapped safetensors file: the mapping plus a directory of its
/// tensors (kept in file order, with a name→index for lookup).
pub struct SafeTensors {
    mmap: Mmap,
    data_start: usize, // = 8 + header_len; where the tensor blob begins in the file
    tensors: Vec<Tensor>,
    index: HashMap<String, usize>,
    metadata: HashMap<String, String>, // the optional `__metadata__` block
}

impl SafeTensors {
    /// Memory-map and parse `path` (a `*.safetensors` file).
    ///
    /// Steps (the whole format, see learning 01):
    /// 1. `Mmap::open(path)` → the raw bytes.
    /// 2. read the first 8 bytes as a little-endian `u64` = header length `N`
    ///    (error if the file is shorter than 8 bytes → `Truncated`).
    /// 3. slice bytes `[8, 8+N)`, parse as a `serde_json::Value` object
    ///    (`HeaderNotUtf8` / `BadHeader` on failure).
    /// 4. for each `(name, info)`:
    ///       - the key `__metadata__` → stash into `metadata`, skip;
    ///       - else read `dtype` (→ `Dtype::parse`), `shape` (array of usize),
    ///         and `data_offsets` `[start, end]`. Validate `end - start ==
    ///         num_elements · dtype.size()` and `end <= blob_len`
    ///         (→ `BadTensorInfo { name, .. }`). Push a `Tensor`.
    /// 5. `data_start = 8 + N`; build the name→index map (duplicate name =
    ///    `BadHeader`). Keep `tensors` in header order for a stable `inspect`.
    pub fn load(path: &str) -> Result<Self, SafeTensorsError> {
        let _ = path;
        todo!("mmap → read u64 header len → parse JSON header → build the directory")
    }

    /// All tensors, in file order (what `inspect` walks).
    pub fn tensors(&self) -> &[Tensor] {
        &self.tensors
    }

    /// Look up a tensor by exact name.
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    /// The raw bytes of a tensor: a lazy slice into the mmap, no copy. Decoding
    /// `bf16 → f32` (via [`bf16_to_f32`]) is the caller's job, and only M2 needs it.
    ///
    /// `t.start`/`t.end` are blob-relative, so we offset by `data_start`.
    pub fn bytes(&self, t: &Tensor) -> &[u8] {
        let all = self.mmap.as_bytes();
        &all[self.data_start + t.start..self.data_start + t.end]
    }

    /// Optional free-form `__metadata__` from the header (format/version strings).
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }
}

/// Everything that can go wrong mapping/parsing a safetensors file.
#[derive(Debug)]
pub enum SafeTensorsError {
    /// `mmap`/open failed (file missing, empty, or the syscall errored).
    MapFailed { path: String, message: String },
    /// File too short to even hold the 8-byte header length / the declared header.
    Truncated { message: String },
    /// The header bytes were not valid UTF-8 JSON.
    HeaderNotUtf8,
    /// The header JSON was malformed or structurally wrong.
    BadHeader { message: String },
    /// A tensor entry was inconsistent (shape vs. byte length, out-of-range).
    BadTensorInfo { name: String, message: String },
    /// A dtype string we don't (yet) recognize.
    UnknownDtype { dtype: String },
}

impl std::fmt::Display for SafeTensorsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SafeTensorsError::MapFailed { path, message } => {
                write!(f, "could not map {path}: {message}")
            }
            SafeTensorsError::Truncated { message } => {
                write!(f, "safetensors file is truncated: {message}")
            }
            SafeTensorsError::HeaderNotUtf8 => {
                write!(f, "safetensors header is not valid UTF-8 JSON")
            }
            SafeTensorsError::BadHeader { message } => {
                write!(f, "safetensors header is malformed: {message}")
            }
            SafeTensorsError::BadTensorInfo { name, message } => {
                write!(f, "tensor '{name}' has inconsistent metadata: {message}")
            }
            SafeTensorsError::UnknownDtype { dtype } => {
                write!(f, "unsupported tensor dtype '{dtype}'")
            }
        }
    }
}

impl std::error::Error for SafeTensorsError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // A unique temp path per test so the (parallel) tests never collide on a file.
    fn temp_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("fs_mmap_{tag}.bin"))
    }

    #[test]
    fn maps_a_file_and_reads_its_exact_bytes() {
        // Include a NUL and high bytes to prove we read raw bytes, not a UTF-8 string.
        let data: &[u8] = b"failed star \x00\x01\xfe\xff mmap";
        let path = temp_path("roundtrip");
        std::fs::File::create(&path).unwrap().write_all(data).unwrap();

        let map = Mmap::open(path.to_str().unwrap()).expect("mapping a real file should succeed");
        assert_eq!(map.len, data.len(), "map length must equal file size");
        assert_eq!(map.as_bytes(), data, "the mapped view is the file's bytes, zero-copy");
        // `map` drops at end of scope → its `munmap` runs (RAII). Then clean up.
        drop(map);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn missing_file_is_a_typed_error_not_a_panic() {
        let err = Mmap::open("/no/such/failed-star/file.bin").unwrap_err();
        assert!(matches!(err, SafeTensorsError::MapFailed { .. }));
    }

    #[test]
    fn empty_file_is_rejected_before_the_syscall() {
        let path = temp_path("empty");
        std::fs::File::create(&path).unwrap(); // zero bytes
        let err = Mmap::open(path.to_str().unwrap()).unwrap_err();
        assert!(matches!(err, SafeTensorsError::MapFailed { .. }), "mmap can't map 0 bytes");
        std::fs::remove_file(&path).ok();
    }
}
