# solana-sbpf

SBPF virtual machine

[![Build Status](https://github.com/anza-xyz/sbpf/actions/workflows/main.yml/badge.svg)](https://github.com/anza-xyz/sbpf/actions/workflows/main.yml)
[![Crates.io](https://img.shields.io/crates/v/solana-sbpf.svg)](https://crates.io/crates/solana-sbpf)

## Description

This is a fork of [RBPF](https://github.com/solana-labs/rbpf) which in turn is a fork of [RBPF](https://github.com/qmonnet/rbpf) by Quentin Monnet.

This crate contains a virtual machine for eBPF program execution. BPF, as in
_Berkeley Packet Filter_, is an assembly-like language initially developed for
BSD systems, in order to filter packets in the kernel with tools such as
tcpdump so as to avoid useless copies to user-space. It was ported to Linux,
where it evolved into eBPF (_extended_ BPF), a faster version with more
features. While BPF programs are originally intended to run in the kernel, the
virtual machine of this crate enables running it in user-space applications;
it contains an interpreter, an x86_64 JIT-compiler for eBPF programs, as well as
an assembler, disassembler and verifier.

The crate is supposed to compile and run on Linux, MacOS X, and Windows,
although the JIT-compiler does not work with Windows at this time.

## SBPF feature description

The Solana-versioned BPF, namely SBPF, has four versions, each with different
functionality. In the table below, we give an overview of which features are
available for each version.

| Feature                       | v0 | v1  | v2  | v3  |
|-------------------------------|----|-----|-----|-----|
| Static syscalls               | no | no  | no  | yes |
| Dynamic stack frames          | no | yes | yes | no  |
| PQR instructions              | no | no  | yes | no  |
| Explicit sign extension       | no | no  | yes | yes |
| Compatibility with eBPFv3 ISA | no | no  | no  | yes |


## aarch64 JIT (this fork)

The `arm64-jit` branch is v0.21.1 plus a JIT compiler for aarch64: Apple silicon
macOS and Linux. The public API is the one of v0.21.1, so a program built
against solana-sbpf 0.21.1 (agave, LiteSVM) picks it up through a patch:

```toml
[patch.crates-io]
solana-sbpf = { git = "https://github.com/Pawel-608/sbpf-arm64-jit", branch = "arm64-jit" }
```

`ExecutionMode::PreferJit` / `Jit` and `Executable::jit_compile()` now work on
aarch64 as they do on x86-64.

### How it works

- `src/jit_aarch64.rs` is compiled as `solana_sbpf::jit` on aarch64. It is a
  port of the x86-64 JIT (`src/jit.rs`, unchanged): the same subroutines
  (anchors), the same instruction meter technique and the same exception
  paths, so results, errors, final pc and instruction counts match the
  interpreter (and the x86-64 JIT).
- `src/aarch64.rs` is the upstream ARM64 emitter stage (anza-xyz/sbpf #490,
  #510), extended with the instructions the JIT needs.
- BPF r0..r9 live in the callee saved X19..X28; r10, the VM pointer and the
  instruction meter live in caller saved registers that are saved around calls
  into Rust (syscalls, `MemoryMapping::load`/`store`). A trampoline at the
  start of the generated code enters it; X29 keeps a valid frame record.
- Executable memory: on macOS the pages are mapped with `MAP_JIT` and written
  between `pthread_jit_write_protect_np(0)` and `(1)` (W^X per thread), then
  `sys_icache_invalidate`. On Linux: `mmap`, `mprotect` and the
  `DC CVAU`/`IC IVAU` cache maintenance sequence.
- Not ported because they never change results: the random no-op padding
  (`Config::noop_instruction_rate`) and the immediate value sanitization
  (`Config::sanitize_user_provided_values`).
- Shared with the x86-64 JIT: the meter is checked at branches and
  checkpoints, so when a program runs out of budget the JIT may have executed
  a few more instructions of a linear run than the interpreter before it
  reports `ExceededMaxInstructions`. Result, pc and instruction count agree;
  memory written after the budget ran out may differ.

### Tests

`cargo test --release` runs the whole suite (tests/execution.rs and
tests/exercise_instructions.rs run every program in the interpreter and the
JIT and compare results, instruction counts, final pc and register traces).
`tests/jit_differential.rs` generates random verifier-valid programs for every
SBPF version (ALU, PQR, endianness, loads and stores to input, stack, heap and
bytecode, branches and loops, calls, callx, syscalls that charge compute units,
fail or write memory) with random configs and compares result, instruction
count, final pc, remaining budget, register trace and all memory. It also runs
a program whose machine code is larger than the reach of conditional branches.

```sh
JIT_DIFF_ITERATIONS=200000 cargo test --release --test jit_differential -- --nocapture
cd fuzz && cargo +nightly fuzz run smarter-jit-diff -- -max_total_time=180
```

### Benchmark

`cargo run --release --example jit_vs_interpreter`, millions of BPF
instructions per second (best of 5):

| program                            | interpreter | JIT   | speedup |
|------------------------------------|------------:|------:|--------:|
| alu (xorshift, mul, div)           |         470 | 9,406 |   20.0x |
| memory (input loads, stack stores) |         292 |   536 |    1.8x |
| calls (recursive fibonacci)        |         310 | 4,023 |   13.0x |
| branchy (collatz lengths)          |         349 | 3,049 |    8.7x |

Apple M4 Max, macOS 15. Linux aarch64 (a VM on the same machine) gives the
same speedups (20.0x, 2.3x, 13.3x, 8.3x). Loads and stores call
`MemoryMapping::load`/`store` like the x86-64 JIT does, so memory heavy code
gains the least.

## Link to the crate

This crate is available from [crates.io](https://crates.io/crates/solana-sbpf),
so it should work out of the box by adding it as a dependency in your
`Cargo.toml` file:

```toml
[dependencies]
solana-sbpf = "0.19.0"
```

You can also use the development version from this GitHub repository. This
should be as simple as putting this inside your `Cargo.toml`:

```toml
[dependencies]
solana-sbpf = { git = "https://github.com/anza-xyz/sbpf", branch = "main" }
```

Of course, if you prefer, you can clone it locally, possibly hack the crate,
and then indicate the path of your local version in `Cargo.toml`:

```toml
[dependencies]
solana-sbpf = { path = "path/to/sbpf" }
```

Then indicate in your source code that you want to use the crate:

```rust,ignore
extern crate solana_sbpf;
```

## API

The API is pretty well documented inside the source code. You should also be
able to access [an online version of the documentation from
here](https://docs.rs/solana-sbpf/), automatically generated from the
[crates.io](https://crates.io/crates/solana-sbpf)
version (may not be up-to-date with master branch).
[Examples](examples), [unit tests](tests) and [performance benchmarks](benches)
should also prove helpful.

Here are the steps to follow to run an SBPF:

1. Create the config and a loader built-in program, add some functions.
2. Create an executable, either from the bytecode or an ELF.
3. If you want a JIT-compiled program, compile it.
4. Create a memory mapping, consisting of multiple memory regions.
5. Create a context object which will also acts as instruction meter.
6. Create a virtual machine using all of the previous steps.
7. Execute your program: Either run the interpreter or call the JIT-compiled
   function.

## Developer

### Dependencies
- rustc version 1.83 or higher

### Build and test instructions
- To build run `cargo build`
- To test run `cargo test`

## License

Following the effort of the Rust language project itself in order to ease
integration with other projects, the sbpf crate is distributed under the terms
of both the MIT license and the Apache License (Version 2.0).

See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) for details.
