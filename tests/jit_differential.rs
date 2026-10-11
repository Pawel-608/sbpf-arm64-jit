#![allow(clippy::arithmetic_side_effects)]
#![cfg(all(
    feature = "jit",
    not(target_os = "windows"),
    any(target_arch = "x86_64", target_arch = "aarch64")
))]

//! Differential tests: random, verifier-valid programs run in the interpreter and in the JIT.
//! Both must agree on the result (including which error), the instruction count, the final pc,
//! the remaining budget, the register trace and every byte of memory (input, stack, heap).
//!
//! `JIT_DIFF_ITERATIONS` (programs per SBPF version, default 3000) and `JIT_DIFF_SEED` control
//! the run, e.g. `JIT_DIFF_ITERATIONS=200000 cargo test --release --test jit_differential`.

use rand::{rngs::SmallRng, Rng, SeedableRng};
use solana_sbpf::{
    aligned_memory::AlignedMemory,
    declare_builtin_function,
    ebpf::{self, HOST_ALIGN},
    elf::Executable,
    memory_region::MemoryRegion,
    program::{BuiltinFunctionDefinition, BuiltinProgram, FunctionRegistry, SBPFVersion},
    verifier::RequisiteVerifier,
    vm::{CallFrame, Config, ContextObject, EbpfVm, ExecutionMode},
};
use std::sync::Arc;
use test_utils::{compare_register_trace, create_memory_mapping, TestContextObject};

declare_builtin_function!(
    /// Mixes its arguments, charges some compute units and fails for one magic value
    SyscallMix,
    fn rust(
        context_object: &mut TestContextObject,
        a: u64,
        b: u64,
        c: u64,
        d: u64,
        e: u64,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        context_object.consume(a % 5);
        if b & 0xff == 0x5a {
            return Err("magic value".into());
        }
        Ok(a.wrapping_mul(31) ^ b ^ c.rotate_left(7) ^ d.wrapping_add(e))
    }
);

declare_builtin_function!(
    /// Stores the first argument at the vm address given by the second one
    SyscallStore,
    fn rust(
        context_object: &mut TestContextObject,
        a: u64,
        b: u64,
        _c: u64,
        _d: u64,
        _e: u64,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        match context_object.memory_mapping.store::<u64>(a, b) {
            solana_sbpf::error::ProgramResult::Ok(_) => Ok(a ^ b),
            solana_sbpf::error::ProgramResult::Err(err) => Err(Box::new(err)),
        }
    }
);

const INPUT_LEN: usize = 256;
const HEAP_LEN: usize = 256;
const FUNCTION_KEY_BASE: u32 = 0x4000_0000;

fn insn(opc: u8, dst: u8, src: u8, off: i16, imm: i32) -> [u8; 8] {
    let mut bytes = [0u8; 8];
    bytes[0] = opc;
    bytes[1] = (src << 4) | (dst & 0xf);
    bytes[2..4].copy_from_slice(&off.to_le_bytes());
    bytes[4..8].copy_from_slice(&imm.to_le_bytes());
    bytes
}

const INTERESTING_IMMEDIATES: [i64; 22] = [
    0,
    1,
    -1,
    2,
    3,
    7,
    31,
    32,
    63,
    64,
    255,
    4095,
    4096,
    -4095,
    -4096,
    0x7fff,
    0x8000,
    0xffff,
    i32::MIN as i64,
    i32::MAX as i64,
    0x5a,
    -0x80,
];

const INTERESTING_VALUES: [u64; 12] = [
    0,
    1,
    u64::MAX,
    i64::MIN as u64,
    i64::MAX as u64,
    i32::MIN as i64 as u64,
    i32::MAX as u64,
    u32::MAX as u64,
    0x8000_0000,
    0x1_0000_0000,
    0xffff_ffff_0000_0000,
    0x5a,
];

struct Generator {
    rng: SmallRng,
    version: SBPFVersion,
    syscall_keys: [u32; 2],
    /// Allow loops and recursion (instruction meter must be enabled)
    allow_loops: bool,
    has_syscalls: bool,
}

/// One instruction to be placed, with branch targets resolved later
enum Slot {
    Insn([u8; 8]),
    /// Jump to an absolute pc: (opc, dst, src, imm, target pc)
    Jump(u8, u8, u8, i32, usize),
    /// Internal call to function number n
    Call(usize),
    /// First / second half of loading the address of function n into a register
    FunctionAddressLow(u8, usize),
    FunctionAddressHigh(u8, usize),
}

struct Program {
    text: Vec<u8>,
    /// (key, pc) of the functions called by key (SBPF versions without static syscalls)
    functions: Vec<(u32, usize)>,
    has_loops: bool,
    has_syscalls: bool,
}

impl Program {
    fn function_registry(&self) -> FunctionRegistry<usize> {
        let mut function_registry = FunctionRegistry::default();
        for (key, pc) in &self.functions {
            function_registry
                .register_function(*key, format!("function_{key:x}"), *pc)
                .unwrap();
        }
        function_registry
    }
}

impl Generator {
    fn imm(&mut self) -> i64 {
        if self.rng.gen_bool(0.6) {
            INTERESTING_IMMEDIATES[self.rng.gen_range(0..INTERESTING_IMMEDIATES.len())]
        } else {
            self.rng.gen::<i32>() as i64
        }
    }

    fn value(&mut self) -> u64 {
        if self.rng.gen_bool(0.5) {
            INTERESTING_VALUES[self.rng.gen_range(0..INTERESTING_VALUES.len())]
        } else if self.rng.gen_bool(0.5) {
            self.rng.gen_range(0..1000)
        } else {
            self.rng.gen()
        }
    }

    /// A destination register, rarely r8 (the input base)
    fn dst(&mut self) -> u8 {
        if self.rng.gen_bool(0.03) {
            8
        } else {
            [0, 1, 2, 3, 4, 5, 6, 7, 9][self.rng.gen_range(0..9)]
        }
    }

    fn src(&mut self) -> u8 {
        self.rng.gen_range(0..=10)
    }

    fn load_constant(&mut self, out: &mut Vec<Slot>, dst: u8, value: u64) {
        if self.version.disable_lddw() {
            out.push(Slot::Insn(insn(
                ebpf::MOV32_IMM,
                dst,
                0,
                0,
                value as u32 as i32,
            )));
            if value >> 32 != 0 {
                out.push(Slot::Insn(insn(
                    ebpf::HOR64_IMM,
                    dst,
                    0,
                    0,
                    (value >> 32) as u32 as i32,
                )));
            }
        } else {
            out.push(Slot::Insn(insn(
                ebpf::LD_DW_IMM,
                dst,
                0,
                0,
                value as u32 as i32,
            )));
            out.push(Slot::Insn(insn(0, 0, 0, 0, (value >> 32) as u32 as i32)));
        }
    }

    fn alu(&mut self, out: &mut Vec<Slot>) {
        let v = self.version;
        let is64 = self.rng.gen_bool(0.5);
        let class = if is64 {
            ebpf::BPF_ALU64_STORE
        } else {
            ebpf::BPF_ALU32_LOAD
        };
        let use_reg = self.rng.gen_bool(0.5);
        let source = if use_reg { ebpf::BPF_X } else { ebpf::BPF_K };
        let dst = self.dst();
        let src = if use_reg { self.src() } else { 0 };
        let bits = if is64 { 64 } else { 32 };
        let mut ops = vec![
            ebpf::BPF_ADD,
            ebpf::BPF_SUB,
            ebpf::BPF_OR,
            ebpf::BPF_AND,
            ebpf::BPF_LSH,
            ebpf::BPF_RSH,
            ebpf::BPF_XOR,
            ebpf::BPF_MOV,
            ebpf::BPF_ARSH,
        ];
        if !v.enable_pqr() {
            ops.extend([ebpf::BPF_MUL, ebpf::BPF_DIV, ebpf::BPF_MOD]);
        }
        if !v.disable_neg() && !use_reg {
            ops.push(ebpf::BPF_NEG);
        }
        let op = ops[self.rng.gen_range(0..ops.len())];
        let imm = match op {
            ebpf::BPF_LSH | ebpf::BPF_RSH | ebpf::BPF_ARSH => self.rng.gen_range(0..bits),
            ebpf::BPF_DIV | ebpf::BPF_MOD => loop {
                let imm = self.imm();
                if imm != 0 {
                    break imm;
                }
            },
            ebpf::BPF_NEG => 0,
            _ => self.imm(),
        };
        out.push(Slot::Insn(insn(
            class | source | op,
            dst,
            src,
            0,
            imm as i32,
        )));
    }

    fn pqr(&mut self, out: &mut Vec<Slot>) {
        let use_reg = self.rng.gen_bool(0.5);
        let mut ops = vec![
            ebpf::BPF_LMUL,
            ebpf::BPF_UDIV,
            ebpf::BPF_UREM,
            ebpf::BPF_SDIV,
            ebpf::BPF_SREM,
        ];
        let is64 = self.rng.gen_bool(0.5);
        if is64 {
            ops.extend([ebpf::BPF_UHMUL, ebpf::BPF_SHMUL]);
        }
        let op = ops[self.rng.gen_range(0..ops.len())];
        let opc = ebpf::BPF_PQR
            | if use_reg { ebpf::BPF_X } else { ebpf::BPF_K }
            | if is64 { ebpf::BPF_B } else { 0 }
            | op;
        let imm = loop {
            let imm = self.imm();
            if imm != 0 || use_reg {
                break imm;
            }
        };
        let dst = self.dst();
        let src = if use_reg { self.src() } else { 0 };
        out.push(Slot::Insn(insn(opc, dst, src, 0, imm as i32)));
    }

    fn endian(&mut self, out: &mut Vec<Slot>) {
        let imm = [16, 32, 64][self.rng.gen_range(0..3)];
        let opc = if !self.version.disable_le() && self.rng.gen_bool(0.5) {
            ebpf::LE
        } else {
            ebpf::BE
        };
        let dst = self.dst();
        out.push(Slot::Insn(insn(opc, dst, 0, 0, imm)));
    }

    fn memory(&mut self, out: &mut Vec<Slot>) {
        let size_index = self.rng.gen_range(0..4);
        let size = 1i16 << size_index;
        let store = self.rng.gen_bool(0.5);
        let store_reg = self.rng.gen_bool(0.5);
        // Pick the base register and an offset which is usually in bounds
        let (base, off) = match self.rng.gen_range(0..10) {
            0..=4 => (
                8,
                if self.rng.gen_bool(0.97) {
                    self.rng.gen_range(0..=(INPUT_LEN as i16 - size))
                } else {
                    self.rng.gen_range(-64..INPUT_LEN as i16 + 64)
                },
            ),
            5..=7 => (
                10,
                if self.rng.gen_bool(0.97) {
                    -self.rng.gen_range(size..=512)
                } else {
                    self.rng.gen_range(-8192..8192)
                },
            ),
            8 => {
                let base = 7;
                let addr = if self.rng.gen_bool(0.5) {
                    ebpf::MM_HEAP_START
                } else {
                    ebpf::MM_BYTECODE_START
                };
                self.load_constant(out, base, addr);
                (base, self.rng.gen_range(0..=(HEAP_LEN as i16 - size)))
            }
            _ => (self.src(), self.rng.gen()),
        };
        let v = self.version;
        let opc = if v.move_memory_instruction_classes() {
            if !store {
                [
                    ebpf::LD_1B_REG,
                    ebpf::LD_2B_REG,
                    ebpf::LD_4B_REG,
                    ebpf::LD_8B_REG,
                ][size_index]
            } else if store_reg {
                [
                    ebpf::ST_1B_REG,
                    ebpf::ST_2B_REG,
                    ebpf::ST_4B_REG,
                    ebpf::ST_8B_REG,
                ][size_index]
            } else {
                [
                    ebpf::ST_1B_IMM,
                    ebpf::ST_2B_IMM,
                    ebpf::ST_4B_IMM,
                    ebpf::ST_8B_IMM,
                ][size_index]
            }
        } else if !store {
            [
                ebpf::LD_B_REG,
                ebpf::LD_H_REG,
                ebpf::LD_W_REG,
                ebpf::LD_DW_REG,
            ][size_index]
        } else if store_reg {
            [
                ebpf::ST_B_REG,
                ebpf::ST_H_REG,
                ebpf::ST_W_REG,
                ebpf::ST_DW_REG,
            ][size_index]
        } else {
            [
                ebpf::ST_B_IMM,
                ebpf::ST_H_IMM,
                ebpf::ST_W_IMM,
                ebpf::ST_DW_IMM,
            ][size_index]
        };
        if store {
            let src = if store_reg { self.src() } else { 0 };
            let imm = self.imm() as i32;
            out.push(Slot::Insn(insn(opc, base, src, off, imm)));
        } else {
            let dst = self.dst();
            out.push(Slot::Insn(insn(opc, dst, base, off, 0)));
        }
    }

    fn branch(&mut self, out: &mut Vec<Slot>, current: usize, body: std::ops::Range<usize>) {
        let target = if self.allow_loops && self.rng.gen_bool(0.3) {
            self.rng.gen_range(body.start..=current.min(body.end))
        } else {
            self.rng.gen_range(current.min(body.end)..=body.end)
        };
        if self.rng.gen_bool(0.1) {
            out.push(Slot::Jump(ebpf::JA, 0, 0, 0, target));
            return;
        }
        let is32 = self.version.enable_jmp32() && self.rng.gen_bool(0.4);
        let class = if is32 {
            ebpf::BPF_JMP32
        } else {
            ebpf::BPF_JMP64
        };
        let ops = [
            ebpf::BPF_JEQ,
            ebpf::BPF_JGT,
            ebpf::BPF_JGE,
            ebpf::BPF_JLT,
            ebpf::BPF_JLE,
            ebpf::BPF_JSET,
            ebpf::BPF_JNE,
            ebpf::BPF_JSGT,
            ebpf::BPF_JSGE,
            ebpf::BPF_JSLT,
            ebpf::BPF_JSLE,
        ];
        let op = ops[self.rng.gen_range(0..ops.len())];
        let use_reg = self.rng.gen_bool(0.5);
        let dst = self.rng.gen_range(0..=9);
        if use_reg {
            let src = self.src();
            out.push(Slot::Jump(class | ebpf::BPF_X | op, dst, src, 0, target));
        } else {
            let imm = self.imm() as i32;
            out.push(Slot::Jump(class | ebpf::BPF_K | op, dst, 0, imm, target));
        }
    }

    fn syscall(&mut self, out: &mut Vec<Slot>) {
        // Arguments in r1 ..= r5 are whatever the program computed
        let key = self.syscall_keys[self.rng.gen_range(0..2)];
        if self.syscall_keys[1] == key {
            // SyscallStore: make the address valid most of the time
            if self.rng.gen_bool(0.8) {
                let addr = ebpf::MM_INPUT_START + self.rng.gen_range(0..(INPUT_LEN as u64 - 8));
                self.load_constant(out, 2, addr);
            }
        }
        out.push(Slot::Insn(insn(ebpf::CALL_IMM, 0, 0, 0, key as i32)));
        self.has_syscalls = true;
    }

    fn callx(&mut self, out: &mut Vec<Slot>, function_count: usize) {
        let reg = self.rng.gen_range(0..=9);
        match self.rng.gen_range(0..10) {
            0 => {
                let value = self.value();
                self.load_constant(out, reg, value)
            }
            1 => {
                let addr = ebpf::MM_BYTECODE_START
                    + self.rng.gen_range(0..64) * 8
                    + self.rng.gen_range(0..8);
                self.load_constant(out, reg, addr)
            }
            _ => {
                let function = self.rng.gen_range(0..function_count);
                out.push(Slot::FunctionAddressLow(reg, function));
                out.push(Slot::FunctionAddressHigh(reg, function));
            }
        }
        let v = self.version;
        out.push(Slot::Insn(if v.callx_uses_src_reg() {
            insn(ebpf::CALL_REG, 0, reg, 0, 0)
        } else if v.callx_uses_dst_reg() {
            insn(ebpf::CALL_REG, reg, 0, 0, 0)
        } else {
            insn(ebpf::CALL_REG, 0, 0, 0, reg as i32)
        }));
    }

    /// Instructions of a basic block of `len` random instructions
    fn block(
        &mut self,
        out: &mut Vec<Slot>,
        len: usize,
        body: std::ops::Range<usize>,
        function_count: usize,
        callee_min: usize,
    ) {
        for _ in 0..len {
            let current = out.len();
            match self.rng.gen_range(0..100) {
                0..=34 => self.alu(out),
                35..=42 if self.version.enable_pqr() => self.pqr(out),
                35..=42 => self.alu(out),
                43..=46 => self.endian(out),
                47..=49 => {
                    let dst = self.dst();
                    let value = self.value();
                    self.load_constant(out, dst, value)
                }
                50..=69 => self.memory(out),
                70..=84 => self.branch(out, current, body.clone()),
                85..=88 if callee_min < function_count => {
                    // Calls only go to later functions unless loops (recursion) are allowed
                    let first = if self.allow_loops { 0 } else { callee_min };
                    out.push(Slot::Call(self.rng.gen_range(first..function_count)));
                }
                89..=92 => self.syscall(out),
                93..=94 if self.allow_loops && function_count > 0 => {
                    self.callx(out, function_count)
                }
                95 if self.version.manual_stack_frame_bump() => {
                    let imm = 64 * self.rng.gen_range(-4..=4);
                    out.push(Slot::Insn(insn(ebpf::ADD64_IMM, 10, 0, 0, imm)));
                }
                96 => out.push(Slot::Insn(insn(ebpf::EXIT, 0, 0, 0, 0))),
                _ => self.alu(out),
            }
        }
    }

    fn generate(&mut self) -> (Vec<Slot>, Vec<usize>) {
        let function_count = self.rng.gen_range(0..4);
        let mut slots = Vec::new();
        // Prologue: initialize the registers
        for reg in 0..=9u8 {
            let value = if reg == 8 {
                ebpf::MM_INPUT_START
            } else {
                self.value()
            };
            self.load_constant(&mut slots, reg, value);
        }
        let body_start = slots.len();
        let body_len = self.rng.gen_range(1..60);
        // Body, the end of the body (the epilogue) is a valid jump target
        let mut body = Vec::new();
        let estimated_end = body_start + body_len * 2;
        self.block(
            &mut body,
            body_len,
            0..estimated_end - body_start,
            function_count,
            0,
        );
        for slot in body.iter_mut() {
            if let Slot::Jump(_, _, _, _, target) = slot {
                *target += body_start;
            }
        }
        slots.extend(body);
        let body_end = slots.len();
        for slot in slots[body_start..].iter_mut() {
            if let Slot::Jump(_, _, _, _, target) = slot {
                *target = (*target).min(body_end);
            }
        }
        // Epilogue: spill r0 ..= r9 to the stack, then exit
        for reg in 0..=9u8 {
            slots.push(Slot::Insn(insn(
                if self.version.move_memory_instruction_classes() {
                    ebpf::ST_8B_REG
                } else {
                    ebpf::ST_DW_REG
                },
                10,
                reg,
                -8 * (reg as i16 + 1),
                0,
            )));
        }
        slots.push(Slot::Insn(insn(ebpf::EXIT, 0, 0, 0, 0)));
        // Functions
        let mut function_pcs = Vec::new();
        for function in 0..function_count {
            function_pcs.push(slots.len());
            let start = slots.len();
            let len = self.rng.gen_range(1..12);
            let mut block = Vec::new();
            self.block(&mut block, len, 0..len * 2, function_count, function + 1);
            let end = start + block.len();
            for slot in block.iter_mut() {
                if let Slot::Jump(_, _, _, _, target) = slot {
                    *target = (*target + start).min(end);
                }
            }
            slots.extend(block);
            slots.push(Slot::Insn(insn(ebpf::EXIT, 0, 0, 0, 0)));
        }
        (slots, function_pcs)
    }

    fn assemble(&mut self) -> Program {
        let (slots, function_pcs) = self.generate();
        let mut functions = Vec::new();
        let mut text = Vec::new();
        let mut has_loops = false;
        for (pc, slot) in slots.iter().enumerate() {
            let bytes = match slot {
                Slot::Insn(bytes) => *bytes,
                Slot::Jump(opc, dst, src, imm, target) => {
                    has_loops |= *target <= pc;
                    insn(
                        *opc,
                        *dst,
                        *src,
                        (*target as i64 - pc as i64 - 1) as i16,
                        *imm,
                    )
                }
                Slot::FunctionAddressLow(reg, function) => {
                    let addr = ebpf::MM_BYTECODE_START + function_pcs[*function] as u64 * 8;
                    if self.version.disable_lddw() {
                        insn(ebpf::MOV32_IMM, *reg, 0, 0, addr as u32 as i32)
                    } else {
                        insn(ebpf::LD_DW_IMM, *reg, 0, 0, addr as u32 as i32)
                    }
                }
                Slot::FunctionAddressHigh(reg, function) => {
                    let addr = ebpf::MM_BYTECODE_START + function_pcs[*function] as u64 * 8;
                    if self.version.disable_lddw() {
                        insn(ebpf::HOR64_IMM, *reg, 0, 0, (addr >> 32) as i32)
                    } else {
                        insn(0, 0, 0, 0, (addr >> 32) as i32)
                    }
                }
                Slot::Call(function) => {
                    let target = function_pcs[*function];
                    has_loops = true;
                    if self.version.static_syscalls() {
                        insn(
                            ebpf::CALL_IMM,
                            0,
                            1,
                            0,
                            (target as i64 - pc as i64 - 1) as i32,
                        )
                    } else {
                        let key = FUNCTION_KEY_BASE + *function as u32;
                        functions.push((key, target));
                        insn(ebpf::CALL_IMM, 0, 0, 0, key as i32)
                    }
                }
            };
            text.extend_from_slice(&bytes);
        }
        Program {
            text,
            functions,
            has_loops,
            has_syscalls: self.has_syscalls,
        }
    }
}

struct Outcome {
    result: String,
    instruction_count: u64,
    final_pc: u64,
    remaining: u64,
    trace: Vec<[u64; 12]>,
    input: Vec<u8>,
    stack: Vec<u8>,
    heap: Vec<u8>,
}

fn run(
    executable: &Executable<TestContextObject>,
    budget: u64,
    mode: ExecutionMode,
    input_seed: u64,
) -> Outcome {
    let config = executable.get_config();
    let mut input = vec![0u8; INPUT_LEN];
    SmallRng::seed_from_u64(input_seed).fill(&mut input[..]);
    let mut stack = AlignedMemory::<{ HOST_ALIGN }>::zero_filled(config.stack_size());
    let mut heap = AlignedMemory::<{ HOST_ALIGN }>::zero_filled(HEAP_LEN);
    let mut context_object = TestContextObject::new(budget);
    let input_region = MemoryRegion::new(&raw mut input[..], ebpf::MM_INPUT_START);
    context_object.memory_mapping =
        create_memory_mapping(executable, &mut stack, &mut heap, vec![input_region], None).unwrap();
    let stack_len = stack.len();
    let mut call_frames = vec![CallFrame::default(); config.max_call_depth];
    let (instruction_count, result, final_pc, trace) = {
        let mut vm = EbpfVm::new(
            executable.get_loader().clone(),
            executable.get_sbpf_version(),
            &mut context_object,
            stack_len,
        );
        vm.registers[1] = ebpf::MM_INPUT_START;
        let mut mode = mode;
        let (instruction_count, result) =
            vm.execute_program(executable, &mut mode, &mut call_frames);
        (
            instruction_count,
            format!("{result:?}"),
            vm.registers[11],
            std::mem::take(&mut vm.register_trace),
        )
    };
    Outcome {
        result,
        instruction_count,
        final_pc,
        remaining: context_object.remaining,
        trace,
        input,
        stack: stack.as_slice().to_vec(),
        heap: heap.as_slice().to_vec(),
    }
}

fn check_program(
    version: SBPFVersion,
    program: &Program,
    config: Config,
    budget: u64,
    input_seed: u64,
    min_machine_code_length: usize,
) -> Option<Outcome> {
    let mut loader = BuiltinProgram::new_loader(config);
    SyscallMix::register(&mut loader, "mix").unwrap();
    SyscallStore::register(&mut loader, "store").unwrap();
    let Ok(executable) = Executable::<TestContextObject>::from_text_bytes(
        &program.text,
        Arc::new(loader),
        version,
        program.function_registry(),
    ) else {
        return None;
    };
    if executable.verify::<RequisiteVerifier>().is_err() {
        return None;
    }
    executable.jit_compile().unwrap();
    let machine_code_length = executable
        .get_compiled_program()
        .unwrap()
        .machine_code_length();
    assert!(
        machine_code_length >= min_machine_code_length,
        "{}",
        machine_code_length
    );
    let interpreter = run(&executable, budget, ExecutionMode::Interpreted, input_seed);
    let jit = run(&executable, budget, ExecutionMode::Jit, input_seed);
    let mut diverged = Vec::new();
    if interpreter.result != jit.result {
        diverged.push(format!(
            "result: interpreter {} JIT {}",
            interpreter.result, jit.result
        ));
    }
    if interpreter.instruction_count != jit.instruction_count {
        diverged.push(format!(
            "instruction count: interpreter {} JIT {}",
            interpreter.instruction_count, jit.instruction_count
        ));
    }
    if interpreter.final_pc != jit.final_pc {
        diverged.push(format!(
            "final pc: interpreter {} JIT {}",
            interpreter.final_pc, jit.final_pc
        ));
    }
    if interpreter.remaining != jit.remaining {
        diverged.push(format!(
            "remaining: interpreter {} JIT {}",
            interpreter.remaining, jit.remaining
        ));
    }
    if !compare_register_trace(&interpreter.trace, &jit.trace) {
        diverged.push(format!(
            "register trace: interpreter {} entries JIT {} entries",
            interpreter.trace.len(),
            jit.trace.len()
        ));
    }
    // The JIT only checks the instruction meter at branches (and checkpoints), so when the
    // budget runs out it may have executed (and stored) a few more instructions of a linear
    // run than the interpreter. Results, pc and counts still agree, memory does not have to.
    if interpreter.result != "Err(ExceededMaxInstructions)" {
        for (name, a, b) in [
            ("input", &interpreter.input, &jit.input),
            ("stack", &interpreter.stack, &jit.stack),
            ("heap", &interpreter.heap, &jit.heap),
        ] {
            if let Some(offset) = a.iter().zip(b.iter()).position(|(a, b)| a != b) {
                diverged.push(format!(
                    "{name} memory at offset {offset}: interpreter {:02x?} JIT {:02x?}",
                    &a[offset & !7..(offset & !7) + 8],
                    &b[offset & !7..(offset & !7) + 8]
                ));
            }
        }
    }
    if !diverged.is_empty() {
        let analysis =
            solana_sbpf::static_analysis::Analysis::from_executable(&executable).unwrap();
        let stdout = std::io::stdout();
        analysis.disassemble(&mut stdout.lock()).unwrap();
        panic!(
            "{version:?} budget {budget} config {:?}\n{}",
            executable.get_config(),
            diverged.join("\n")
        );
    }
    Some(jit)
}

fn random_config(rng: &mut SmallRng, version: SBPFVersion, meter: bool) -> Config {
    Config {
        enabled_sbpf_versions: version..=version,
        enable_register_tracing: rng.gen_bool(0.3),
        enable_instruction_meter: meter,
        instruction_meter_checkpoint_distance: if rng.gen_bool(0.5) {
            rng.gen_range(1..20)
        } else {
            10000
        },
        max_call_depth: rng.gen_range(2..=8),
        enable_stack_frame_gaps: rng.gen_bool(0.5),
        aligned_memory_mapping: rng.gen_bool(0.5),
        noop_instruction_rate: if rng.gen_bool(0.5) { 0 } else { 256 },
        sanitize_user_provided_values: rng.gen_bool(0.5),
        ..Config::default()
    }
}

fn iterations() -> usize {
    std::env::var("JIT_DIFF_ITERATIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3000)
}

fn seed() -> u64 {
    std::env::var("JIT_DIFF_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0x05ee_da64_u64)
}

fn differential(version: SBPFVersion) {
    let mut rng = SmallRng::seed_from_u64(seed() ^ version as u64);
    let syscall_keys = [
        ebpf::hash_symbol_name(b"mix"),
        ebpf::hash_symbol_name(b"store"),
    ];
    let mut checked = 0;
    let mut attempts = 0;
    let mut results = std::collections::BTreeMap::<String, usize>::new();
    let mut executed_instructions = 0;
    while checked < iterations() {
        attempts += 1;
        assert!(attempts < iterations() * 20, "too many invalid programs");
        let allow_loops = rng.gen_bool(0.8);
        let mut generator = Generator {
            rng: SmallRng::seed_from_u64(rng.gen()),
            version,
            syscall_keys,
            allow_loops,
            has_syscalls: false,
        };
        let program = generator.assemble();
        // The interpreter can not run syscalls without the instruction meter (debug overflow)
        let meter = program.has_loops || program.has_syscalls || rng.gen_bool(0.8);
        let config = random_config(&mut rng, version, meter);
        let budget = if rng.gen_bool(0.5) {
            rng.gen_range(0..200)
        } else {
            rng.gen_range(200..5000)
        };
        if let Some(outcome) = check_program(version, &program, config, budget, rng.gen(), 0) {
            checked += 1;
            executed_instructions += outcome.instruction_count;
            let kind = if outcome.result.starts_with("Ok") {
                "Ok".to_string()
            } else {
                outcome.result[4..]
                    .split(['(', ')'])
                    .next()
                    .unwrap_or_default()
                    .to_string()
            };
            *results.entry(kind).or_default() += 1;
        }
    }
    println!(
        "{version:?}: {checked} programs ({attempts} generated), {executed_instructions} instructions, results {results:?}"
    );
}

#[test]
fn test_differential_v0() {
    differential(SBPFVersion::V0);
}

#[test]
fn test_differential_v1() {
    differential(SBPFVersion::V1);
}

#[test]
fn test_differential_v2() {
    differential(SBPFVersion::V2);
}

#[test]
fn test_differential_v3() {
    differential(SBPFVersion::V3);
}

#[test]
fn test_differential_v4() {
    differential(SBPFVersion::V4);
}

/// Programs larger than the reach of conditional branches (+-1 MiB of machine code), with
/// conditional branches forward and backward across all of it.
#[test]
fn test_differential_large_program() {
    for version in [SBPFVersion::V0, SBPFVersion::V3] {
        let filler = 32_000usize;
        let mut text = Vec::new();
        text.extend_from_slice(&insn(ebpf::MOV64_IMM, 1, 0, 0, 0));
        text.extend_from_slice(&insn(ebpf::MOV64_IMM, 2, 0, 0, 0));
        // pc 2: on the first round jump forward over the whole filler
        text.extend_from_slice(&insn(ebpf::JEQ64_IMM, 1, 0, filler as i16, 0));
        for pc in 3..3 + filler {
            let bytes = match pc % 8 {
                0 => insn(ebpf::ADD64_IMM, 2, 0, 0, pc as i32),
                1 => insn(ebpf::JSGT64_REG, 2, 1, 0, 0),
                2 | 5 => insn(ebpf::JSET64_IMM, 2, 0, 0, 0x1234_5678 << 1),
                _ => insn(ebpf::JEQ64_IMM, 2, 0, 0, 0x1234_5678),
            };
            text.extend_from_slice(&bytes);
        }
        // pc 3 + filler: second round goes back to pc 2 (backward, far), the third one exits
        let pc = 3 + filler;
        text.extend_from_slice(&insn(ebpf::ADD64_IMM, 1, 0, 0, 1));
        text.extend_from_slice(&insn(ebpf::JLT64_IMM, 1, 0, 2 - (pc as i16 + 2), 2));
        text.extend_from_slice(&insn(ebpf::MOV64_REG, 0, 2, 0, 0));
        text.extend_from_slice(&insn(ebpf::EXIT, 0, 0, 0, 0));
        let program = Program {
            text,
            functions: Vec::new(),
            has_loops: true,
            has_syscalls: false,
        };
        for (budget, tracing) in [
            (10, true),
            (20_000, false),
            (40_000, true),
            (1_000_000, false),
        ] {
            let config = Config {
                enabled_sbpf_versions: version..=version,
                enable_register_tracing: tracing,
                instruction_meter_checkpoint_distance: if tracing { 3 } else { 1 },
                ..Config::default()
            };
            assert!(check_program(version, &program, config, budget, 0, 17 << 16).is_some());
        }
    }
}
