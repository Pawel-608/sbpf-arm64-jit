//! Just-in-time compiler (Linux aarch64, macOS aarch64)
//!
//! A port of the x86-64 JIT (`jit.rs`) to aarch64. It keeps the same structure, the same
//! anchors (subroutines), the same instruction meter ("integral") technique and the same
//! exception paths, so that both JITs (and the interpreter) agree on results, errors, final pc
//! and instruction meter consumption.
//!
//! Not ported (they never change results): the random no-op padding
//! (`Config::noop_instruction_rate`) and the immediate value sanitization
//! (`Config::sanitize_user_provided_values`).

// Derived from uBPF <https://github.com/iovisor/ubpf>
// Copyright 2015 Big Switch Networks, Inc
//      (uBPF: JIT algorithm, originally in C)
// Copyright 2016 6WIND S.A. <quentin.monnet@6wind.com>
//      (Translation to Rust, MetaBuff addition)
// Copyright 2020 Solana Maintainers <maintainers@solana.com>
//
// Licensed under the Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> or
// the MIT license <http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#![allow(clippy::arithmetic_side_effects)]

use std::{fmt::Debug, mem, ptr};

use crate::{
    aarch64::{
        ARM64Instruction, ARM64MemoryOperand, Condition, FP as X29, LR, SP_XZR, X0, X1, X10, X11,
        X12, X13, X14, X15, X16, X17, X19, X2, X20, X21, X22, X23, X24, X25, X26, X27, X28, X3, X4,
        X5, X9,
    },
    ebpf::{self, FIRST_SCRATCH_REG, FRAME_PTR_REG, INSN_SIZE, SCRATCH_REGS},
    elf::Executable,
    error::{EbpfError, ProgramResult},
    memory_management::{
        allocate_pages_pooled, flush_instruction_cache, free_pages_pooled, get_system_page_size,
        protect_pages, round_to_page_size, JitWriteGuard, PagePermissions,
    },
    memory_region::MemoryMapping,
    program::BuiltinFunction,
    static_analysis::RegisterTraceEntry,
    vm::{get_runtime_environment_key, Config, ContextObject, EbpfVm, RuntimeEnvironmentSlot},
};

/// The maximum machine code length in bytes of a program with no guest instructions
pub const MAX_EMPTY_PROGRAM_MACHINE_CODE_LENGTH: usize = 4096;
/// The maximum machine code length in bytes of a single guest instruction
pub const MAX_MACHINE_CODE_LENGTH_PER_INSTRUCTION: usize = 160;
/// The maximum machine code length in bytes of an instruction meter checkpoint
pub const MACHINE_CODE_PER_INSTRUCTION_METER_CHECKPOINT: usize = 24;
/// The maximum machine code length of the randomized padding (unused on aarch64)
pub const MAX_START_PADDING_LENGTH: usize = 256;

/// B and BL reach +-128 MiB
const MAX_BRANCH_DISTANCE: isize = 1 << 27;
/// B.cond, CBZ and CBNZ reach +-1 MiB
const MAX_CONDITIONAL_BRANCH_DISTANCE: isize = 1 << 20;
/// BRK #0, fills the unused tail of the text section
const TRAP_INSTRUCTION: u32 = 0xd420_0000;

/// The program compiled to native host machinecode
pub struct JitProgram {
    /// OS page size in bytes and the alignment of the sections
    page_size: usize,
    /// Full pooled allocation size backing the pc and text sections.
    allocation_size: usize,
    /// Byte offset in the text_section for each BPF instruction
    pc_section: &'static mut [u32],
    /// The aarch64 machinecode.
    ///
    /// Before sealing this is the full code capacity; after sealing
    /// this is the emitted code.
    text_section: &'static mut [u8],
}

impl JitProgram {
    fn new(pc: usize, code_size: usize) -> Result<Self, EbpfError> {
        let page_size = get_system_page_size();
        let pc_loc_table_size = round_to_page_size(pc * std::mem::size_of::<u32>(), page_size);
        let over_allocated_code_size = round_to_page_size(code_size, page_size);
        let (raw, allocation_size) =
            allocate_pages_pooled(pc_loc_table_size + over_allocated_code_size);

        unsafe {
            let pc_section = std::slice::from_raw_parts_mut(raw.cast::<u32>(), pc);
            // pc_section relies on zero-initialization to distinguish unfilled
            // forward-jump targets from filled backward-jump targets in
            // relative_to_target_pc. The pool may hand back recycled memory, so
            // zero just the pc_section bytes here.
            pc_section.fill(0);
            Ok(Self {
                page_size,
                allocation_size,
                pc_section,
                text_section: std::slice::from_raw_parts_mut(
                    raw.add(pc_loc_table_size),
                    over_allocated_code_size,
                ),
            })
        }
    }

    fn seal(&mut self, text_section_usage: usize) -> Result<(), EbpfError> {
        if self.page_size == 0 {
            return Ok(());
        }
        let raw = self.pc_section.as_ptr() as *mut u8;
        let pc_loc_table_size =
            round_to_page_size(std::mem::size_of_val(self.pc_section), self.page_size);
        let code_size = round_to_page_size(text_section_usage, self.page_size);
        unsafe {
            // Fill with debugger traps
            let mut offset = text_section_usage;
            while offset < code_size {
                ptr::write_unaligned(
                    raw.add(pc_loc_table_size).add(offset).cast::<u32>(),
                    TRAP_INSTRUCTION,
                );
                offset += mem::size_of::<u32>();
            }
            protect_pages(
                self.pc_section.as_mut_ptr().cast::<u8>(),
                pc_loc_table_size,
                PagePermissions::Read,
            )?;
            protect_pages(
                self.text_section.as_mut_ptr(),
                code_size,
                PagePermissions::ReadExecute,
            )?;
            flush_instruction_cache(self.text_section.as_ptr(), code_size);
            self.text_section =
                std::slice::from_raw_parts_mut(self.text_section.as_mut_ptr(), text_section_usage);
        }
        Ok(())
    }

    pub(crate) fn invoke<C: ContextObject>(
        &self,
        _config: &Config,
        vm: &mut EbpfVm<C>,
        registers: [u64; 12],
    ) {
        unsafe {
            let instruction_meter =
                (vm.previous_instruction_meter as i64).wrapping_add(registers[11] as i64);
            let entrypoint = &self.text_section
                [self.pc_section[registers[11] as usize] as usize & (i32::MAX as u32 as usize)]
                as *const u8;
            // The trampoline at the start of the text section (see `emit_subroutines()`)
            let trampoline: unsafe extern "C" fn(*mut EbpfVm<C>, *const u64, *const u8, i64) =
                mem::transmute(self.text_section.as_ptr());
            trampoline(vm, registers.as_ptr(), entrypoint, instruction_meter);
        }
    }

    /// The length of the host machinecode in bytes
    pub fn machine_code_length(&self) -> usize {
        self.text_section.len()
    }

    /// The total pooled allocation size retained by the compiled program.
    pub fn mem_size(&self) -> usize {
        self.allocation_size
    }
}

impl Drop for JitProgram {
    fn drop(&mut self) {
        unsafe {
            free_pages_pooled(
                self.pc_section.as_mut_ptr().cast::<u8>(),
                self.allocation_size,
            );
        }
    }
}

impl Debug for JitProgram {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.write_fmt(format_args!("JitProgram {:?}", self as *const _))
    }
}

impl PartialEq for JitProgram {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self as *const _, other as *const _)
    }
}

// Used to define subroutines and then call them
// See JitCompiler::set_anchor() and JitCompiler::anchor_offset()
const ANCHOR_TRACE: usize = 0;
const ANCHOR_THROW_EXCEEDED_MAX_INSTRUCTIONS: usize = 1;
const ANCHOR_EPILOGUE: usize = 2;
const ANCHOR_THROW_EXCEPTION_UNCHECKED: usize = 3;
const ANCHOR_EXIT: usize = 4;
const ANCHOR_THROW_EXCEPTION: usize = 5;
const ANCHOR_CALL_DEPTH_EXCEEDED: usize = 6;
const ANCHOR_CALL_REG_OUTSIDE_TEXT_SEGMENT: usize = 7;
const ANCHOR_DIV_BY_ZERO: usize = 8;
const ANCHOR_DIV_OVERFLOW: usize = 9;
const ANCHOR_CALL_UNSUPPORTED_INSTRUCTION: usize = 11;
const ANCHOR_EXTERNAL_FUNCTION_CALL: usize = 12;
const ANCHOR_INTERNAL_FUNCTION_CALL_PROLOGUE: usize = 13;
const ANCHOR_INTERNAL_FUNCTION_CALL_REG: usize = 14;
const ANCHOR_TRANSLATE_MEMORY_ADDRESS: usize = 21;
const ANCHOR_COUNT: usize = 34; // Update me when adding or removing anchors

/* Register allocation

    BPF r0 ..= r9 live in the callee saved registers X19 ..= X28, so they survive calls into
    Rust (syscalls and memory address translation). BPF r10 (the frame pointer), the pointer to
    the VM and the instruction meter live in caller saved registers and are saved around calls.
    X29 (FP) keeps pointing at the frame record of the trampoline, so backtraces stay intact.
    X18 is never touched (platform register on Apple).

    X12 (REGISTER_SCRATCH) has the same role as R11 in the x86 JIT: it holds the guest pc when
    jumping to an exception handler. X13 ..= X17 are temporaries:
    - X13: guest pc passed to the memory translation subroutines
    - X14: value to store / guest target address of callx
    - X15: immediate operands of ALU instructions
    - X16, X17: everything else (never live across instructions)
*/
const REGISTER_MAP: [u8; 11] = [X19, X20, X21, X22, X23, X24, X25, X26, X27, X28, X9];

/// X10: Pointer to the EbpfVm (not encrypted)
const REGISTER_PTR_TO_VM: u8 = X10;
/// X11: Program counter limit
const REGISTER_INSTRUCTION_METER: u8 = X11;
/// X12: Scratch register
const REGISTER_SCRATCH: u8 = X12;
/// X13: Guest pc for the memory translation subroutines
const REGISTER_TRANSLATION_PC: u8 = X13;
/// X14: Value to store / callx target
const REGISTER_OTHER: u8 = X14;
/// X15: Immediate operands
const REGISTER_IMMEDIATE: u8 = X15;
/// X16: Temporary
const REGISTER_TEMP: u8 = X16;
/// X17: Temporary
const REGISTER_TEMP2: u8 = X17;

/// Size of the host stack frame of a BPF to BPF call:
/// [return address, r10, r6, r7, r8, r9]
const CALL_FRAME_SIZE: u16 = 8 * (SCRATCH_REGS as u16 + 2);

/// Bit width of an instruction operand
#[derive(Copy, Clone, Debug)]
pub enum OperandSize {
    /// Empty
    S0 = 0,
    /// 8 bit
    S8 = 8,
    /// 16 bit
    S16 = 16,
    /// 32 bit
    S32 = 32,
    /// 64 bit
    S64 = 64,
}

#[derive(Debug)]
struct Jump {
    /// Offset of a B / BL instruction in the text section
    location: usize,
    target_pc: usize,
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum AluOp {
    Add,
    Sub,
    Or,
    And,
    Xor,
}

/// Result of a memory access as returned (in X0 and X1) to the JIT code
#[repr(C)]
struct TranslationResult {
    is_err: u64,
    value: u64,
}

macro_rules! translation_functions {
    ($($load:ident, $store:ident, $T:ty;)*) => {
        $(
            unsafe extern "C" fn $load(
                memory_mapping: *mut MemoryMapping,
                vm_addr: u64,
                program_result: *mut ProgramResult,
            ) -> TranslationResult {
                match unsafe { (*memory_mapping).load::<$T>(vm_addr) } {
                    ProgramResult::Ok(value) => TranslationResult { is_err: 0, value },
                    err => {
                        unsafe { ptr::write(program_result, err) };
                        TranslationResult { is_err: 1, value: 0 }
                    }
                }
            }

            unsafe extern "C" fn $store(
                memory_mapping: *mut MemoryMapping,
                value: u64,
                vm_addr: u64,
                program_result: *mut ProgramResult,
            ) -> u64 {
                match unsafe { (*memory_mapping).store::<$T>(value as $T, vm_addr) } {
                    ProgramResult::Ok(_) => 0,
                    err => {
                        unsafe { ptr::write(program_result, err) };
                        1
                    }
                }
            }
        )*
    };
}

translation_functions! {
    load_u8, store_u8, u8;
    load_u16, store_u16, u16;
    load_u32, store_u32, u32;
    load_u64, store_u64, u64;
}

unsafe extern "C" fn push_register_trace_entry(
    register_trace: *mut Vec<RegisterTraceEntry>,
    entry: *const RegisterTraceEntry,
) {
    unsafe { (*register_trace).push(*entry) }
}

/* Explanation of the Instruction Meter

    See jit.rs, the same technique is used here: Every branch subtracts its current pc and adds
    its target pc to the remaining instruction counter (REGISTER_INSTRUCTION_METER), which then
    measures the maximum pc that can be reached with the remaining budget.
*/

/// Temporary object which stores the compilation context
pub struct JitCompiler<'a, C: ContextObject> {
    result: JitProgram,
    text_section_jumps: Vec<Jump>,
    anchors: [usize; ANCHOR_COUNT],
    offset_in_text_section: usize,
    executable: &'a Executable<C>,
    program: &'a [u8],
    program_vm_addr: u64,
    config: &'a Config,
    pc: usize,
    last_instruction_meter_validation_pc: usize,
    runtime_environment_key: i32,
    branch_out_of_range: bool,
    // Dropped last: makes the code pages executable again (macOS MAP_JIT)
    _write_guard: JitWriteGuard,
}

#[rustfmt::skip]
impl<'a, C: ContextObject> JitCompiler<'a, C> {
    /// Constructs a new compiler and allocates memory for the compilation output
    pub fn new(executable: &'a Executable<C>) -> Result<Self, EbpfError> {
        let config = executable.get_config();
        let (program_vm_addr, program) = executable.get_text_bytes();

        // Scan through program to find actual number of instructions
        let mut pc = 0;
        if !executable.get_sbpf_version().disable_lddw() {
            while (pc + 1) * ebpf::INSN_SIZE <= program.len() {
                let insn = ebpf::get_insn_unchecked(program, pc);
                pc += match insn.opc {
                    ebpf::LD_DW_IMM => 2,
                    _ => 1,
                };
            }
        } else {
            pc = program.len() / ebpf::INSN_SIZE;
        }

        let mut code_length_estimate = MAX_EMPTY_PROGRAM_MACHINE_CODE_LENGTH + MAX_MACHINE_CODE_LENGTH_PER_INSTRUCTION * pc;
        if let Some(q) = pc.checked_div(config.instruction_meter_checkpoint_distance) {
            code_length_estimate += q * MACHINE_CODE_PER_INSTRUCTION_METER_CHECKPOINT;
        }
        // Relative jump destinations limit the maximum output size
        debug_assert!(code_length_estimate < (i32::MAX as usize));

        let write_guard = JitWriteGuard::new();
        Ok(Self {
            result: JitProgram::new(pc, code_length_estimate)?,
            text_section_jumps: vec![],
            anchors: [usize::MAX; ANCHOR_COUNT],
            offset_in_text_section: 0,
            executable,
            program_vm_addr,
            program,
            config,
            pc: 0,
            last_instruction_meter_validation_pc: 0,
            runtime_environment_key: get_runtime_environment_key(),
            branch_out_of_range: false,
            _write_guard: write_guard,
        })
    }

    /// Compiles the given executable, consuming the compiler
    pub fn compile(mut self) -> Result<JitProgram, EbpfError> {
        self.emit_subroutines();

        while self.pc * ebpf::INSN_SIZE < self.program.len() {
            if self.offset_in_text_section + MAX_MACHINE_CODE_LENGTH_PER_INSTRUCTION * 2 >= self.result.text_section.len() {
                return Err(EbpfError::ExhaustedTextSegment(self.pc));
            }
            let mut insn = ebpf::get_insn_unchecked(self.program, self.pc);
            self.result.pc_section[self.pc] = self.offset_in_text_section as u32;

            // Regular instruction meter checkpoints to prevent long linear runs from exceeding their budget
            if self.last_instruction_meter_validation_pc + self.config.instruction_meter_checkpoint_distance <= self.pc {
                self.emit_validate_instruction_count(Some(self.pc));
            }

            if self.config.enable_register_tracing {
                self.emit_load_immediate(REGISTER_SCRATCH, self.pc as i64);
                self.emit_call_anchor(ANCHOR_TRACE);
            }

            let dst = REGISTER_MAP[insn.dst as usize];
            let src = REGISTER_MAP[insn.src as usize];
            let target_pc = (self.pc as isize + insn.off as isize + 1) as usize;

            match insn.opc {
                ebpf::LD_DW_IMM if !self.executable.get_sbpf_version().disable_lddw() => {
                    self.emit_validate_and_profile_instruction_count(self.pc + 2);
                    self.pc += 1;
                    self.result.pc_section[self.pc] = self.anchors[ANCHOR_CALL_UNSUPPORTED_INSTRUCTION] as u32;
                    ebpf::augment_lddw_unchecked(self.program, &mut insn);
                    self.emit_load_immediate(dst, insn.imm);
                },

                // BPF_LDX class
                ebpf::LD_B_REG  if !self.executable.get_sbpf_version().move_memory_instruction_classes() => {
                    self.emit_address_translation(Some(dst), src, insn.off, 1, None);
                },
                ebpf::LD_H_REG  if !self.executable.get_sbpf_version().move_memory_instruction_classes() => {
                    self.emit_address_translation(Some(dst), src, insn.off, 2, None);
                },
                ebpf::LD_W_REG  if !self.executable.get_sbpf_version().move_memory_instruction_classes() => {
                    self.emit_address_translation(Some(dst), src, insn.off, 4, None);
                },
                ebpf::LD_DW_REG if !self.executable.get_sbpf_version().move_memory_instruction_classes() => {
                    self.emit_address_translation(Some(dst), src, insn.off, 8, None);
                },

                // BPF_ST class
                ebpf::ST_B_IMM  if !self.executable.get_sbpf_version().move_memory_instruction_classes() => {
                    self.emit_address_translation(None, dst, insn.off, 1, Some(StoreValue::Constant(insn.imm)));
                },
                ebpf::ST_H_IMM  if !self.executable.get_sbpf_version().move_memory_instruction_classes() => {
                    self.emit_address_translation(None, dst, insn.off, 2, Some(StoreValue::Constant(insn.imm)));
                },
                ebpf::ST_W_IMM  if !self.executable.get_sbpf_version().move_memory_instruction_classes() => {
                    self.emit_address_translation(None, dst, insn.off, 4, Some(StoreValue::Constant(insn.imm)));
                },
                ebpf::ST_DW_IMM if !self.executable.get_sbpf_version().move_memory_instruction_classes() => {
                    self.emit_address_translation(None, dst, insn.off, 8, Some(StoreValue::Constant(insn.imm)));
                },

                // BPF_STX class
                ebpf::ST_B_REG  if !self.executable.get_sbpf_version().move_memory_instruction_classes() => {
                    self.emit_address_translation(None, dst, insn.off, 1, Some(StoreValue::Register(src)));
                },
                ebpf::ST_H_REG  if !self.executable.get_sbpf_version().move_memory_instruction_classes() => {
                    self.emit_address_translation(None, dst, insn.off, 2, Some(StoreValue::Register(src)));
                },
                ebpf::ST_W_REG  if !self.executable.get_sbpf_version().move_memory_instruction_classes() => {
                    self.emit_address_translation(None, dst, insn.off, 4, Some(StoreValue::Register(src)));
                },
                ebpf::ST_DW_REG if !self.executable.get_sbpf_version().move_memory_instruction_classes() => {
                    self.emit_address_translation(None, dst, insn.off, 8, Some(StoreValue::Register(src)));
                },

                // BPF_ALU32_LOAD class
                ebpf::ADD32_IMM  => {
                    self.emit_alu_immediate(OperandSize::S32, AluOp::Add, dst, insn.imm);
                    self.emit_sign_extension_of_result(dst);
                },
                ebpf::ADD32_REG  => {
                    self.emit_ins(ARM64Instruction::add(OperandSize::S32, dst, src, dst));
                    self.emit_sign_extension_of_result(dst);
                },
                ebpf::SUB32_IMM  => {
                    if self.executable.get_sbpf_version().swap_sub_reg_imm_operands() {
                        // dst = imm - dst
                        self.emit_load_immediate(REGISTER_IMMEDIATE, insn.imm);
                        self.emit_ins(ARM64Instruction::sub(OperandSize::S32, REGISTER_IMMEDIATE, dst, dst));
                    } else {
                        self.emit_alu_immediate(OperandSize::S32, AluOp::Sub, dst, insn.imm);
                    }
                    self.emit_sign_extension_of_result(dst);
                },
                ebpf::SUB32_REG  => {
                    self.emit_ins(ARM64Instruction::sub(OperandSize::S32, dst, src, dst));
                    self.emit_sign_extension_of_result(dst);
                },
                ebpf::MUL32_IMM if !self.executable.get_sbpf_version().enable_pqr() => {
                    self.emit_load_immediate(REGISTER_IMMEDIATE, insn.imm);
                    self.emit_ins(ARM64Instruction::mul(OperandSize::S32, dst, REGISTER_IMMEDIATE, dst));
                    self.emit_sign_extension_of_result(dst);
                },
                ebpf::DIV32_IMM | ebpf::MOD32_IMM if !self.executable.get_sbpf_version().enable_pqr() =>
                    self.emit_product_quotient_remainder(
                        OperandSize::S32,
                        (insn.opc & ebpf::BPF_ALU_OP_MASK) == ebpf::BPF_MOD,
                        (insn.opc & ebpf::BPF_ALU_OP_MASK) != ebpf::BPF_MUL,
                        (insn.opc & ebpf::BPF_ALU_OP_MASK) == ebpf::BPF_MUL,
                        dst, dst, Some(insn.imm),
                    ),
                ebpf::LD_1B_REG  if self.executable.get_sbpf_version().move_memory_instruction_classes() => {
                    self.emit_address_translation(Some(dst), src, insn.off, 1, None);
                },
                ebpf::MUL32_REG if !self.executable.get_sbpf_version().enable_pqr() => {
                    self.emit_ins(ARM64Instruction::mul(OperandSize::S32, dst, src, dst));
                    self.emit_sign_extension_of_result(dst);
                },
                ebpf::DIV32_REG | ebpf::MOD32_REG if !self.executable.get_sbpf_version().enable_pqr() =>
                    self.emit_product_quotient_remainder(
                        OperandSize::S32,
                        (insn.opc & ebpf::BPF_ALU_OP_MASK) == ebpf::BPF_MOD,
                        (insn.opc & ebpf::BPF_ALU_OP_MASK) != ebpf::BPF_MUL,
                        (insn.opc & ebpf::BPF_ALU_OP_MASK) == ebpf::BPF_MUL,
                        src, dst, None,
                    ),
                ebpf::LD_2B_REG  if self.executable.get_sbpf_version().move_memory_instruction_classes() => {
                    self.emit_address_translation(Some(dst), src, insn.off, 2, None);
                },
                ebpf::OR32_IMM   => self.emit_alu_immediate(OperandSize::S32, AluOp::Or, dst, insn.imm),
                ebpf::OR32_REG   => self.emit_ins(ARM64Instruction::LogicalRegister(logical_register(OperandSize::S32, 1, dst, src, dst))),
                ebpf::AND32_IMM  => self.emit_alu_immediate(OperandSize::S32, AluOp::And, dst, insn.imm),
                ebpf::AND32_REG  => self.emit_ins(ARM64Instruction::and(OperandSize::S32, dst, src, dst)),
                ebpf::LSH32_IMM  => self.emit_shift_immediate(OperandSize::S32, ShiftOp::Lsl, dst, insn.imm),
                ebpf::LSH32_REG  => self.emit_ins(ARM64Instruction::lsl_reg(OperandSize::S32, dst, src, dst)),
                ebpf::RSH32_IMM  => self.emit_shift_immediate(OperandSize::S32, ShiftOp::Lsr, dst, insn.imm),
                ebpf::RSH32_REG  => self.emit_ins(ARM64Instruction::lsr_reg(OperandSize::S32, dst, src, dst)),
                ebpf::NEG32      if !self.executable.get_sbpf_version().disable_neg() => self.emit_ins(ARM64Instruction::sub(OperandSize::S32, SP_XZR, dst, dst)),
                ebpf::LD_4B_REG  if self.executable.get_sbpf_version().move_memory_instruction_classes() => {
                    self.emit_address_translation(Some(dst), src, insn.off, 4, None);
                },
                ebpf::LD_8B_REG  if self.executable.get_sbpf_version().move_memory_instruction_classes() => {
                    self.emit_address_translation(Some(dst), src, insn.off, 8, None);
                },
                ebpf::XOR32_IMM  => self.emit_alu_immediate(OperandSize::S32, AluOp::Xor, dst, insn.imm),
                ebpf::XOR32_REG  => self.emit_ins(ARM64Instruction::LogicalRegister(logical_register(OperandSize::S32, 2, dst, src, dst))),
                ebpf::MOV32_IMM  => self.emit_load_immediate(dst, insn.imm as u32 as u64 as i64),
                ebpf::MOV32_REG  => {
                    if self.executable.get_sbpf_version().explicit_sign_extension_of_results() {
                        self.emit_ins(ARM64Instruction::sign_extend_to_i64(OperandSize::S32, src, dst));
                    } else {
                        self.emit_ins(ARM64Instruction::mov(OperandSize::S32, src, dst));
                    }
                }
                ebpf::ARSH32_IMM => self.emit_shift_immediate(OperandSize::S32, ShiftOp::Asr, dst, insn.imm),
                ebpf::ARSH32_REG => self.emit_ins(ARM64Instruction::asr_reg(OperandSize::S32, dst, src, dst)),
                ebpf::LE if !self.executable.get_sbpf_version().disable_le() => {
                    match insn.imm {
                        16 => self.emit_ins(ARM64Instruction::zero_extend_to_u64(OperandSize::S16, dst, dst)),
                        32 => self.emit_ins(ARM64Instruction::mov(OperandSize::S32, dst, dst)),
                        64 => {}
                        _ => {
                            return Err(EbpfError::InvalidInstruction);
                        }
                    }
                },
                ebpf::BE         => {
                    match insn.imm {
                        16 => {
                            self.emit_ins(ARM64Instruction::rev(OperandSize::S16, dst, dst));
                            self.emit_ins(ARM64Instruction::zero_extend_to_u64(OperandSize::S16, dst, dst));
                        }
                        32 => self.emit_ins(ARM64Instruction::rev(OperandSize::S32, dst, dst)),
                        64 => self.emit_ins(ARM64Instruction::rev(OperandSize::S64, dst, dst)),
                        _ => {
                            return Err(EbpfError::InvalidInstruction);
                        }
                    }
                },

                // BPF_ALU64_STORE class
                ebpf::ADD64_IMM  => self.emit_alu_immediate(OperandSize::S64, AluOp::Add, dst, insn.imm),
                ebpf::ADD64_REG  => self.emit_ins(ARM64Instruction::add(OperandSize::S64, dst, src, dst)),
                ebpf::SUB64_IMM  => {
                    if self.executable.get_sbpf_version().swap_sub_reg_imm_operands() {
                        // dst = imm - dst
                        self.emit_load_immediate(REGISTER_IMMEDIATE, insn.imm);
                        self.emit_ins(ARM64Instruction::sub(OperandSize::S64, REGISTER_IMMEDIATE, dst, dst));
                    } else {
                        self.emit_alu_immediate(OperandSize::S64, AluOp::Sub, dst, insn.imm);
                    }
                }
                ebpf::SUB64_REG  => self.emit_ins(ARM64Instruction::sub(OperandSize::S64, dst, src, dst)),
                ebpf::MUL64_IMM if !self.executable.get_sbpf_version().enable_pqr() => {
                    self.emit_load_immediate(REGISTER_IMMEDIATE, insn.imm);
                    self.emit_ins(ARM64Instruction::mul(OperandSize::S64, dst, REGISTER_IMMEDIATE, dst));
                },
                ebpf::DIV64_IMM | ebpf::MOD64_IMM if !self.executable.get_sbpf_version().enable_pqr() =>
                    self.emit_product_quotient_remainder(
                        OperandSize::S64,
                        (insn.opc & ebpf::BPF_ALU_OP_MASK) == ebpf::BPF_MOD,
                        (insn.opc & ebpf::BPF_ALU_OP_MASK) != ebpf::BPF_MUL,
                        (insn.opc & ebpf::BPF_ALU_OP_MASK) == ebpf::BPF_MUL,
                        dst, dst, Some(insn.imm),
                    ),
                ebpf::ST_1B_IMM  if self.executable.get_sbpf_version().move_memory_instruction_classes() => {
                    self.emit_address_translation(None, dst, insn.off, 1, Some(StoreValue::Constant(insn.imm)));
                },
                ebpf::ST_2B_IMM  if self.executable.get_sbpf_version().move_memory_instruction_classes() => {
                    self.emit_address_translation(None, dst, insn.off, 2, Some(StoreValue::Constant(insn.imm)));
                },
                ebpf::MUL64_REG if !self.executable.get_sbpf_version().enable_pqr() => {
                    self.emit_ins(ARM64Instruction::mul(OperandSize::S64, dst, src, dst));
                },
                ebpf::DIV64_REG | ebpf::MOD64_REG if !self.executable.get_sbpf_version().enable_pqr() =>
                    self.emit_product_quotient_remainder(
                        OperandSize::S64,
                        (insn.opc & ebpf::BPF_ALU_OP_MASK) == ebpf::BPF_MOD,
                        (insn.opc & ebpf::BPF_ALU_OP_MASK) != ebpf::BPF_MUL,
                        (insn.opc & ebpf::BPF_ALU_OP_MASK) == ebpf::BPF_MUL,
                        src, dst, None,
                    ),
                ebpf::ST_1B_REG  if self.executable.get_sbpf_version().move_memory_instruction_classes() => {
                    self.emit_address_translation(None, dst, insn.off, 1, Some(StoreValue::Register(src)));
                },
                ebpf::ST_2B_REG  if self.executable.get_sbpf_version().move_memory_instruction_classes() => {
                    self.emit_address_translation(None, dst, insn.off, 2, Some(StoreValue::Register(src)));
                },
                ebpf::OR64_IMM   => self.emit_alu_immediate(OperandSize::S64, AluOp::Or, dst, insn.imm),
                ebpf::OR64_REG   => self.emit_ins(ARM64Instruction::LogicalRegister(logical_register(OperandSize::S64, 1, dst, src, dst))),
                ebpf::AND64_IMM  => self.emit_alu_immediate(OperandSize::S64, AluOp::And, dst, insn.imm),
                ebpf::AND64_REG  => self.emit_ins(ARM64Instruction::and(OperandSize::S64, dst, src, dst)),
                ebpf::LSH64_IMM  => self.emit_shift_immediate(OperandSize::S64, ShiftOp::Lsl, dst, insn.imm),
                ebpf::LSH64_REG  => self.emit_ins(ARM64Instruction::lsl_reg(OperandSize::S64, dst, src, dst)),
                ebpf::RSH64_IMM  => self.emit_shift_immediate(OperandSize::S64, ShiftOp::Lsr, dst, insn.imm),
                ebpf::RSH64_REG  => self.emit_ins(ARM64Instruction::lsr_reg(OperandSize::S64, dst, src, dst)),
                ebpf::ST_4B_IMM  if self.executable.get_sbpf_version().move_memory_instruction_classes() => {
                    self.emit_address_translation(None, dst, insn.off, 4, Some(StoreValue::Constant(insn.imm)));
                },
                ebpf::NEG64      if !self.executable.get_sbpf_version().disable_neg() => self.emit_ins(ARM64Instruction::sub(OperandSize::S64, SP_XZR, dst, dst)),
                ebpf::ST_4B_REG  if self.executable.get_sbpf_version().move_memory_instruction_classes() => {
                    self.emit_address_translation(None, dst, insn.off, 4, Some(StoreValue::Register(src)));
                },
                ebpf::ST_8B_IMM  if self.executable.get_sbpf_version().move_memory_instruction_classes() => {
                    self.emit_address_translation(None, dst, insn.off, 8, Some(StoreValue::Constant(insn.imm)));
                },
                ebpf::ST_8B_REG  if self.executable.get_sbpf_version().move_memory_instruction_classes() => {
                    self.emit_address_translation(None, dst, insn.off, 8, Some(StoreValue::Register(src)));
                },
                ebpf::XOR64_IMM  => self.emit_alu_immediate(OperandSize::S64, AluOp::Xor, dst, insn.imm),
                ebpf::XOR64_REG  => self.emit_ins(ARM64Instruction::LogicalRegister(logical_register(OperandSize::S64, 2, dst, src, dst))),
                ebpf::MOV64_IMM  => self.emit_load_immediate(dst, insn.imm),
                ebpf::MOV64_REG  => self.emit_ins(ARM64Instruction::mov(OperandSize::S64, src, dst)),
                ebpf::ARSH64_IMM => self.emit_shift_immediate(OperandSize::S64, ShiftOp::Asr, dst, insn.imm),
                ebpf::ARSH64_REG => self.emit_ins(ARM64Instruction::asr_reg(OperandSize::S64, dst, src, dst)),
                ebpf::HOR64_IMM if self.executable.get_sbpf_version().disable_lddw() => {
                    self.emit_alu_immediate(OperandSize::S64, AluOp::Or, dst, (insn.imm as u64).wrapping_shl(32) as i64);
                }

                // BPF_PQR class
                ebpf::LMUL32_IMM | ebpf::LMUL64_IMM | ebpf::UHMUL64_IMM | ebpf::SHMUL64_IMM |
                ebpf::UDIV32_IMM | ebpf::UDIV64_IMM | ebpf::UREM32_IMM | ebpf::UREM64_IMM |
                ebpf::SDIV32_IMM | ebpf::SDIV64_IMM | ebpf::SREM32_IMM | ebpf::SREM64_IMM
                if self.executable.get_sbpf_version().enable_pqr() => {
                    let signed = insn.opc & (1 << 7) != 0;
                    let mut imm = insn.imm;
                    if !signed {
                        imm &= u32::MAX as i64;
                    }
                    self.emit_product_quotient_remainder(
                        if insn.opc & (1 << 4) != 0 { OperandSize::S64 } else { OperandSize::S32 },
                        insn.opc & (1 << 5) != 0,
                        insn.opc & (1 << 6) != 0,
                        signed,
                        dst, dst, Some(imm),
                    )
                }
                ebpf::LMUL32_REG | ebpf::LMUL64_REG | ebpf::UHMUL64_REG | ebpf::SHMUL64_REG |
                ebpf::UDIV32_REG | ebpf::UDIV64_REG | ebpf::UREM32_REG | ebpf::UREM64_REG |
                ebpf::SDIV32_REG | ebpf::SDIV64_REG | ebpf::SREM32_REG | ebpf::SREM64_REG
                if self.executable.get_sbpf_version().enable_pqr() =>
                    self.emit_product_quotient_remainder(
                        if insn.opc & (1 << 4) != 0 { OperandSize::S64 } else { OperandSize::S32 },
                        insn.opc & (1 << 5) != 0,
                        insn.opc & (1 << 6) != 0,
                        insn.opc & (1 << 7) != 0,
                        src, dst, None,
                    ),

                // BPF_JMP32 class
                ebpf::JEQ32_IMM   if self.executable.get_sbpf_version().enable_jmp32() => self.emit_conditional_branch_imm(OperandSize::S32, Condition::EQ, false, insn.imm, dst, target_pc),
                ebpf::JEQ32_REG   if self.executable.get_sbpf_version().enable_jmp32() => self.emit_conditional_branch_reg(OperandSize::S32, Condition::EQ, false, src, dst, target_pc),
                ebpf::JGT32_IMM   if self.executable.get_sbpf_version().enable_jmp32() => self.emit_conditional_branch_imm(OperandSize::S32, Condition::HI, false, insn.imm, dst, target_pc),
                ebpf::JGT32_REG   if self.executable.get_sbpf_version().enable_jmp32() => self.emit_conditional_branch_reg(OperandSize::S32, Condition::HI, false, src, dst, target_pc),
                ebpf::JGE32_IMM   if self.executable.get_sbpf_version().enable_jmp32() => self.emit_conditional_branch_imm(OperandSize::S32, Condition::HS, false, insn.imm, dst, target_pc),
                ebpf::JGE32_REG   if self.executable.get_sbpf_version().enable_jmp32() => self.emit_conditional_branch_reg(OperandSize::S32, Condition::HS, false, src, dst, target_pc),
                ebpf::JLT32_IMM   if self.executable.get_sbpf_version().enable_jmp32() => self.emit_conditional_branch_imm(OperandSize::S32, Condition::LO, false, insn.imm, dst, target_pc),
                ebpf::JLT32_REG   if self.executable.get_sbpf_version().enable_jmp32() => self.emit_conditional_branch_reg(OperandSize::S32, Condition::LO, false, src, dst, target_pc),
                ebpf::JLE32_IMM   if self.executable.get_sbpf_version().enable_jmp32() => self.emit_conditional_branch_imm(OperandSize::S32, Condition::LS, false, insn.imm, dst, target_pc),
                ebpf::JLE32_REG   if self.executable.get_sbpf_version().enable_jmp32() => self.emit_conditional_branch_reg(OperandSize::S32, Condition::LS, false, src, dst, target_pc),
                ebpf::JSET32_IMM  if self.executable.get_sbpf_version().enable_jmp32() => self.emit_conditional_branch_imm(OperandSize::S32, Condition::NE, true, insn.imm, dst, target_pc),
                ebpf::JSET32_REG  if self.executable.get_sbpf_version().enable_jmp32() => self.emit_conditional_branch_reg(OperandSize::S32, Condition::NE, true, src, dst, target_pc),
                ebpf::JNE32_IMM   if self.executable.get_sbpf_version().enable_jmp32() => self.emit_conditional_branch_imm(OperandSize::S32, Condition::NE, false, insn.imm, dst, target_pc),
                ebpf::JNE32_REG   if self.executable.get_sbpf_version().enable_jmp32() => self.emit_conditional_branch_reg(OperandSize::S32, Condition::NE, false, src, dst, target_pc),
                ebpf::JSGT32_IMM  if self.executable.get_sbpf_version().enable_jmp32() => self.emit_conditional_branch_imm(OperandSize::S32, Condition::GT, false, insn.imm, dst, target_pc),
                ebpf::JSGT32_REG  if self.executable.get_sbpf_version().enable_jmp32() => self.emit_conditional_branch_reg(OperandSize::S32, Condition::GT, false, src, dst, target_pc),
                ebpf::JSGE32_IMM  if self.executable.get_sbpf_version().enable_jmp32() => self.emit_conditional_branch_imm(OperandSize::S32, Condition::GE, false, insn.imm, dst, target_pc),
                ebpf::JSGE32_REG  if self.executable.get_sbpf_version().enable_jmp32() => self.emit_conditional_branch_reg(OperandSize::S32, Condition::GE, false, src, dst, target_pc),
                ebpf::JSLT32_IMM  if self.executable.get_sbpf_version().enable_jmp32() => self.emit_conditional_branch_imm(OperandSize::S32, Condition::LT, false, insn.imm, dst, target_pc),
                ebpf::JSLT32_REG  if self.executable.get_sbpf_version().enable_jmp32() => self.emit_conditional_branch_reg(OperandSize::S32, Condition::LT, false, src, dst, target_pc),
                ebpf::JSLE32_IMM  if self.executable.get_sbpf_version().enable_jmp32() => self.emit_conditional_branch_imm(OperandSize::S32, Condition::LE, false, insn.imm, dst, target_pc),
                ebpf::JSLE32_REG  if self.executable.get_sbpf_version().enable_jmp32() => self.emit_conditional_branch_reg(OperandSize::S32, Condition::LE, false, src, dst, target_pc),

                // BPF_JMP64 class
                ebpf::JA         => {
                    self.emit_validate_and_profile_instruction_count(target_pc);
                    self.emit_jump_to_target_pc(target_pc, false);
                },
                ebpf::JEQ64_IMM    => self.emit_conditional_branch_imm(OperandSize::S64, Condition::EQ, false, insn.imm, dst, target_pc),
                ebpf::JEQ64_REG    => self.emit_conditional_branch_reg(OperandSize::S64, Condition::EQ, false, src, dst, target_pc),
                ebpf::JGT64_IMM    => self.emit_conditional_branch_imm(OperandSize::S64, Condition::HI, false, insn.imm, dst, target_pc),
                ebpf::JGT64_REG    => self.emit_conditional_branch_reg(OperandSize::S64, Condition::HI, false, src, dst, target_pc),
                ebpf::JGE64_IMM    => self.emit_conditional_branch_imm(OperandSize::S64, Condition::HS, false, insn.imm, dst, target_pc),
                ebpf::JGE64_REG    => self.emit_conditional_branch_reg(OperandSize::S64, Condition::HS, false, src, dst, target_pc),
                ebpf::JLT64_IMM    => self.emit_conditional_branch_imm(OperandSize::S64, Condition::LO, false, insn.imm, dst, target_pc),
                ebpf::JLT64_REG    => self.emit_conditional_branch_reg(OperandSize::S64, Condition::LO, false, src, dst, target_pc),
                ebpf::JLE64_IMM    => self.emit_conditional_branch_imm(OperandSize::S64, Condition::LS, false, insn.imm, dst, target_pc),
                ebpf::JLE64_REG    => self.emit_conditional_branch_reg(OperandSize::S64, Condition::LS, false, src, dst, target_pc),
                ebpf::JSET64_IMM   => self.emit_conditional_branch_imm(OperandSize::S64, Condition::NE, true, insn.imm, dst, target_pc),
                ebpf::JSET64_REG   => self.emit_conditional_branch_reg(OperandSize::S64, Condition::NE, true, src, dst, target_pc),
                ebpf::JNE64_IMM    => self.emit_conditional_branch_imm(OperandSize::S64, Condition::NE, false, insn.imm, dst, target_pc),
                ebpf::JNE64_REG    => self.emit_conditional_branch_reg(OperandSize::S64, Condition::NE, false, src, dst, target_pc),
                ebpf::JSGT64_IMM   => self.emit_conditional_branch_imm(OperandSize::S64, Condition::GT, false, insn.imm, dst, target_pc),
                ebpf::JSGT64_REG   => self.emit_conditional_branch_reg(OperandSize::S64, Condition::GT, false, src, dst, target_pc),
                ebpf::JSGE64_IMM   => self.emit_conditional_branch_imm(OperandSize::S64, Condition::GE, false, insn.imm, dst, target_pc),
                ebpf::JSGE64_REG   => self.emit_conditional_branch_reg(OperandSize::S64, Condition::GE, false, src, dst, target_pc),
                ebpf::JSLT64_IMM   => self.emit_conditional_branch_imm(OperandSize::S64, Condition::LT, false, insn.imm, dst, target_pc),
                ebpf::JSLT64_REG   => self.emit_conditional_branch_reg(OperandSize::S64, Condition::LT, false, src, dst, target_pc),
                ebpf::JSLE64_IMM   => self.emit_conditional_branch_imm(OperandSize::S64, Condition::LE, false, insn.imm, dst, target_pc),
                ebpf::JSLE64_REG   => self.emit_conditional_branch_reg(OperandSize::S64, Condition::LE, false, src, dst, target_pc),
                ebpf::CALL_IMM     => {
                    let mut resolved = false;
                    // External syscall
                    if !self.executable.get_sbpf_version().static_syscalls() || insn.src == 0 {
                        if let Some((_, (_, callback))) = self.executable.get_loader().get_function_registry().lookup_by_key(insn.imm as u32) {
                            callback(&mut self);
                            resolved = true;
                        }
                    }
                    // Internal call
                    if self.executable.get_sbpf_version().static_syscalls() {
                        let target_pc = (self.pc as i64).saturating_add(insn.imm).saturating_add(1);
                        if ebpf::is_pc_in_program(self.program, target_pc as usize) && insn.src == 1 {
                            self.emit_internal_call(CallTarget::Constant(target_pc));
                            resolved = true;
                        }
                    } else if let Some((_function_name, target_pc)) =
                        self.executable
                            .get_function_registry()
                            .lookup_by_key(insn.imm as u32) {
                        self.emit_internal_call(CallTarget::Constant(target_pc as i64));
                        resolved = true;
                    }
                    if !resolved {
                        self.emit_load_immediate(REGISTER_SCRATCH, self.pc as i64);
                        self.emit_jump_to_anchor(ANCHOR_CALL_UNSUPPORTED_INSTRUCTION);
                    }
                },
                ebpf::CALL_REG  => {
                    let target_pc = if self.executable.get_sbpf_version().callx_uses_src_reg() {
                        src
                    } else if self.executable.get_sbpf_version().callx_uses_dst_reg() {
                        dst
                    } else {
                        REGISTER_MAP[insn.imm as usize]
                    };
                    self.emit_internal_call(CallTarget::Register(target_pc));
                },
                ebpf::EXIT      => {
                    self.emit_validate_and_profile_instruction_count(0);

                    // If env.call_depth == 0, we've reached the exit instruction of the entry point
                    self.emit_ldr_vm(REGISTER_TEMP, RuntimeEnvironmentSlot::CallDepth, 0);
                    // we're done
                    self.emit_cbz_to_anchor(OperandSize::S32, REGISTER_TEMP, false, ANCHOR_EXIT);

                    // else decrement and update env.call_depth
                    self.emit_ins(ARM64Instruction::sub_imm(OperandSize::S64, REGISTER_TEMP, 1, REGISTER_TEMP));
                    self.emit_str_vm(REGISTER_TEMP, RuntimeEnvironmentSlot::CallDepth, 0);

                    // and return
                    self.emit_ins(ARM64Instruction::load(OperandSize::S64, SP_XZR, ARM64MemoryOperand::OffsetScaled(0), LR));
                    self.emit_ins(ARM64Instruction::ret());
                },

                _               => return Err(EbpfError::UnsupportedInstruction),
            }

            self.pc += 1;
        }

        // Bumper in case there was no final exit
        if self.offset_in_text_section + MAX_MACHINE_CODE_LENGTH_PER_INSTRUCTION * 2 >= self.result.text_section.len() {
            return Err(EbpfError::ExhaustedTextSegment(self.pc));
        }
        self.emit_validate_and_profile_instruction_count(self.pc + 1);
        self.emit_load_immediate(REGISTER_SCRATCH, self.pc as i64); // Save pc
        self.emit_set_exception_kind(EbpfError::ExecutionOverrun);
        self.emit_jump_to_anchor(ANCHOR_THROW_EXCEPTION);

        self.resolve_jumps();
        if self.branch_out_of_range {
            return Err(EbpfError::ExhaustedTextSegment(self.pc));
        }
        self.result.seal(self.offset_in_text_section)?;
        Ok(self.result)
    }

    fn slot_in_vm(&self, slot: RuntimeEnvironmentSlot) -> usize {
        slot as usize
    }

    pub(crate) fn emit<T>(&mut self, data: T) {
        unsafe {
            let ptr = self.result.text_section.as_ptr().add(self.offset_in_text_section);
            #[allow(clippy::cast_ptr_alignment)]
            ptr::write_unaligned(ptr as *mut T, data as T);
        }
        self.offset_in_text_section += mem::size_of::<T>();
    }

    #[inline(always)]
    fn emit_ins(&mut self, instruction: ARM64Instruction) {
        instruction.emit(self);
    }

    /// Loads a 64 bit immediate into a register with as few MOVZ / MOVN / MOVK as possible
    fn emit_load_immediate(&mut self, destination: u8, value: i64) {
        let value = value as u64;
        let halfwords = [
            value as u16,
            (value >> 16) as u16,
            (value >> 32) as u16,
            (value >> 48) as u16,
        ];
        let zero_halfwords = halfwords.iter().filter(|h| **h == 0).count();
        let ones_halfwords = halfwords.iter().filter(|h| **h == 0xffff).count();
        if ones_halfwords > zero_halfwords {
            let first = halfwords.iter().position(|h| *h != 0xffff).unwrap_or(0);
            self.emit_ins(ARM64Instruction::movn(destination, first as u8, !halfwords[first]));
            for (i, halfword) in halfwords.iter().enumerate().skip(first + 1) {
                if *halfword != 0xffff {
                    self.emit_ins(ARM64Instruction::movk(destination, i as u8, *halfword));
                }
            }
        } else {
            let first = halfwords.iter().position(|h| *h != 0).unwrap_or(0);
            self.emit_ins(ARM64Instruction::movz(destination, first as u8, halfwords[first]));
            for (i, halfword) in halfwords.iter().enumerate().skip(first + 1) {
                if *halfword != 0 {
                    self.emit_ins(ARM64Instruction::movk(destination, i as u8, *halfword));
                }
            }
        }
    }

    /// destination = source + value (64 bit), uses REGISTER_TEMP if value does not fit
    fn emit_add_immediate(&mut self, source: u8, value: i64, destination: u8) {
        if (0..4096).contains(&value) {
            if value != 0 || source != destination {
                self.emit_ins(ARM64Instruction::add_imm(OperandSize::S64, source, value as u16, destination));
            }
        } else if (-4095..0).contains(&value) {
            self.emit_ins(ARM64Instruction::sub_imm(OperandSize::S64, source, (-value) as u16, destination));
        } else {
            self.emit_load_immediate(REGISTER_TEMP, value);
            self.emit_ins(ARM64Instruction::add(OperandSize::S64, source, REGISTER_TEMP, destination));
        }
    }

    fn emit_alu_immediate(&mut self, size: OperandSize, op: AluOp, destination: u8, immediate: i64) {
        // 32 bit operations only look at the lower half of the immediate
        let immediate = if let OperandSize::S32 = size { immediate as i32 as i64 } else { immediate };
        match op {
            AluOp::Add | AluOp::Sub => {
                let value = if op == AluOp::Sub { immediate.wrapping_neg() } else { immediate };
                if (0..4096).contains(&value) {
                    self.emit_ins(ARM64Instruction::add_imm(size, destination, value as u16, destination));
                    return;
                } else if (-4095..0).contains(&value) {
                    self.emit_ins(ARM64Instruction::sub_imm(size, destination, (-value) as u16, destination));
                    return;
                }
                self.emit_load_immediate(REGISTER_IMMEDIATE, immediate);
                if op == AluOp::Add {
                    self.emit_ins(ARM64Instruction::add(size, destination, REGISTER_IMMEDIATE, destination));
                } else {
                    self.emit_ins(ARM64Instruction::sub(size, destination, REGISTER_IMMEDIATE, destination));
                }
            }
            AluOp::Or | AluOp::And | AluOp::Xor => {
                self.emit_load_immediate(REGISTER_IMMEDIATE, immediate);
                let opcode = match op {
                    AluOp::And => 0,
                    AluOp::Or => 1,
                    _ => 2,
                };
                self.emit_ins(ARM64Instruction::LogicalRegister(logical_register(size, opcode, destination, REGISTER_IMMEDIATE, destination)));
            }
        }
    }

    fn emit_shift_immediate(&mut self, size: OperandSize, op: ShiftOp, destination: u8, immediate: i64) {
        let bits: u8 = if let OperandSize::S64 = size { 64 } else { 32 };
        let shift = (immediate as u8) & (bits - 1);
        match op {
            ShiftOp::Lsl if shift != 0 => {
                self.emit_ins(ARM64Instruction::ubfm(size, destination, bits - shift, bits - 1 - shift, destination));
            }
            // LSL #0 is LSR #0, which is a zero extending move for 32 bit
            ShiftOp::Lsl | ShiftOp::Lsr => {
                self.emit_ins(ARM64Instruction::ubfm(size, destination, shift, bits - 1, destination));
            }
            ShiftOp::Asr => {
                self.emit_ins(ARM64Instruction::sbfm(size, destination, shift, bits - 1, destination));
            }
        }
    }

    fn emit_sign_extension_of_result(&mut self, destination: u8) {
        if !self.executable.get_sbpf_version().explicit_sign_extension_of_results() {
            self.emit_ins(ARM64Instruction::sign_extend_to_i64(OperandSize::S32, destination, destination)); // sign extend i32 to i64
        }
    }

    fn emit_ldr_vm(&mut self, destination: u8, slot: RuntimeEnvironmentSlot, offset: usize) {
        let offset = self.slot_in_vm(slot) + offset;
        debug_assert!(offset % 8 == 0 && offset < 8 * 4096);
        self.emit_ins(ARM64Instruction::load(OperandSize::S64, REGISTER_PTR_TO_VM, ARM64MemoryOperand::OffsetScaled((offset / 8) as u16), destination));
    }

    fn emit_str_vm(&mut self, source: u8, slot: RuntimeEnvironmentSlot, offset: usize) {
        let offset = self.slot_in_vm(slot) + offset;
        debug_assert!(offset % 8 == 0 && offset < 8 * 4096);
        self.emit_ins(ARM64Instruction::store(OperandSize::S64, source, REGISTER_PTR_TO_VM, ARM64MemoryOperand::OffsetScaled((offset / 8) as u16)));
    }

    fn emit_validate_instruction_count(&mut self, pc: Option<usize>) {
        if !self.config.enable_instruction_meter {
            return;
        }
        // Update `MACHINE_CODE_PER_INSTRUCTION_METER_CHECKPOINT` if you change the code generation here
        if let Some(pc) = pc {
            self.last_instruction_meter_validation_pc = pc;
            self.emit_load_immediate(REGISTER_SCRATCH, pc as i64);
        }
        // If instruction_meter <= pc, throw ExceededMaxInstructions
        self.emit_ins(ARM64Instruction::cmp(OperandSize::S64, REGISTER_SCRATCH, REGISTER_INSTRUCTION_METER));
        self.emit_conditional_jump_to_anchor(Condition::LS, ANCHOR_THROW_EXCEEDED_MAX_INSTRUCTIONS);
    }

    fn emit_profile_instruction_count(&mut self, target_pc: usize) {
        if !self.config.enable_instruction_meter {
            return;
        }
        self.emit_add_immediate(REGISTER_INSTRUCTION_METER, target_pc as i64 - self.pc as i64 - 1, REGISTER_INSTRUCTION_METER); // instruction_meter += target_pc - (self.pc + 1);
    }

    fn emit_undo_profile_instruction_count(&mut self, target_pc: usize) {
        if self.config.enable_instruction_meter {
            self.emit_add_immediate(REGISTER_INSTRUCTION_METER, self.pc as i64 + 1 - target_pc as i64, REGISTER_INSTRUCTION_METER); // instruction_meter += (self.pc + 1) - target_pc;
        }
    }

    fn emit_validate_and_profile_instruction_count(&mut self, target_pc: usize) {
        self.emit_validate_instruction_count(Some(self.pc));
        self.emit_profile_instruction_count(target_pc);
    }

    fn emit_internal_call(&mut self, dst: CallTarget) {
        if let CallTarget::Register(reg) = dst {
            // Read the target before the prologue changes the frame pointer
            self.emit_ins(ARM64Instruction::mov(OperandSize::S64, reg, REGISTER_OTHER));
        }
        // Store PC in case the bounds check fails
        self.emit_load_immediate(REGISTER_SCRATCH, self.pc as i64);
        self.last_instruction_meter_validation_pc = self.pc;
        self.emit_call_anchor(ANCHOR_INTERNAL_FUNCTION_CALL_PROLOGUE);

        match dst {
            CallTarget::Register(_reg) => {
                // Returns the host target address in REGISTER_TEMP
                self.emit_call_anchor(ANCHOR_INTERNAL_FUNCTION_CALL_REG);
                // Push the return address and call
                self.emit_ins(ARM64Instruction::adr(REGISTER_TEMP2, 12));
                self.emit_ins(ARM64Instruction::store(OperandSize::S64, REGISTER_TEMP2, SP_XZR, ARM64MemoryOperand::OffsetScaled(0)));
                self.emit_ins(ARM64Instruction::blr(REGISTER_TEMP));
            },
            CallTarget::Constant(target_pc) => {
                self.emit_profile_instruction_count(target_pc as usize);
                // Push the return address and call
                self.emit_ins(ARM64Instruction::adr(REGISTER_TEMP2, 12));
                self.emit_ins(ARM64Instruction::store(OperandSize::S64, REGISTER_TEMP2, SP_XZR, ARM64MemoryOperand::OffsetScaled(0)));
                self.emit_jump_to_target_pc(target_pc as usize, true);
            },
        }

        self.emit_undo_profile_instruction_count(0);

        // Restore the previous frame pointer and the scratch registers
        self.emit_ins(ARM64Instruction::load(OperandSize::S64, SP_XZR, ARM64MemoryOperand::OffsetScaled(1), REGISTER_MAP[FRAME_PTR_REG]));
        for i in (0..SCRATCH_REGS).step_by(2) {
            self.emit_ins(ARM64Instruction::ldp(
                REGISTER_MAP[FIRST_SCRATCH_REG + i],
                REGISTER_MAP[FIRST_SCRATCH_REG + i + 1],
                SP_XZR,
                ARM64MemoryOperand::Offset(16 + 8 * i as i16),
            ));
        }
        self.emit_ins(ARM64Instruction::add_imm(OperandSize::S64, SP_XZR, CALL_FRAME_SIZE, SP_XZR));
    }

    /// Emits a syscall handler invocation
    pub fn emit_external_call(&mut self, function: BuiltinFunction<C>) {
        self.emit_validate_and_profile_instruction_count(0);
        self.emit_load_immediate(REGISTER_SCRATCH, function as usize as i64);
        self.emit_call_anchor(ANCHOR_EXTERNAL_FUNCTION_CALL);
        self.emit_undo_profile_instruction_count(0);
    }

    fn emit_address_translation(&mut self, dst: Option<u8>, base: u8, offset: i16, len: u64, value: Option<StoreValue>) {
        debug_assert_ne!(dst.is_some(), value.is_some());

        match value {
            Some(StoreValue::Register(reg)) => {
                self.emit_ins(ARM64Instruction::mov(OperandSize::S64, reg, REGISTER_OTHER));
            }
            Some(StoreValue::Constant(constant)) => {
                self.emit_load_immediate(REGISTER_OTHER, constant);
            }
            None => {}
        }

        // REGISTER_SCRATCH = vm_addr
        let offset = offset as i64;
        if (0..4096).contains(&offset) {
            self.emit_ins(ARM64Instruction::add_imm(OperandSize::S64, base, offset as u16, REGISTER_SCRATCH));
        } else if (-4095..0).contains(&offset) {
            self.emit_ins(ARM64Instruction::sub_imm(OperandSize::S64, base, (-offset) as u16, REGISTER_SCRATCH));
        } else {
            self.emit_load_immediate(REGISTER_SCRATCH, offset);
            self.emit_ins(ARM64Instruction::add(OperandSize::S64, REGISTER_SCRATCH, base, REGISTER_SCRATCH));
        }

        let size = match len {
            1 => OperandSize::S8,
            2 => OperandSize::S16,
            4 => OperandSize::S32,
            8 => OperandSize::S64,
            _ => unreachable!(),
        };
        if self.config.enable_address_translation {
            let anchor_base = if value.is_some() { 4 } else { 0 };
            let anchor = ANCHOR_TRANSLATE_MEMORY_ADDRESS + anchor_base + len.trailing_zeros() as usize;
            self.emit_load_immediate(REGISTER_TRANSLATION_PC, self.pc as i64);
            self.emit_call_anchor(anchor);
            if let Some(dst) = dst {
                self.emit_ins(ARM64Instruction::mov(OperandSize::S64, REGISTER_SCRATCH, dst));
            }
        } else if let Some(dst) = dst {
            self.emit_ins(ARM64Instruction::load(size, REGISTER_SCRATCH, ARM64MemoryOperand::OffsetScaled(0), dst));
        } else {
            self.emit_ins(ARM64Instruction::store(size, REGISTER_OTHER, REGISTER_SCRATCH, ARM64MemoryOperand::OffsetScaled(0)));
        }
    }

    fn emit_conditional_branch_reg(&mut self, size: OperandSize, condition: Condition, bitwise: bool, first_operand: u8, second_operand: u8, target_pc: usize) {
        self.emit_validate_and_profile_instruction_count(target_pc);
        if bitwise { // Logical
            self.emit_ins(ARM64Instruction::tst(size, first_operand, second_operand));
        } else { // Arithmetic
            self.emit_ins(ARM64Instruction::cmp(size, first_operand, second_operand));
        }
        self.emit_conditional_jump_to_target_pc(condition, target_pc);
        self.emit_undo_profile_instruction_count(target_pc);
    }

    fn emit_conditional_branch_imm(&mut self, size: OperandSize, condition: Condition, bitwise: bool, immediate: i64, second_operand: u8, target_pc: usize) {
        self.emit_validate_and_profile_instruction_count(target_pc);
        if !bitwise && (0..4096).contains(&immediate) {
            self.emit_ins(ARM64Instruction::cmp_imm(size, second_operand, immediate as u16));
        } else {
            self.emit_load_immediate(REGISTER_IMMEDIATE, immediate);
            if bitwise { // Logical
                self.emit_ins(ARM64Instruction::tst(size, REGISTER_IMMEDIATE, second_operand));
            } else { // Arithmetic
                self.emit_ins(ARM64Instruction::cmp(size, REGISTER_IMMEDIATE, second_operand));
            }
        }
        self.emit_conditional_jump_to_target_pc(condition, target_pc);
        self.emit_undo_profile_instruction_count(target_pc);
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_product_quotient_remainder(
        &mut self,
        size: OperandSize,
        alt_dst: bool,
        division: bool,
        signed: bool,
        src: u8,
        dst: u8,
        imm: Option<i64>,
    ) {
        //         LMUL UHMUL SHMUL UDIV SDIV UREM SREM
        // alt_dst 0    1     1     0    0    1    1
        // signed  ?    0     1     0    1    0    1

        if division {
            // Prevent division by zero
            if imm.is_none() {
                self.emit_load_immediate(REGISTER_SCRATCH, self.pc as i64); // Save pc
                self.emit_cbz_to_anchor(size, src, false, ANCHOR_DIV_BY_ZERO); // src == 0
            }

            // Signed division overflows with MIN / -1.
            // If we have an immediate and it's not -1, we can skip the following check.
            if signed && imm.unwrap_or(-1) == -1 {
                self.emit_load_immediate(REGISTER_SCRATCH, self.pc as i64);
                self.emit_load_immediate(REGISTER_TEMP2, if let OperandSize::S64 = size { i64::MIN } else { i32::MIN as i64 });
                self.emit_ins(ARM64Instruction::cmp(size, REGISTER_TEMP2, dst)); // dst == MIN
                if imm.is_none() {
                    // The exception case is: dst == MIN && src == -1
                    let skip = self.emit_placeholder();
                    self.emit_ins(ARM64Instruction::cmn_imm(size, src, 1)); // src == -1
                    self.emit_conditional_jump_to_anchor(Condition::EQ, ANCHOR_DIV_OVERFLOW);
                    self.patch_conditional_branch(skip, Condition::NE);
                } else {
                    // MIN / -1, raise EbpfError::DivideOverflow
                    self.emit_conditional_jump_to_anchor(Condition::EQ, ANCHOR_DIV_OVERFLOW);
                }
            }
        }

        let operand = if let Some(imm) = imm {
            self.emit_load_immediate(REGISTER_IMMEDIATE, imm);
            REGISTER_IMMEDIATE
        } else {
            src
        };

        match (division, alt_dst, signed) {
            (false, false, _) => self.emit_ins(ARM64Instruction::mul(size, dst, operand, dst)),
            (false, true, false) => self.emit_ins(ARM64Instruction::umulh(dst, operand, dst)),
            (false, true, true) => self.emit_ins(ARM64Instruction::smulh(dst, operand, dst)),
            (true, false, false) => self.emit_ins(ARM64Instruction::udiv(size, dst, operand, dst)),
            (true, false, true) => self.emit_ins(ARM64Instruction::sdiv(size, dst, operand, dst)),
            (true, true, signed) => {
                if signed {
                    self.emit_ins(ARM64Instruction::sdiv(size, dst, operand, REGISTER_TEMP));
                } else {
                    self.emit_ins(ARM64Instruction::udiv(size, dst, operand, REGISTER_TEMP));
                }
                // dst = dst - quotient * operand
                self.emit_ins(ARM64Instruction::msub(size, REGISTER_TEMP, operand, dst, dst));
            }
        }

        if let OperandSize::S32 = size {
            if signed && !self.executable.get_sbpf_version().explicit_sign_extension_of_results() {
                self.emit_ins(ARM64Instruction::sign_extend_to_i64(OperandSize::S32, dst, dst)); // sign extend i32 to i64
            }
        }
    }

    fn emit_set_exception_kind(&mut self, err: EbpfError) {
        let err_kind = unsafe { *std::ptr::addr_of!(err).cast::<u64>() };
        let err_discriminant = ProgramResult::Err(err).discriminant();
        self.emit_load_immediate(REGISTER_TEMP, err_discriminant as i64);
        self.emit_str_vm(REGISTER_TEMP, RuntimeEnvironmentSlot::ProgramResult, 0); // result.discriminant = err_discriminant;
        self.emit_load_immediate(REGISTER_TEMP, err_kind as i64);
        self.emit_str_vm(REGISTER_TEMP, RuntimeEnvironmentSlot::ProgramResult, std::mem::size_of::<u64>()); // err.kind = err_kind;
    }

    /// Sets the flags: EQ if the ProgramResult is Ok, NE if it is an Err
    fn emit_result_is_err(&mut self) {
        let ok = ProgramResult::Ok(0);
        let ok_discriminant = ok.discriminant();
        self.emit_ldr_vm(REGISTER_TEMP, RuntimeEnvironmentSlot::ProgramResult, 0);
        self.emit_load_immediate(REGISTER_TEMP2, ok_discriminant as i64);
        self.emit_ins(ARM64Instruction::cmp(OperandSize::S64, REGISTER_TEMP2, REGISTER_TEMP));
    }

    fn emit_subroutines(&mut self) {
        // Trampoline from Rust (extern "C") into the JIT code, must be at offset 0.
        // X0: EbpfVm, X1: registers, X2: host entrypoint, X3: instruction meter
        debug_assert_eq!(self.offset_in_text_section, 0);
        self.emit_ins(ARM64Instruction::stp(X29, LR, SP_XZR, ARM64MemoryOperand::OffsetPreIndex(-96)));
        self.emit_ins(ARM64Instruction::mov_sp(SP_XZR, X29));
        for (i, pair) in [[X19, X20], [X21, X22], [X23, X24], [X25, X26], [X27, X28]].iter().enumerate() {
            self.emit_ins(ARM64Instruction::stp(pair[0], pair[1], SP_XZR, ARM64MemoryOperand::Offset(16 + 16 * i as i16)));
        }
        self.emit_ins(ARM64Instruction::mov(OperandSize::S64, X0, REGISTER_PTR_TO_VM));
        self.emit_ins(ARM64Instruction::mov(OperandSize::S64, X3, REGISTER_INSTRUCTION_METER));
        self.emit_ins(ARM64Instruction::mov_sp(SP_XZR, REGISTER_TEMP));
        self.emit_str_vm(REGISTER_TEMP, RuntimeEnvironmentSlot::HostStackPointer, 0);
        for i in (0..10).step_by(2) {
            self.emit_ins(ARM64Instruction::ldp(REGISTER_MAP[i], REGISTER_MAP[i + 1], X1, ARM64MemoryOperand::Offset(8 * i as i16)));
        }
        self.emit_ins(ARM64Instruction::load(OperandSize::S64, X1, ARM64MemoryOperand::OffsetScaled(10), REGISTER_MAP[10]));
        self.emit_ins(ARM64Instruction::br(X2));

        // Routine for instruction tracing
        if self.config.enable_register_tracing {
            self.set_anchor(ANCHOR_TRACE);
            // Save registers on stack: [r0 ..= r10, pc] then the caller saved registers in use
            self.emit_ins(ARM64Instruction::sub_imm(OperandSize::S64, SP_XZR, 128, SP_XZR));
            for i in (0..10).step_by(2) {
                self.emit_ins(ARM64Instruction::stp(REGISTER_MAP[i], REGISTER_MAP[i + 1], SP_XZR, ARM64MemoryOperand::Offset(8 * i as i16)));
            }
            self.emit_ins(ARM64Instruction::stp(REGISTER_MAP[10], REGISTER_SCRATCH, SP_XZR, ARM64MemoryOperand::Offset(80)));
            self.emit_ins(ARM64Instruction::stp(REGISTER_PTR_TO_VM, REGISTER_INSTRUCTION_METER, SP_XZR, ARM64MemoryOperand::Offset(96)));
            self.emit_ins(ARM64Instruction::store(OperandSize::S64, LR, SP_XZR, ARM64MemoryOperand::OffsetScaled(14)));
            self.emit_add_immediate(REGISTER_PTR_TO_VM, self.slot_in_vm(RuntimeEnvironmentSlot::RegisterTrace) as i64, X0);
            self.emit_ins(ARM64Instruction::mov_sp(SP_XZR, X1));
            self.emit_rust_call(push_register_trace_entry as *const u8 as i64);
            self.emit_ins(ARM64Instruction::ldp(REGISTER_PTR_TO_VM, REGISTER_INSTRUCTION_METER, SP_XZR, ARM64MemoryOperand::Offset(96)));
            self.emit_ins(ARM64Instruction::ldp(REGISTER_MAP[10], REGISTER_SCRATCH, SP_XZR, ARM64MemoryOperand::Offset(80)));
            self.emit_ins(ARM64Instruction::load(OperandSize::S64, SP_XZR, ARM64MemoryOperand::OffsetScaled(14), LR));
            self.emit_ins(ARM64Instruction::add_imm(OperandSize::S64, SP_XZR, 128, SP_XZR));
            self.emit_ins(ARM64Instruction::ret());
        }

        // Epilogue
        self.set_anchor(ANCHOR_EPILOGUE);
        if self.config.enable_instruction_meter {
            self.emit_ins(ARM64Instruction::sub_imm(OperandSize::S64, REGISTER_INSTRUCTION_METER, 1, REGISTER_INSTRUCTION_METER)); // REGISTER_INSTRUCTION_METER -= 1;
            self.emit_ins(ARM64Instruction::sub(OperandSize::S64, REGISTER_INSTRUCTION_METER, REGISTER_SCRATCH, REGISTER_INSTRUCTION_METER)); // REGISTER_INSTRUCTION_METER -= pc;
            // *DueInsnCount = *PreviousInstructionMeter - REGISTER_INSTRUCTION_METER;
            self.emit_ldr_vm(REGISTER_TEMP, RuntimeEnvironmentSlot::PreviousInstructionMeter, 0);
            self.emit_ins(ARM64Instruction::sub(OperandSize::S64, REGISTER_TEMP, REGISTER_INSTRUCTION_METER, REGISTER_INSTRUCTION_METER));
            self.emit_str_vm(REGISTER_INSTRUCTION_METER, RuntimeEnvironmentSlot::DueInsnCount, 0);
        }
        // Restore stack pointer in case we did not exit gracefully
        self.emit_ldr_vm(REGISTER_TEMP, RuntimeEnvironmentSlot::HostStackPointer, 0);
        self.emit_ins(ARM64Instruction::mov_sp(REGISTER_TEMP, SP_XZR));
        for (i, pair) in [[X19, X20], [X21, X22], [X23, X24], [X25, X26], [X27, X28]].iter().enumerate() {
            self.emit_ins(ARM64Instruction::ldp(pair[0], pair[1], SP_XZR, ARM64MemoryOperand::Offset(16 + 16 * i as i16)));
        }
        self.emit_ins(ARM64Instruction::ldp(X29, LR, SP_XZR, ARM64MemoryOperand::OffsetPostIndex(96)));
        self.emit_ins(ARM64Instruction::ret());

        // Handler for EbpfError::ExceededMaxInstructions
        self.set_anchor(ANCHOR_THROW_EXCEEDED_MAX_INSTRUCTIONS);
        self.emit_set_exception_kind(EbpfError::ExceededMaxInstructions);
        self.emit_ins(ARM64Instruction::mov(OperandSize::S64, REGISTER_INSTRUCTION_METER, REGISTER_SCRATCH)); // REGISTER_SCRATCH = REGISTER_INSTRUCTION_METER;
        // Fall through

        // Epilogue for errors
        self.set_anchor(ANCHOR_THROW_EXCEPTION_UNCHECKED);
        self.emit_str_vm(REGISTER_SCRATCH, RuntimeEnvironmentSlot::Registers, 11 * std::mem::size_of::<u64>()); // registers[11] = pc;
        self.emit_jump_to_anchor(ANCHOR_EPILOGUE);

        // Quit gracefully
        self.set_anchor(ANCHOR_EXIT);
        if self.config.enable_instruction_meter {
            self.emit_ins(ARM64Instruction::add_imm(OperandSize::S64, REGISTER_INSTRUCTION_METER, 1, REGISTER_INSTRUCTION_METER)); // REGISTER_INSTRUCTION_METER += 1;
        }
        self.emit_str_vm(REGISTER_MAP[0], RuntimeEnvironmentSlot::ProgramResult, std::mem::size_of::<u64>()); // result.return_value = R0;
        self.emit_load_immediate(REGISTER_SCRATCH, 0);
        self.emit_jump_to_anchor(ANCHOR_EPILOGUE);

        // Handler for exceptions which report their pc
        self.set_anchor(ANCHOR_THROW_EXCEPTION);
        // Validate that we did not reach the instruction meter limit before the exception occured
        self.emit_validate_instruction_count(None);
        self.emit_jump_to_anchor(ANCHOR_THROW_EXCEPTION_UNCHECKED);

        // Handler for EbpfError::CallDepthExceeded
        self.set_anchor(ANCHOR_CALL_DEPTH_EXCEEDED);
        self.emit_set_exception_kind(EbpfError::CallDepthExceeded);
        self.emit_jump_to_anchor(ANCHOR_THROW_EXCEPTION);

        // Handler for EbpfError::CallOutsideTextSegment
        self.set_anchor(ANCHOR_CALL_REG_OUTSIDE_TEXT_SEGMENT);
        self.emit_set_exception_kind(EbpfError::CallOutsideTextSegment);
        self.emit_jump_to_anchor(ANCHOR_THROW_EXCEPTION);

        // Handler for EbpfError::DivideByZero
        self.set_anchor(ANCHOR_DIV_BY_ZERO);
        self.emit_set_exception_kind(EbpfError::DivideByZero);
        self.emit_jump_to_anchor(ANCHOR_THROW_EXCEPTION);

        // Handler for EbpfError::DivideOverflow
        self.set_anchor(ANCHOR_DIV_OVERFLOW);
        self.emit_set_exception_kind(EbpfError::DivideOverflow);
        self.emit_jump_to_anchor(ANCHOR_THROW_EXCEPTION);

        // Handler for EbpfError::UnsupportedInstruction
        // Also the target of callx into the second half of an lddw (see pc_section),
        // in which case REGISTER_SCRATCH holds the target pc.
        self.set_anchor(ANCHOR_CALL_UNSUPPORTED_INSTRUCTION);
        if self.config.enable_register_tracing {
            self.emit_call_anchor(ANCHOR_TRACE);
        }
        self.emit_set_exception_kind(EbpfError::UnsupportedInstruction);
        self.emit_jump_to_anchor(ANCHOR_THROW_EXCEPTION);

        // Routine for external functions
        // Inputs: function pointer in REGISTER_SCRATCH
        self.set_anchor(ANCHOR_EXTERNAL_FUNCTION_CALL);
        self.emit_ins(ARM64Instruction::stp(REGISTER_MAP[10], REGISTER_PTR_TO_VM, SP_XZR, ARM64MemoryOperand::OffsetPreIndex(-32)));
        self.emit_ins(ARM64Instruction::stp(REGISTER_INSTRUCTION_METER, LR, SP_XZR, ARM64MemoryOperand::Offset(16)));
        if self.config.enable_instruction_meter {
            self.emit_str_vm(REGISTER_INSTRUCTION_METER, RuntimeEnvironmentSlot::DueInsnCount, 0); // *DueInsnCount = REGISTER_INSTRUCTION_METER;
        }
        // The syscall expects the encrypted address of the VM
        self.emit_add_immediate(REGISTER_PTR_TO_VM, self.runtime_environment_key as i64, X0);
        for (i, argument) in [X1, X2, X3, X4, X5].iter().enumerate() {
            self.emit_ins(ARM64Instruction::mov(OperandSize::S64, REGISTER_MAP[1 + i], *argument));
        }
        self.emit_ins(ARM64Instruction::blr(REGISTER_SCRATCH));
        self.emit_ins(ARM64Instruction::ldp(REGISTER_INSTRUCTION_METER, LR, SP_XZR, ARM64MemoryOperand::Offset(16)));
        self.emit_ins(ARM64Instruction::ldp(REGISTER_MAP[10], REGISTER_PTR_TO_VM, SP_XZR, ARM64MemoryOperand::OffsetPostIndex(32)));
        if self.config.enable_instruction_meter {
            self.emit_ldr_vm(REGISTER_INSTRUCTION_METER, RuntimeEnvironmentSlot::PreviousInstructionMeter, 0); // REGISTER_INSTRUCTION_METER = *PreviousInstructionMeter;
        }
        // Test if result indicates that an error occured
        self.emit_result_is_err();
        // Used as PC value in error case
        self.emit_load_immediate(REGISTER_SCRATCH, -1);
        self.emit_conditional_jump_to_anchor(Condition::NE, ANCHOR_EPILOGUE);
        // Store Ok value in result register
        self.emit_ldr_vm(REGISTER_MAP[0], RuntimeEnvironmentSlot::ProgramResult, 8);
        self.emit_ins(ARM64Instruction::ret());

        // Routine for prologue of emit_internal_call()
        // Inputs: guest current pc in REGISTER_SCRATCH, guest return address is stored by the caller
        self.set_anchor(ANCHOR_INTERNAL_FUNCTION_CALL_PROLOGUE);
        self.emit_validate_instruction_count(None);
        self.emit_ins(ARM64Instruction::sub_imm(OperandSize::S64, SP_XZR, CALL_FRAME_SIZE, SP_XZR)); // alloca
        // [return address, r10, r6, r7, r8, r9]
        self.emit_ins(ARM64Instruction::store(OperandSize::S64, REGISTER_MAP[FRAME_PTR_REG], SP_XZR, ARM64MemoryOperand::OffsetScaled(1)));
        for i in (0..SCRATCH_REGS).step_by(2) {
            self.emit_ins(ARM64Instruction::stp(
                REGISTER_MAP[FIRST_SCRATCH_REG + i],
                REGISTER_MAP[FIRST_SCRATCH_REG + i + 1],
                SP_XZR,
                ARM64MemoryOperand::Offset(16 + 8 * i as i16),
            ));
        }
        // Increase env.call_depth
        self.emit_ldr_vm(REGISTER_TEMP, RuntimeEnvironmentSlot::CallDepth, 0);
        self.emit_ins(ARM64Instruction::add_imm(OperandSize::S64, REGISTER_TEMP, 1, REGISTER_TEMP)); // env.call_depth += 1;
        self.emit_str_vm(REGISTER_TEMP, RuntimeEnvironmentSlot::CallDepth, 0);
        // If env.call_depth == self.config.max_call_depth, throw CallDepthExceeded
        self.emit_load_immediate(REGISTER_TEMP2, self.config.max_call_depth as i64);
        self.emit_ins(ARM64Instruction::cmp(OperandSize::S32, REGISTER_TEMP2, REGISTER_TEMP));
        self.emit_conditional_jump_to_anchor(Condition::HS, ANCHOR_CALL_DEPTH_EXCEEDED);
        // Setup the frame pointer for the new frame. What we do depends on whether we're using dynamic or fixed frames.
        if !self.executable.get_sbpf_version().manual_stack_frame_bump() {
            // With fixed frames we start the new frame at the next fixed offset
            let num_frames = if self.executable.get_sbpf_version().stack_frame_gaps()
                && self.config.enable_stack_frame_gaps {
                2
            } else {
                1
            };
            let stack_frame_size = self.config.stack_frame_size as i64 * num_frames;
            self.emit_add_immediate(REGISTER_MAP[FRAME_PTR_REG], stack_frame_size, REGISTER_MAP[FRAME_PTR_REG]); // REGISTER_MAP[FRAME_PTR_REG] += stack_frame_size;
        }
        self.emit_ins(ARM64Instruction::ret());

        // Routine for emit_internal_call(CallTarget::Register())
        // Inputs: Guest current pc in REGISTER_SCRATCH, Guest target address in REGISTER_OTHER
        // Outputs: Guest current pc in REGISTER_SCRATCH, Host target address in REGISTER_TEMP
        self.set_anchor(ANCHOR_INTERNAL_FUNCTION_CALL_REG);
        // Calculate offset relative to program_vm_addr
        self.emit_load_immediate(REGISTER_TEMP, self.program_vm_addr as i64);
        self.emit_ins(ARM64Instruction::sub(OperandSize::S64, REGISTER_OTHER, REGISTER_TEMP, REGISTER_OTHER)); // guest_target_address -= self.program_vm_addr;
        // guest_target_pc = guest_target_address / INSN_SIZE;
        let shift_amount = INSN_SIZE.trailing_zeros() as u8;
        debug_assert_eq!(INSN_SIZE, 1 << shift_amount);
        self.emit_ins(ARM64Instruction::lsr_imm(REGISTER_OTHER, shift_amount, REGISTER_OTHER));
        // Bound check
        // if(guest_target_pc >= number_of_instructions) throw CALL_OUTSIDE_TEXT_SEGMENT;
        let number_of_instructions = self.result.pc_section.len();
        self.emit_load_immediate(REGISTER_TEMP, number_of_instructions as i64);
        self.emit_ins(ARM64Instruction::cmp(OperandSize::S64, REGISTER_TEMP, REGISTER_OTHER));
        self.emit_conditional_jump_to_anchor(Condition::HS, ANCHOR_CALL_REG_OUTSIDE_TEXT_SEGMENT);
        // Load host_target_address offset from self.result.pc_section
        self.emit_load_immediate(REGISTER_TEMP, self.result.pc_section.as_ptr() as i64); // host_target_address = self.result.pc_section;
        self.emit_ins(ARM64Instruction::load(OperandSize::S32, REGISTER_TEMP, ARM64MemoryOperand::OffsetIndexShift(REGISTER_OTHER, true), REGISTER_TEMP2)); // host_target_address = self.result.pc_section[guest_target_pc];
        // Check destination is valid
        // If host_target_address & (1 << 31) != 0, throw UnsupportedInstruction
        self.emit_tbz_to_anchor(REGISTER_TEMP2, 31, true, ANCHOR_CALL_UNSUPPORTED_INSTRUCTION);
        // A version of `self.emit_profile_instruction_count(None);` which reads self.pc from REGISTER_SCRATCH
        self.emit_ins(ARM64Instruction::sub(OperandSize::S64, REGISTER_INSTRUCTION_METER, REGISTER_SCRATCH, REGISTER_INSTRUCTION_METER)); // instruction_meter -= guest_current_pc;
        self.emit_ins(ARM64Instruction::sub_imm(OperandSize::S64, REGISTER_INSTRUCTION_METER, 1, REGISTER_INSTRUCTION_METER)); // instruction_meter -= 1;
        self.emit_ins(ARM64Instruction::add(OperandSize::S64, REGISTER_INSTRUCTION_METER, REGISTER_OTHER, REGISTER_INSTRUCTION_METER)); // instruction_meter += guest_target_pc;
        // REGISTER_SCRATCH = guest_target_pc (the pc reported if the target is the second half of an lddw)
        self.emit_ins(ARM64Instruction::mov(OperandSize::S64, REGISTER_OTHER, REGISTER_SCRATCH));
        // Offset host_target_address by self.result.text_section
        self.emit_load_immediate(REGISTER_TEMP, self.result.text_section.as_ptr() as i64);
        self.emit_ins(ARM64Instruction::add(OperandSize::S64, REGISTER_TEMP, REGISTER_TEMP2, REGISTER_TEMP)); // host_target_address += self.result.text_section;
        self.emit_ins(ARM64Instruction::ret());

        // Translates a vm memory address to a host memory address
        // Inputs: vm_addr in REGISTER_SCRATCH, pc in REGISTER_TRANSLATION_PC, value to store in REGISTER_OTHER
        // Outputs: loaded value in REGISTER_SCRATCH
        for (anchor_base, len) in &[
            (0, 1i32), (0, 2i32), (0, 4i32), (0, 8i32),
            (4, 1i32), (4, 2i32), (4, 4i32), (4, 8i32),
        ] {
            let target_offset = *anchor_base + len.trailing_zeros() as usize;
            self.set_anchor(ANCHOR_TRANSLATE_MEMORY_ADDRESS + target_offset);
            self.emit_ins(ARM64Instruction::stp(REGISTER_MAP[10], REGISTER_PTR_TO_VM, SP_XZR, ARM64MemoryOperand::OffsetPreIndex(-48)));
            self.emit_ins(ARM64Instruction::stp(REGISTER_INSTRUCTION_METER, LR, SP_XZR, ARM64MemoryOperand::Offset(16)));
            self.emit_ins(ARM64Instruction::store(OperandSize::S64, REGISTER_TRANSLATION_PC, SP_XZR, ARM64MemoryOperand::OffsetScaled(4)));
            self.emit_ldr_vm(X0, RuntimeEnvironmentSlot::MemoryMapping, 0);
            if *anchor_base == 0 { // AccessType::Load
                let load = match len {
                    1 => load_u8 as *const u8 as i64,
                    2 => load_u16 as *const u8 as i64,
                    4 => load_u32 as *const u8 as i64,
                    8 => load_u64 as *const u8 as i64,
                    _ => unreachable!()
                };
                self.emit_ins(ARM64Instruction::mov(OperandSize::S64, REGISTER_SCRATCH, X1));
                self.emit_add_immediate(REGISTER_PTR_TO_VM, self.slot_in_vm(RuntimeEnvironmentSlot::ProgramResult) as i64, X2);
                self.emit_rust_call(load);
            } else { // AccessType::Store
                let store = match len {
                    1 => store_u8 as *const u8 as i64,
                    2 => store_u16 as *const u8 as i64,
                    4 => store_u32 as *const u8 as i64,
                    8 => store_u64 as *const u8 as i64,
                    _ => unreachable!()
                };
                self.emit_ins(ARM64Instruction::mov(OperandSize::S64, REGISTER_OTHER, X1));
                self.emit_ins(ARM64Instruction::mov(OperandSize::S64, REGISTER_SCRATCH, X2));
                self.emit_add_immediate(REGISTER_PTR_TO_VM, self.slot_in_vm(RuntimeEnvironmentSlot::ProgramResult) as i64, X3);
                self.emit_rust_call(store);
            }
            self.emit_ins(ARM64Instruction::load(OperandSize::S64, SP_XZR, ARM64MemoryOperand::OffsetScaled(4), REGISTER_TRANSLATION_PC));
            self.emit_ins(ARM64Instruction::ldp(REGISTER_INSTRUCTION_METER, LR, SP_XZR, ARM64MemoryOperand::Offset(16)));
            self.emit_ins(ARM64Instruction::ldp(REGISTER_MAP[10], REGISTER_PTR_TO_VM, SP_XZR, ARM64MemoryOperand::OffsetPostIndex(48)));

            // Throw error if the result indicates one
            if *anchor_base == 0 { // AccessType::Load
                // unwrap() the result into REGISTER_SCRATCH
                self.emit_ins(ARM64Instruction::mov(OperandSize::S64, X1, REGISTER_SCRATCH));
            }
            self.emit_ins(ARM64Instruction::cbz(OperandSize::S64, X0, true, 2)); // if is_err goto throw
            self.emit_ins(ARM64Instruction::ret());
            self.emit_ins(ARM64Instruction::mov(OperandSize::S64, REGISTER_TRANSLATION_PC, REGISTER_SCRATCH)); // REGISTER_SCRATCH = pc
            self.emit_jump_to_anchor(ANCHOR_THROW_EXCEPTION);
        }
    }

    /// Calls a function at a constant host address, all caller saved registers are clobbered
    fn emit_rust_call(&mut self, function: i64) {
        self.emit_load_immediate(REGISTER_TEMP, function);
        self.emit_ins(ARM64Instruction::blr(REGISTER_TEMP));
    }

    fn set_anchor(&mut self, anchor: usize) {
        self.anchors[anchor] = self.offset_in_text_section;
    }

    /// Byte distance from the current position to the anchor
    fn relative_to_anchor(&self, anchor: usize) -> isize {
        let destination = self.anchors[anchor];
        debug_assert_ne!(destination, usize::MAX);
        destination as isize - self.offset_in_text_section as isize
    }

    fn check_branch_distance(&mut self, distance: isize, limit: isize) -> i32 {
        if distance < -limit || distance >= limit {
            self.branch_out_of_range = true;
            return 0;
        }
        (distance / 4) as i32
    }

    fn emit_jump_to_anchor(&mut self, anchor: usize) {
        let distance = self.relative_to_anchor(anchor);
        let imm26 = self.check_branch_distance(distance, MAX_BRANCH_DISTANCE);
        self.emit_ins(ARM64Instruction::b(imm26));
    }

    fn emit_call_anchor(&mut self, anchor: usize) {
        let distance = self.relative_to_anchor(anchor);
        let imm26 = self.check_branch_distance(distance, MAX_BRANCH_DISTANCE);
        self.emit_ins(ARM64Instruction::bl(imm26));
    }

    fn emit_conditional_jump_to_anchor(&mut self, condition: Condition, anchor: usize) {
        let distance = self.relative_to_anchor(anchor);
        if (-MAX_CONDITIONAL_BRANCH_DISTANCE..MAX_CONDITIONAL_BRANCH_DISTANCE).contains(&distance) {
            self.emit_ins(ARM64Instruction::b_cond(condition, (distance / 4) as i32));
        } else {
            self.emit_ins(ARM64Instruction::b_cond(invert_condition(condition), 2));
            self.emit_jump_to_anchor(anchor);
        }
    }

    /// CBZ (or CBNZ if non_zero) to an anchor
    fn emit_cbz_to_anchor(&mut self, size: OperandSize, reg: u8, non_zero: bool, anchor: usize) {
        let distance = self.relative_to_anchor(anchor);
        if (-MAX_CONDITIONAL_BRANCH_DISTANCE..MAX_CONDITIONAL_BRANCH_DISTANCE).contains(&distance) {
            self.emit_ins(ARM64Instruction::cbz(size, reg, non_zero, (distance / 4) as i32));
        } else {
            self.emit_ins(ARM64Instruction::cbz(size, reg, !non_zero, 2));
            self.emit_jump_to_anchor(anchor);
        }
    }

    /// TBZ (or TBNZ if non_zero) to an anchor
    fn emit_tbz_to_anchor(&mut self, reg: u8, bit: u8, non_zero: bool, anchor: usize) {
        // TBZ only reaches +-32 KiB, the anchors are all close to each other though
        let distance = self.relative_to_anchor(anchor);
        if (-(1 << 15)..(1 << 15)).contains(&distance) {
            self.emit_ins(ARM64Instruction::tbz(reg, bit, non_zero, (distance / 4) as i32));
        } else {
            self.emit_ins(ARM64Instruction::tbz(reg, bit, !non_zero, 2));
            self.emit_jump_to_anchor(anchor);
        }
    }

    /// Reserves space for a conditional branch to be patched by `patch_conditional_branch()`
    fn emit_placeholder(&mut self) -> usize {
        let location = self.offset_in_text_section;
        self.emit::<u32>(TRAP_INSTRUCTION);
        location
    }

    /// Turns a placeholder into a conditional branch to the current position
    fn patch_conditional_branch(&mut self, location: usize, condition: Condition) {
        let end = self.offset_in_text_section;
        self.offset_in_text_section = location;
        self.emit_ins(ARM64Instruction::b_cond(condition, ((end - location) / 4) as i32));
        self.offset_in_text_section = end;
    }

    /// Emits a B (or BL if link) to the code of target_pc
    fn emit_jump_to_target_pc(&mut self, target_pc: usize, link: bool) {
        let imm26 = if self.result.pc_section[target_pc] != 0 {
            // Backward jump
            let destination = (self.result.pc_section[target_pc] & (i32::MAX as u32)) as isize;
            let distance = destination - self.offset_in_text_section as isize;
            self.check_branch_distance(distance, MAX_BRANCH_DISTANCE)
        } else {
            // Forward jump, needs relocation
            self.text_section_jumps.push(Jump { location: self.offset_in_text_section, target_pc });
            0
        };
        if link {
            self.emit_ins(ARM64Instruction::bl(imm26));
        } else {
            self.emit_ins(ARM64Instruction::b(imm26));
        }
    }

    fn emit_conditional_jump_to_target_pc(&mut self, condition: Condition, target_pc: usize) {
        if self.result.pc_section[target_pc] != 0 {
            // Backward jump
            let destination = (self.result.pc_section[target_pc] & (i32::MAX as u32)) as isize;
            let distance = destination - self.offset_in_text_section as isize;
            if (-MAX_CONDITIONAL_BRANCH_DISTANCE..MAX_CONDITIONAL_BRANCH_DISTANCE).contains(&distance) {
                self.emit_ins(ARM64Instruction::b_cond(condition, (distance / 4) as i32));
                return;
            }
        }
        self.emit_ins(ARM64Instruction::b_cond(invert_condition(condition), 2));
        self.emit_jump_to_target_pc(target_pc, false);
    }

    fn resolve_jumps(&mut self) {
        // Relocate forward jumps
        for jump in std::mem::take(&mut self.text_section_jumps) {
            let destination = (self.result.pc_section[jump.target_pc] & (i32::MAX as u32)) as isize;
            let distance = destination - jump.location as isize;
            let imm26 = self.check_branch_distance(distance, MAX_BRANCH_DISTANCE) as u32;
            unsafe {
                let location = self.result.text_section.as_mut_ptr().add(jump.location).cast::<u32>();
                let instruction = ptr::read_unaligned(location);
                ptr::write_unaligned(location, (instruction & 0xfc00_0000) | (imm26 & 0x03ff_ffff));
            }
        }
    }
}

#[derive(Copy, Clone)]
enum StoreValue {
    Register(u8),
    Constant(i64),
}

#[derive(Copy, Clone)]
enum CallTarget {
    Register(u8),
    Constant(i64),
}

#[derive(Copy, Clone)]
enum ShiftOp {
    Lsl,
    Lsr,
    Asr,
}

fn invert_condition(condition: Condition) -> Condition {
    match condition {
        Condition::EQ => Condition::NE,
        Condition::NE => Condition::EQ,
        Condition::CS => Condition::CC,
        Condition::CC => Condition::CS,
        Condition::HI => Condition::LS,
        Condition::LS => Condition::HI,
        Condition::GE => Condition::LT,
        Condition::LT => Condition::GE,
        Condition::GT => Condition::LE,
        Condition::LE => Condition::GT,
    }
}

/// AND (0), ORR (1), EOR (2) with registers: destination = src1 op src2
fn logical_register(
    size: OperandSize,
    opcode: u8,
    src1: u8,
    src2: u8,
    destination: u8,
) -> crate::aarch64::ARM64InstructionLogicalShiftedRegister {
    crate::aarch64::ARM64InstructionLogicalShiftedRegister {
        size,
        opcode,
        n: 0,
        dest: destination,
        src1,
        src2,
        ..crate::aarch64::ARM64InstructionLogicalShiftedRegister::default()
    }
}
