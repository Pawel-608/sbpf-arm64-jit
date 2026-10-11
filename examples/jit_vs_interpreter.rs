//! Compares the interpreter with the JIT on a few compute heavy programs.
//!
//! cargo run --release --example jit_vs_interpreter

#[cfg(all(
    feature = "jit",
    not(target_os = "windows"),
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
fn main() {
    use solana_sbpf::{
        assembler::assemble,
        ebpf,
        memory_region::MemoryRegion,
        program::{BuiltinProgram, SBPFVersion},
        verifier::RequisiteVerifier,
        vm::{CallFrame, Config, ExecutionMode},
    };
    use std::{sync::Arc, time::Instant};
    use test_utils::{create_vm, TestContextObject};

    let programs: [(&str, &str); 4] = [
        (
            "alu (xorshift, mul, div)",
            "
            mov64 r0, 0
            lddw r1, 0x139408DCBBF7A44
            mov64 r2, 0
        loop:
            mov64 r3, r1
            lsh64 r3, 13
            xor64 r1, r3
            mov64 r3, r1
            rsh64 r3, 7
            xor64 r1, r3
            mov64 r3, r1
            lsh64 r3, 17
            xor64 r1, r3
            mov64 r4, r1
            mul64 r4, 0x2545F491
            mov64 r5, r4
            and64 r5, 0xffff
            or64 r5, 1
            div64 r4, r5
            add64 r0, r4
            add64 r2, 1
            jlt r2, 1000000, loop
            exit",
        ),
        (
            "memory (input loads, stack stores)",
            "
            mov64 r0, 0
            mov64 r2, 0
        loop:
            mov64 r3, r2
            and64 r3, 0x3ff8
            mov64 r4, r1
            add64 r4, r3
            ldxdw r5, [r4]
            add64 r0, r5
            stxdw [r10-8], r0
            ldxdw r6, [r10-8]
            xor64 r0, r6
            stxdw [r4], r0
            add64 r2, 8
            jlt r2, 4000000, loop
            exit",
        ),
        (
            "calls (recursive fibonacci)",
            "
            mov64 r1, 24
            call function_fib
            exit
        function_fib:
            mov64 r0, r1
            jlt r1, 2, done
            mov64 r6, r1
            sub64 r1, 1
            call function_fib
            mov64 r7, r0
            mov64 r1, r6
            sub64 r1, 2
            call function_fib
            add64 r0, r7
        done:
            exit",
        ),
        (
            "branchy (collatz lengths)",
            "
            mov64 r0, 0
            mov64 r1, 1
        outer:
            mov64 r2, r1
        inner:
            jeq r2, 1, next
            add64 r0, 1
            jset r2, 1, odd
            rsh64 r2, 1
            ja inner
        odd:
            mul64 r2, 3
            add64 r2, 1
            ja inner
        next:
            add64 r1, 1
            jlt r1, 30000, outer
            exit",
        ),
    ];

    println!(
        "{:<36} {:>12} {:>14} {:>14} {:>9} {:>11}",
        "program", "instructions", "interpreter", "JIT", "speedup", "compile"
    );
    for (name, source) in programs {
        let config = Config {
            enabled_sbpf_versions: SBPFVersion::V0..=SBPFVersion::V0,
            max_call_depth: 64,
            ..Config::default()
        };
        let executable =
            assemble::<TestContextObject>(source, Arc::new(BuiltinProgram::new_loader(config)))
                .unwrap();
        executable.verify::<RequisiteVerifier>().unwrap();
        let compile_start = Instant::now();
        executable.jit_compile().unwrap();
        let compile_time = compile_start.elapsed();

        let mut input = vec![0u8; 0x4000];
        let budget = u64::MAX >> 1;
        let mut measure = |jit: bool| {
            let mut best = f64::MAX;
            let mut outcome = None;
            for _ in 0..5 {
                let mut context_object = TestContextObject::new(budget);
                let input_region = MemoryRegion::new(&raw mut input[..], ebpf::MM_INPUT_START);
                create_vm!(
                    vm,
                    &executable,
                    &mut context_object,
                    stack,
                    heap,
                    vec![input_region],
                    None
                );
                let mut call_frames = vec![CallFrame::default(); 64];
                let mut mode = if jit {
                    ExecutionMode::Jit
                } else {
                    ExecutionMode::Interpreted
                };
                let start = Instant::now();
                let (instruction_count, result) =
                    vm.execute_program(&executable, &mut mode, &mut call_frames);
                best = best.min(start.elapsed().as_secs_f64());
                outcome = Some((instruction_count, format!("{result:?}")));
            }
            (best, outcome.unwrap())
        };
        let (interpreter_time, interpreter) = measure(false);
        let (jit_time, jit) = measure(true);
        assert_eq!(interpreter, jit, "{name}: interpreter and JIT disagree");
        if !interpreter.1.starts_with("Ok") {
            println!("{name}: {}", interpreter.1);
        }
        let instructions = interpreter.0 as f64;
        println!(
            "{:<36} {:>12} {:>9.1} Mi/s {:>9.1} Mi/s {:>8.1}x {:>8.2} ms",
            name,
            interpreter.0,
            instructions / interpreter_time / 1e6,
            instructions / jit_time / 1e6,
            interpreter_time / jit_time,
            compile_time.as_secs_f64() * 1e3,
        );
    }
}

#[cfg(not(all(
    feature = "jit",
    not(target_os = "windows"),
    any(target_arch = "x86_64", target_arch = "aarch64")
)))]
fn main() {
    println!("The JIT is not available on this platform");
}
