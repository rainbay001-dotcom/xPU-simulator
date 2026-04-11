"""Minimal CA sim cube timing test — single tiny matmul on 1 core."""
from __future__ import annotations
import os, sys, time, shutil
import te.platform as tp
tp.te_set_version("Ascend910B1", core_type="AiCore")
from te import tik
import numpy as np

SOC = "Ascend910B1"
SIM_LIB = "/usr/local/Ascend/cann-8.5.0/tools/simulator"
DUMP = "/home/Ray/ca_sim_test/ca_profiling_dumps/cube_mini"


def build_matmul(M: int, K: int, N: int):
    t = tik.Tik(disable_debug=False)
    gm_a = t.Tensor("float16", (M, K), name="gm_a", scope=tik.scope_gm)
    gm_b = t.Tensor("float16", (K, N), name="gm_b", scope=tik.scope_gm)
    gm_c = t.Tensor("float16", (M, N), name="gm_c", scope=tik.scope_gm)

    # single-core version: block_dim=1, all data fits in L1 at once
    with t.for_range(0, 1, block_num=1) as _bid:
        l1_a = t.Tensor("float16", (M, K), name="l1_a", scope=tik.scope_cbuf)
        l1_b = t.Tensor("float16", (K, N), name="l1_b", scope=tik.scope_cbuf)
        l0c_c = t.Tensor("float32", (M, N), name="l0c_c", scope=tik.scope_cbuf_out)

        a_burst = (M * K * 2) // 32
        b_burst = (K * N * 2) // 32
        t.data_move(l1_a, gm_a, 0, 1, a_burst, 0, 0)
        t.data_move(l1_b, gm_b, 0, 1, b_burst, 0, 0)
        t.matmul(l0c_c, l1_a, l1_b, M, K, N, init_l1out=True)
        t.fixpipe(
            gm_c, l0c_c, M, (N * 4) // 32, 0, 0,
            extend_params={"quantize_params": {"mode": "fp322fp16", "mode_param": None}},
        )
    return t, f"mm_{M}x{K}x{N}", [gm_a, gm_b], [gm_c]


def run(M, K, N):
    from op_test_frame.common.ascend_tbe_op import AscendOpKernel, AscendOpKernelRunner
    dump = f"{DUMP}_{M}x{K}x{N}"
    if os.path.exists(dump):
        shutil.rmtree(dump)
    os.makedirs(dump)

    t, kname, inputs, outputs = build_matmul(M, K, N)
    t.BuildCCE(kernel_name=kname, inputs=inputs, outputs=outputs)
    bin_path = os.path.join("kernel_meta", f"{kname}.o")
    json_path = os.path.join("kernel_meta", f"{kname}.json")

    inp_data = [np.random.uniform(-1, 1, size=tuple(int(d) for d in tnsr.shape)).astype("float16")
                for tnsr in inputs]
    out_info = [{"shape": tuple(int(d) for d in tnsr.shape), "dtype": "float16"} for tnsr in outputs]

    op_kernel = AscendOpKernel(bin_path, json_path)
    op_kernel.set_input_info([{"shape": d.shape, "dtype": "float16", "value": d} for d in inp_data])
    op_kernel.set_output_info(out_info)

    t0 = time.time()
    with AscendOpKernelRunner(
        simulator_mode="ca", soc_version=SOC,
        simulator_lib_path=SIM_LIB, simulator_dump_path=dump,
    ) as runner:
        runner.run(op_kernel, inputs=inp_data, block_dim=1)
    elapsed = time.time() - t0
    print(f"[{M}x{K}x{N}] wall_time={elapsed:.2f}s")
    return elapsed


if __name__ == "__main__":
    # Shapes that fit L0C (32KB fp32 = 8192 elts per core → M*N ≤ 8192).
    shapes = [(128, 128, 64), (128, 256, 64), (128, 512, 64), (256, 256, 128)]
    for M, K, N in shapes:
        try:
            run(M, K, N)
        except BaseException as e:
            msg = str(e).replace("\n", " | ")[:200]
            print(f"[{M}x{K}x{N}] FAILED: {type(e).__name__}: {msg}")
