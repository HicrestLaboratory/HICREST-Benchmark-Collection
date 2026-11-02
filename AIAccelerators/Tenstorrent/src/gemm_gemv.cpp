/*

GeMM / GeMV addmm/matmul benchmark

Usage:
    ./gemm M K N [options]

Required arguments:
    M                      Rows of A and C
    K                      Columns of A / Rows of B
    N                      Columns of B and C

Optional arguments:
    --op <value>
        Operation to benchmark.

        Valid values:
            addmm
            matmul, mm

        Default: addmm

    -w, --warmup <int>
        Number of warmup iterations.
        Default: 5

    -r, --run <int>
        Number of timed iterations.
        Default: 10

    --dtype <value>
        Tensor data type.

        Valid values:
            bf16, bfloat16
            fp32, float32
            bfp8, bfloat8_b
            bfp4, bfloat4_b

        Default: bf16

    --math-fidelity <value>
        Compute math fidelity.

        Valid values:
            lofi,  m0
            hifi2, m2
            hifi3, m3
            hifi4, m4

        Recommended combinations:
            fp32 : hifi4
            bf16 : hifi2, hifi4
            bfp8 : lofi, hifi2
            bfp4 : lofi

        Note:
            Not all dtype/fidelity combinations are supported or meaningful.
            BFLOAT8_B (bfp8) and BFLOAT4_B (bfp4) are typically intended
            to be used with reduced math fidelity.

    --layout <value>
        Input tensor layout.

        Valid values:
            tile
            row_major, row-major, rm

        Default: tile

        Notes:
            - MatMul/addmm executes only on TILE layout.
            - If ROW_MAJOR is selected, tensors are automatically converted
              to TILE before execution.
            - BFLOAT8_B (bfp8) and BFLOAT4_B (bfp4) only support TILE layout.
              If ROW_MAJOR is selected, it is automatically overridden to TILE.

    --memory-layout <value>
        Tensor memory layout.

        Valid values:
            interleaved
            sharded

        Default: interleaved

    --measurement-mode <value>
        Measurement boundary: kernel or end-to-end.
        End-to-end creates host inputs once, warms up the complete transfer and
        execution pipeline, then measures H2D -> operation -> D2H per iteration.
        Default: kernel

        Notes:
            - INTERLEAVED is the default TTNN memory layout.
            - SHARDED distributes tensor A across the selected core grid
              using BLOCK_SHARDED L1 memory.
            - Tensor B always remains INTERLEAVED because the current TTNN
              addmm implementation requires it.

    --compute-grid-y <int>
        Compute grid height used by addmm/matmul.
        Default: 8

    --compute-grid-x <int>
        Compute grid width used by addmm/matmul.
        Default: 8

    --shard-grid-y <int>
        Shard grid height used only when --memory-layout=sharded.
        Default: 8

    --shard-grid-x <int>
        Shard grid width used only when --memory-layout=sharded.
        Default: 8

Examples:

    # addmm, BF16
    ./gemm 4096 4096 4096 \
        --op addmm \
        --dtype bf16 \
        --math-fidelity hifi4

    # matmul, BF16
    ./gemm 4096 4096 4096 \
        --op matmul \
        --dtype bf16 \
        --math-fidelity hifi4

*/

/* GeMM / GeMV addmm/matmul benchmark with trace replay */

#include "energy_markers.hpp"
#include <argparse/argparse.hpp>

#include <ttnn/device.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/types.hpp>

#include <ttnn/operations/creation/creation.hpp>
#include <ttnn/operations/rand/rand.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/core/core.hpp>

#include <tt-metalium/device.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/host_api.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>

#include "profiling.hpp"

using namespace ttnn;
using tt::tt_metal::distributed::MeshDevice;

enum class BenchmarkOp {
    ADDMM,
    MATMUL
};

static std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

static BenchmarkOp parse_op(const std::string& value) {
    const auto s = lower(value);
    if (s == "addmm") return BenchmarkOp::ADDMM;
    if (s == "matmul" || s == "mm") return BenchmarkOp::MATMUL;
    throw std::runtime_error("Unsupported --op: " + value);
}

static DataType parse_dtype(const std::string& value) {
    const auto s = lower(value);
    if (s == "bf16" || s == "bfloat16") return DataType::BFLOAT16;
    if (s == "fp32" || s == "float32") return DataType::FLOAT32;
    if (s == "bfp8" || s == "bfloat8_b") return DataType::BFLOAT8_B;
    if (s == "bfp4" || s == "bfloat4_b") return DataType::BFLOAT4_B;
    throw std::runtime_error("Unsupported --dtype: " + value);
}

static MathFidelity parse_math_fidelity(const std::string& value) {
    const auto s = lower(value);
    if (s == "lofi" || s == "m0") return MathFidelity::LoFi;
    if (s == "hifi2" || s == "m2") return MathFidelity::HiFi2;
    if (s == "hifi3" || s == "m3") return MathFidelity::HiFi3;
    if (s == "hifi4" || s == "m4") return MathFidelity::HiFi4;
    throw std::runtime_error("Unsupported --math-fidelity: " + value);
}

static Layout parse_layout(const std::string& value) {
    const auto s = lower(value);
    if (s == "tile") return Layout::TILE;
    if (s == "row_major" || s == "row-major" || s == "rm") return Layout::ROW_MAJOR;
    throw std::runtime_error("Unsupported --layout: " + value);
}

int main(int argc, char* argv[]) {
    benchmark::energy::Interval application_interval("HICREST_APPLICATION_INTERVAL");
    argparse::ArgumentParser program("GeMM and GeMV addmm/matmul test");

    program.add_argument("M").required().scan<'i', int>();
    program.add_argument("K").required().scan<'i', int>();
    program.add_argument("N").required().scan<'i', int>();

    program.add_argument("--op")
        .help("Operation to benchmark: addmm or matmul")
        .default_value(std::string("addmm"));

    program.add_argument("-w", "--warmup").default_value(5).scan<'i', int>();
    program.add_argument("-r", "--run").default_value(10).scan<'i', int>();

    program.add_argument("--dtype").default_value(std::string("bf16"));
    program.add_argument("--math-fidelity").default_value(std::string("hifi4"));
    program.add_argument("--layout").default_value(std::string("tile"));
    program.add_argument("--memory-layout").default_value(std::string("interleaved"));
    program.add_argument("--measurement-mode")
        .help("Measurement boundary: kernel or end-to-end (repeated H2D, operation, D2H)")
        .default_value(std::string("kernel"));

    program.add_argument("--compute-grid-y").default_value(8).scan<'i', int>();
    program.add_argument("--compute-grid-x").default_value(8).scan<'i', int>();
    program.add_argument("--shard-grid-y").default_value(8).scan<'i', int>();
    program.add_argument("--shard-grid-x").default_value(8).scan<'i', int>();

    constexpr int device_id = 0;
    std::shared_ptr<MeshDevice> device = ttnn::open_mesh_device(device_id);

    try {
        program.parse_args(argc, argv);

        const uint32_t M = static_cast<uint32_t>(program.get<int>("M"));
        const uint32_t K = static_cast<uint32_t>(program.get<int>("K"));
        const uint32_t N = static_cast<uint32_t>(program.get<int>("N"));

        const uint32_t warmup_iterations =
            static_cast<uint32_t>(program.get<int>("--warmup"));
        const uint32_t run_iterations =
            static_cast<uint32_t>(program.get<int>("--run"));

        const auto op = parse_op(program.get<std::string>("--op"));
        const auto dtype = parse_dtype(program.get<std::string>("--dtype"));
        const auto requested_layout = parse_layout(program.get<std::string>("--layout"));
        const auto math_fidelity =
            parse_math_fidelity(program.get<std::string>("--math-fidelity"));

        const auto memory_layout =
            lower(program.get<std::string>("--memory-layout"));
        const auto measurement_mode = lower(program.get<std::string>("--measurement-mode"));
        if (measurement_mode != "kernel" && measurement_mode != "end-to-end") {
            throw std::runtime_error("Unsupported --measurement-mode: " + measurement_mode);
        }
        if (measurement_mode == "end-to-end" && warmup_iterations == 0) {
            throw std::runtime_error("end-to-end mode requires at least one warmup iteration");
        }

        const uint32_t compute_grid_y =
            static_cast<uint32_t>(program.get<int>("--compute-grid-y"));
        const uint32_t compute_grid_x =
            static_cast<uint32_t>(program.get<int>("--compute-grid-x"));

        const uint32_t shard_grid_y =
            static_cast<uint32_t>(program.get<int>("--shard-grid-y"));
        const uint32_t shard_grid_x =
            static_cast<uint32_t>(program.get<int>("--shard-grid-x"));

        Layout input_layout = requested_layout;

        if ((dtype == DataType::BFLOAT8_B || dtype == DataType::BFLOAT4_B) &&
            input_layout != Layout::TILE) {
            std::cerr << "Warning: BFLOAT8_B/BFLOAT4_B only support TILE layout. "
                      << "Overriding --layout to tile.\n";
            input_layout = Layout::TILE;
        }

        const auto compute_grid =
            ttnn::types::CoreGrid(compute_grid_y, compute_grid_x);

        constexpr float beta = 1.0f;
        constexpr float alpha = 1.0f;

        tt::tt_metal::MemoryConfig output_mem_cfg = DRAM_MEMORY_CONFIG;
        tt::tt_metal::MemoryConfig a_mem_cfg = DRAM_MEMORY_CONFIG;
        tt::tt_metal::MemoryConfig b_mem_cfg = DRAM_MEMORY_CONFIG;

        if (memory_layout == "sharded") {
            using namespace tt::tt_metal;

            const CoreRangeSet shard_grid{
                CoreRange{
                    CoreCoord{0, 0},
                    CoreCoord{shard_grid_x - 1, shard_grid_y - 1}
                }
            };

            const std::array<uint32_t, 2> shard_shape = {
                static_cast<uint32_t>((M + shard_grid_y - 1) / shard_grid_y),
                static_cast<uint32_t>((N + shard_grid_x - 1) / shard_grid_x)
            };

            output_mem_cfg = MemoryConfig(
                TensorMemoryLayout::BLOCK_SHARDED,
                BufferType::L1,
                ShardSpec{
                    shard_grid,
                    shard_shape,
                    ShardOrientation::ROW_MAJOR
                }
            );

            a_mem_cfg = output_mem_cfg;
            b_mem_cfg = DRAM_MEMORY_CONFIG;
        } else if (memory_layout != "interleaved") {
            throw std::runtime_error("Unsupported --memory-layout: " + memory_layout);
        }

        auto compute_cfg = ttnn::ComputeKernelConfig{
            math_fidelity,
            false,
            false,
            false
        };

        std::cout << "M=" << M << ", K=" << K << ", N=" << N << "\n";
        std::cout << "Operation: "
                  << (op == BenchmarkOp::ADDMM ? "addmm" : "matmul")
                  << "\n";
        std::cout << "Compute grid: "
                  << compute_grid_y << " x " << compute_grid_x
                  << " = " << compute_grid_y * compute_grid_x
                  << " requested Tensix cores\n";

        if (memory_layout == "sharded") {
            std::cout << "Shard grid: "
                      << shard_grid_y << " x " << shard_grid_x
                      << " = " << shard_grid_y * shard_grid_x
                      << " L1 shards\n";
        } else {
            std::cout << "Memory layout: interleaved\n";
        }

        Tensor a;
        Tensor b;
        Tensor c;
        Tensor host_a;
        Tensor host_b;
        Tensor host_c;
        if (measurement_mode == "end-to-end") {
            {
                ZoneScopedN("host_input_creation");
                host_a = ttnn::full(ttnn::Shape({M, K}), 0.25f, dtype, input_layout);
                host_b = ttnn::full(ttnn::Shape({K, N}), 0.25f, dtype, input_layout);
                host_c = ttnn::full(ttnn::Shape({M, N}), 0.5f, dtype, input_layout);
            }
        } else {
            ZoneScopedN("device_input_creation");
            a = ttnn::rand(ttnn::Shape({M, K}), *device, dtype, input_layout);
            b = ttnn::rand(ttnn::Shape({K, N}), *device, dtype, input_layout);
            c = ttnn::full(ttnn::Shape({M, N}), 0.5f, dtype, input_layout,
                std::optional<std::reference_wrapper<MeshDevice>>(std::ref(*device)));
        }
        std::cout << "Measurement mode: " << measurement_mode << "\n";
        if (measurement_mode == "end-to-end") {
            std::cout << "End-to-end protocol: repeated-transfer-v2\n";
        }

        if (measurement_mode == "kernel") {
            benchmark::profiling::signpost("layout_conversion", "start");
            if (input_layout == Layout::ROW_MAJOR) {
                ZoneScopedN("layout_conversion_row_major_to_tile");
                a = ttnn::to_layout(a, Layout::TILE, dtype, a_mem_cfg);
                b = ttnn::to_layout(b, Layout::TILE, dtype, b_mem_cfg);
                c = ttnn::to_layout(c, Layout::TILE, dtype, output_mem_cfg);
            } else if (memory_layout == "sharded") {
                ZoneScopedN("layout_conversion_interleaved_to_sharded");
                a = ttnn::to_memory_config(a, a_mem_cfg);
                b = ttnn::to_memory_config(b, b_mem_cfg);
                c = ttnn::to_memory_config(c, output_mem_cfg);
            }
            benchmark::profiling::signpost("layout_conversion", "end");
        }

        auto run_op = [&]() -> Tensor {
            if (op == BenchmarkOp::ADDMM) {
                return ttnn::addmm(
                    c,
                    a,
                    b,
                    beta,
                    alpha,
                    std::optional<const tt::tt_metal::MemoryConfig>(output_mem_cfg),
                    std::optional<const DataType>(dtype),
                    std::nullopt,
                    std::optional<const ttnn::ComputeKernelConfig>(compute_cfg),
                    std::optional<const ttnn::types::CoreGrid>(compute_grid),
                    std::nullopt
                );
            }

            return ttnn::matmul(
                a,
                b,
                false,
                false,
                std::optional<const tt::tt_metal::MemoryConfig>(output_mem_cfg),
                std::optional<const DataType>(dtype),
                std::nullopt,
                std::nullopt,
                std::optional<const ttnn::ComputeKernelConfig>(compute_cfg),
                std::optional<const ttnn::types::CoreGrid>(compute_grid),
                std::nullopt
            );
        };

        Tensor out;
        const uint8_t cq_id = 0;
        auto& cq = device->mesh_command_queue(cq_id);

        if (measurement_mode == "end-to-end") {
            // Warm up the same complete pipeline that is measured below. This
            // prevents a single cold allocation/transfer from representing H2D.
            for (uint32_t i = 0; i < warmup_iterations; ++i) {
                if (i == 0) {
                    ZoneScopedN("first_run_host_to_device_transfer");
                    a = host_a.to_device(device.get(), DRAM_MEMORY_CONFIG);
                    b = host_b.to_device(device.get(), DRAM_MEMORY_CONFIG);
                    c = host_c.to_device(device.get(), DRAM_MEMORY_CONFIG);
                    tt::tt_metal::distributed::Finish(cq);
                } else {
                    ZoneScopedN("warmup_host_to_device_transfer");
                    a = host_a.to_device(device.get(), DRAM_MEMORY_CONFIG);
                    b = host_b.to_device(device.get(), DRAM_MEMORY_CONFIG);
                    c = host_c.to_device(device.get(), DRAM_MEMORY_CONFIG);
                    tt::tt_metal::distributed::Finish(cq);
                }
                if (input_layout == Layout::ROW_MAJOR) {
                    a = ttnn::to_layout(a, Layout::TILE, dtype, a_mem_cfg);
                    b = ttnn::to_layout(b, Layout::TILE, dtype, b_mem_cfg);
                    c = ttnn::to_layout(c, Layout::TILE, dtype, output_mem_cfg);
                    tt::tt_metal::distributed::Finish(cq);
                } else if (memory_layout == "sharded") {
                    a = ttnn::to_memory_config(a, a_mem_cfg);
                    b = ttnn::to_memory_config(b, b_mem_cfg);
                    c = ttnn::to_memory_config(c, output_mem_cfg);
                    tt::tt_metal::distributed::Finish(cq);
                }
                if (i == 0) {
                    benchmark::profiling::signpost("first_run_compile_and_execute", "start");
                    {
                        ZoneScopedN("first_run_dispatch");
                        out = run_op();
                    }
                    {
                        ZoneScopedN("first_run_synchronize");
                        tt::tt_metal::distributed::Finish(cq);
                    }
                    benchmark::profiling::signpost("first_run_compile_and_execute", "end");
                } else {
                    {
                        ZoneScopedN("warmup_dispatch");
                        out = run_op();
                    }
                    {
                        ZoneScopedN("warmup_synchronize");
                        tt::tt_metal::distributed::Finish(cq);
                    }
                }
                {
                    ZoneScopedN("warmup_device_to_host_transfer");
                    auto host_output = out.cpu(/*blocking=*/true);
                    (void)host_output;
                }
            }

            benchmark::profiling::signpost("steady_state", "start");
            for (uint32_t i = 0; i < run_iterations; ++i) {
                {
                    ZoneScopedN("host_to_device_transfer");
                    a = host_a.to_device(device.get(), DRAM_MEMORY_CONFIG);
                    b = host_b.to_device(device.get(), DRAM_MEMORY_CONFIG);
                    c = host_c.to_device(device.get(), DRAM_MEMORY_CONFIG);
                    tt::tt_metal::distributed::Finish(cq);
                }
                benchmark::profiling::signpost("layout_conversion", "start");
                if (input_layout == Layout::ROW_MAJOR) {
                    ZoneScopedN("layout_conversion_row_major_to_tile");
                    a = ttnn::to_layout(a, Layout::TILE, dtype, a_mem_cfg);
                    b = ttnn::to_layout(b, Layout::TILE, dtype, b_mem_cfg);
                    c = ttnn::to_layout(c, Layout::TILE, dtype, output_mem_cfg);
                    tt::tt_metal::distributed::Finish(cq);
                } else if (memory_layout == "sharded") {
                    ZoneScopedN("layout_conversion_interleaved_to_sharded");
                    a = ttnn::to_memory_config(a, a_mem_cfg);
                    b = ttnn::to_memory_config(b, b_mem_cfg);
                    c = ttnn::to_memory_config(c, output_mem_cfg);
                    tt::tt_metal::distributed::Finish(cq);
                }
                benchmark::profiling::signpost("layout_conversion", "end");
                {
                    ZoneScopedN("steady_state_dispatch");
                    out = run_op();
                }
                {
                    ZoneScopedN("steady_state_synchronize");
                    tt::tt_metal::distributed::Finish(cq);
                }
                {
                    ZoneScopedN("device_to_host_transfer");
                    auto host_output = out.cpu(/*blocking=*/true);
                    (void)host_output;
                }
            }
            benchmark::profiling::signpost("steady_state", "end");
            benchmark::energy::work(run_iterations, 2ULL * M * K * N, "flop");
        } else {
            uint32_t completed_warmups = 0;
            if (warmup_iterations > 0) {
                benchmark::profiling::signpost("first_run_compile_and_execute", "start");
                {
                    ZoneScopedN("first_run_dispatch");
                    out = run_op();
                }
                {
                    ZoneScopedN("first_run_synchronize");
                    tt::tt_metal::distributed::Finish(cq);
                }
                benchmark::profiling::signpost("first_run_compile_and_execute", "end");
                completed_warmups = 1;
            }

            benchmark::profiling::signpost("remaining_warmup", "start");
            for (uint32_t i = completed_warmups; i < warmup_iterations; ++i) {
                ZoneScopedN("warmup_dispatch");
                out = run_op();
            }
            {
                ZoneScopedN("warmup_synchronize");
                tt::tt_metal::distributed::Finish(cq);
            }
            benchmark::profiling::signpost("remaining_warmup", "end");

            benchmark::profiling::signpost("steady_state", "start");
            for (uint32_t i = 0; i < run_iterations; ++i) {
                ZoneScopedN("steady_state_dispatch");
                out = run_op();
            }
            {
                ZoneScopedN("steady_state_synchronize");
                tt::tt_metal::distributed::Finish(cq);
            }
            benchmark::profiling::signpost("steady_state", "end");
            benchmark::energy::work(run_iterations, 2ULL * M * K * N, "flop");

            if (run_iterations > 0 || warmup_iterations > 0) {
                ZoneScopedN("device_to_host_transfer");
                auto host_output = out.cpu(/*blocking=*/true);
                (void)host_output;
            }
        }

        std::cout << "Completed " << run_iterations << " "
                  << (op == BenchmarkOp::ADDMM ? "addmm" : "matmul")
                  << " trace replay iterations successfully.\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        ttnn::close_device(*device);
        return -1;
    }

    ttnn::close_device(*device);
    return 0;
}
