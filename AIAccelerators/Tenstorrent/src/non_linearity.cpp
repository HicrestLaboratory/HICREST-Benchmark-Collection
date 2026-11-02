/*
Activation benchmark for TTNN relu / gelu / softmax

Usage:
    ./activation_bench H W [options]

Required arguments:
    H                      Tensor height / rows
    W                      Tensor width / columns

Optional arguments:
    --op <value>
        Operation to benchmark.

        Valid values:
            relu
            gelu
            softmax

        Default: relu

    --softmax-dim <int>
        Dimension for softmax.
        Default: -1

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

    --layout <value>
        Input tensor layout.

        Valid values:
            tile
            row_major, row-major, rm

        Default: tile

        Notes:
            - These ops generally run fastest from TILE layout.
            - If ROW_MAJOR is selected, this benchmark converts to TILE before execution.
            - BFLOAT8_B/BFLOAT4_B are forced to TILE.

    --memory-layout <value>
        Tensor memory layout.

        Valid values:
            interleaved
            sharded

        Default: interleaved

    --measurement-mode <value>
        Measurement boundary: kernel or end-to-end.
        End-to-end creates a host input once, warms up the complete transfer and
        execution pipeline, then measures H2D -> operation -> D2H per iteration.
        Default: kernel

    --shard-grid-y <int>
        Shard grid height used only when --memory-layout=sharded.
        Default: 8

    --shard-grid-x <int>
        Shard grid width used only when --memory-layout=sharded.
        Default: 8

Examples:

    ./activation_bench 4096 4096 --op relu --dtype bf16
    ./activation_bench 4096 4096 --op gelu --dtype bf16
    ./activation_bench 4096 4096 --op softmax --softmax-dim -1 --dtype bf16
*/

#include "energy_markers.hpp"
#include <argparse/argparse.hpp>

#include <ttnn/device.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/types.hpp>

#include <ttnn/operations/creation/creation.hpp>
#include <ttnn/operations/rand/rand.hpp>
#include <ttnn/operations/core/core.hpp>

#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <ttnn/operations/normalization/softmax/softmax.hpp>

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
    RELU,
    GELU,
    SOFTMAX
};

static std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

static BenchmarkOp parse_op(const std::string& value) {
    const auto s = lower(value);
    if (s == "relu") return BenchmarkOp::RELU;
    if (s == "gelu") return BenchmarkOp::GELU;
    if (s == "softmax") return BenchmarkOp::SOFTMAX;
    throw std::runtime_error("Unsupported --op: " + value);
}

static const char* op_name(BenchmarkOp op) {
    switch (op) {
        case BenchmarkOp::RELU: return "relu";
        case BenchmarkOp::GELU: return "gelu";
        case BenchmarkOp::SOFTMAX: return "softmax";
    }
    return "unknown";
}

static DataType parse_dtype(const std::string& value) {
    const auto s = lower(value);
    if (s == "bf16" || s == "bfloat16") return DataType::BFLOAT16;
    if (s == "fp32" || s == "float32") return DataType::FLOAT32;
    if (s == "bfp8" || s == "bfloat8_b") return DataType::BFLOAT8_B;
    if (s == "bfp4" || s == "bfloat4_b") return DataType::BFLOAT4_B;
    throw std::runtime_error("Unsupported --dtype: " + value);
}

static Layout parse_layout(const std::string& value) {
    const auto s = lower(value);
    if (s == "tile") return Layout::TILE;
    if (s == "row_major" || s == "row-major" || s == "rm") return Layout::ROW_MAJOR;
    throw std::runtime_error("Unsupported --layout: " + value);
}

int main(int argc, char* argv[]) {
    benchmark::energy::Interval application_interval("HICREST_APPLICATION_INTERVAL");
    argparse::ArgumentParser program("TTNN relu/gelu/softmax benchmark");

    program.add_argument("H").required().scan<'i', int>();
    program.add_argument("W").required().scan<'i', int>();

    program.add_argument("--op")
        .help("Operation to benchmark: relu, gelu, or softmax")
        .default_value(std::string("relu"));

    program.add_argument("--softmax-dim")
        .help("Dimension for softmax")
        .default_value(-1)
        .scan<'i', int>();

    program.add_argument("-w", "--warmup").default_value(5).scan<'i', int>();
    program.add_argument("-r", "--run").default_value(10).scan<'i', int>();

    program.add_argument("--dtype").default_value(std::string("bf16"));
    program.add_argument("--layout").default_value(std::string("tile"));
    program.add_argument("--memory-layout").default_value(std::string("interleaved"));
    program.add_argument("--measurement-mode")
        .help("Measurement boundary: kernel or end-to-end (explicit host creation and H2D)")
        .default_value(std::string("kernel"));

    program.add_argument("--shard-grid-y").default_value(8).scan<'i', int>();
    program.add_argument("--shard-grid-x").default_value(8).scan<'i', int>();

    constexpr int device_id = 0;
    std::shared_ptr<MeshDevice> device = ttnn::open_mesh_device(device_id);

    try {
        program.parse_args(argc, argv);

        const uint32_t H = static_cast<uint32_t>(program.get<int>("H"));
        const uint32_t W = static_cast<uint32_t>(program.get<int>("W"));

        const uint32_t warmup_iterations =
            static_cast<uint32_t>(program.get<int>("--warmup"));
        const uint32_t run_iterations =
            static_cast<uint32_t>(program.get<int>("--run"));

        const auto op = parse_op(program.get<std::string>("--op"));
        const auto dtype = parse_dtype(program.get<std::string>("--dtype"));
        const auto requested_layout = parse_layout(program.get<std::string>("--layout"));
        const int softmax_dim = program.get<int>("--softmax-dim");
        const auto memory_layout = lower(program.get<std::string>("--memory-layout"));
        const auto measurement_mode = lower(program.get<std::string>("--measurement-mode"));
        if (measurement_mode != "kernel" && measurement_mode != "end-to-end") {
            throw std::runtime_error("Unsupported --measurement-mode: " + measurement_mode);
        }
        if (measurement_mode == "end-to-end" && warmup_iterations == 0) {
            throw std::runtime_error("end-to-end mode requires at least one warmup iteration");
        }

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

        tt::tt_metal::MemoryConfig output_mem_cfg = DRAM_MEMORY_CONFIG;
        tt::tt_metal::MemoryConfig input_mem_cfg = DRAM_MEMORY_CONFIG;

        if (memory_layout == "sharded") {
            using namespace tt::tt_metal;

            const CoreRangeSet shard_grid{
                CoreRange{
                    CoreCoord{0, 0},
                    CoreCoord{shard_grid_x - 1, shard_grid_y - 1}
                }
            };

            const std::array<uint32_t, 2> shard_shape = {
                static_cast<uint32_t>(((H + shard_grid_y * 32 - 1) / (shard_grid_y * 32)) * 32),
                static_cast<uint32_t>(((W + shard_grid_x * 32 - 1) / (shard_grid_x * 32)) * 32)
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
            input_mem_cfg = output_mem_cfg;
        } else if (memory_layout != "interleaved") {
            throw std::runtime_error("Unsupported --memory-layout: " + memory_layout);
        }

        std::cout << "H=" << H << ", W=" << W << "\n";
        std::cout << "Operation: " << op_name(op) << "\n";
        if (op == BenchmarkOp::SOFTMAX) {
            std::cout << "Softmax dim: " << softmax_dim << "\n";
        }

        if (memory_layout == "sharded") {
            std::cout << "Shard grid: "
                      << shard_grid_y << " x " << shard_grid_x
                      << " = " << shard_grid_y * shard_grid_x
                      << " L1 shards\n";
        } else {
            std::cout << "Memory layout: interleaved\n";
        }

        Tensor input;
        Tensor host_input;
        if (measurement_mode == "end-to-end") {
            {
                ZoneScopedN("host_input_creation");
                host_input = ttnn::full(ttnn::Shape({H, W}), 0.25f, dtype, input_layout);
            }
        } else {
            ZoneScopedN("device_input_creation");
            input = ttnn::rand(ttnn::Shape({H, W}), *device, dtype, input_layout);
        }
        std::cout << "Measurement mode: " << measurement_mode << "\n";
        if (measurement_mode == "end-to-end") {
            std::cout << "End-to-end protocol: repeated-transfer-v2\n";
        }

        if (measurement_mode == "kernel") {
            benchmark::profiling::signpost("layout_conversion", "start");
            if (input_layout == Layout::ROW_MAJOR) {
                ZoneScopedN("layout_conversion_row_major_to_tile");
                input = ttnn::to_layout(input, Layout::TILE, dtype, input_mem_cfg);
            } else if (memory_layout == "sharded") {
                ZoneScopedN("layout_conversion_interleaved_to_sharded");
                input = ttnn::to_memory_config(input, input_mem_cfg);
            }
            benchmark::profiling::signpost("layout_conversion", "end");
        }

        auto run_op = [&]() -> Tensor {
            if (op == BenchmarkOp::RELU) {
                return ttnn::relu(
                    input,
                    std::optional<const tt::tt_metal::MemoryConfig>(output_mem_cfg));
            }
            
            if (op == BenchmarkOp::GELU) {
                return ttnn::gelu(
                    input,
                    true,
                    std::optional<const tt::tt_metal::MemoryConfig>(output_mem_cfg));
            }

            return ttnn::softmax(
                input,
                softmax_dim,
                std::optional<const tt::tt_metal::MemoryConfig>(output_mem_cfg));
        };

        Tensor out;
        const uint8_t cq_id = 0;
        auto& cq = device->mesh_command_queue(cq_id);

        if (measurement_mode == "end-to-end") {
            for (uint32_t i = 0; i < warmup_iterations; ++i) {
                if (i == 0) {
                    ZoneScopedN("first_run_host_to_device_transfer");
                    input = host_input.to_device(device.get(), DRAM_MEMORY_CONFIG);
                    tt::tt_metal::distributed::Finish(cq);
                } else {
                    ZoneScopedN("warmup_host_to_device_transfer");
                    input = host_input.to_device(device.get(), DRAM_MEMORY_CONFIG);
                    tt::tt_metal::distributed::Finish(cq);
                }
                if (input_layout == Layout::ROW_MAJOR) {
                    input = ttnn::to_layout(input, Layout::TILE, dtype, input_mem_cfg);
                    tt::tt_metal::distributed::Finish(cq);
                } else if (memory_layout == "sharded") {
                    input = ttnn::to_memory_config(input, input_mem_cfg);
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
                    input = host_input.to_device(device.get(), DRAM_MEMORY_CONFIG);
                    tt::tt_metal::distributed::Finish(cq);
                }
                benchmark::profiling::signpost("layout_conversion", "start");
                if (input_layout == Layout::ROW_MAJOR) {
                    ZoneScopedN("layout_conversion_row_major_to_tile");
                    input = ttnn::to_layout(input, Layout::TILE, dtype, input_mem_cfg);
                    tt::tt_metal::distributed::Finish(cq);
                } else if (memory_layout == "sharded") {
                    ZoneScopedN("layout_conversion_interleaved_to_sharded");
                    input = ttnn::to_memory_config(input, input_mem_cfg);
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
            benchmark::energy::work(run_iterations, std::uint64_t(H) * W, "elements");
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
            for (uint32_t i = completed_warmups; i < warmup_iterations; ++i) {
                ZoneScopedN("warmup_dispatch");
                out = run_op();
            }
            {
                ZoneScopedN("warmup_synchronize");
                tt::tt_metal::distributed::Finish(cq);
            }
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
            benchmark::energy::work(run_iterations, std::uint64_t(H) * W, "elements");
            if (run_iterations > 0 || warmup_iterations > 0) {
                ZoneScopedN("device_to_host_transfer");
                auto host_output = out.cpu(/*blocking=*/true);
                (void)host_output;
            }
        }

        std::cout << "Completed " << run_iterations << " "
                  << op_name(op)
                  << " iterations successfully.\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        ttnn::close_device(*device);
        return -1;
    }

    ttnn::close_device(*device);
    return 0;
}
