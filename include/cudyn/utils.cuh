#pragma once

typedef struct {
    uint64_t total_tasks;
    uint64_t grid_dimensions;
    uint64_t block_dimensions;
    
} KernelConfig;


namespace grid_configruation {
    __host__ KernelConfig manual_grid_configuration(long long total_tasks, int blocks_count, int threads_per_block){
        KernelConfig config;
        config.total_tasks = total_tasks;
        config.grid_dimensions = blocks_count;
        config.block_dimensions = threads_per_block;

        validate_KernelConfig();
    }
    
        // namespace for functions that do not define the public facing API
        namespace details {

            __host__ constexpr void validate_KernelConfig(KernelConfig config){
                static_assert(config.total_tasks > 0, "[cudyn bad argument] - The number of total tasks must be greater than 0");
                static_assert(config.grid_dimensions > 0, "[cudyn bad argument] - The number of blocks must be greater than 0");
                static_assert(config.block_dimensions > 0, "[cudyn bad argument] - The number of threads per block must be greater than 0");
            }
        }
    
}