#pragma once

#include <cstdint>

namespace pankey{

    namespace Library{

        namespace AI{

            struct PkeConfig{
                float learningRate = 0.05f;
                uint32_t batchSize = 1;
                uint32_t microBatchSize = 1;
                uint32_t steps = 1;
                uint32_t seed = 0;
                uint64_t memoryBudgetBytes = 6ull * 1024ull * 1024ull * 1024ull;
            };

        }

    }

}
