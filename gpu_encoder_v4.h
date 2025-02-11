namespace gpu_v4 {
#define BATCHES_PER_BLOCK 16

/*
v: input data
|-----sizeof(T)*8-----|
.
.
n
.
.
.
|-----sizeof(T)*8-----|

encoded: encoded data
|-----n/(sizeof(T_bp)*8)-----|
.
.
b
.
.
.
|-----n/(sizeof(T_bp)*8)-----|

n: number of elements in input data
b: number of bitplanes to encode
*/


    template<typename T>
    __device__ void print_bits(T v, int num_bits, bool reverse = false) {
        for (int j = 0; j < num_bits; j++) {
            if (!reverse)
                printf("%u", (v >> (num_bits - 1 - j)) & 1u);
            else
                printf("%u", (v >> j) & 1u);
        }
        printf("\n");
    }


    // resultBytes = b * BATCHES_PER_BLOCK * 2 * sizeof(T_bp)
    // shared_ memory_size = dataBytes + signBytes + resultBytes
    template<typename T, typename T_fp, typename T_sfp, typename T_bp>
    __global__ void encode(T *v, T_bp *encoded_bitplanes, int n, int b, int exp) {
        constexpr int batch_size = sizeof(T_bp) * 8;
        const int num_batches = (n + batch_size - 1) / batch_size;

        extern __shared__ unsigned char sMem[];
        size_t dataBytes = batch_size * sizeof(T_fp);
        size_t signBytes = batch_size * sizeof(T_sfp);

        T_fp *shared_data = reinterpret_cast<T_fp *>(sMem);
        T_sfp *shared_sign = reinterpret_cast<T_sfp *>(sMem + dataBytes);
        T_bp *result = reinterpret_cast<T_bp *>(sMem + dataBytes + signBytes);

        const int tid = threadIdx.x;

        // empty the result shared memory
        int resultSize = b * (BATCHES_PER_BLOCK * 2);
        for (int idx = tid; idx < resultSize; idx += blockDim.x) {
            result[idx] = 0;
        }

        // Get range and perform data fetch
        int starting_batch = blockIdx.x * BATCHES_PER_BLOCK;
        int ending_batch = min(starting_batch + BATCHES_PER_BLOCK, num_batches);

        for (int i = starting_batch; i < ending_batch; i++) {
            T data = v[i * batch_size + tid];
            T shifted_data = ldexp(data, b - exp);
            shared_data[tid] = (T_fp) fabs(shifted_data);
            shared_sign[tid] = ((T_sfp) signbit(data));
            // After loading when should have a whole batch of data in our shared memory
            __syncthreads();

            int local_batch = i - starting_batch;
            if (tid < batch_size) {
                result[(i - starting_batch) * 2 + 1] |= shared_sign[tid] << tid;
            }

            for (int bp = 0; bp < b; bp++) {
                int local_index_data = bp * (BATCHES_PER_BLOCK * 2) + local_batch * 2;
                result[local_index_data] |= (((shared_data[tid] >> (b - 1 - bp)) & 1) << tid);
            }
            __syncthreads();
        }

        __syncthreads();

        for (int bp = 0; bp < b; bp++) {
            for (int idx = tid; idx < BATCHES_PER_BLOCK * 2; idx += blockDim.x) {
                int localIndex = bp * (BATCHES_PER_BLOCK * 2) + idx;
                int local_batch = idx / 2;
                int offset      = idx % 2;

                int global_batch = starting_batch + local_batch;
                if (global_batch < num_batches) {
                    int globalIndex = bp * (num_batches * 2) + global_batch * 2 + offset;
                    encoded_bitplanes[globalIndex] = result[localIndex];
                }
            }
            __syncthreads();
        }
    }

    template<typename T, typename T_fp, typename T_sfp, typename T_bp>
    __global__ void decode(T *v, const T_bp *encoded_bitplanes, int batches_per_warp, int n, int b, int exp) {
        constexpr int batch_size = sizeof(T_bp) * 8;
        const int num_batches = (n + batch_size - 1) / batch_size;

        const int tid = threadIdx.x;
        const int warp_id = tid / 32;
        const int lane_id = tid % 32;

        const int warps_per_block = blockDim.x / 32;
        const int starting_batch = (blockIdx.x * warps_per_block + warp_id) * batches_per_warp;

        for (int i = 0; i < batches_per_warp; i++) {
            const int batch_idx = starting_batch + i;
            if (batch_idx >= num_batches) break;

            const int global_idx = batch_idx * batch_size + lane_id;
            if (global_idx >= n) continue;

            const T_bp sign_mask = encoded_bitplanes[batch_idx * 2 + 1];
            const bool sign = (sign_mask >> lane_id) & 1;

            T_fp fp_val = 0;
            for (int bp = 0; bp < b; ++bp) {
                const int plane_idx = bp * num_batches * 2 + batch_idx * 2;
                const T_bp data_mask = encoded_bitplanes[plane_idx];
                const unsigned bit = (data_mask >> lane_id) & 1;
                fp_val |= (bit << (b - 1 - bp));
            }

            const T shifted = ldexp(static_cast<T>(fp_val), -b + exp);
            v[global_idx] = sign ? -shifted : shifted;
        }
    }


} // namespace gpu_v4