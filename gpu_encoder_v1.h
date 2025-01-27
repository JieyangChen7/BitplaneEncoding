
namespace gpu_v1 {


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
|-----n/(sizeof(T_bitplane)*8)-----|
.
.
b
.
.
.
|-----n/(sizeof(T_bitplane)*8)-----|

n: number of elements in input data
b: number of bitplanes to encode
*/

template <typename T>
__device__ void print_bits(T v, int num_bits, bool reverse = false) {
  for (int j = 0; j < num_bits; j++) {
    if (!reverse)
      printf("%u", (v >> num_bits - 1 - j) & 1u);
    else
      printf("%u", (v >> j) & 1u);
  }
  printf("\n");
}

template <typename T_fp, typename T_bp>
__device__ void encode_batch(T_fp *v, T_bp *encoded, int batch_size, int b) {
    for (int bp_idx = 0; bp_idx < b; bp_idx ++) {
        T_bp buffer = 0;
        for (int data_idx = 0; data_idx < batch_size; data_idx++) {
            T_bp bit = (v[data_idx] >> (sizeof(T_fp) * 8 - 1 - bp_idx)) & 1u;
            buffer += bit << sizeof(T_bp) * 8 - 1 - data_idx;
        }
        encoded[bp_idx] = buffer;
    }
}

template <typename T_fp, typename T_bp>
__device__ void decode_batch(T_fp *v, T_bp *encoded, int batch_size, int b) {
    for (int data_idx = 0; data_idx < batch_size; data_idx++) {
        T_fp buffer = 0;
        for (int bp_idx = 0; bp_idx < b; bp_idx ++) {
            T_fp bit = (encoded[bp_idx] >> (sizeof(T_bp) * 8 - 1 - data_idx)) & 1u;
            buffer += bit << (b - 1 - bp_idx);
        }
        v[data_idx] = buffer;
    }
}



template <typename T, typename T_fp, typename T_sfp, typename T_bp>
__global__ void encode (T *v, T_bp * encoded_bitplanes, int n, int b, int exp) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    constexpr int batch_size = sizeof(T_bp) * 8;
    int num_batches = (n-1) / batch_size + 1;

    T shifted_data[batch_size];
    T_fp fp_data[batch_size];
    T_fp signs[batch_size];
    T_bp encoded_data[32];
    T_bp encoded_sign[32];

    if (batch_idx < num_batches) {
    // for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        for (int data_idx = 0; data_idx < batch_size; data_idx++) {
            T data = v[batch_idx * batch_size + data_idx];
            shifted_data[data_idx] = ldexp(data, b - exp);
            fp_data[data_idx] = (T_fp)fabs(shifted_data[data_idx]);
            signs[data_idx] = ((T_sfp)signbit(data)) << (sizeof(T_fp) * 8 - 1);
            // printf("%f: ", data); print_bits(fp_data[data_idx], b);
        }
        // encode data
        encode_batch(fp_data, encoded_data, batch_size, b);
        // encode sign
        encode_batch(signs, encoded_sign, batch_size, 1);
        for (int bp_idx = 0; bp_idx < b; bp_idx++) {
            encoded_bitplanes[bp_idx * num_batches * 2 + batch_idx * 2] = encoded_data[bp_idx];
            // print_bits(encoded_bitplanes[bp_idx * b + batch_idx * 2], batch_size);       
        }
        encoded_bitplanes[0 * num_batches * 2 + batch_idx * 2 + 1] = encoded_sign[0];
        // print_bits(encoded_bitplanes[0 * b + batch_idx * 2 + 1], batch_size);
    }
}


template <typename T, typename T_fp, typename T_sfp, typename T_bp>
__global__ void decode (T *v, T_bp * encoded_bitplanes, int n, int b, int exp) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    constexpr int batch_size = sizeof(T_bp) * 8;
    int num_batches = (n-1) / batch_size + 1;

    T shifted_data[batch_size];
    T_fp fp_data[batch_size];
    T_fp signs[batch_size];
    T_bp encoded_data[32];
    T_bp encoded_sign[32];

    // for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
    if (batch_idx < num_batches) {

        for (int bp_idx = 0; bp_idx < b; bp_idx++) {
            encoded_data[bp_idx] = encoded_bitplanes[bp_idx * num_batches * 2 + batch_idx * 2];
            // print_bits(encoded_data[bp_idx], batch_size);
        }
        encoded_sign[0] = encoded_bitplanes[0 * num_batches * 2 + batch_idx * 2 + 1];
        // print_bits(encoded_sign[0], batch_size);

        // encode data
        decode_batch(fp_data, encoded_data, batch_size, b);
        // encode sign
        decode_batch(signs, encoded_sign, batch_size, 1);
        for (int data_idx = 0; data_idx < batch_size; data_idx++) {
            
            T data  = ldexp((T)fp_data[data_idx], - b + exp);
            v[batch_idx * batch_size + data_idx] = signs[data_idx] ? -data : data;
            // printf("%f: ", data); print_bits(fp_data[data_idx], b);
        }
    }

}

}