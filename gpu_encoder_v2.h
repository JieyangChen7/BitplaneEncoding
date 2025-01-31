namespace gpu_v2 {

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


template <typename T>
__device__ void print_bits(T v, int num_bits, bool reverse = false) {
    for (int j = 0; j < num_bits; j++) {
        if (!reverse)
            printf("%u", (v >> (num_bits - 1 - j)) & 1u);
        else
            printf("%u", (v >> j) & 1u);
    }
    printf("\n");
}


template <typename T, typename T_fp, typename T_sfp, typename T_bp>
__global__ void encode(T* v, T_bp* encoded_bitplanes, int n, int b, int exp) {
  constexpr int batch_size = sizeof(T_bp) * 8;  // Most likely 32 here
  const int num_batches = (n + batch_size - 1) / batch_size;
  const int batch_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int global_idx = batch_idx * batch_size + tid;

  T data = (global_idx < n) ? v[global_idx] : T(0);
  T shifted = ldexp(data, b - exp);
  bool sign = signbit(data);


  unsigned sign_mask = __ballot_sync(0xFFFFFFFF, sign? 1 : 0);
  if (tid == 0) {
    encoded_bitplanes[batch_idx * 2 + 1] = static_cast<T_bp>(sign_mask);
  }

  // Process bit planes of data
  for (int bp = 0; bp < b; ++bp) {
    unsigned bit = (static_cast<T_bp>(shifted) >> (b - 1 - bp)) & 1;
    unsigned data_mask = __ballot_sync(0xFFFFFFFF, bit);
    if (tid == 0) {
      const int plane_idx = bp * num_batches * 2 + batch_idx * 2;
      encoded_bitplanes[plane_idx] = static_cast<T_bp>(data_mask);
    }
  }
}

template <typename T, typename T_fp, typename T_sfp, typename T_bp>
__global__ void decode(T* v, const T_bp*  encoded_bitplanes, int n, int b, int exp) {
  constexpr int batch_size = sizeof(T_bp) * 8;
  const int num_batches = (n + batch_size - 1) / batch_size;
  const int batch_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int global_idx = batch_idx * batch_size + tid;

  if (global_idx >= n) return;

  const T_bp sign_mask = encoded_bitplanes[batch_idx * 2 + 1];
  const bool sign = (sign_mask >> tid) & 1;

  T_fp fp_val = 0;

  for (int bp = 0; bp < b; ++bp) {
    const int plane_idx = bp * num_batches * 2 + batch_idx * 2;
    const T_bp data_mask = encoded_bitplanes[plane_idx];
    const unsigned bit = (data_mask >> tid) & 1;
    fp_val |= (bit << (b - 1 - bp));
  }

  const T shifted = ldexp(static_cast<T>(fp_val), -b + exp);
  v[global_idx] = sign ? -shifted : shifted;
}

} // namespace gpu_v2