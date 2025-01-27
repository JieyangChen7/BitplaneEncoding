#include <iostream>
#include <chrono>
#include "cpu_encoder.h"
#include "gpu_encoder_v1.h"


using namespace std::chrono;

template <typename T, typename T_fp, typename T_sfp, typename T_bp>
void test_cpu(int n, int b) {


    printf("Begin test\n");
    T * data = new T[n];
    T * decoded_data = new T[n];
    int bitplane_length = n/(sizeof(T_bp)*8)*2;
    T_bp * encoded_bitplanes = new T_bp[bitplane_length*b];
    T max_abs = 0;
    int exp = 0;
    for (int i = 0; i < n; i++) {
        data[i] = i;
        if (fabs(data[i] > max_abs)) {
            max_abs = fabs(data[i]);
        }
        
    }
    frexp(max_abs, &exp);
    
    printf("Encode: \n");
    auto start = std::chrono::high_resolution_clock::now();
    cpu::encode<T, T_fp, T_sfp, T_bp>(data, encoded_bitplanes, n, b, exp);
    auto stop = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<microseconds>(stop - start).count();
    printf("time: %f s, %f GB/s\n", float(time)/1e6, float(n * sizeof(T))/1e3/time);


    printf("Decode: \n");
    start = std::chrono::high_resolution_clock::now();
    cpu::decode<T, T_fp, T_sfp, T_bp>(decoded_data, encoded_bitplanes, n, b, exp);
    stop = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<microseconds>(stop - start).count();
    printf("time: %f s, %f GB/s\n", float(time)/1e6, float(n * sizeof(T))/1e3/time);

    bool pass = true;
    for (int i = 0; i < n; i++) {
        if (data[i] !=  decoded_data[i]) {
            pass = false;
            // printf("%f, %f\n", data[i], decoded_data[i]);
            break;
        }
    }
    
    delete[] data;
    delete[] decoded_data;
    delete[] encoded_bitplanes;

    printf("Pass: %d\n", pass);

    printf("Finish test\n");
}
template <typename T, typename T_fp, typename T_sfp, typename T_bp>
void test_gpu_v1(int n, int b) {
    printf("Begin test\n");
    T * data = new T[n];
    T * decoded_data = new T[n];
    int bitplane_length = n/(sizeof(T_bp)*8)*2;
    T_bp * encoded_bitplanes = new T_bp[bitplane_length*b];

    T * d_data;
    T * d_decoded_data;
    T_bp * d_encoded_bitplanes;
    cudaMalloc(&d_data, n*sizeof(T));
    cudaMalloc(&d_decoded_data, n*sizeof(T));
    cudaMalloc(&d_encoded_bitplanes, bitplane_length*b*sizeof(T_bp));

    T max_abs = 0;
    int exp = 0;
    for (int i = 0; i < n; i++) {
        data[i] = i;
        if (fabs(data[i] > max_abs)) {
            max_abs = fabs(data[i]);
        }
        
    }
    frexp(max_abs, &exp);

    cudaMemcpy(d_data, data, n*sizeof(T), cudaMemcpyDefault);

    // warmup gpu
    for (int i = 0; i < 10; i++) {
        gpu_v1::encode<T, T_fp, T_sfp, T_bp> <<<(n-1)/256+1, 256>>>(d_data, d_encoded_bitplanes, n, b, exp);
    }

    printf("Encode: \n");
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
        gpu_v1::encode<T, T_fp, T_sfp, T_bp> <<<(n-1)/256+1, 256>>>(d_data, d_encoded_bitplanes, n, b, exp);
    }
    cudaDeviceSynchronize();
    auto stop = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<microseconds>(stop - start).count() / 10;
    printf("time: %f s, %f GB/s\n", float(time)/1e6, float(n * sizeof(T))/1e3/time);

    cudaMemcpy(encoded_bitplanes, d_encoded_bitplanes, bitplane_length*b*sizeof(T_bp), cudaMemcpyDefault);
    cudaMemcpy(d_encoded_bitplanes, encoded_bitplanes, bitplane_length*b*sizeof(T_bp), cudaMemcpyDefault);

    printf("Decode: \n");
    cudaDeviceSynchronize();
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
        gpu_v1::decode<T, T_fp, T_sfp, T_bp> <<<(n-1)/256+1, 256>>>(d_decoded_data, d_encoded_bitplanes, n, b, exp);
    }
    cudaDeviceSynchronize();
    stop = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<microseconds>(stop - start).count() / 10;
    printf("time: %f s, %f GB/s\n", float(time)/1e6, float(n * sizeof(T))/1e3/time);

    cudaMemcpy(decoded_data, d_decoded_data, n*sizeof(T), cudaMemcpyDefault);

    bool pass = true;
    for (int i = 0; i < n; i++) {
        // printf("%f, %f\n", data[i], decoded_data[i]);
        if (data[i] !=  decoded_data[i]) {
            pass = false;
            break;
        }
    }

    delete[] data;
    delete[] decoded_data;
    delete[] encoded_bitplanes;

    cudaFree(d_data);
    cudaFree(d_decoded_data);
    cudaFree(d_encoded_bitplanes);

    printf("Pass: %d\n", pass);

    printf("Finish test\n");
}


template <typename T>
void test(int n, int b) {
    using T_sfp = typename std::conditional<std::is_same<T, double>::value,
                                          int64_t, int32_t>::type;
    using T_fp = typename std::conditional<std::is_same<T, double>::value,
                                            uint64_t, uint32_t>::type;
    using T_bp = uint32_t;


    test_cpu<float, T_fp, T_sfp, T_bp>(n, b);
    test_gpu_v1<float, T_fp, T_sfp, T_bp>(n, b);
}


int main() {
    int n = 32 * 1024; // needs to be a multiple of 32
    int b = 32;
    test<float>(n, b);
}