/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

// need to enable unstable api
#define ROCBLAS_NO_DEPRECATED_WARNINGS
#define ROCBLAS_BETA_FEATURES_API
//#include "client_utility.hpp"
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <cassert>
#include <chrono>
#include <map>
#include <random>
#include <vector>
#include <iostream>
#include <thread> // for std::this_thread::sleep_for if needed

// Simple error checking macro for rocBLAS API calls
#define CHECK_ROCBLAS_ERROR(status)                                             \
    do {                                                                       \
        rocblas_status _status = (status);                                     \
        if(_status != rocblas_status_success) {                                \
            std::cerr << "rocBLAS error: " << _status                         \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while(0)

// Simple error checking macro for HIP API calls
#define CHECK_HIP_ERROR(status)                                                 \
    do {                                                                       \
        hipError_t _status = (status);                                         \
        if(_status != hipSuccess) {                                            \
            std::cerr << "HIP error: " << hipGetErrorString(_status)           \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while(0)

#define DIM1 12 // nproj
#define DIM2 20 // nst_linear
#define DIM3 23071 // npoints

#define rocblas_gemm_exM(...) rocblas_gemm_ex(__VA_ARGS__)

constexpr bool debug = false;

// Utility function to get current time in microseconds and synchronize the stream
inline double get_time_us_sync(hipStream_t stream)
{
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(now.time_since_epoch()).count();
}

template <typename T>
struct GEMMExParams
{
    // Group params for convenience
    rocblas_handle    handle;
    rocblas_operation transa;
    rocblas_operation transb;
    rocblas_int       m;
    rocblas_int       n;
    rocblas_int       k;
    T                 alpha;
    T                 beta;
    rocblas_datatype  input_type;
    rocblas_datatype  output_type;
    rocblas_datatype  compute_type;
    T*                da;
    T*                db;
    T*                dc;
    rocblas_int       lda;
    rocblas_int       ldb;
    rocblas_int       ldc;
};

template <typename T>
bool is_subset(std::vector<T> A, std::vector<T> B)
{
    std::sort(A.begin(), A.end());
    std::sort(B.begin(), B.end());
    return std::includes(A.begin(), A.end(), B.begin(), B.end());
}

template <typename T>
rocblas_int benchmark_solutions(std::vector<rocblas_int> const& solutions,
                                GEMMExParams<T> const&          gemmParams,
                                rocblas_int                     cold_calls = 2,
                                rocblas_int                     hot_calls  = 10)
{
// Note: `cold_calls` and 'hot_calls' defaults match rocblas-bench
//       Higher values give more consistent benchmarking results

// macros
#define GEMM_EX_ARGS_BM                                                                        \
    gemmParams.handle, gemmParams.transa, gemmParams.transb, gemmParams.m, gemmParams.n,       \
        gemmParams.k, &gemmParams.alpha, gemmParams.da, gemmParams.input_type, gemmParams.lda, \
        gemmParams.db, gemmParams.input_type, gemmParams.ldb, &gemmParams.beta, gemmParams.dc, \
        gemmParams.output_type, gemmParams.ldc, gemmParams.dc, gemmParams.output_type,         \
        gemmParams.ldc, gemmParams.compute_type, rocblas_gemm_algo_solution_index

    double         bestTime = std::numeric_limits<double>::max();
    rocblas_int    bestSol  = -1;
    rocblas_status status;
    for(auto sol : solutions)
    {
        // Check solution is valid
        status = rocblas_gemm_exM(GEMM_EX_ARGS_BM, sol, rocblas_gemm_flags_none);
        if(status == rocblas_status_invalid_value)
        {
            //std::cout << "Solution " << sol << " not valid for this problem." << std::endl;
            continue;
        }

        // warmup
        for(rocblas_int c = 0; c < cold_calls; ++c)
        {
            CHECK_ROCBLAS_ERROR(rocblas_gemm_exM(GEMM_EX_ARGS_BM, sol, rocblas_gemm_flags_none));
        }
        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(gemmParams.handle, &stream));
        double time = get_time_us_sync(stream); // in microseconds

        // timing loop
        for(rocblas_int c = 0; c < hot_calls; ++c)
        {
            CHECK_ROCBLAS_ERROR(rocblas_gemm_exM(GEMM_EX_ARGS_BM, sol, rocblas_gemm_flags_none));
        }
        time = get_time_us_sync(stream) - time;

        double avg_time = hot_calls ? (time / hot_calls) : 0;
        if constexpr (debug) std::cout << "Solution " << sol << ": " << avg_time << " us" << std::endl;

        // track winner
        if(avg_time < bestTime)
        {
            bestSol  = sol;
            bestTime = avg_time;
        }
    }
    if constexpr (debug) std::cout << "Winner: " << bestSol << " in " << bestTime << " us" << std::endl << std::endl;

    return bestSol;
}

// Helper for random initialization
template <typename T>
void fill_random(std::vector<T>& v)
{
    std::mt19937 rng;
    std::uniform_real_distribution<typename std::conditional<std::is_same<T, rocblas_float_complex>::value || std::is_same<T, rocblas_double_complex>::value, float, T>::type>
        dist(-0.5, 0.5);
    for(auto& x : v)
        x = dist(rng);
}

// Specialization for rocblas_float_complex
template <>
void fill_random<rocblas_float_complex>(std::vector<rocblas_float_complex>& v)
{
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for(auto& x : v)
        x = rocblas_float_complex{dist(rng), dist(rng)};
}

// Specialization for rocblas_double_complex
template <>
void fill_random<rocblas_double_complex>(std::vector<rocblas_double_complex>& v)
{
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    for(auto& x : v)
        x = rocblas_double_complex{dist(rng), dist(rng)};
}

// Helper for tolerance
template <typename T>
T get_tol() { return T(1e-4); }
template <>
rocblas_float_complex get_tol<rocblas_float_complex>() { return rocblas_float_complex{1e-4f, 1e-4f}; }
template <>
rocblas_double_complex get_tol<rocblas_double_complex>() { return rocblas_double_complex{1e-12, 1e-12}; }

// Helper for alpha/beta
template <typename T>
T get_alpha() { return T(1.1); }
template <>
rocblas_float_complex get_alpha<rocblas_float_complex>() { return rocblas_float_complex{1.1f, 0.2f}; }
template <>
rocblas_double_complex get_alpha<rocblas_double_complex>() { return rocblas_double_complex{1.1, 0.2}; }

template <typename T>
T get_beta() { return T(0.9); }
template <>
rocblas_float_complex get_beta<rocblas_float_complex>() { return rocblas_float_complex{0.9f, -0.1f}; }
template <>
rocblas_double_complex get_beta<rocblas_double_complex>() { return rocblas_double_complex{0.9, -0.1}; }

// Helper for abs (for complex types)
template <typename T>
auto abs_diff(const T& a, const T& b) -> decltype(std::abs(a-b)) { return std::abs(a-b); }
template <>
float abs_diff<rocblas_float_complex>(const rocblas_float_complex& a, const rocblas_float_complex& b)
{
    std::complex<float> ca(a.real(), a.imag()), cb(b.real(), b.imag());
    return std::abs(ca - cb);
}
template <>
double abs_diff<rocblas_double_complex>(const rocblas_double_complex& a, const rocblas_double_complex& b)
{
    std::complex<double> ca(a.real(), a.imag()), cb(b.real(), b.imag());
    return std::abs(ca - cb);
}


template <typename T>
void benchmark()
{
    if constexpr (std::is_same<T, float>::value)
        std::cout << "Data type: float" << std::endl;
    else if constexpr (std::is_same<T, double>::value)
        std::cout << "Data type: double" << std::endl;
    else if constexpr (std::is_same<T, rocblas_float_complex>::value)
        std::cout << "Data type: rocblas_float_complex" << std::endl;
    else if constexpr (std::is_same<T, rocblas_double_complex>::value)
        std::cout << "Data type: rocblas_double_complex" << std::endl;
    else
        std::cerr << "Unsupported data type." << std::endl;

    // Construct GEMM
    rocblas_operation transa = rocblas_operation_none, transb = rocblas_operation_transpose;
    const T alpha = get_alpha<T>(), beta = get_beta<T>();

    rocblas_int    m = DIM1, n = DIM2, k = DIM3;
    rocblas_int    lda, ldb, ldc;
    size_t         size_a, size_b, size_c;
    rocblas_stride a_stride_1, a_stride_2, b_stride_1, b_stride_2;
    std::cout << "user driven tuning example" << std::endl;
    if(transa == rocblas_operation_none)
    {
        lda        = m;
        size_a     = size_t(k) * lda;
        a_stride_1 = 1;
        a_stride_2 = lda;
    }
    else
    {
        lda        = k;
        size_a     = size_t(m) * lda;
        a_stride_1 = lda;
        a_stride_2 = 1;
    }
    if(transb == rocblas_operation_none)
    {
        ldb        = k;
        size_b     = size_t(n) * ldb;
        b_stride_1 = 1;
        b_stride_2 = ldb;
    }
    else
    {
        ldb        = n;
        size_b     = size_t(k) * ldb;
        b_stride_1 = ldb;
        b_stride_2 = 1;
    }
    ldc    = m;
    size_c = size_t(n) * ldc;

    // Naming: da is in GPU (device) memory. ha is in CPU (host) memory
    
    // Host memory
    std::vector<T> ha(size_a), hb(size_b), hc(size_c);
    fill_random(ha);
    fill_random(hb);
    fill_random(hc);

    // allocate memory on device
    T *da, *db, *dc;
    CHECK_HIP_ERROR(hipMalloc(&da, size_a * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&db, size_b * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dc, size_c * sizeof(T)));

    // copy matrices from host to device
    CHECK_HIP_ERROR(hipMemcpy(da, ha.data(), sizeof(T) * size_a, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(db, hb.data(), sizeof(T) * size_b, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dc, hc.data(), sizeof(T) * size_c, hipMemcpyHostToDevice));

    rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

    rocblas_datatype input_type, output_type, compute_type;
    if constexpr (std::is_same<T, float>::value)
    {
        input_type   = rocblas_datatype_f32_r;
        output_type  = rocblas_datatype_f32_r;
        compute_type = rocblas_datatype_f32_r;
    }
    else if constexpr (std::is_same<T, double>::value)
    {
        input_type   = rocblas_datatype_f64_r;
        output_type  = rocblas_datatype_f64_r;
        compute_type = rocblas_datatype_f64_r;
    }
    else if constexpr (std::is_same<T, rocblas_float_complex>::value)
    {
        input_type   = rocblas_datatype_f32_c;
        output_type  = rocblas_datatype_f32_c;
        compute_type = rocblas_datatype_f32_c;
    }
    else if constexpr (std::is_same<T, rocblas_double_complex>::value)
    {
        input_type   = rocblas_datatype_f64_c;
        output_type  = rocblas_datatype_f64_c;
        compute_type = rocblas_datatype_f64_c;
    }
    else
    {
        std::cerr << "Unsupported data type." << std::endl;
    }

    GEMMExParams<T> params {handle,
                        transa,
                        transb,
                        m,
                        n,
                        k,
                        alpha,
                        beta,
                        input_type,
                        output_type,
                        compute_type,
                        da,
                        db,
                        dc,
                        lda,
                        ldb,
                        ldc};

    /*
     * Get solutions by type example
    */
    // Get number of solutions that match this GEMM problem's type
    // NOTE: for batched problems use 'rocblas_gemm_batched_ex_get_solutions_by_type'
    //       for strided/batched problems use 'rocblas_gemm_ex_get_solutions_by_type'
    rocblas_int sizeType;
    CHECK_ROCBLAS_ERROR(rocblas_gemm_ex_get_solutions_by_type(
        handle, input_type, output_type, compute_type, rocblas_gemm_flags_none, NULL, &sizeType));
    std::cout << sizeType << " solution(s) found that match this GEMM's type." << std::endl;

    // Fill array with list of solutions that match type
    // Note: some of these may be invalid
    std::vector<rocblas_int> solutionsType(sizeType);
    CHECK_ROCBLAS_ERROR(rocblas_gemm_ex_get_solutions_by_type(handle,
                                                              input_type,
                                                              output_type,
                                                              compute_type,
                                                              rocblas_gemm_flags_none,
                                                              solutionsType.data(),
                                                              &sizeType));

    std::cout << "Benchmarking..." << std::endl;
    rocblas_int bestSolutionType = benchmark_solutions<T>(solutionsType, params);

/*
     * Get solutions that can solve only
     */
#define GEMM_EX_ARGS                                                                              \
    handle, transa, transb, m, n, k, &alpha, da, input_type, lda, db, input_type, ldb, &beta, dc, \
        output_type, ldc, dc, output_type, ldc, compute_type, rocblas_gemm_algo_solution_index

    // Get number of solutions that can solve this GEMM problem
    // NOTE: for batched problems use 'rocblas_gemm_batched_ex_get_solutions'
    //       for strided/batched problems use 'rocblas_gemm_strided_batched_ex_get_solutions'
    rocblas_int sizeSolve;
    CHECK_ROCBLAS_ERROR(
        rocblas_gemm_ex_get_solutions(GEMM_EX_ARGS, rocblas_gemm_flags_none, NULL, &sizeSolve));
    std::cout << sizeSolve << " solution(s) found that can solve this GEMM." << std::endl;

    // Fill array with list of solutions that match type
    // Note: some of these may be invalid
    std::vector<rocblas_int> solutionsSolve(sizeSolve);
    CHECK_ROCBLAS_ERROR(rocblas_gemm_ex_get_solutions(
        GEMM_EX_ARGS, rocblas_gemm_flags_none, solutionsSolve.data(), &sizeSolve));

    std::cout << "Benchmarking..." << std::endl;
    rocblas_int bestSolutionSolve = benchmark_solutions(solutionsSolve, params);

    // NOTE: bestSolutionType may be different to bestSolutionSolve, due to benchmarking noise
    assert(is_subset(solutionsType, solutionsSolve));

    // Check if solution is valid for problem (success case)
    CHECK_ROCBLAS_ERROR(
        rocblas_gemm_exM(GEMM_EX_ARGS, bestSolutionSolve, rocblas_gemm_flags_check_solution_index));

    // Allocate host buffers for results
    std::vector<T> hc_exm(size_c);
    std::vector<T> hc_sgemm(size_c);
    std::vector<T> hc_dft(size_c);

    // Solve using winner (rocblas_gemm_ex)
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    double exm_start = get_time_us_sync(stream);
    CHECK_ROCBLAS_ERROR(rocblas_gemm_exM(GEMM_EX_ARGS, bestSolutionSolve, rocblas_gemm_flags_none));
    double exm_end = get_time_us_sync(stream);
    double exm_time = exm_end - exm_start;
    std::cout << "[Timing] rocblas_gemm_ex winner: " << exm_time << " us" << std::endl;
    // Copy result to host
    CHECK_HIP_ERROR(hipMemcpy(hc_exm.data(), dc, sizeof(T) * size_c, hipMemcpyDeviceToHost));

    // Solve using standard rocblas_sgemm
    double sgemm_start = get_time_us_sync(stream);
    if constexpr (std::is_same<T, float>::value)
    {
        CHECK_ROCBLAS_ERROR(rocblas_sgemm(handle, transa, transb, m, n, k,
            reinterpret_cast<const float*>(&alpha),
            reinterpret_cast<const float*>(da), lda,
            reinterpret_cast<const float*>(db), ldb,
            reinterpret_cast<const float*>(&beta),
            reinterpret_cast<float*>(dc), ldc));
    }
    else if constexpr (std::is_same<T, double>::value)
    {
        CHECK_ROCBLAS_ERROR(rocblas_dgemm(handle, transa, transb, m, n, k,
            reinterpret_cast<const double*>(&alpha),
            reinterpret_cast<const double*>(da), lda,
            reinterpret_cast<const double*>(db), ldb,
            reinterpret_cast<const double*>(&beta),
            reinterpret_cast<double*>(dc), ldc));
    }
    else if constexpr (std::is_same<T, rocblas_float_complex>::value)
    {
        CHECK_ROCBLAS_ERROR(rocblas_cgemm(handle, transa, transb, m, n, k,
            reinterpret_cast<const rocblas_float_complex*>(&alpha),
            reinterpret_cast<const rocblas_float_complex*>(da), lda,
            reinterpret_cast<const rocblas_float_complex*>(db), ldb,
            reinterpret_cast<const rocblas_float_complex*>(&beta),
            reinterpret_cast<rocblas_float_complex*>(dc), ldc));
    }
    else if constexpr (std::is_same<T, rocblas_double_complex>::value)
    {
        CHECK_ROCBLAS_ERROR(rocblas_zgemm(handle, transa, transb, m, n, k,
            reinterpret_cast<const rocblas_double_complex*>(&alpha),
            reinterpret_cast<const rocblas_double_complex*>(da), lda,
            reinterpret_cast<const rocblas_double_complex*>(db), ldb,
            reinterpret_cast<const rocblas_double_complex*>(&beta),
            reinterpret_cast<rocblas_double_complex*>(dc), ldc));
    }
    else
    {
        std::cerr << "Unsupported data type for sgemm." << std::endl;
    }
    double sgemm_end = get_time_us_sync(stream);
    double sgemm_time = sgemm_end - sgemm_start;
    std::cout << "[Timing] rocblas_gemm (no ex): " << sgemm_time << " us" << std::endl;
    // Copy result to host
    CHECK_HIP_ERROR(hipMemcpy(hc_sgemm.data(), dc, sizeof(T) * size_c, hipMemcpyDeviceToHost));

    // Solve using default solution
    double dft_start = get_time_us_sync(stream);
    CHECK_ROCBLAS_ERROR(rocblas_gemm_exM(GEMM_EX_ARGS, 0, rocblas_gemm_flags_none));
    double dft_end = get_time_us_sync(stream);
    double dft_time = dft_end - dft_start;
    std::cout << "[Timing] rocblas_gemm_ex default: " << dft_time << " us" << std::endl;
    // Copy result to host
    CHECK_HIP_ERROR(hipMemcpy(hc_dft.data(), dc, sizeof(T) * size_c, hipMemcpyDeviceToHost));

    // Compare results
    auto compare_results = [](const std::vector<T>& a, const std::vector<T>& b, double tol, const char* name1, const char* name2) {
        size_t mismatches = 0;
        for(size_t i = 0; i < a.size(); ++i) {
            if(abs_diff(a[i], b[i]) > tol) {
                if(mismatches < 10) // only print first 10 mismatches
                    std::cout << "Mismatch at " << i << ": " << name1 << "=" << a[i] << ", " << name2 << "=" << b[i] << std::endl;
                ++mismatches;
            }
        }
        if(mismatches == 0)
            std::cout << "[Check] " << name1 << " and " << name2 << " match within tolerance." << std::endl;
        else
            std::cout << "[Check] " << name1 << " and " << name2 << " differ at " << mismatches << " positions." << std::endl;
    };

    double tol = 1e-4f;
    compare_results(hc_exm, hc_sgemm, tol, "rocblas_gemm_ex", "rocblas_gemm");
    compare_results(hc_exm, hc_dft, tol, "rocblas_gemm_ex", "rocblas_gemm_ex_default");
    compare_results(hc_sgemm, hc_dft, tol, "rocblas_gemm", "rocblas_gemm_ex_default");

    CHECK_HIP_ERROR(hipFree(da));
    CHECK_HIP_ERROR(hipFree(db));
    CHECK_HIP_ERROR(hipFree(dc));
    CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));

}

int main(int argc, char* argv[])
{

    // Run benchmarks for different data types
    benchmark<float>();
    benchmark<double>();
    benchmark<rocblas_float_complex>();
    benchmark<rocblas_double_complex>();

    return EXIT_SUCCESS;
}
