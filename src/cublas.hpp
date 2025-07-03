#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <cublasLt.h>
#include <iostream>
#include <vector>
#include <type_traits>
#include <chrono>
#include <random>
#include <complex>

// Helper for error checking
#define CHECK_CUBLAS_ERROR(status)                                             \
    do {                                                                       \
        cublasStatus_t err = (status);                                         \
        if (err != CUBLAS_STATUS_SUCCESS) {                                    \
            std::cerr << "CUBLAS error: " << err                               \
                      << " at line " << __LINE__ << std::endl;                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CHECK_CUDA_ERROR(status)                                               \
    do {                                                                       \
        cudaError_t err = (status);                                            \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)             \
                      << " at line " << __LINE__ << std::endl;                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CHECK_CUBLASLT_ERROR(status)                                           \
    do {                                                                       \
        cublasStatus_t err = (status);                                         \
        if (err != CUBLAS_STATUS_SUCCESS) {                                    \
            std::cerr << "cuBLASLt error: " << err                             \
                      << " at line " << __LINE__ << std::endl;                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Fill random
template <typename T>
void fill_random(std::vector<T>& v) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-1, 1);
    for (auto& x : v) x = static_cast<T>(dis(gen));
}
template <>
void fill_random<std::complex<float>>(std::vector<std::complex<float>>& v) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-1, 1);
    for (auto& x : v) x = std::complex<float>(dis(gen), dis(gen));
}
template <>
void fill_random<std::complex<double>>(std::vector<std::complex<double>>& v) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-1, 1);
    for (auto& x : v) x = std::complex<double>(dis(gen), dis(gen));
}

// Get alpha/beta
template <typename T> T get_alpha() { return T(1); }
template <typename T> T get_beta() { return T(0); }
template <> cuComplex get_alpha<cuComplex>() { return make_cuComplex(1,0); }
template <> cuComplex get_beta<cuComplex>() { return make_cuComplex(0,0); }
template <> cuDoubleComplex get_alpha<cuDoubleComplex>() { return make_cuDoubleComplex(1,0); }
template <> cuDoubleComplex get_beta<cuDoubleComplex>() { return make_cuDoubleComplex(0,0); }

// Find and return the best cuBLASLt algorithm for GEMM
template <typename T>
bool find_best_cublaslt_gemm_algo(
    int m, int n, int k,
    cudaDataType_t dtype,
    cublasOperation_t transa, cublasOperation_t transb,
    cublasLtMatmulAlgo_t& bestAlgoOut, float& bestTimeOut)
{
    size_t size_a = m * k * sizeof(T);
    size_t size_b = k * n * sizeof(T);
    size_t size_c = m * n * sizeof(T);
    
    T* da = nullptr; T* db = nullptr; T* dc = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&da, size_a));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&db, size_b));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&dc, size_c));

    std::vector<T> ha(m * k), hb(k * n), hc(m * n);
    fill_random(ha);
    fill_random(hb);
    fill_random(hc);
    
    cublasLtHandle_t ltHandle;
    CHECK_CUBLASLT_ERROR(cublasLtCreate(&ltHandle));

    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;

    cublasComputeType_t computeType;
    if (dtype == CUDA_R_64F || dtype == CUDA_C_64F)
        computeType = CUBLAS_COMPUTE_64F;
    else
        computeType = CUBLAS_COMPUTE_32F;
    CHECK_CUBLASLT_ERROR(cublasLtMatmulDescCreate(&operationDesc, computeType, dtype));

    CHECK_CUBLASLT_ERROR(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CHECK_CUBLASLT_ERROR(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    int lda = (transa == CUBLAS_OP_N) ? m : k;
    int ldb = (transb == CUBLAS_OP_N) ? k : n;
    int ldc = m;

    // Set matrix dimensions according to transpose flags
    int a_rows = (transa == CUBLAS_OP_N) ? m : k;
    int a_cols = (transa == CUBLAS_OP_N) ? k : m;
    int b_rows = (transb == CUBLAS_OP_N) ? k : n;
    int b_cols = (transb == CUBLAS_OP_N) ? n : k;

    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutCreate(&Adesc, dtype, a_rows, a_cols, lda));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutCreate(&Bdesc, dtype, b_rows, b_cols, ldb));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutCreate(&Cdesc, dtype, m, n, ldc));

    T alpha = get_alpha<T>(), beta = get_beta<T>();

    // Heuristic search
    const int requestAlgoCount = 32;
    cublasLtMatmulHeuristicResult_t heuristicResults[requestAlgoCount];
    int returnedAlgoCount = 0;
    
    cublasLtMatmulPreference_t preference;
    CHECK_CUBLASLT_ERROR(cublasLtMatmulPreferenceCreate(&preference));
    size_t workspaceSize = 256 * 1024 * 1024; //256 mb
    void* workspace;
    cudaMalloc(&workspace, workspaceSize);
    CHECK_CUBLASLT_ERROR(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));
    
    CHECK_CUBLASLT_ERROR(cublasLtMatmulAlgoGetHeuristic(
        ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference,
        requestAlgoCount, heuristicResults, &returnedAlgoCount));
    
    cublasLtMatmulPreferenceDestroy(preference);

    float bestTime = 1e30f;
    int bestAlgoIdx = -1;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int num_runs = 50;

    for (int i = 0; i < returnedAlgoCount; ++i) {
        std::vector<float> times(num_runs);
        for (int run = 0; run < num_runs; ++run) {
            CHECK_CUDA_ERROR(cudaMemcpy(da, ha.data(), size_a, cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(db, hb.data(), size_b, cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(dc, hc.data(), size_c, cudaMemcpyHostToDevice));

            cudaEventRecord(start, 0);
            cublasStatus_t status = cublasLtMatmul(
                ltHandle, operationDesc,
                &alpha, da, Adesc, db, Bdesc,
                &beta, dc, Cdesc, dc, Cdesc,
                &heuristicResults[i].algo, workspace, workspaceSize, 0);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);

            // Only record successful runs
            if (status == CUBLAS_STATUS_SUCCESS)
                times[run] = ms;
            else
                times[run] = 1e30f;
        }
        // Discard first 10%
        int discard = num_runs / 10;
        int valid_runs = num_runs - discard;
        float mean = 0.0f;
        for (int run = discard; run < num_runs; ++run) mean += times[run];
        mean /= valid_runs;

        if (mean < bestTime) {
            bestTime = mean;
            bestAlgoIdx = i;
        }
    }

    bool found = false;
    if (bestAlgoIdx >= 0) {
        bestAlgoOut = heuristicResults[bestAlgoIdx].algo;
        bestTimeOut = bestTime;
        found = true;
        std::cout << "Best cuBLASLt algorithm index: " << bestAlgoIdx
            << " time: " << bestTime*1e3 << " us" << std::endl;
    } else {
        std::cout << "No valid cuBLASLt algorithm found!" << std::endl;
    }

    CHECK_CUDA_ERROR(cudaFree(da));
    CHECK_CUDA_ERROR(cudaFree(db));
    CHECK_CUDA_ERROR(cudaFree(dc));

    cudaFree(workspace);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtDestroy(ltHandle);

    return found;
}

// Main benchmark
template <typename T>
void benchmark(const size_t dim1, const size_t dim2, const size_t dim3, const cublasOperation_t transa, const cublasOperation_t transb) {
    std::cout << " ============================== " << std::endl;
    std::cout << "Benchmarking GEMMEx with dimensions: " << dim1 << " x " << dim2 << " x " << dim3 << std::endl;

    // Write the transpose operations
    switch(transa) {
        case CUBLAS_OP_N:
            std::cout << "N x ";
            break;
        case CUBLAS_OP_T:
            std::cout << "T x ";
            break;
        case CUBLAS_OP_C:
            std::cout << "C x ";
            break;
        default:
            std::cerr << "Unsupported transpose operation for A." << std::endl;
            return;
    }
    switch(transb) {
        case CUBLAS_OP_N:
            std::cout << "N" << std::endl;
            break;
        case CUBLAS_OP_T:
            std::cout << "T" << std::endl;
            break;
        case CUBLAS_OP_C:
            std::cout << "C" << std::endl;
            break;
        default:
            std::cerr << "Unsupported transpose operation for B." << std::endl;
            return;
    }
    std::cout << std::endl;


    cudaDataType_t dtype;
    if constexpr (std::is_same<T, float>::value) {
        dtype = CUDA_R_32F;
        std::cout << "Data type: float" << std::endl;
    } else if constexpr (std::is_same<T, double>::value) {
        dtype = CUDA_R_64F;
        std::cout << "Data type: double" << std::endl;
    } else if constexpr (std::is_same<T, cuComplex>::value) {
        dtype = CUDA_C_32F;
        std::cout << "Data type: cuComplex" << std::endl;
    } else if constexpr (std::is_same<T, cuDoubleComplex>::value) {
        dtype = CUDA_C_64F;
        std::cout << "Data type: cuDoubleComplex" << std::endl;
    } else {
        static_assert(sizeof(T) == 0, "Unsupported data type");
    }

    const T alpha = get_alpha<T>(), beta = get_beta<T>();
    int m = static_cast<int>(dim1), n = static_cast<int>(dim2), k = static_cast<int>(dim3);

    // Set leading dimensions based on transpose status
    int lda = (transa == CUBLAS_OP_N) ? m : k;
    int ldb = (transb == CUBLAS_OP_N) ? k : n;
    int ldc = m;
    size_t size_a = m * k * sizeof(T);
    size_t size_b = k * n * sizeof(T);
    size_t size_c = m * n * sizeof(T);

    T* da = nullptr; T* db = nullptr; T* dc = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&da, size_a));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&db, size_b));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&dc, size_c));

    std::vector<T> ha(m * k), hb(k * n), hc(m * n);
    cublasHandle_t handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&handle));
    CHECK_CUBLAS_ERROR(cublasSetStream(handle, 0));

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    const int num_runs = 50;
    std::vector<float> times(num_runs);

    fill_random(ha);
    fill_random(hb);
    fill_random(hc);

    // GEMMEx timing
    for (int run = 0; run < num_runs; ++run) {
        CHECK_CUDA_ERROR(cudaMemcpy(da, ha.data(), size_a, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(db, hb.data(), size_b, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(dc, hc.data(), size_c, cudaMemcpyHostToDevice));

        CHECK_CUDA_ERROR(cudaEventRecord(start, 0));
        cublasStatus_t status = cublasGemmEx(
            handle, transa, transb, m, n, k,
            &alpha, da, dtype, lda,
            db, dtype, ldb,
            &beta, dc, dtype, ldc,
            dtype, CUBLAS_GEMM_DEFAULT);
        CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        float ms = 0;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, start, stop));
        times[run] = ms;

        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cublasGemmEx failed with status: " << status << std::endl;
        }
    }

    // Compute mean and stddev in nanoseconds
    int discard = num_runs / 10;
    if (discard < 1) discard = 1;
    float mean = 0.0f, stddev = 0.0f;
    int valid_runs = num_runs - discard;
    for (int i = discard; i < num_runs; ++i) mean += times[i];
    mean /= valid_runs;
    for (int i = discard; i < num_runs; ++i) stddev += (times[i] - mean) * (times[i] - mean);
    stddev = std::sqrt(stddev / valid_runs);

    // Convert ms to ns
    mean *= 1e3f;
    stddev *= 1e3f;

    std::cout << "GEMMEx mean time: " << mean << " us, stddev: " << stddev << " us" << std::endl;

    // Standard cuBLAS GEMM timing
    for (int run = 0; run < num_runs; ++run) {
        CHECK_CUDA_ERROR(cudaMemcpy(da, ha.data(), size_a, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(db, hb.data(), size_b, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(dc, hc.data(), size_c, cudaMemcpyHostToDevice));

        CHECK_CUDA_ERROR(cudaEventRecord(start, 0));
        cublasStatus_t status;

        if constexpr (std::is_same<T, float>::value) {
            status = cublasSgemm(
                handle, transa, transb, m, n, k,
                &alpha, da, lda, db, ldb, &beta, dc, ldc);
        } else if constexpr (std::is_same<T, double>::value) {
            status = cublasDgemm(
                handle, transa, transb, m, n, k,
                &alpha, da, lda, db, ldb, &beta, dc, ldc);
        } else if constexpr (std::is_same<T, cuComplex>::value) {
            status = cublasCgemm(
                handle, transa, transb, m, n, k,
                &alpha, da, lda, db, ldb, &beta, dc, ldc);
        } else if constexpr (std::is_same<T, cuDoubleComplex>::value) {
            status = cublasZgemm(
                handle, transa, transb, m, n, k,
                &alpha, da, lda, db, ldb, &beta, dc, ldc);
        } else {
            static_assert(sizeof(T) == 0, "Unsupported data type for standard GEMM");
        }

        CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        float ms = 0;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, start, stop));
        times[run] = ms;

        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "Standard cuBLAS GEMM failed with status: " << status << std::endl;
        }
    }

    // Compute mean and stddev in nanoseconds for standard GEMM
    mean = 0.0f;
    stddev = 0.0f;
    for (int i = discard; i < num_runs; ++i) mean += times[i];
    mean /= valid_runs;
    for (int i = discard; i < num_runs; ++i) stddev += (times[i] - mean) * (times[i] - mean);
    stddev = std::sqrt(stddev / valid_runs);

    mean *= 1e3f;
    stddev *= 1e3f;

    std::cout << "Standard GEMM mean time: " << mean << " us, stddev: " << stddev << " us" << std::endl;

    // cuBLASLt algorithm search
    cublasLtMatmulAlgo_t bestAlgo;
    float bestTime;
    bool found = find_best_cublaslt_gemm_algo<T>(
        m, n, k,
        dtype,
        transa, transb,
        bestAlgo, bestTime
    );

    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(hc.data(), dc, size_c, cudaMemcpyDeviceToHost));

    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(da));
    CHECK_CUDA_ERROR(cudaFree(db));
    CHECK_CUDA_ERROR(cudaFree(dc));
    CHECK_CUBLAS_ERROR(cublasDestroy(handle));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
}

