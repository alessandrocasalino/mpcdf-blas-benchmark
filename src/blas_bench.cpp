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

#ifdef __HIP_PLATFORM_HCC__
// need to enable unstable api
#define ROCBLAS_NO_DEPRECATED_WARNINGS
#define ROCBLAS_BETA_FEATURES_API

#include "rocblas.hpp"
#else
#include "cublas.hpp"
#include <cuComplex.h>
#endif

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{

    // Run benchmarks for different data types
#ifdef __HIP_PLATFORM_HCC__
    benchmark<rocblas_float_complex>();
    benchmark<rocblas_double_complex>();
#else
    benchmark<float>(12, 20, 23071, CUBLAS_OP_N, CUBLAS_OP_N);
    benchmark<double>(12, 20, 23071, CUBLAS_OP_N, CUBLAS_OP_N);
    benchmark<float>(12, 20, 23071, CUBLAS_OP_N, CUBLAS_OP_T);
    benchmark<double>(12, 20, 23071, CUBLAS_OP_N, CUBLAS_OP_T);
    benchmark<float>(12, 20, 23071, CUBLAS_OP_T, CUBLAS_OP_T);
    benchmark<double>(12, 20, 23071, CUBLAS_OP_T, CUBLAS_OP_T);
    benchmark<cuComplex>(12, 20, 23071, CUBLAS_OP_N, CUBLAS_OP_N);
    benchmark<cuDoubleComplex>(12, 20, 23071, CUBLAS_OP_N, CUBLAS_OP_N);
    benchmark<cuComplex>(12, 20, 23071, CUBLAS_OP_N, CUBLAS_OP_T);
    benchmark<cuDoubleComplex>(12, 20, 23071, CUBLAS_OP_N, CUBLAS_OP_T);
    benchmark<cuComplex>(12, 20, 23071, CUBLAS_OP_T, CUBLAS_OP_C);
    benchmark<cuDoubleComplex>(12, 20, 23071, CUBLAS_OP_T, CUBLAS_OP_C);
#endif

    return EXIT_SUCCESS;
}
