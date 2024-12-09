//
// Created by yiwei on 24-12-9.
//
#include <cublas_v2.h>
#include <math.h>
#ifndef COMPUTE_H
#define COMPUTE_H
cudaError_t perform_matrix_multiplication(float* A, float* B, float* C, int matrix_size, cudaStream_t stream, cudaStream_t receive_stream) {
	printf("Starting matrix multiplication debug\n");

    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Check initial CUDA status
    cudaError_t cuda_status = cudaGetLastError();
    printf("Initial CUDA status: %s\n", cudaGetErrorString(cuda_status));

    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    printf("cuBLAS creation status: %d\n", status);

    // Set stream
    status = cublasSetStream(handle, stream);
    printf("Stream set status: %d\n", status);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Record start time
    cudaEventRecord(start, stream);
    printf("Starting SGEMM operation...\n");
	cudaStreamSynchronize(receive_stream);
    // Perform multiplication
    status = cublasSgemm(handle,
                        CUBLAS_OP_T,
                        CUBLAS_OP_T,
                        matrix_size,
                        matrix_size,
                        matrix_size,
                        &alpha,
                        A,
                        matrix_size,
                        B,
                        matrix_size,
                        &beta,
                        C,
                        matrix_size);

    printf("SGEMM status: %d\n", status);

    // Record end time
    cudaEventRecord(stop, stream);

    // Wait for completion
    cudaStreamSynchronize(stream);

    // Calculate execution time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("SGEMM execution time: %f ms\n", milliseconds);

    // Final status check
    cuda_status = cudaGetLastError();
    printf("Final CUDA status: %s\n", cudaGetErrorString(cuda_status));

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);

    return cuda_status;
}
#endif //COMPUTE_H
