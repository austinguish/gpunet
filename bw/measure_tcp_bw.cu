//#include <stdlib.h>
//#include <string.h>
//#include <doca_gpunetio_dev_buf.cuh>
//#include <doca_gpunetio_dev_sem.cuh>
//#include <doca_gpunetio_dev_eth_rxq.cuh>
//#include "common.h"
//
///* Add to common.h */
//struct stats_bandwidth {
//    uint64_t bytes;
//    uint64_t packets;
//    uint64_t start_time_ns;
//    uint64_t end_time_ns;
//    float gbps;
//};
//
//struct rxq_bandwidth_queues {
//    uint32_t numq;
//    uint32_t nums;
//    struct doca_gpu_eth_rxq *eth_rxq_gpu[MAX_QUEUES];
//    struct doca_gpu_semaphore_gpu *sem_gpu[MAX_QUEUES];
//};
//
//
//DOCA_LOG_REGISTER(GPU_SANITY::KernelReceiveBandwidth);
//
//__global__ void cuda_kernel_receive_bandwidth(
//    uint32_t *exit_cond,
//    struct doca_gpu_eth_rxq *rxq0, struct doca_gpu_eth_rxq *rxq1,
//    struct doca_gpu_eth_rxq *rxq2, struct doca_gpu_eth_rxq *rxq3,
//    int sem_num,
//    struct doca_gpu_semaphore_gpu *sem_stats0, struct doca_gpu_semaphore_gpu *sem_stats1,
//    struct doca_gpu_semaphore_gpu *sem_stats2, struct doca_gpu_semaphore_gpu *sem_stats3
//) {
//    __shared__ uint32_t rx_pkt_num;
//    __shared__ uint64_t rx_buf_idx;
//    __shared__ struct stats_bandwidth stats_sh;
//
//    doca_error_t ret;
//    struct doca_gpu_eth_rxq *rxq = NULL;
//    struct doca_gpu_semaphore_gpu *sem_stats = NULL;
//    struct doca_gpu_buf *buf_ptr;
//    struct stats_bandwidth stats_thread;
//    struct stats_bandwidth *stats_global;
//    uintptr_t buf_addr;
//    uint64_t buf_idx = 0;
//    uint32_t laneId = threadIdx.x % WARP_SIZE;
//    uint32_t sem_stats_idx = 0;
//    uint32_t packet_size;
//
//    if (blockIdx.x == 0) {
//        rxq = rxq0;
//        sem_stats = sem_stats0;
//    } else if (blockIdx.x == 1) {
//        rxq = rxq1;
//        sem_stats = sem_stats1;
//    } else if (blockIdx.x == 2) {
//        rxq = rxq2;
//        sem_stats = sem_stats2;
//    } else if (blockIdx.x == 3) {
//        rxq = rxq3;
//        sem_stats = sem_stats3;
//    } else
//        return;
//
//    if (threadIdx.x == 0) {
//        DOCA_GPUNETIO_VOLATILE(stats_sh.bytes) = 0;
//        DOCA_GPUNETIO_VOLATILE(stats_sh.packets) = 0;
//        DOCA_GPUNETIO_VOLATILE(stats_sh.start_time_ns) = clock64();
//    }
//    __syncthreads();
//
//    while (DOCA_GPUNETIO_VOLATILE(*exit_cond) == 0) {
//        stats_thread.bytes = 0;
//        stats_thread.packets = 0;
//
//        ret = doca_gpu_dev_eth_rxq_receive_block(rxq, MAX_RX_NUM_PKTS, MAX_RX_TIMEOUT_NS, &rx_pkt_num, &rx_buf_idx);
//        if (ret != DOCA_SUCCESS) {
//            if (threadIdx.x == 0) {
//                printf("Receive Bandwidth kernel error %d Block %d rxpkts %d error %d\n",
//                       ret, blockIdx.x, rx_pkt_num, ret);
//                DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
//            }
//            break;
//        }
//
//        if (rx_pkt_num == 0)
//            continue;
//
//        buf_idx = threadIdx.x;
//        while (buf_idx < rx_pkt_num) {
//            ret = doca_gpu_dev_eth_rxq_get_buf(rxq, rx_buf_idx + buf_idx, &buf_ptr);
//            if (ret != DOCA_SUCCESS) {
//                printf("BW Error %d doca_gpu_dev_eth_rxq_get_buf block %d thread %d\n",
//                       ret, blockIdx.x, threadIdx.x);
//                DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
//                break;
//            }
//
//            ret = doca_gpu_dev_buf_get_size(buf_ptr, &packet_size);
//            if (ret != DOCA_SUCCESS) {
//                printf("BW Error %d doca_gpu_dev_buf_get_size block %d thread %d\n",
//                       ret, blockIdx.x, threadIdx.x);
//                DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
//                break;
//            }
//
//            stats_thread.bytes += packet_size;
//            stats_thread.packets++;
//            buf_idx += blockDim.x;
//        }
//
//        #pragma unroll
//        for (int offset = 16; offset > 0; offset /= 2) {
//            stats_thread.bytes += __shfl_down_sync(WARP_FULL_MASK, stats_thread.bytes, offset);
//            stats_thread.packets += __shfl_down_sync(WARP_FULL_MASK, stats_thread.packets, offset);
//            __syncwarp();
//        }
//
//        if (laneId == 0) {
//            atomicAdd_block(&(stats_sh.bytes), stats_thread.bytes);
//            atomicAdd_block(&(stats_sh.packets), stats_thread.packets);
//        }
//        __syncthreads();
//
//        if (threadIdx.x == 0 && rx_pkt_num > 0) {
//            ret = doca_gpu_dev_semaphore_get_custom_info_addr(sem_stats, sem_stats_idx, (void **)&stats_global);
//            if (ret != DOCA_SUCCESS) {
//                printf("BW Error %d doca_gpu_dev_semaphore_get_custom_info_addr block %d thread %d\n",
//                       ret, blockIdx.x, threadIdx.x);
//                DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
//                break;
//            }
//
//            stats_sh.end_time_ns = clock64();
//            float duration_sec = (stats_sh.end_time_ns - stats_sh.start_time_ns) * 1e-9;
//            stats_sh.gbps = (stats_sh.bytes * 8.0f) / (duration_sec * 1e9);
//
//            DOCA_GPUNETIO_VOLATILE(stats_global->bytes) = DOCA_GPUNETIO_VOLATILE(stats_sh.bytes);
//            DOCA_GPUNETIO_VOLATILE(stats_global->packets) = DOCA_GPUNETIO_VOLATILE(stats_sh.packets);
//            DOCA_GPUNETIO_VOLATILE(stats_global->start_time_ns) = DOCA_GPUNETIO_VOLATILE(stats_sh.start_time_ns);
//            DOCA_GPUNETIO_VOLATILE(stats_global->end_time_ns) = DOCA_GPUNETIO_VOLATILE(stats_sh.end_time_ns);
//            DOCA_GPUNETIO_VOLATILE(stats_global->gbps) = DOCA_GPUNETIO_VOLATILE(stats_sh.gbps);
//
//            doca_gpu_dev_semaphore_set_status(sem_stats, sem_stats_idx, DOCA_GPU_SEMAPHORE_STATUS_READY);
//            __threadfence_system();
//
//            sem_stats_idx = (sem_stats_idx + 1) % sem_num;
//
//            DOCA_GPUNETIO_VOLATILE(stats_sh.bytes) = 0;
//            DOCA_GPUNETIO_VOLATILE(stats_sh.packets) = 0;
//            DOCA_GPUNETIO_VOLATILE(stats_sh.start_time_ns) = clock64();
//        }
//        __syncthreads();
//    }
//}
//
//extern "C" {
//doca_error_t kernel_receive_bandwidth(cudaStream_t stream, uint32_t *exit_cond,
//                                    struct rxq_bandwidth_queues *bw_queues)
//{
//    cudaError_t result = cudaSuccess;
//
//    if (exit_cond == 0 || bw_queues == NULL || bw_queues->numq == 0) {
//        DOCA_LOG_ERR("kernel_receive_bandwidth invalid input values");
//        return DOCA_ERROR_INVALID_VALUE;
//    }
//
//    result = cudaGetLastError();
//    if (cudaSuccess != result) {
//        DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__,
//                     cudaGetErrorString(result));
//        return DOCA_ERROR_BAD_STATE;
//    }
//
//    cuda_kernel_receive_bandwidth<<<bw_queues->numq, CUDA_THREADS, 0, stream>>>(
//        exit_cond,
//        bw_queues->eth_rxq_gpu[0], bw_queues->eth_rxq_gpu[1],
//        bw_queues->eth_rxq_gpu[2], bw_queues->eth_rxq_gpu[3],
//        bw_queues->nums,
//        bw_queues->sem_gpu[0], bw_queues->sem_gpu[1],
//        bw_queues->sem_gpu[2], bw_queues->sem_gpu[3]
//    );
//
//    result = cudaGetLastError();
//    if (cudaSuccess != result) {
//        DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__,
//                     cudaGetErrorString(result));
//        return DOCA_ERROR_BAD_STATE;
//    }
//
//    return DOCA_SUCCESS;
//}
//}