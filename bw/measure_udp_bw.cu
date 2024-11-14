/* Main kernel file */
#include <stdlib.h>
#include <string.h>
#include <doca_gpunetio_dev_buf.cuh>
#include <doca_gpunetio_dev_sem.cuh>
#include <doca_gpunetio_dev_eth_rxq.cuh>
#include "common.h"
#include "packets.h"
#include "filters.cuh"

DOCA_LOG_REGISTER(GPU_SANITY::KernelReceiveUdpBw);

__global__ void cuda_kernel_receive_udp_bw(uint32_t *exit_cond,
    struct doca_gpu_eth_rxq *rxq0, struct doca_gpu_eth_rxq *rxq1,
    struct doca_gpu_eth_rxq *rxq2, struct doca_gpu_eth_rxq *rxq3,
    int sem_num,
    struct doca_gpu_semaphore_gpu *sem0, struct doca_gpu_semaphore_gpu *sem1,
    struct doca_gpu_semaphore_gpu *sem2, struct doca_gpu_semaphore_gpu *sem3
) {
    __shared__ uint32_t rx_pkt_num;
    __shared__ uint64_t rx_buf_idx;
    __shared__ struct stats_udp stats_sh;
    __shared__ unsigned long long int bytes_sh;
    __shared__ uint64_t start_time_sh;
    __shared__ float gbps_sh;

    doca_error_t ret;
    struct doca_gpu_eth_rxq *rxq = NULL;
    struct doca_gpu_semaphore_gpu *sem = NULL;
    struct doca_gpu_buf *buf_ptr;
    struct stats_udp stats_thread;
    struct stats_udp *stats_global;
    struct eth_ip_udp_hdr *hdr;
    uintptr_t buf_addr;
    uint64_t buf_idx = 0;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint8_t *payload;
    uint32_t sem_idx = 0;
    unsigned long long int bytes_thread = 0;

    if (blockIdx.x == 0) {
        rxq = rxq0;
        sem = sem0;
    } else if (blockIdx.x == 1) {
        rxq = rxq1;
        sem = sem1;
    } else if (blockIdx.x == 2) {
        rxq = rxq2;
        sem = sem2;
    } else if (blockIdx.x == 3) {
        rxq = rxq3;
        sem = sem3;
    } else
        return;

    if (threadIdx.x == 0) {
        DOCA_GPUNETIO_VOLATILE(stats_sh.dns) = 0;
        DOCA_GPUNETIO_VOLATILE(stats_sh.others) = 0;
        DOCA_GPUNETIO_VOLATILE(bytes_sh) = 0;
        start_time_sh = clock64();
    }
    __syncthreads();

    while (DOCA_GPUNETIO_VOLATILE(*exit_cond) == 0) {
        stats_thread.dns = 0;
        stats_thread.others = 0;
        bytes_thread = 0;

        ret = doca_gpu_dev_eth_rxq_receive_block(rxq, 0, MAX_RX_TIMEOUT_NS, &rx_pkt_num, &rx_buf_idx);
        if (ret != DOCA_SUCCESS) {
            if (threadIdx.x == 0) {
                printf("Receive UDP BW kernel error %d Block %d rxpkts %d error %d\n",
                       ret, blockIdx.x, rx_pkt_num, ret);
                DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
            }
            break;
        }

        if (rx_pkt_num == 0)
            continue;

        buf_idx = threadIdx.x;
        while (buf_idx < rx_pkt_num) {
            ret = doca_gpu_dev_eth_rxq_get_buf(rxq, rx_buf_idx + buf_idx, &buf_ptr);
            if (ret != DOCA_SUCCESS) {
                printf("UDP BW Error %d get_buf block %d thread %d\n", ret, blockIdx.x, threadIdx.x);
                DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
                break;
            }

            ret = doca_gpu_dev_buf_get_addr(buf_ptr, &buf_addr);
            if (ret != DOCA_SUCCESS) {
                printf("UDP BW Error %d get_addr block %d thread %d\n", ret, blockIdx.x, threadIdx.x);
                DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
                break;
            }

            raw_to_udp(buf_addr, &hdr, &payload);

            // Get packet size from IP total length
            bytes_thread += (unsigned long long int)hdr->l3_hdr.total_length;

            if (filter_is_dns(&(hdr->l4_hdr), payload))
                stats_thread.dns++;
            else
                stats_thread.others++;

            wipe_packet_32b((uint8_t*)&(hdr->l4_hdr));
            buf_idx += blockDim.x;
        }

#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            stats_thread.dns += __shfl_down_sync(WARP_FULL_MASK, stats_thread.dns, offset);
            stats_thread.others += __shfl_down_sync(WARP_FULL_MASK, stats_thread.others, offset);
            bytes_thread += __shfl_down_sync(WARP_FULL_MASK, bytes_thread, offset);
            __syncwarp();
        }

        if (lane_id == 0) {
            atomicAdd_block((uint32_t *)&(stats_sh.dns), stats_thread.dns);
            atomicAdd_block((uint32_t *)&(stats_sh.others), stats_thread.others);
            atomicAdd((unsigned long long int*)&bytes_sh, bytes_thread);
        }
        __syncthreads();

        if (threadIdx.x == 0 && rx_pkt_num > 0) {
            uint64_t end_time = clock64();
            float duration_sec = (end_time - start_time_sh) * 1e-9;
            gbps_sh = (bytes_sh * 8.0f) / (duration_sec * 1e9);

            ret = doca_gpu_dev_semaphore_get_custom_info_addr(sem, sem_idx, (void **)&stats_global);
            if (ret != DOCA_SUCCESS) {
                printf("UDP BW Error %d get_custom_info block %d thread %d\n",
                       ret, blockIdx.x, threadIdx.x);
                DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
                break;
            }

            DOCA_GPUNETIO_VOLATILE(stats_global->dns) = DOCA_GPUNETIO_VOLATILE(stats_sh.dns);
            DOCA_GPUNETIO_VOLATILE(stats_global->others) = DOCA_GPUNETIO_VOLATILE(stats_sh.others);
            DOCA_GPUNETIO_VOLATILE(stats_global->total) = rx_pkt_num;

            // Log bandwidth info
            printf("Block %d: Bandwidth %.2f Gbps, Total bytes: %llu\n",
                   blockIdx.x, gbps_sh, bytes_sh);

            doca_gpu_dev_semaphore_set_status(sem, sem_idx, DOCA_GPU_SEMAPHORE_STATUS_READY);
            __threadfence_system();

            sem_idx = (sem_idx + 1) % sem_num;

            DOCA_GPUNETIO_VOLATILE(stats_sh.dns) = 0;
            DOCA_GPUNETIO_VOLATILE(stats_sh.others) = 0;
            bytes_sh = 0;
            start_time_sh = clock64();
        }
        __syncthreads();
    }
}

extern "C" {
doca_error_t kernel_receive_udp_bw(cudaStream_t stream, uint32_t *exit_cond, struct rxq_udp_queues *udp_queues)
{
    cudaError_t result = cudaSuccess;

    if (udp_queues == NULL || udp_queues->numq == 0 || udp_queues->numq > MAX_QUEUES || exit_cond == 0) {
        DOCA_LOG_ERR("kernel_receive_udp_bw invalid input values");
        return DOCA_ERROR_INVALID_VALUE;
    }

    result = cudaGetLastError();
    if (cudaSuccess != result) {
        DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    cuda_kernel_receive_udp_bw<<<udp_queues->numq, CUDA_THREADS, 0, stream>>>(
        exit_cond,
        udp_queues->eth_rxq_gpu[0], udp_queues->eth_rxq_gpu[1],
        udp_queues->eth_rxq_gpu[2], udp_queues->eth_rxq_gpu[3],
        udp_queues->nums,
        udp_queues->sem_gpu[0], udp_queues->sem_gpu[1],
        udp_queues->sem_gpu[2], udp_queues->sem_gpu[3]
    );

    result = cudaGetLastError();
    if (cudaSuccess != result) {
        DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}
}