// 在文件开头添加设备端字节序转换函数
__device__ inline uint16_t cuda_ntohs(uint16_t netshort) {
    return (netshort >> 8) | (netshort << 8);
}

#include <stdlib.h>
#include <string.h>

#include <doca_gpunetio_dev_buf.cuh>
#include <doca_gpunetio_dev_sem.cuh>
#include <doca_gpunetio_dev_eth_rxq.cuh>

#include "common.h"
#include "packets.h"
#include "filters.cuh"

#define UDP_WARP_MODE 0

DOCA_LOG_REGISTER(GPU_SANITY::KernelReceiveUdp);

struct bandwidth_stats {
    unsigned long long total_bytes;
    unsigned long long start_time;
    unsigned long long end_time;
    unsigned long long total_packets;
};

__global__ void cuda_kernel_receive_udp(uint32_t *exit_cond,
                    struct doca_gpu_eth_rxq *rxq0, struct doca_gpu_eth_rxq *rxq1,
                    struct doca_gpu_eth_rxq *rxq2, struct doca_gpu_eth_rxq *rxq3,
                    int sem_num,
                    struct doca_gpu_semaphore_gpu *sem0, struct doca_gpu_semaphore_gpu *sem1,
                    struct doca_gpu_semaphore_gpu *sem2, struct doca_gpu_semaphore_gpu *sem3,
                    struct bandwidth_stats *bw_stats)
{
    __shared__ uint32_t rx_pkt_num;
    __shared__ uint64_t rx_buf_idx;
    __shared__ struct stats_udp stats_sh;
    __shared__ unsigned long long block_bytes;

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
    unsigned long long thread_bytes = 0;

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
    }
    else
        return;

    if (threadIdx.x == 0) {
        DOCA_GPUNETIO_VOLATILE(stats_sh.dns) = 0;
        DOCA_GPUNETIO_VOLATILE(stats_sh.others) = 0;
        block_bytes = 0;

        if (blockIdx.x == 0) {
            bw_stats->start_time = clock64();
            bw_stats->total_bytes = 0;
            bw_stats->total_packets = 0;
        }
    }
    __syncthreads();

    while (DOCA_GPUNETIO_VOLATILE(*exit_cond) == 0) {
        stats_thread.dns = 0;
        stats_thread.others = 0;
        thread_bytes = 0;

        ret = doca_gpu_dev_eth_rxq_receive_block(rxq, 0, MAX_RX_TIMEOUT_NS, &rx_pkt_num, &rx_buf_idx);
        if (ret != DOCA_SUCCESS) {
            if (threadIdx.x == 0) {
                printf("Receive UDP kernel error %d Block %d rxpkts %d error %d\n",
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
                printf("UDP Error %d doca_gpu_dev_eth_rxq_get_buf block %d thread %d\n",
                       ret, blockIdx.x, threadIdx.x);
                DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
                break;
            }

            ret = doca_gpu_dev_buf_get_addr(buf_ptr, &buf_addr);
            if (ret != DOCA_SUCCESS) {
                printf("UDP Error %d doca_gpu_dev_eth_rxq_get_buf block %d thread %d\n",
                       ret, blockIdx.x, threadIdx.x);
                DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
                break;
            }

            raw_to_udp(buf_addr, &hdr, &payload);

            // 使用设备端字节序转换函数
            thread_bytes += (unsigned long long)cuda_ntohs(hdr->l3_hdr.total_length);

            if (filter_is_dns(&(hdr->l4_hdr), payload))
                stats_thread.dns++;
            else
                stats_thread.others++;

            wipe_packet_32b((uint8_t*)&(hdr->l4_hdr));
            buf_idx += blockDim.x;
        }
        __syncthreads();

#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            stats_thread.dns += __shfl_down_sync(WARP_FULL_MASK, stats_thread.dns, offset);
            stats_thread.others += __shfl_down_sync(WARP_FULL_MASK, stats_thread.others, offset);
            thread_bytes += __shfl_down_sync(WARP_FULL_MASK, thread_bytes, offset);
            __syncwarp();
        }

        if (lane_id == 0) {
            atomicAdd_block((uint32_t *)&(stats_sh.dns), stats_thread.dns);
            atomicAdd_block((uint32_t *)&(stats_sh.others), stats_thread.others);
            atomicAdd_block(&block_bytes, thread_bytes);
        }
        __syncthreads();

        if (threadIdx.x == 0 && rx_pkt_num > 0) {
            ret = doca_gpu_dev_semaphore_get_custom_info_addr(sem, sem_idx, (void **)&stats_global);
            if (ret != DOCA_SUCCESS) {
                printf("UDP Error %d doca_gpu_dev_semaphore_get_custom_info_addr block %d thread %d\n",
                       ret, blockIdx.x, threadIdx.x);
                DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
                break;
            }

            DOCA_GPUNETIO_VOLATILE(stats_global->dns) = DOCA_GPUNETIO_VOLATILE(stats_sh.dns);
            DOCA_GPUNETIO_VOLATILE(stats_global->others) = DOCA_GPUNETIO_VOLATILE(stats_sh.others);
            DOCA_GPUNETIO_VOLATILE(stats_global->total) = rx_pkt_num;

            atomicAdd_system((unsigned long long*)&bw_stats->total_bytes, block_bytes);
            atomicAdd_system((unsigned long long*)&bw_stats->total_packets, (unsigned long long)rx_pkt_num);
            bw_stats->end_time = clock64();

            doca_gpu_dev_semaphore_set_status(sem, sem_idx, DOCA_GPU_SEMAPHORE_STATUS_READY);
            __threadfence_system();

            sem_idx = (sem_idx + 1) % sem_num;

            DOCA_GPUNETIO_VOLATILE(stats_sh.dns) = 0;
            DOCA_GPUNETIO_VOLATILE(stats_sh.others) = 0;
            block_bytes = 0;
        }

        __syncthreads();
    }
}

extern "C" {

doca_error_t kernel_receive_udp(cudaStream_t stream, uint32_t *exit_cond,
                               struct rxq_udp_queues *udp_queues)
{
    cudaError_t result = cudaSuccess;
    struct bandwidth_stats *bw_stats;

    if (udp_queues == NULL || udp_queues->numq == 0 ||
        udp_queues->numq > MAX_QUEUES || exit_cond == 0) {
        DOCA_LOG_ERR("kernel_receive_udp invalid input values");
        return DOCA_ERROR_INVALID_VALUE;
    }

    result = cudaMalloc(&bw_stats, sizeof(struct bandwidth_stats));
    if (result != cudaSuccess) {
        DOCA_LOG_ERR("Failed to allocate bandwidth stats memory");
        return DOCA_ERROR_NO_MEMORY;
    }

    result = cudaMemset(bw_stats, 0, sizeof(struct bandwidth_stats));
    if (result != cudaSuccess) {
        DOCA_LOG_ERR("Failed to initialize bandwidth stats memory");
        cudaFree(bw_stats);
        return DOCA_ERROR_NO_MEMORY;
    }

    result = cudaGetLastError();
    if (cudaSuccess != result) {
        DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__,
                     cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    cuda_kernel_receive_udp<<<udp_queues->numq, CUDA_THREADS, 0, stream>>>(
        exit_cond,
        udp_queues->eth_rxq_gpu[0], udp_queues->eth_rxq_gpu[1],
        udp_queues->eth_rxq_gpu[2], udp_queues->eth_rxq_gpu[3],
        udp_queues->nums,
        udp_queues->sem_gpu[0], udp_queues->sem_gpu[1],
        udp_queues->sem_gpu[2], udp_queues->sem_gpu[3],
        bw_stats
    );

    result = cudaGetLastError();
    if (cudaSuccess != result) {
        DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__,
                     cudaGetErrorString(result));
        cudaFree(bw_stats);
        return DOCA_ERROR_BAD_STATE;
    }

    struct bandwidth_stats host_stats;
    cudaMemcpy(&host_stats, bw_stats, sizeof(struct bandwidth_stats),
               cudaMemcpyDeviceToHost);

    double duration_sec = (host_stats.end_time - host_stats.start_time) * 1e-9;
    double bandwidth_gbps = (host_stats.total_bytes * 8.0) / (duration_sec * 1e9);
    double pps = host_stats.total_packets / duration_sec;

    printf("Bandwidth Statistics:\n");
    printf("Total Bytes: %llu\n", host_stats.total_bytes);
    printf("Total Packets: %llu\n", host_stats.total_packets);
    printf("Duration: %.2f seconds\n", duration_sec);
    printf("Bandwidth: %.2f Gbps\n", bandwidth_gbps);
    printf("Packet Rate: %.2f pps\n", pps);

    cudaFree(bw_stats);
    return DOCA_SUCCESS;
}

} /* extern C */