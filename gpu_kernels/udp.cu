//
// Created by yiwei on 24-12-2.
//
/*
 * Copyright (c) 2023 NVIDIA CORPORATION AND AFFILIATES.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <stdlib.h>
#include <string.h>

#include <doca_gpunetio_dev_buf.cuh>
#include <doca_gpunetio_dev_sem.cuh>
#include <doca_gpunetio_dev_eth_rxq.cuh>

#include "common.h"
#include "packets.h"
#include "filters.cuh"
#include "matmul/mat_message.h"
#define UDP_WARP_MODE 0

DOCA_LOG_REGISTER(GPU_SANITY::KernelReceiveUdp);
__device__ void parse_matrix_packet(const uint8_t* payload, float *mat_a, float *mat_b) {
    // 首先验证输入指针
    if (payload == NULL) {
        if (threadIdx.x == 0) {
            printf("Error: NULL payload pointer\n");
        }
        return;
    }

    MatrixPacketHeader* matrix_header = (MatrixPacketHeader*)payload;

    // 打印原始header信息
    if (threadIdx.x == 0) {
        printf("Header info: matrix_id=%u, chunk_id=%u, chunk_size=%u\n",
               matrix_header->matrix_id,
               matrix_header->chunk_id,
               matrix_header->chunk_size);
    }

    // 转换字节序
    uint32_t matrix_id = BYTE_SWAP32(matrix_header->matrix_id);
    uint32_t chunk_id = BYTE_SWAP32(matrix_header->chunk_id);
    uint32_t chunk_size = BYTE_SWAP32(matrix_header->chunk_size);

    // 验证 chunk_size
    if (chunk_size == 0 || chunk_size > MAX_FLOATS_PER_PACKET) {
        if (threadIdx.x == 0) {
            printf("Invalid chunk size: %u\n", chunk_size);
        }
        return;
    }

    // 计算实际的数据指针位置
    float* mat_ptr = (float*)(payload + sizeof(MatrixPacketHeader));

    // 打印指针信息
    if (threadIdx.x == 0) {
        printf("Payload address: %p, Matrix data address: %p\n",
               payload, mat_ptr);
        printf("First float value at mat_ptr: %f\n", mat_ptr[0]);
    }

    // 选择目标矩阵
    float* dest_matrix = (matrix_id == 0) ? mat_a : mat_b;
    if (dest_matrix == NULL) {
        if (threadIdx.x == 0) {
            printf("Error: NULL destination matrix pointer\n");
        }
        return;
    }

    // 计算偏移，并验证
    size_t offset = chunk_id * chunk_size;

    // 逐个元素安全复制
    for (uint32_t i = 0; i < chunk_size; i++) {
        if (threadIdx.x == 0 && i == 0) {
            printf("Copying first element: source=%f\n", mat_ptr[0]);
        }

        float val;
        // 使用更安全的复制方式
        if (i < chunk_size) {  // 额外的边界检查
            val = __ldg(&mat_ptr[i]);  // 使用 __ldg 做全局内存加载
            dest_matrix[offset + i] = val;
        }
    }

    // 验证写入
    if (threadIdx.x == 0) {
        printf("First value written: %f\n", dest_matrix[offset]);
    }
}







__global__ void cuda_kernel_receive_udp_bw(uint32_t *exit_cond,
					struct doca_gpu_eth_rxq *rxq0, struct doca_gpu_eth_rxq *rxq1, struct doca_gpu_eth_rxq *rxq2, struct doca_gpu_eth_rxq *rxq3,
					int sem_num,
					struct doca_gpu_semaphore_gpu *sem0, struct doca_gpu_semaphore_gpu *sem1, struct doca_gpu_semaphore_gpu *sem2, struct doca_gpu_semaphore_gpu *sem3
				,float* mat_a)
{
	__shared__ uint32_t rx_pkt_num;
	__shared__ uint64_t rx_buf_idx;
	__shared__ struct stats_udp stats_sh;

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
    uint32_t worker_number = blockIdx.x*blockDim.x+threadIdx.x;
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
	}
	__syncthreads();

	while (DOCA_GPUNETIO_VOLATILE(*exit_cond) == 0) {
		stats_thread.dns = 0;
		stats_thread.others = 0;

		/* No need to impose packet limit here as we want the max number of packets every time */
		ret = doca_gpu_dev_eth_rxq_receive_block(rxq, 0, MAX_RX_TIMEOUT_NS, &rx_pkt_num, &rx_buf_idx);
		/* If any thread returns receive error, the whole execution stops */
		if (ret != DOCA_SUCCESS) {
			if (threadIdx.x == 0) {
				/*
				 * printf in CUDA kernel may be a good idea only to report critical errors or debugging.
				 * If application prints this message on the console, something bad happened and
				 * applications needs to exit
				 */
				printf("Receive UDP kernel error %d Block %d rxpkts %d error %d\n", ret, blockIdx.x, rx_pkt_num, ret);
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
				printf("UDP Error %d doca_gpu_dev_eth_rxq_get_buf block %d thread %d\n", ret, blockIdx.x, threadIdx.x);
				DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
				break;
			}

			ret = doca_gpu_dev_buf_get_addr(buf_ptr, &buf_addr);
			if (ret != DOCA_SUCCESS) {
				printf("UDP Error %d doca_gpu_dev_eth_rxq_get_buf block %d thread %d\n", ret, blockIdx.x, threadIdx.x);
				DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
				break;
			}

			raw_to_udp(buf_addr, &hdr, &payload);
			parse_matrix_packet(payload,mat_a,NULL);
			// try to print out the hdr info
			// printf("the src addr is %u,and the src port is %u \n",hdr->l3_hdr.src_addr,hdr->l4_hdr.src_port);
			// printf("the dst addr is %u,and the dst port is %u\n",hdr->l3_hdr.dst_addr,hdr->l4_hdr.dst_port);
			stats_thread.others++;
			/* Double-proof it's not reading old packets */
			wipe_packet_32b((uint8_t*)&(hdr->l4_hdr));
			buf_idx += blockDim.x;
			//mat_a[worker_number] = 123.4f;
		}
		__syncthreads();

#pragma unroll
		for (int offset = 16; offset > 0; offset /= 2) {
			stats_thread.dns += __shfl_down_sync(WARP_FULL_MASK, stats_thread.dns, offset);
			stats_thread.others += __shfl_down_sync(WARP_FULL_MASK, stats_thread.others, offset);
			__syncwarp();
		}

		if (lane_id == 0) {
			atomicAdd_block((uint32_t *)&(stats_sh.dns), stats_thread.dns);
			atomicAdd_block((uint32_t *)&(stats_sh.others), stats_thread.others);
		}
		__syncthreads();

		if (threadIdx.x == 0 && rx_pkt_num > 0) {
			ret = doca_gpu_dev_semaphore_get_custom_info_addr(sem, sem_idx, (void **)&stats_global);
			if (ret != DOCA_SUCCESS) {
				printf("UDP Error %d doca_gpu_dev_semaphore_get_custom_info_addr block %d thread %d\n", ret, blockIdx.x, threadIdx.x);
				DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
				break;
			}

			DOCA_GPUNETIO_VOLATILE(stats_global->dns) = DOCA_GPUNETIO_VOLATILE(stats_sh.dns);
			DOCA_GPUNETIO_VOLATILE(stats_global->others) = DOCA_GPUNETIO_VOLATILE(stats_sh.others);
			DOCA_GPUNETIO_VOLATILE(stats_global->total) = rx_pkt_num;
			doca_gpu_dev_semaphore_set_status(sem, sem_idx, DOCA_GPU_SEMAPHORE_STATUS_READY);
			__threadfence_system();

			sem_idx = (sem_idx + 1) % sem_num;

			DOCA_GPUNETIO_VOLATILE(stats_sh.dns) = 0;
			DOCA_GPUNETIO_VOLATILE(stats_sh.others) = 0;
		}

		__syncthreads();
	}
}

extern "C" {

doca_error_t kernel_receive_udp_bw(cudaStream_t stream, uint32_t *exit_cond, struct rxq_udp_bw_queues *udp_queues,float* mat_a)
{
	cudaError_t result = cudaSuccess;

	if (udp_queues == NULL || udp_queues->numq == 0 || udp_queues->numq > MAX_QUEUES || exit_cond == 0) {
		DOCA_LOG_ERR("kernel_receive_udp invalid input values");
		return DOCA_ERROR_INVALID_VALUE;
	}

	/* Check no previous CUDA errors */
	result = cudaGetLastError();
	if (cudaSuccess != result) {
		DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	/* Assume MAX_QUEUES == 4 */
	cuda_kernel_receive_udp_bw<<<udp_queues->numq, CUDA_THREADS, 0, stream>>>(exit_cond,
									udp_queues->eth_rxq_gpu[0], udp_queues->eth_rxq_gpu[1], udp_queues->eth_rxq_gpu[2], udp_queues->eth_rxq_gpu[3],
									udp_queues->nums,
									udp_queues->sem_gpu[0], udp_queues->sem_gpu[1], udp_queues->sem_gpu[2], udp_queues->sem_gpu[3],mat_a
									);
	result = cudaGetLastError();
	if (cudaSuccess != result) {
		DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	return DOCA_SUCCESS;
}

} /* extern C */