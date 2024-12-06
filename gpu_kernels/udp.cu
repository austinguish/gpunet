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

__device__ __forceinline__ uint16_t cuda_ntohs(uint16_t netshort) {
    return (netshort >> 8) | (netshort << 8);
}

__device__ __forceinline__ uint32_t cuda_ntohl(uint32_t netlong) {
    return ((netlong >> 24) & 0xff) |
           ((netlong << 8) & 0xff0000) |
           ((netlong >> 8) & 0xff00) |
           ((netlong << 24) & 0xff000000);
}

__device__ void print_eth_ip_udp_headers(const struct eth_ip_udp_hdr *hdr) {
    // Print Ethernet header
    printf("\n=== Ethernet Header ===\n");
    printf("Destination MAC: %02x:%02x:%02x:%02x:%02x:%02x\n",
           hdr->l2_hdr.d_addr_bytes[0], hdr->l2_hdr.d_addr_bytes[1],
           hdr->l2_hdr.d_addr_bytes[2], hdr->l2_hdr.d_addr_bytes[3],
           hdr->l2_hdr.d_addr_bytes[4], hdr->l2_hdr.d_addr_bytes[5]);
    printf("Source MAC: %02x:%02x:%02x:%02x:%02x:%02x\n",
           hdr->l2_hdr.s_addr_bytes[0], hdr->l2_hdr.s_addr_bytes[1],
           hdr->l2_hdr.s_addr_bytes[2], hdr->l2_hdr.s_addr_bytes[3],
           hdr->l2_hdr.s_addr_bytes[4], hdr->l2_hdr.s_addr_bytes[5]);
    printf("Ether Type: 0x%04x\n", cuda_ntohs(hdr->l2_hdr.ether_type));

    // Print IPv4 header
    printf("\n=== IPv4 Header ===\n");
    printf("Version: %u\n", (hdr->l3_hdr.version_ihl >> 4) & 0x0F);
    printf("Header Length: %u bytes\n", ((hdr->l3_hdr.version_ihl & 0x0F) * 4));
    printf("Type of Service: 0x%02x\n", hdr->l3_hdr.type_of_service);
    printf("Total Length: %u\n", cuda_ntohs(hdr->l3_hdr.total_length));
    printf("Packet ID: 0x%04x\n", cuda_ntohs(hdr->l3_hdr.packet_id));
    printf("Fragment Offset: 0x%04x\n", cuda_ntohs(hdr->l3_hdr.fragment_offset));
    printf("Time to Live: %u\n", hdr->l3_hdr.time_to_live);
    printf("Protocol: %u\n", hdr->l3_hdr.next_proto_id);
    printf("Header Checksum: 0x%04x\n", cuda_ntohs(hdr->l3_hdr.hdr_checksum));

    // Print IP addresses in dotted decimal format
    uint32_t src_ip = cuda_ntohl(hdr->l3_hdr.src_addr);
    uint32_t dst_ip = cuda_ntohl(hdr->l3_hdr.dst_addr);
    printf("Source IP: %u.%u.%u.%u\n",
           (src_ip >> 24) & 0xFF,
           (src_ip >> 16) & 0xFF,
           (src_ip >> 8) & 0xFF,
           src_ip & 0xFF);
    printf("Destination IP: %u.%u.%u.%u\n",
           (dst_ip >> 24) & 0xFF,
           (dst_ip >> 16) & 0xFF,
           (dst_ip >> 8) & 0xFF,
           dst_ip & 0xFF);

    // Print UDP header
    printf("\n=== UDP Header ===\n");
    printf("Source Port: %u\n", cuda_ntohs(hdr->l4_hdr.src_port));
    printf("Destination Port: %u\n", cuda_ntohs(hdr->l4_hdr.dst_port));
    printf("Datagram Length: %u\n", cuda_ntohs(hdr->l4_hdr.dgram_len));
    printf("Checksum: 0x%04x\n", cuda_ntohs(hdr->l4_hdr.dgram_cksum));
}

__device__ void parse_matrix_packet(const uint8_t* payload, float *mat_a, float *mat_b,MatrixCompletionInfo *stat_thread) {
	MatrixPacketHeader header;
	stat_thread->received_chunk_num++;
	memcpy(&header, payload, sizeof(MatrixPacketHeader));
	stat_thread->total_chunks_a = header.total_chunks;
	if (header.matrix_id == 0)
		stat_thread->received_a_elems+=header.chunk_size;
	else if (header.matrix_id == 1)
		stat_thread->received_b_elems+=header.chunk_size;
	const float* data = reinterpret_cast<const float*>(payload + sizeof(MatrixPacketHeader));
	float* target_matrix = (header.matrix_id == 0) ? mat_a : mat_b;
	size_t offset = header.chunk_id * header.chunk_size;

	// copy the data to the target matrix
	memcpy(target_matrix + offset, data, header.chunk_size * sizeof(float));

	// // print some value for debug
	// for (int i = 0; i < min(5, (int)header.chunk_size); i++) {
	// 	printf("data[%d] = %f\n", i, target_matrix[offset + i]);
	// }

}







__global__ void cuda_kernel_receive_udp_bw(uint32_t *exit_cond,
					struct doca_gpu_eth_rxq *rxq0, struct doca_gpu_eth_rxq *rxq1, struct doca_gpu_eth_rxq *rxq2, struct doca_gpu_eth_rxq *rxq3,
					int sem_num,
					struct doca_gpu_semaphore_gpu *sem0, struct doca_gpu_semaphore_gpu *sem1, struct doca_gpu_semaphore_gpu *sem2, struct doca_gpu_semaphore_gpu *sem3
				,float* mat_a,float* mat_b)
{
	__shared__ uint32_t rx_pkt_num;
	__shared__ uint64_t rx_buf_idx;
	__shared__ struct MatrixCompletionInfo stats_sh;

	doca_error_t ret;
	struct doca_gpu_eth_rxq *rxq = NULL;
	struct doca_gpu_semaphore_gpu *sem = NULL;
	struct doca_gpu_buf *buf_ptr;
	struct MatrixCompletionInfo stats_thread;
	struct MatrixCompletionInfo *stats_global;
	struct eth_ip_udp_hdr *hdr;
	uintptr_t buf_addr;
	uint64_t buf_idx = 0;
	uint32_t lane_id = threadIdx.x % WARP_SIZE;
	uint8_t *payload;
	uint32_t sem_idx = 0;
    //uint32_t worker_number = blockIdx.x*blockDim.x+threadIdx.x;
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
		DOCA_GPUNETIO_VOLATILE(stats_sh.received_a_elems) = 0;
		DOCA_GPUNETIO_VOLATILE(stats_sh.received_b_elems) = 0;
		DOCA_GPUNETIO_VOLATILE(stats_sh.received_chunk_num) = 0;
	}
	__syncthreads();

	while (DOCA_GPUNETIO_VOLATILE(*exit_cond) == 0) {
		stats_thread.received_a_elems = 0;
		stats_thread.received_b_elems = 0;
		stats_thread.received_chunk_num=0;

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
			// if ( threadIdx.x==0)printf("received a bunch of packets %d\n",rx_pkt_num);
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
            // use one of the thread to copy the hdr to the resp_hdr
            // if (threadIdx.x==0)
            // {
	           //  print_eth_ip_udp_headers(hdr);
            // }

			// print the packet size
			// printf("received a packet with size %d\n",hdr->l4_hdr.dgram_len);
			stats_thread.net_info.ip_src_addr = hdr->l3_hdr.src_addr;
			stats_thread.net_info.ip_dst_addr = hdr->l3_hdr.dst_addr;
			stats_thread.net_info.eth_src_addr_bytes [0] = hdr->l2_hdr.s_addr_bytes[0];
			stats_thread.net_info.eth_src_addr_bytes [1] = hdr->l2_hdr.s_addr_bytes[1];
			stats_thread.net_info.eth_src_addr_bytes [2] = hdr->l2_hdr.s_addr_bytes[2];
			stats_thread.net_info.eth_dst_addr_bytes [0] = hdr->l2_hdr.d_addr_bytes[0];
			stats_thread.net_info.eth_dst_addr_bytes [1] = hdr->l2_hdr.d_addr_bytes[1];
			stats_thread.net_info.eth_dst_addr_bytes [2] = hdr->l2_hdr.d_addr_bytes[2];
			stats_thread.net_info.eth_src_addr_bytes [3] = hdr->l2_hdr.s_addr_bytes[3];
			stats_thread.net_info.eth_src_addr_bytes [4] = hdr->l2_hdr.s_addr_bytes[4];
			stats_thread.net_info.eth_src_addr_bytes [5] = hdr->l2_hdr.s_addr_bytes[5];
			stats_thread.net_info.eth_dst_addr_bytes [3] = hdr->l2_hdr.d_addr_bytes[3];
			stats_thread.net_info.eth_dst_addr_bytes [4] = hdr->l2_hdr.d_addr_bytes[4];
			stats_thread.net_info.eth_dst_addr_bytes [5] = hdr->l2_hdr.d_addr_bytes[5];
			parse_matrix_packet(payload,mat_a,mat_b,&stats_thread);
			// try to print out the hdr inf
			/* Double-proof it's not reading old packets */
			// wipe_packet_32b((uint8_t*)&(hdr->l4_hdr));
			buf_idx += blockDim.x;
			//mat_a[worker_number] = 123.4f;
		}
		__syncthreads();

#pragma unroll
		for (int offset = 16; offset > 0; offset /= 2) {
			stats_thread.received_a_elems += __shfl_down_sync(WARP_FULL_MASK, stats_thread.received_a_elems, offset);
			stats_thread.received_b_elems += __shfl_down_sync(WARP_FULL_MASK, stats_thread.received_b_elems, offset);
			stats_thread.received_chunk_num += __shfl_down_sync(WARP_FULL_MASK, stats_thread.received_chunk_num, offset);
			__syncwarp();
		}

		if (lane_id == 0) {
			atomicAdd_block((uint32_t *)&(stats_sh.received_a_elems), stats_thread.received_a_elems);
			atomicAdd_block((uint32_t *)&(stats_sh.received_b_elems), stats_thread.received_b_elems);
			atomicCAS_block((uint32_t *)&(stats_sh.total_chunks_a),0,stats_thread.total_chunks_a);
			atomicAdd_block((uint32_t *)&(stats_sh.received_chunk_num), stats_thread.received_chunk_num);
		}
		__syncthreads();

		if (threadIdx.x == 0 && rx_pkt_num > 0) {
			ret = doca_gpu_dev_semaphore_get_custom_info_addr(sem, sem_idx, (void **)&stats_global);
			if (ret != DOCA_SUCCESS) {
				printf("UDP Error %d doca_gpu_dev_semaphore_get_custom_info_addr block %d thread %d\n", ret, blockIdx.x, threadIdx.x);
				DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
				break;
			}

			DOCA_GPUNETIO_VOLATILE(stats_global->received_a_elems) = DOCA_GPUNETIO_VOLATILE(stats_sh.received_a_elems);
			DOCA_GPUNETIO_VOLATILE(stats_global->received_b_elems) = DOCA_GPUNETIO_VOLATILE(stats_sh.received_b_elems);
			DOCA_GPUNETIO_VOLATILE(stats_global->received_chunk_num) = DOCA_GPUNETIO_VOLATILE(stats_sh.received_chunk_num);
			DOCA_GPUNETIO_VOLATILE(stats_global->total_chunks_a) = DOCA_GPUNETIO_VOLATILE(stats_sh.total_chunks_a);
			DOCA_GPUNETIO_VOLATILE(stats_global->total_chunks_b) = DOCA_GPUNETIO_VOLATILE(stats_sh.total_chunks_a);
			stats_global->net_info.ip_src_addr = stats_thread.net_info.ip_src_addr;
			stats_global->net_info.ip_dst_addr = stats_thread.net_info.ip_dst_addr;
			stats_global->net_info.eth_src_addr_bytes[0] = stats_thread.net_info.eth_src_addr_bytes[0];
			stats_global->net_info.eth_src_addr_bytes[1] = stats_thread.net_info.eth_src_addr_bytes[1];
			stats_global->net_info.eth_src_addr_bytes[2] = stats_thread.net_info.eth_src_addr_bytes[2];
			stats_global->net_info.eth_dst_addr_bytes[0] = stats_thread.net_info.eth_dst_addr_bytes[0];
			stats_global->net_info.eth_dst_addr_bytes[1] = stats_thread.net_info.eth_dst_addr_bytes[1];
			stats_global->net_info.eth_dst_addr_bytes[2] = stats_thread.net_info.eth_dst_addr_bytes[2];
			stats_global->net_info.eth_src_addr_bytes[3] = stats_thread.net_info.eth_src_addr_bytes[3];
			stats_global->net_info.eth_src_addr_bytes[4] = stats_thread.net_info.eth_src_addr_bytes[4];
			stats_global->net_info.eth_src_addr_bytes[5] = stats_thread.net_info.eth_src_addr_bytes[5];
			stats_global->net_info.eth_dst_addr_bytes[3] = stats_thread.net_info.eth_dst_addr_bytes[3];
			stats_global->net_info.eth_dst_addr_bytes[4] = stats_thread.net_info.eth_dst_addr_bytes[4];
			stats_global->net_info.eth_dst_addr_bytes[5] = stats_thread.net_info.eth_dst_addr_bytes[5];

			doca_gpu_dev_semaphore_set_status(sem, sem_idx, DOCA_GPU_SEMAPHORE_STATUS_READY);
			__threadfence_system();

			sem_idx = (sem_idx + 1) % sem_num;

			DOCA_GPUNETIO_VOLATILE(stats_sh.received_a_elems) = 0;
			DOCA_GPUNETIO_VOLATILE(stats_sh.received_b_elems) = 0;
			DOCA_GPUNETIO_VOLATILE(stats_sh.received_chunk_num) = 0;
		}

		__syncthreads();
	}
}

extern "C" {

doca_error_t kernel_receive_udp_bw(cudaStream_t stream, uint32_t *exit_cond, struct rxq_udp_bw_queues *udp_queues, float* mat_a, float* mat_b)
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
									,mat_b);
	result = cudaGetLastError();
	if (cudaSuccess != result) {
		DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	return DOCA_SUCCESS;
}

} /* extern C */