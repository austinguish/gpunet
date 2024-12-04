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
#include <doca_gpunetio_dev_eth_txq.cuh>
#include <doca_gpunetio_dev_eth_rxq.cuh>

#include "common.h"
#include "packets.h"
#include "filters.cuh"

DOCA_LOG_REGISTER(GPU_SANITY::KernelTCPBW);

static
__device__ void tcp_swap_mac_addr(struct eth_ip_tcp_hdr* hdr)
{
    uint16_t addr_bytes[3];

    addr_bytes[0] = ((uint16_t*)hdr->l2_hdr.s_addr_bytes)[0];
    addr_bytes[1] = ((uint16_t*)hdr->l2_hdr.s_addr_bytes)[1];
    addr_bytes[2] = ((uint16_t*)hdr->l2_hdr.s_addr_bytes)[2];

    ((uint16_t*)hdr->l2_hdr.s_addr_bytes)[0] = ((uint16_t*)hdr->l2_hdr.d_addr_bytes)[0];
    ((uint16_t*)hdr->l2_hdr.s_addr_bytes)[1] = ((uint16_t*)hdr->l2_hdr.d_addr_bytes)[1];
    ((uint16_t*)hdr->l2_hdr.s_addr_bytes)[2] = ((uint16_t*)hdr->l2_hdr.d_addr_bytes)[2];

    ((uint16_t*)hdr->l2_hdr.d_addr_bytes)[0] = addr_bytes[0];
    ((uint16_t*)hdr->l2_hdr.d_addr_bytes)[1] = addr_bytes[1];
    ((uint16_t*)hdr->l2_hdr.d_addr_bytes)[2] = addr_bytes[2];
}

static
__device__ void tcp_swap_ip_addr(struct eth_ip_tcp_hdr* hdr)
{
    uint32_t tmp;

    tmp = hdr->l3_hdr.src_addr;
    hdr->l3_hdr.src_addr = hdr->l3_hdr.dst_addr;
    hdr->l3_hdr.dst_addr = tmp;
}

static
__device__ void tcp_swap_ports(struct eth_ip_tcp_hdr* hdr)
{
    uint16_t tmp;

    tmp = hdr->l4_hdr.src_port;
    hdr->l4_hdr.src_port = hdr->l4_hdr.dst_port;
    hdr->l4_hdr.dst_port = tmp;
}

static
__device__ void create_ack_packet(struct eth_ip_tcp_hdr* hdr, uint32_t* nbytes) {
    // 1. MAC layer: swap addresses
    tcp_swap_mac_addr(hdr);

    // 2. IP layer: swap addresses and set basic fields
    tcp_swap_ip_addr(hdr);
    hdr->l3_hdr.version_ihl = (4 << 4) | 5;
    hdr->l3_hdr.time_to_live = 64;
    hdr->l3_hdr.type_of_service = 0;

    // 3. TCP layer: handle ports and sequence numbers
    tcp_swap_ports(hdr);
    uint32_t tcp_header_len = (hdr->l4_hdr.dt_off >> 4) * 4;
    uint32_t data_len = BYTE_SWAP16(hdr->l3_hdr.total_length) - sizeof(struct ipv4_hdr) - tcp_header_len;

    // Set sequence and ack numbers
    uint32_t rcv_seq = BYTE_SWAP32(hdr->l4_hdr.sent_seq);
    uint32_t rcv_ack = BYTE_SWAP32(hdr->l4_hdr.recv_ack);
    hdr->l4_hdr.sent_seq = BYTE_SWAP32(rcv_ack);
    hdr->l4_hdr.recv_ack = BYTE_SWAP32(rcv_seq + data_len);
    printf("set the ack to %u\n",BYTE_SWAP32(hdr->l4_hdr.recv_ack));

    // 4. Set TCP flags and window
    hdr->l4_hdr.tcp_flags = TCP_FLAG_ACK;
    hdr->l4_hdr.rx_win = BYTE_SWAP16(5670);  // Same window size as reference
    hdr->l4_hdr.tcp_urp = 0;

    // 5. Set packet lengths
    hdr->l3_hdr.total_length = BYTE_SWAP16(sizeof(struct ipv4_hdr) + tcp_header_len);
    *nbytes = sizeof(struct ether_hdr) + sizeof(struct ipv4_hdr) + tcp_header_len;

    // 6. Reset checksums for hardware offload
    hdr->l3_hdr.hdr_checksum = 0;
    hdr->l4_hdr.cksum = 0;
}

__global__ void cuda_kernel_tcp_bw(uint32_t* exit_cond, struct doca_gpu_eth_rxq* rxq1, struct doca_gpu_eth_rxq* rxq2,
                                   struct doca_gpu_eth_rxq* rxq3,
                                   struct doca_gpu_eth_rxq* rxq4, struct doca_gpu_eth_txq* txq1,
                                   struct doca_gpu_eth_txq* txq2, struct doca_gpu_eth_txq* txq3,
                                   struct doca_gpu_eth_txq* txq4)
{
    __shared__ uint32_t rx_pkt_num;
    __shared__ uint64_t rx_buf_idx;

    doca_error_t ret;
    uint64_t buf_idx = 0;
    uintptr_t buf_addr;
    struct doca_gpu_buf* buf_ptr;
    struct eth_ip_tcp_hdr* hdr;
    uint8_t* payload;
    uint32_t nbytes;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    struct doca_gpu_eth_rxq* rxq = NULL;
    struct doca_gpu_eth_txq* txq = NULL;

    if (warp_id > 0)
        return;

    // Select queue based on block index
    if (blockIdx.x == 0)
    {
        rxq = rxq1;
        txq = txq1;
    }
    else if (blockIdx.x == 1)
    {
        rxq = rxq2;
        txq = txq2;
    }
    else if (blockIdx.x == 2)
    {
        rxq = rxq3;
        txq = txq3;
    }
    else if (blockIdx.x == 3)
    {
        rxq = rxq4;
        txq = txq4;
    }

    while (DOCA_GPUNETIO_VOLATILE(*exit_cond)==0)
    {
        ret = doca_gpu_dev_eth_rxq_receive_block(rxq, MAX_RX_NUM_PKTS, MAX_RX_TIMEOUT_NS, &rx_pkt_num, &rx_buf_idx);
        if (ret != DOCA_SUCCESS)
        {
            if (lane_id == 0)
            {
                //only lane 0 report the error
                printf("Receive TCP kernel error %d warp %d lane %d error %d\n", ret, warp_id, rx_pkt_num, ret);

                DOCA_GPUNETIO_VOLATILE(*exit_cond)=1;
            }
            break;
        }

        if (rx_pkt_num == 0)
            continue;

        buf_idx = lane_id;
        while (buf_idx < rx_pkt_num)
        {
            ret = doca_gpu_dev_eth_rxq_get_buf(rxq, rx_buf_idx + buf_idx, &buf_ptr);
            if (ret != DOCA_SUCCESS)
            {
                printf("Error %d doca_gpu_dev_eth_rxq_get_buf warp %d lane %d\n", ret, warp_id, lane_id);
                DOCA_GPUNETIO_VOLATILE(*exit_cond)=1;
                break;
            }

            ret = doca_gpu_dev_buf_get_addr(buf_ptr, &buf_addr);
            if (ret != DOCA_SUCCESS)
            {
                printf("Error %d doca_gpu_dev_buf_get_addr warp %d lane %d\n", ret, warp_id, lane_id);
                DOCA_GPUNETIO_VOLATILE(*exit_cond)=1;
                break;
            }

            raw_to_tcp(buf_addr, &hdr, &payload);
            // check the seq number and the ack number
            // print the payload here


            // printf("the flag of the packet is %d\n",hdr->l4_hdr.tcp_flags);
            // Prepare TCP ACK packet
            // tcp_swap_mac_addr(hdr);
            // tcp_swap_ip_addr(hdr);
            // tcp_swap_ports(hdr);
            //
            // // 1. TCP header length
            // uint32_t tcp_header_len = (hdr->l4_hdr.dt_off >> 4) * 4;
            //
            // // data length
            // uint32_t data_len = BYTE_SWAP16(hdr->l3_hdr.total_length) - sizeof(struct ipv4_hdr) - tcp_header_len;
            // // printf("datalength is %d",data_len);
            // // set the sequence number and ack number
            // uint32_t rcv_seq = BYTE_SWAP32(hdr->l4_hdr.sent_seq);
            // uint32_t rcv_ack = BYTE_SWAP32(hdr->l4_hdr.recv_ack);
            // hdr->l4_hdr.sent_seq = BYTE_SWAP32(rcv_ack);
            // hdr->l4_hdr.recv_ack = BYTE_SWAP32(rcv_seq + data_len);
            // // uint32_t prev_pkt_sz = BYTE_SWAP16(hdr->l3_hdr.total_length) - sizeof(struct ipv4_hdr) - ((hdr->l4_hdr.dt_off >> 4) * sizeof(uint32_t));
            // // hdr->l4_hdr.recv_ack = BYTE_SWAP32(BYTE_SWAP32(hdr->l4_hdr.sent_seq) + prev_pkt_sz);
            //
            // // set tcp flags and window
            // hdr->l4_hdr.tcp_flags = TCP_FLAG_ACK|TCP_FLAG_PSH;
            // hdr->l4_hdr.rx_win = BYTE_SWAP16(6000);
            // hdr->l4_hdr.cksum = 0;
            // hdr->l4_hdr.tcp_urp = 0;
            // // set IP header
            // hdr->l3_hdr.version_ihl = (4 << 4) | 5;
            // hdr->l3_hdr.time_to_live = 64;
            // hdr->l3_hdr.total_length = BYTE_SWAP16(sizeof(struct ipv4_hdr) + tcp_header_len);
            // hdr->l3_hdr.type_of_service=0x0;
            // hdr->l3_hdr.hdr_checksum = 0;
            //
            // nbytes = sizeof(struct ether_hdr) + sizeof(struct ipv4_hdr) + tcp_header_len;

            printf("Seq: %u, Ack: %u\n",
                   BYTE_SWAP32(hdr->l4_hdr.sent_seq),
                   BYTE_SWAP32(hdr->l4_hdr.recv_ack));

            create_ack_packet(hdr,&nbytes);
            //Send ACK packet
            doca_gpu_dev_eth_txq_send_enqueue_strong(txq, buf_ptr, nbytes, DOCA_GPU_SEND_FLAG_NOTIFY);

            buf_idx += WARP_SIZE;
        }
        __syncwarp();

        if (lane_id == 0)
        {
            doca_gpu_dev_eth_txq_commit_strong(txq);
            doca_gpu_dev_eth_txq_push(txq);
        }
        __syncwarp();
    }
}


extern "C" {
doca_error_t kernel_tcp_bw_test(
    cudaStream_t stream,
    uint32_t* exit_cond,
    struct tcp_bw_queues* tcp_bw_queues)
{
    //receive a packet then just send ack back
    cudaError_t result = cudaSuccess;

    if (exit_cond == 0 || tcp_bw_queues == NULL || tcp_bw_queues->numq == 0 || tcp_bw_queues->numq > MAX_QUEUES)
    {
        DOCA_LOG_ERR("kernel_tcp_bw invalid input values");
        return DOCA_ERROR_INVALID_VALUE;
    }

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (cudaSuccess != result)
    {
        DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    /*  MAX_QUEUES == 4 */
    cuda_kernel_tcp_bw<<<tcp_bw_queues->numq, CUDA_THREADS, 0, stream>>>(
        exit_cond, tcp_bw_queues->eth_rxq_gpu[0], tcp_bw_queues->eth_rxq_gpu[1], tcp_bw_queues->eth_rxq_gpu[2],
        tcp_bw_queues->eth_rxq_gpu[3]
        , tcp_bw_queues->eth_txq_gpu[0], tcp_bw_queues->eth_txq_gpu[1], tcp_bw_queues->eth_txq_gpu[2],
        tcp_bw_queues->eth_txq_gpu[3]);
    result = cudaGetLastError();
    if (cudaSuccess != result)
    {
        DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}
}

// extern "C" {
//
// doca_error_t kernel_receive_icmp(cudaStream_t stream, uint32_t *exit_cond, struct rxq_icmp_queues *icmp_queues)
// {
// 	cudaError_t result = cudaSuccess;
//
// 	if (exit_cond == 0 || icmp_queues == NULL || icmp_queues->numq == 0 || icmp_queues->numq > MAX_QUEUES_ICMP) {
// 		DOCA_LOG_ERR("kernel_receive_icmp invalid input values");
// 		return DOCA_ERROR_INVALID_VALUE;
// 	}
//
// 	/* Check no previous CUDA errors */
// 	result = cudaGetLastError();
// 	if (cudaSuccess != result) {
// 		DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
// 		return DOCA_ERROR_BAD_STATE;
// 	}
//
// 	/* Assume MAX_QUEUES_ICMP == 1 */
// 	cuda_kernel_receive_icmp<<<1, WARP_SIZE, 0, stream>>>(exit_cond, icmp_queues->eth_rxq_gpu[0], icmp_queues->eth_txq_gpu[0]);
// 	result = cudaGetLastError();
// 	if (cudaSuccess != result) {
// 		DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
// 		return DOCA_ERROR_BAD_STATE;
// 	}
//
// 	return DOCA_SUCCESS;
// }

/* extern C */
