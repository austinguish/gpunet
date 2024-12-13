//
// Created by yiwei on 24-12-6.
//
#include <stdlib.h>
#include <string.h>

#include <doca_gpunetio_dev_buf.cuh>
#include <doca_gpunetio_dev_sem.cuh>
#include <doca_gpunetio_dev_eth_txq.cuh>
#include "filters.cuh"
#include "common.h"
#include "packets.h"
#include "matmul/mat_message.h"

#define WARP_SIZE 32
#define BLOCKS_PER_GRID 4
#define MAX_THREADS_PER_BLOCK 512

struct PacketContext
{
    struct doca_gpu_eth_txq* txq;
    struct doca_gpu_buf_arr* buf_arr;
    uint32_t total_numbers;
    float* matrix_data;
    uint32_t total_chunks;
    uint32_t chunk_size;
    uint32_t matrix_id;
    uint8_t eth_src_addr_bytes[ETHER_ADDR_LEN]; /* Source addr bytes in tx order */
    uint8_t eth_dst_addr_bytes[ETHER_ADDR_LEN]; /* Destination addr bytes in tx order */
    uint32_t ip_src_addr; /* IP source address */
    uint32_t ip_dst_addr; /* IP destination address */
    uint8_t* debug_buf;
};

__device__ void prepare_packet(uintptr_t buf_addr,
                               uint32_t chunk_idx,
                               const PacketContext* ctx,
                               uint32_t* total_size)
{
    // prepare header
    struct eth_ip_udp_hdr* net_header = reinterpret_cast<struct eth_ip_udp_hdr*>(buf_addr);
#pragma unroll
    for (int i = 0; i < ETHER_ADDR_LEN; i++)
    {
        net_header->l2_hdr.s_addr_bytes[i] = ctx->eth_dst_addr_bytes[i];
        net_header->l2_hdr.d_addr_bytes[i] = ctx->eth_src_addr_bytes[i];
    }
    net_header->l2_hdr.ether_type = 0x8;
    net_header->l3_hdr.src_addr = ctx->ip_dst_addr;
    net_header->l3_hdr.dst_addr = ctx->ip_src_addr;
    // net_header->l4_hdr.src_port = gpu_htons(1234);
    // net_header->l4_hdr.dst_port = gpu_htons(5678);
    // reset the checksum

    net_header->l3_hdr.hdr_checksum = 0;

    uint8_t* header_end = (uint8_t*)(net_header + 1);
    // add 2 bytes 0 for the padding
    memset(header_end,0,2);
    header_end+=2;

    // prepare the matrix header
    struct MatrixPacketHeader* matrix_header = (struct MatrixPacketHeader*)header_end;
        matrix_header->matrix_id = ctx->matrix_id;
        matrix_header->chunk_id = chunk_idx;
        matrix_header->total_chunks = ctx->total_chunks;
        matrix_header->chunk_size = ctx->chunk_size;

    // copy the matrix data
   float* data_payload = (float*)(matrix_header + 1);
       uint32_t start_idx = chunk_idx * ctx->chunk_size;
       uint32_t actual_size = min(ctx->chunk_size,
                                 ctx->total_numbers - start_idx);

    for (uint32_t i = 0; i < actual_size; i++)
    {
        data_payload[i] = ctx->matrix_data[start_idx + i];
    }
    uint32_t udp_payload_size = sizeof(struct MatrixPacketHeader) + actual_size * sizeof(float)+2;
    *total_size = sizeof(struct eth_ip_udp_hdr) + udp_payload_size;
    // set the header length
    net_header->l4_hdr.dgram_len = gpu_htons(udp_payload_size + sizeof(udp_hdr));
    // set the checksum to 0
    net_header->l4_hdr.dgram_cksum = 0;
    // set the ip header length
    net_header->l3_hdr.total_length = gpu_htons(udp_payload_size + sizeof(ipv4_hdr) + sizeof(udp_hdr));
    //set the checksum to 0
    net_header->l3_hdr.hdr_checksum = 0;
    // if (chunk_idx == 0) {
    //         printf("Debug Info:\n");
    //         printf("UDP payload size: %u bytes\n", udp_payload_size);
    //         printf("Total packet size: %u bytes\n", *total_size);
    //         printf("Data payload offset: %lu bytes\n", (uint8_t*)data_payload - (uint8_t*)net_header);
    //     }
}

__global__ void cuda_kernel_send_matrix(
    struct doca_gpu_eth_txq* txq0,
    struct doca_gpu_eth_txq* txq1,
    struct doca_gpu_eth_txq* txq2,
    struct doca_gpu_eth_txq* txq3,
    PacketContext ctx
)
{
    struct doca_gpu_eth_txq* txq = NULL;
    if (blockIdx.x == 0)
    {
        txq = txq0;
    }
    else if (blockIdx.x == 1)
    {
        txq = txq1;
    }
    else if (blockIdx.x == 2)
    {
        txq = txq2;
    }
    else if (blockIdx.x == 3)
    {
        txq = txq3;
    }

    if (txq == NULL) return;
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint16_t send_pkts = 0;
    // calculate the number of the chunks per thread
    uint32_t chunks_per_thread = (ctx.total_chunks + blockDim.x * gridDim.x - 1) /
        (blockDim.x * gridDim.x);

    for (uint32_t i = 0; i < chunks_per_thread; i++)
    {
        uint32_t chunk_idx = tid * chunks_per_thread + i;
        if (chunk_idx >= ctx.total_chunks) break;

        // get the buffer for this chunk
        struct doca_gpu_buf* buf = nullptr;
        doca_error_t ret = doca_gpu_dev_buf_get_buf(ctx.buf_arr,
                                                    chunk_idx % TX_BUF_NUM,
                                                    &buf);
        if (ret != DOCA_SUCCESS)
        {
            if (lane_id == 0)
            {
                printf("Buffer allocation failed for chunk %d\n", chunk_idx);
            }
            return;
        }

        // get the buffer address for this chunk
        uintptr_t buf_addr;
        ret = doca_gpu_dev_buf_get_addr(buf, &buf_addr);
        if (ret != DOCA_SUCCESS)
        {
            if (lane_id == 0)
            {
                printf("Failed to get buffer address for chunk %d\n", chunk_idx);
            }
            return;
        }

        // prepare the packet
        uint32_t total_size;
        prepare_packet(buf_addr, chunk_idx, &ctx, &total_size);
        // for debug, the worker_id==0 will copy the packet to the specific location,
        // and the host will check what is the packet looks like
        if (tid == 0)
        {
            memcpy(ctx.debug_buf, (uint8_t*)buf_addr, total_size);
        }
        ret = doca_gpu_dev_eth_txq_send_enqueue_strong(txq, buf, total_size, DOCA_GPU_SEND_FLAG_NOTIFY);
        if (ret != DOCA_SUCCESS)
        {
            if (lane_id == 0)
            {
                printf("Packet enqueue failed for chunk %d\n", chunk_idx);
            }
            return;
        }

        send_pkts++;
        __syncwarp();

        // Sum up send_pkts within warp
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
        {
            send_pkts += __shfl_down_sync(WARP_FULL_MASK, send_pkts, offset);
        }

        // the first thread in every warp will commit and send
        if (lane_id == 0 && send_pkts > 0)
        {
            doca_gpu_dev_eth_txq_commit_strong(txq);
            doca_gpu_dev_eth_txq_push(txq);
        }
    }
    // sychronize the each thread
}


// static __device__ void prepare_udp_packet(uintptr_t buf_addr, struct udp_packet *pkt) {
//     pkt->hdr = (struct eth_ip_udp_hdr *)buf_addr;
//     pkt->payload = (uint8_t *)(pkt->hdr + 1);
// }


extern "C" {
doca_error_t kernel_send_matrix_c(cudaStream_t stream,
                                  rxq_udp_bw_queues* queues, float* mat_c,
                                  uint32_t matrix_size, struct NetInfo* net_info, uint8_t* debug_buf)
{
    if (!queues || !mat_c)
    {
        return DOCA_ERROR_INVALID_VALUE;
    }

    cudaError_t cuda_ret = cudaGetLastError();
    if (cuda_ret != cudaSuccess)
    {
        return DOCA_ERROR_BAD_STATE;
    }
    PacketContext ctx;
    uint32_t total_numbers = matrix_size * matrix_size;
    ctx.total_numbers = total_numbers;
    if (total_numbers < MAX_FLOATS_PER_PACKET) {
        ctx.chunk_size = total_numbers;
        ctx.total_chunks = 1;  // 只需要一个chunk
    } else{
        ctx.chunk_size = MAX_FLOATS_PER_PACKET;
        ctx.total_chunks = (total_numbers + ctx.chunk_size - 1) / ctx.chunk_size;
    }
    ctx.matrix_data = mat_c;
    ctx.matrix_id = 0x2;
    ctx.buf_arr = queues->buf_response.buf_arr_gpu;
    ctx.debug_buf = debug_buf;
    for (int i = 0; i < ETHER_ADDR_LEN; i++)
    {
        ctx.eth_src_addr_bytes[i] = net_info->eth_src_addr_bytes[i];
        ctx.eth_dst_addr_bytes[i] = net_info->eth_dst_addr_bytes[i];
    }
    ctx.ip_src_addr = net_info->ip_src_addr;
    ctx.ip_dst_addr = net_info->ip_dst_addr;

    cuda_kernel_send_matrix<<<queues->numq, CUDA_THREADS, 0, stream>>>(
        queues->eth_txq_gpu[0],
        queues->eth_txq_gpu[1],
        queues->eth_txq_gpu[2],
        queues->eth_txq_gpu[3],
        ctx
    );
    // 同步流
    cudaStreamSynchronize(stream);

    cuda_ret = cudaGetLastError();
    if (cuda_ret != cudaSuccess)
    {
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}
}
