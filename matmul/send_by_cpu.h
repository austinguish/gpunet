//
// Created by yiwei on 24-12-9.
//

#ifndef SEND_BY_CPU_H
#define SEND_BY_CPU_H
//
// Created by yiwei on 24-12-9.
//

#include "rte_mbuf.h"
#include "packets.h"
#include "rte_ip.h"
static struct rte_mbuf* prepare_matrix_packet(struct rte_mempool *mp,
                                            const float *data,
                                            uint32_t matrix_id,
                                            uint32_t chunk_id,
                                            uint32_t total_chunks,
                                            uint32_t chunk_size,
                                            const struct MatrixCompletionInfo *net_info) {
    struct rte_mbuf *pkt = rte_pktmbuf_alloc(mp);
    // if (pkt == NULL) {
    //     DOCA_LOG_ERR("Failed to allocate mbuf");
    //     return NULL;
    // }

    // 为整个包分配空间
    size_t total_size = sizeof(struct ether_hdr) + sizeof(struct ipv4_hdr) +
                       sizeof(struct udp_hdr) + sizeof(struct MatrixPacketHeader) +
                       chunk_size * sizeof(float);

    // prepare the packet
    char *payload = rte_pktmbuf_mtod(pkt, char *);

    // 1. prepare ethernet header
    struct ether_hdr *eth_hdr = (struct ether_hdr *)payload;
    memcpy(&eth_hdr->d_addr_bytes, net_info->net_info.eth_src_addr_bytes, ETHER_ADDR_LEN);
    memcpy(&eth_hdr->s_addr_bytes, net_info->net_info.eth_dst_addr_bytes, ETHER_ADDR_LEN);
    eth_hdr->ether_type = htons(DOCA_FLOW_ETHER_TYPE_IPV4);

    // 2. prepare ip header
    struct ipv4_hdr *ip_hdr = (struct ipv4_hdr *)(eth_hdr + 1);
    ip_hdr->version_ihl = 0x45; // IPv4, 5 * 4 bytes header length
    ip_hdr->type_of_service = 0;
    ip_hdr->total_length = htons(total_size - sizeof(struct ether_hdr));
    ip_hdr->packet_id = 0;
    ip_hdr->fragment_offset = 0;
    ip_hdr->time_to_live = 64;
    ip_hdr->next_proto_id = IPPROTO_UDP;
    ip_hdr->src_addr = inet_addr("10.134.11.66");
    ip_hdr->dst_addr = inet_addr("10.134.11.61");
    ip_hdr->hdr_checksum = 0;
    ip_hdr->hdr_checksum = rte_ipv4_cksum(ip_hdr);

    // 3. prepare the udp header
    struct udp_hdr *udp_hdr = (struct udp_hdr *)(ip_hdr + 1);
    udp_hdr->src_port = htons(1234);
    udp_hdr->dst_port = htons(5678);
    udp_hdr->dgram_len = htons(sizeof(struct udp_hdr) + sizeof(struct MatrixPacketHeader) +
                              chunk_size * sizeof(float));

    // 4. data header
    struct MatrixPacketHeader *mat_hdr = (struct MatrixPacketHeader *)(udp_hdr + 1);
    mat_hdr->matrix_id = matrix_id;
    mat_hdr->chunk_id = chunk_id;
    mat_hdr->total_chunks = total_chunks;
    mat_hdr->chunk_size = chunk_size;

    // 5. copy matrix
    float *matrix_data = (float *)(mat_hdr + 1);
    memcpy(matrix_data, data, chunk_size * sizeof(float));

    // 6. set the checksum
    udp_hdr->dgram_cksum = 0;
    udp_hdr->dgram_cksum = rte_ipv4_udptcp_cksum(ip_hdr, udp_hdr);

    // sent the packet
    pkt->data_len = total_size;
    pkt->pkt_len = pkt->data_len;

    return pkt;
}

/*
 * CPU thread to print statistics from GPU filtering on the console
 *
 * @args [in]: thread input args
 */
void send_by_cpu(int matrix_size, const struct MatrixCompletionInfo* completion_info,float* C,uint16_t dpdk_dev_port_id,struct rte_mempool *mp)
{
    float *C_CPU = (float *)malloc(matrix_size * matrix_size * sizeof(float));
    if (C_CPU) {
        cudaMemcpy(C_CPU, C, matrix_size * matrix_size * sizeof(float), cudaMemcpyDeviceToHost);
        size_t total_floats = matrix_size * matrix_size;
        uint32_t total_chunks = (total_floats + MAX_FLOATS_PER_PACKET - 1) / MAX_FLOATS_PER_PACKET;

        printf("Sending result matrix in %u chunks\n", total_chunks);

        // 分块发送结果矩阵
        for (uint32_t chunk_id = 0; chunk_id < total_chunks; chunk_id++) {
            size_t start_idx = chunk_id * MAX_FLOATS_PER_PACKET;
            size_t chunk_size = (start_idx + MAX_FLOATS_PER_PACKET > total_floats) ?
                              (total_floats - start_idx) : MAX_FLOATS_PER_PACKET;

            struct rte_mbuf *pkt = prepare_matrix_packet(
                mp,
                C_CPU + start_idx,
                2,
                chunk_id,
                total_chunks,
                chunk_size,
                completion_info
            );

            if (pkt == NULL) {
                printf("Failed to prepare packet for chunk %u\n", chunk_id);
                continue;
            }

            // 发送包
            uint16_t nb_tx = rte_eth_tx_burst(dpdk_dev_port_id, 0, &pkt, 1);
            if (nb_tx == 0) {
                printf("Failed to send packet for chunk %u\n", chunk_id);
                rte_pktmbuf_free(pkt);
            } else {
                //printf("Sent chunk %u/%u for result matrix\n", chunk_id + 1, total_chunks);
            }

            // add a small delay[
            rte_delay_us_block(1000);
        }


        printf("First few elements of result matrix C:\n");
        int elements_to_print = (5 < matrix_size * matrix_size) ? 5 : (matrix_size * matrix_size);
        for (int i = 0; i < elements_to_print; i++) {
            printf("C[%d]=%f\n", i, C_CPU[i]);
        }
        free(C_CPU);
    }
};
#endif //SEND_BY_CPU_H
