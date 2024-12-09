// main.c
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_malloc.h>
#include <cublas_v2.h>
#include "mat_message.h"
#include "defines.h"
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_udp.h>

static inline int is_target_packet(struct rte_mbuf *mbuf) {
    // 1. 获取以太网头
    struct rte_ether_hdr *eth_hdr = rte_pktmbuf_mtod(mbuf, struct rte_ether_hdr *);

    // 检查是否是IPv4包
    if (rte_be_to_cpu_16(eth_hdr->ether_type) != RTE_ETHER_TYPE_IPV4) {
        return 0;
    }

    // 2. 获取IP头
    struct rte_ipv4_hdr *ip_hdr = (struct rte_ipv4_hdr *)(eth_hdr + 1);

    // 检查是否是UDP包
    if (ip_hdr->next_proto_id != IPPROTO_UDP) {
        return 0;
    }

    // 3. 获取UDP头
    struct rte_udp_hdr *udp_hdr = (struct rte_udp_hdr *)((unsigned char *)ip_hdr +
                                  (ip_hdr->version_ihl & RTE_IPV4_HDR_IHL_MASK) * 4);

    // 检查目标端口是否为2574
    if (rte_be_to_cpu_16(udp_hdr->dst_port) != 2574) {
        return 0;
    }

    return 1;
}
#define RX_RING_SIZE 1024
#define TX_RING_SIZE 1024
#define NUM_MBUFS 8191
#define MBUF_CACHE_SIZE 250
#define BURST_SIZE 32

// GPU config
#define MAX_BLOCKS 65535
#define THREADS_PER_BLOCK 256

static struct rte_mempool *mbuf_pool;
static cublasHandle_t cublas_handle;

// Matrix storage
static float *matrix_a = NULL;
static float *matrix_b = NULL;
static float *matrix_c = NULL;
static float *d_matrix_a = NULL;
static float *d_matrix_b = NULL;
static float *d_matrix_c = NULL;

static struct MatrixCompletionInfo completion_info = {0};

static int port_init(uint16_t port, struct rte_mempool *mbuf_pool) {
    struct rte_eth_conf port_conf = {0};  // 使用memset方式初始化

    // 配置基本参数
    port_conf.rxmode.mtu = RTE_ETHER_MAX_LEN;

    // 如果需要配置RSS等功能，可以在这里添加
    /*
    port_conf.rx_adv_conf.rss_conf.rss_key = NULL;
    port_conf.rx_adv_conf.rss_conf.rss_hf = ETH_RSS_IP;
    */

    const uint16_t rx_rings = 1, tx_rings = 1;

    // 配置网卡
    int ret = rte_eth_dev_configure(port, rx_rings, tx_rings, &port_conf);
    if (ret < 0) {
        printf("port configure failed, err=%d\n", ret);
        return ret;
    }

    // 配置接收队列
    struct rte_eth_rxconf rxconf = {0};
    ret = rte_eth_rx_queue_setup(port, 0, RX_RING_SIZE,
                                rte_eth_dev_socket_id(port),
                                &rxconf, mbuf_pool);
    if (ret < 0) {
        printf("rx queue setup failed, err=%d\n", ret);
        return ret;
    }

    // 配置发送队列
    struct rte_eth_txconf txconf = {0};
    ret = rte_eth_tx_queue_setup(port, 0, TX_RING_SIZE,
                                rte_eth_dev_socket_id(port),
                                &txconf);
    if (ret < 0) {
        printf("tx queue setup failed, err=%d\n", ret);
        return ret;
    }

    // 启动设备
    ret = rte_eth_dev_start(port);
    if (ret < 0) {
        printf("port start failed, err=%d\n", ret);
        return ret;
    }

    // 设置混杂模式（如果需要的话）
    // rte_eth_promiscuous_enable(port);

    return 0;
}

static void process_matrix_packet(struct MatrixMessage *msg) {
    if (!isValidMatrixPacket(&msg->header)) {
        return;
    }

    float *dst_matrix = (msg->header.matrix_id == 0) ? matrix_a : matrix_b;
    uint32_t offset = msg->header.chunk_id * MAX_FLOATS_PER_PACKET;
    memcpy(dst_matrix + offset, msg->payload, msg->header.chunk_size * sizeof(float));

    if (msg->header.matrix_id == 0) {
        completion_info.received_a_elems += msg->header.chunk_size;
    } else {
        completion_info.received_b_elems += msg->header.chunk_size;
    }
    completion_info.received_chunk_num++;
}

static int check_matrix_completion(void) {
    return (completion_info.received_a_elems == completion_info.total_chunks_a * MAX_FLOATS_PER_PACKET &&
            completion_info.received_b_elems == completion_info.total_chunks_b * MAX_FLOATS_PER_PACKET);
}

static void perform_matrix_multiplication(uint32_t m, uint32_t n, uint32_t k) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Copy matrices to GPU
    cudaMemcpy(d_matrix_a, matrix_a, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix_b, matrix_b, k * n * sizeof(float), cudaMemcpyHostToDevice);

    // Perform SGEMM
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,
                &alpha,
                d_matrix_b, n,
                d_matrix_a, k,
                &beta,
                d_matrix_c, n);

    // Copy result back to host
    cudaMemcpy(matrix_c, d_matrix_c, m * n * sizeof(float), cudaMemcpyDeviceToHost);
}

static void send_result_matrix(uint16_t port, uint32_t m, uint32_t n) {
    struct rte_mbuf *mbufs[BURST_SIZE];
    struct MatrixMessage *msg;
    uint32_t total_elements = m * n;
    uint32_t total_chunks = calcRequiredChunks(total_elements);
    uint32_t elements_sent = 0;

    while (elements_sent < total_elements) {
        uint32_t chunk_size = RTE_MIN(MAX_FLOATS_PER_PACKET, total_elements - elements_sent);
        struct rte_mbuf *mbuf = rte_pktmbuf_alloc(mbuf_pool);
        if (mbuf == NULL) {
            continue;
        }

        msg = (struct MatrixMessage *)rte_pktmbuf_append(mbuf,
            sizeof(struct MatrixPacketHeader) + chunk_size * sizeof(float));

        msg->header.matrix_id = 2; // Matrix C
        msg->header.chunk_id = elements_sent / MAX_FLOATS_PER_PACKET;
        msg->header.total_chunks = total_chunks;
        msg->header.chunk_size = chunk_size;

        memcpy(msg->payload, matrix_c + elements_sent, chunk_size * sizeof(float));
        elements_sent += chunk_size;

        // Set packet metadata (UDP/IP headers)
        // This is simplified - you need to implement proper header setting

        rte_eth_tx_burst(port, 0, &mbuf, 1);
    }
}

int main(int argc, char *argv[]) {
    uint16_t port_id = 0;

    // Initialize EAL
    if (rte_eal_init(argc, argv) < 0) {
        return -1;
    }

    // Create mempool
    mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL", NUM_MBUFS,
        MBUF_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());

    if (mbuf_pool == NULL) {
        return -1;
    }

    // Initialize port
    if (port_init(port_id, mbuf_pool) != 0) {
        return -1;
    }

    // Initialize CUBLAS
    cublasCreate(&cublas_handle);

    // Allocate host matrices
    matrix_a = (float *)rte_malloc(NULL, MAX_MATRIX_ELEMENTS * sizeof(float), 64);
    matrix_b = (float *)rte_malloc(NULL, MAX_MATRIX_ELEMENTS * sizeof(float), 64);
    matrix_c = (float *)rte_malloc(NULL, MAX_MATRIX_ELEMENTS * sizeof(float), 64);

    // Allocate device matrices
    cudaMalloc(&d_matrix_a, MAX_MATRIX_ELEMENTS * sizeof(float));
    cudaMalloc(&d_matrix_b, MAX_MATRIX_ELEMENTS * sizeof(float));
    cudaMalloc(&d_matrix_c, MAX_MATRIX_ELEMENTS * sizeof(float));

    struct rte_mbuf *bufs[BURST_SIZE];

    while (1) {
        struct rte_mbuf *bufs[BURST_SIZE];
        const uint16_t nb_rx = rte_eth_rx_burst(port_id, 0, bufs, BURST_SIZE);

        if (nb_rx == 0) {
            continue;
        }

        for (uint16_t i = 0; i < nb_rx; i++) {
            if (!is_target_packet(bufs[i])) {
                // 不是目标包，直接释放
                rte_pktmbuf_free(bufs[i]);
                continue;
            }

            // 是目标包，进行处理
            struct MatrixMessage *msg = rte_pktmbuf_mtod_offset(bufs[i],
                                      struct MatrixMessage *,
                                      sizeof(struct rte_ether_hdr) +
                                      sizeof(struct rte_ipv4_hdr) +
                                      sizeof(struct rte_udp_hdr));

            process_matrix_packet(msg);
            rte_pktmbuf_free(bufs[i]);
        }
    }

    // Cleanup
    cublasDestroy(cublas_handle);
    rte_free(matrix_a);
    rte_free(matrix_b);
    rte_free(matrix_c);
    cudaFree(d_matrix_a);
    cudaFree(d_matrix_b);
    cudaFree(d_matrix_c);

    return 0;
}