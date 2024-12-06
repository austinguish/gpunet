//
// Created by yiwei on 24-12-2.
//
/*
 * Copyright (c) 2023-2024 NVIDIA CORPORATION AND AFFILIATES.  All rights reserved.
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
#include <rte_ethdev.h>

#include "common.h"
#include "dpdk_tcp/tcp_session_table.h"
#include "dpdk_tcp/tcp_cpu_rss_func.h"
#include "matmul/mat_message.h"
#include "packets.h"


#define SLEEP_IN_NANOS (10 * 1000) /* Sample the PE every 10 microseconds  */

DOCA_LOG_REGISTER(GPU_PACKET_PROCESSING);
static struct rte_mbuf* prepare_matrix_packet(struct rte_mempool *mp,
                                            const float *data,
                                            uint32_t matrix_id,
                                            uint32_t chunk_id,
                                            uint32_t total_chunks,
                                            uint32_t chunk_size,
                                            const struct MatrixCompletionInfo *net_info) {
    struct rte_mbuf *pkt = rte_pktmbuf_alloc(mp);
    if (pkt == NULL) {
        DOCA_LOG_ERR("Failed to allocate mbuf");
        return NULL;
    }

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
bool force_quit;
static struct doca_gpu *gpu_dev;
static struct app_gpu_cfg app_cfg = {0};
static struct doca_dev *ddev;
static uint16_t dpdk_dev_port_id;
static struct rxq_udp_bw_queues udp_queues;
static struct doca_flow_port *df_port;
static struct doca_pe *pe;
#include <cublas_v2.h>
#include <math.h>
float *A,*B,*C;
cudaStream_t compute_stream;
cudaStream_t rx_udp_stream;
cudaStream_t tx_udp_stream;
uint32_t *rx_cpu_exit_condition;
uint32_t *rx_gpu_exit_condition;
/* Function to perform matrix multiplication using cuBLAS with dynamic matrix size */
cudaError_t perform_matrix_multiplication(float* A, float* B, float* C, int matrix_size, cudaStream_t stream) {
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

    // Verify input data
 //    float *test_A = (float *)malloc(matrix_size * matrix_size * sizeof(float));
 //    cudaMemcpy(test_A, A, matrix_size * matrix_size * sizeof(float), cudaMemcpyDeviceToHost);
 //    printf("First 5 elements of A: ");
 //    for (int i = 0; i < 16; i++) {
 //        printf("%f ", test_A[i]);
 //    }
 //    printf("\n");
 //    free(test_A);
 //
	// float *test_B = (float *)malloc(matrix_size * matrix_size * sizeof(float));
	// cudaMemcpy(test_B, B, matrix_size * matrix_size * sizeof(float), cudaMemcpyDeviceToHost);
	// printf("First 5 elements of B: ");
	// for (int i = 0; i < 16; i++) {
	// 	printf("%f ", test_B[i]);
	// }
	// printf("\n");
	// free(test_B);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Record start time
    cudaEventRecord(start, stream);
    printf("Starting SGEMM operation...\n");
	cudaStreamSynchronize(rx_udp_stream);
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

/*
 * DOCA PE callback to be invoked if any Eth Txq get an error
 * sending packets.
 *
 * @event_error [in]: DOCA PE event error handler
 * @event_user_data [in]: custom user data set at registration time
 */
void error_send_udp_packet_cb(struct doca_eth_txq_gpu_event_error_send_packet *event_error, union doca_data event_user_data)
{
	uint16_t packet_index;

	doca_eth_txq_gpu_event_error_send_packet_get_position(event_error, &packet_index);
	DOCA_LOG_INFO("Error in send queue %ld, packet %d. Gracefully killing the app",
		      event_user_data.u64,
		      packet_index);
	DOCA_GPUNETIO_VOLATILE(force_quit) = true;
}

/*
 * DOCA PE callback to be invoked on ICMP Eth Txq to get the debug info
 * when sending packets
 *
 * @event_notify [in]: DOCA PE event debug handler
 * @event_user_data [in]: custom user data set at registration time
 */
/*
 * Get timestamp in nanoseconds
 *
 * @sec [out]: seconds
 * @return: UTC timestamp
 */
static uint64_t get_ns(uint64_t *sec)
{
	struct timespec t;
	int ret;

	ret = clock_gettime(CLOCK_REALTIME, &t);
	if (ret != 0)
		exit(EXIT_FAILURE);

	(*sec) = (uint64_t)t.tv_sec;

	return (uint64_t)t.tv_nsec + (uint64_t)t.tv_sec * 1000 * 1000 * 1000;
}

/*
 * CPU thread to print statistics from GPU filtering on the console
 *
 * @args [in]: thread input args
 */
static void stats_core(void *args)
{
	(void)args;

	doca_error_t result = DOCA_SUCCESS;
	enum doca_gpu_semaphore_status status;
	struct MatrixCompletionInfo udp_st[MAX_QUEUES] = {0};
	uint32_t sem_idx_udp[MAX_QUEUES] = {0};
	uint64_t start_time_sec = 0;
	uint64_t interval_print = 0;
	uint64_t interval_sec = 0;
	struct MatrixCompletionInfo *completion_info;

	DOCA_LOG_INFO("Core %u is reporting filter stats", rte_lcore_id());
	get_ns(&start_time_sec);
	interval_print = get_ns(&interval_sec);
	while (DOCA_GPUNETIO_VOLATILE(force_quit) == false) {
		/* Check UDP packets */
		for (int idxq = 0; idxq < udp_queues.numq; idxq++) {
			result = doca_gpu_semaphore_get_status(udp_queues.sem_cpu[idxq], sem_idx_udp[idxq], &status);
			if (result != DOCA_SUCCESS) {
				DOCA_LOG_ERR("UDP semaphore error");
				DOCA_GPUNETIO_VOLATILE(force_quit) = true;
				return;
			}

			if (status == DOCA_GPU_SEMAPHORE_STATUS_READY) {
				// printf("all the packets are received\n");
				// print the ip address

				result = doca_gpu_semaphore_get_custom_info_addr(udp_queues.sem_cpu[idxq],
										 sem_idx_udp[idxq],
										 (void **)&(completion_info));


				if (result != DOCA_SUCCESS) {
					DOCA_LOG_ERR("UDP semaphore get address error");
					DOCA_GPUNETIO_VOLATILE(force_quit) = true;
					return;
				}
				printf("\n=== Ethernet Header ===\n");
				printf("Destination MAC: %02x:%02x:%02x:%02x:%02x:%02x\n",
					   completion_info->net_info.eth_dst_addr_bytes[0],  completion_info->net_info.eth_dst_addr_bytes[1],
					    completion_info->net_info.eth_dst_addr_bytes[2],  completion_info->net_info.eth_dst_addr_bytes[3],
					    completion_info->net_info.eth_dst_addr_bytes[4],  completion_info->net_info.eth_dst_addr_bytes[5]);
				printf("Source MAC: %02x:%02x:%02x:%02x:%02x:%02x\n",
					    completion_info->net_info.eth_src_addr_bytes[0], completion_info->net_info.eth_src_addr_bytes[1],
					   completion_info->net_info.eth_src_addr_bytes[2], completion_info->net_info.eth_src_addr_bytes[3],
					   completion_info->net_info.eth_src_addr_bytes[4], completion_info->net_info.eth_src_addr_bytes[5]);
				printf("\n=== IPv4 Header ===\n");
				unsigned char *src_ip = (unsigned char *)&completion_info->net_info.ip_src_addr;
				unsigned char *dst_ip = (unsigned char *)&completion_info->net_info.ip_dst_addr;
				printf("Source IP: %u.%u.%u.%u\n",
					   src_ip[0], src_ip[1], src_ip[2], src_ip[3]);
				printf("Destination IP: %u.%u.%u.%u\n",
					   dst_ip[0], dst_ip[1], dst_ip[2], dst_ip[3]);

				udp_st[idxq].received_a_elems += completion_info->received_a_elems;
				udp_st[idxq].received_b_elems += completion_info->received_b_elems;
				udp_st[idxq].total_chunks_a = completion_info->total_chunks_a;
				udp_st[idxq].total_chunks_b = completion_info->total_chunks_b;
				// todo maybe in different block
				udp_st[idxq].received_chunk_num += completion_info->received_chunk_num;
                if (udp_st[idxq].received_chunk_num == 2*udp_st[idxq].total_chunks_a)
                {

                	printf("received all the chunks great\n");
                	printf("UDP receive paused, starting matrix multiplication\n");
                	DOCA_GPUNETIO_VOLATILE(*rx_cpu_exit_condition) = 1;
                	// Calculate matrix dimensions based on received chunks
                	int matrix_size = (int)sqrt((double)udp_st[idxq].received_a_elems);
                	printf("Matrix size calculated as %d x %d\n", matrix_size, matrix_size);

                	// Perform matrix multiplication in separate stream
                	cudaError_t err = perform_matrix_multiplication(A, B, C, matrix_size, compute_stream);
                	if (err != cudaSuccess) {
                		printf("Matrix multiplication failed: %s\n", cudaGetErrorString(err));
                		DOCA_GPUNETIO_VOLATILE(force_quit) = true;
                		return;
                	}

                	// Wait for computation to complete
                	cudaStreamSynchronize(compute_stream);
                	DOCA_GPUNETIO_VOLATILE(*rx_cpu_exit_condition) = 0;
                	// call the send function
                	// todo need to debug to see what happend
                	//kernel_send_matrix_c(tx_udp_stream,&udp_queues,C,udp_st[idxq].total_chunks_a,udp_st[idxq].received_a_elems,rx_gpu_exit_condition,completion_info->net_info);
                	// Copy results back to CPU for verification
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

                            // 准备并发送包
                            struct rte_mbuf *pkt = prepare_matrix_packet(
                                &udp_queues.send_pkt_pool[0],
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

                            // 添加小延迟避免包丢失
                            rte_delay_us_block(1000);  // 1ms delay
                        }


                		printf("First few elements of result matrix C:\n");
                		int elements_to_print = (5 < matrix_size * matrix_size) ? 5 : (matrix_size * matrix_size);
                		for (int i = 0; i < elements_to_print; i++) {
                			printf("C[%d]=%f\n", i, C_CPU[i]);
                		}
                		free(C_CPU);
                	}

                	DOCA_GPUNETIO_VOLATILE(*rx_cpu_exit_condition) = 0;
                	// kernel_receive_udp_bw(rx_udp_stream, gpu_exit_condition, &udp_queues,A,B);
                }



				result = doca_gpu_semaphore_set_status(udp_queues.sem_cpu[idxq],
								       sem_idx_udp[idxq],
								       DOCA_GPU_SEMAPHORE_STATUS_FREE);
				if (result != DOCA_SUCCESS) {
					DOCA_LOG_ERR("UDP semaphore %d error", sem_idx_udp[idxq]);
					DOCA_GPUNETIO_VOLATILE(force_quit) = true;
					return;
				}

				sem_idx_udp[idxq] = (sem_idx_udp[idxq] + 1) % udp_queues.nums;
			}
		}

		/* Check TCP packets */

		if ((get_ns(&interval_sec) - interval_print) > 5000000000) {
			printf("\nSeconds %ld\n", interval_sec - start_time_sec);

			for (int idxq = 0; idxq < udp_queues.numq; idxq++) {
				printf("[UDP] QUEUE: %d A: %u B: %u TOTAL A: %u TOTAL B: %u\n",
				       idxq,
				       udp_st[idxq].received_a_elems,
				       udp_st[idxq].received_b_elems,
				       udp_st[idxq].total_chunks_a,udp_st[idxq].total_chunks_b);
			}

			interval_print = get_ns(&interval_sec);
		}
	}
}

/*
 * Signal handler to quit application gracefully
 *
 * @signum [in]: signal received
 */
static void signal_handler(int signum)
{
	if (signum == SIGINT || signum == SIGTERM) {
		DOCA_LOG_INFO("Signal %d received, preparing to exit!", signum);
		DOCA_GPUNETIO_VOLATILE(force_quit) = true;
	}
}

/*
 * GPU packet processing application main function
 *
 * @argc [in]: command line arguments size
 * @argv [in]: array of command line arguments
 * @return: EXIT_SUCCESS on success and EXIT_FAILURE otherwise
 */
int main(int argc, char **argv)
{
	doca_error_t result;
	int current_lcore = 0;
	int cuda_id;
	cudaError_t cuda_ret;
	struct doca_log_backend *sdk_log;
	struct timespec ts = {
		.tv_sec = 0,
		.tv_nsec = SLEEP_IN_NANOS,
	};

	/* Register a logger backend */
	result = doca_log_backend_create_standard();
	if (result != DOCA_SUCCESS)
		return EXIT_FAILURE;

	/* Register a logger backend for internal SDK errors and warnings */
	result = doca_log_backend_create_with_file_sdk(stderr, &sdk_log);
	if (result != DOCA_SUCCESS)
		return EXIT_FAILURE;
	result = doca_log_backend_set_sdk_level(sdk_log, DOCA_LOG_LEVEL_WARNING);
	if (result != DOCA_SUCCESS)
		return EXIT_FAILURE;

	DOCA_LOG_INFO("===========================================================");
	DOCA_LOG_INFO("DOCA version: %s", doca_version());
	DOCA_LOG_INFO("===========================================================");

	/* Basic DPDK initialization */
	result = doca_argp_init("doca_gpu_packet_processing", &app_cfg);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init ARGP resources: %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	result = register_application_params();
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to parse application input: %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	result = doca_argp_start(argc, argv);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to parse application input: %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	DOCA_LOG_INFO("Options enabled:\n\tGPU %s\n\tNIC %s\n\tGPU Rx queues %d\n\tGPU HTTP server enabled %s",
		      app_cfg.gpu_pcie_addr,
		      app_cfg.nic_pcie_addr,
		      app_cfg.queue_num,
		      (app_cfg.http_server == true ? "Yes" : "No"));

	/* In a multi-GPU system, ensure CUDA refers to the right GPU device */
	cuda_ret = cudaDeviceGetByPCIBusId(&cuda_id, app_cfg.gpu_pcie_addr);
	if (cuda_ret != cudaSuccess) {
		DOCA_LOG_ERR("Invalid GPU bus id provided %s", app_cfg.gpu_pcie_addr);
		return DOCA_ERROR_INVALID_VALUE;
	}

	cudaFree(0);
	cudaSetDevice(cuda_id);

	result = init_doca_device(app_cfg.nic_pcie_addr, &ddev, &dpdk_dev_port_id);
	DOCA_LOG_INFO("port id is %u",dpdk_dev_port_id);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function init_doca_device returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	/* Initialize DOCA GPU instance */
	result = doca_gpu_create(app_cfg.gpu_pcie_addr, &gpu_dev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function doca_gpu_create returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	df_port = init_doca_flow(dpdk_dev_port_id, app_cfg.queue_num);
	if (df_port == NULL) {
		DOCA_LOG_ERR("FAILED: init_doca_flow");
		return EXIT_FAILURE;
	}

	result = doca_pe_create(&pe);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to create pe queue: %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}


	result = create_udp_bw_queues(&udp_queues, df_port, gpu_dev, ddev, pe, app_cfg.queue_num, SEMAPHORES_PER_QUEUE, error_send_udp_packet_cb);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function create_udp_queues returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	udp_queues.send_pkt_pool = rte_pktmbuf_pool_create("tcp_ack_pkt_pool",
									  1023,
									  0,
									  0,
									  RTE_MBUF_DEFAULT_BUF_SIZE,
									  rte_socket_id());

	/* Create root control pipe to route tcp/udp/OS packets */
	result = create_udp_only_root_pipe(&udp_queues, df_port);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function create_root_pipe returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	/* Gracefully terminate app if ctrlc */
	DOCA_GPUNETIO_VOLATILE(force_quit) = false;
	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);


	cudaError_t res_rt = cudaSuccess;


	res_rt = cudaStreamCreateWithFlags(&rx_udp_stream, cudaStreamNonBlocking);
	if (res_rt != cudaSuccess) {
		DOCA_LOG_ERR("Function cudaStreamCreateWithFlags error %d", res_rt);
		return EXIT_FAILURE;
	}


	// In main function, create compute_stream (near where rx_udp_stream is created)
	res_rt = cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);
	if (res_rt != cudaSuccess) {
		DOCA_LOG_ERR("Function cudaStreamCreateWithFlags for compute_stream error %d", res_rt);
		return EXIT_FAILURE;
	}
	res_rt = cudaStreamCreateWithFlags(&tx_udp_stream, cudaStreamNonBlocking);
	if (res_rt != cudaSuccess) {
		DOCA_LOG_ERR("Function cudaStreamCreateWithFlags for tx_udp_stream error %d", res_rt);
		return EXIT_FAILURE;
	}



	result = doca_gpu_mem_alloc(gpu_dev,
				    sizeof(uint32_t),
				    4096,
				    DOCA_GPU_MEM_TYPE_GPU_CPU,
				    (void **)&rx_gpu_exit_condition,
				    (void **)&rx_cpu_exit_condition);
	if (result != DOCA_SUCCESS || rx_gpu_exit_condition == NULL || rx_cpu_exit_condition == NULL) {
		DOCA_LOG_ERR("Function doca_gpu_mem_alloc returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}
	rx_cpu_exit_condition[0] = 0;

	size_t size = MAX_MATRIX_DIMENSION*MAX_MATRIX_DIMENSION*sizeof(float);
	cudaError_t err = cudaMalloc((void **)&A,size);
	if (err != cudaSuccess) {
		printf("allocate cuda mem failed: %s\n", cudaGetErrorString(err));
		return -1;
	}

	err = cudaMalloc((void **)&B,size);
	if (err != cudaSuccess) {
		printf("allocate cuda mem for mat C failed: %s\n", cudaGetErrorString(err));
		return -1;
	}

	err = cudaMalloc((void **)&C,size);
	if (err != cudaSuccess) {
		printf("allocate cuda mem for mat C failed: %s\n", cudaGetErrorString(err));
		return -1;
	}
	// allocate one part for record the packet info
	struct eth_ip_udp_hdr *resp_hdr;
	err = cudaMalloc((void**)&resp_hdr,TX_BUF_MAX_SZ);
	if (err != cudaSuccess) {
		printf("allocate cuda mem for single packet failed: %s\n", cudaGetErrorString(err));
		return -1;
	}
	/*
	 * Some GPUs may require an initial warmup without doing any real operation.
	 */
	DOCA_LOG_INFO("Warm up CUDA kernels");
	DOCA_GPUNETIO_VOLATILE(*rx_cpu_exit_condition) = 1;
	 kernel_receive_udp_bw(rx_udp_stream, rx_gpu_exit_condition, &udp_queues,A,B);
	//kernel_receive_udp_bw(rx_udp_stream, gpu_exit_condition, &udp_queues);
	cudaStreamSynchronize(rx_udp_stream);
	DOCA_GPUNETIO_VOLATILE(*rx_cpu_exit_condition) = 0;

	DOCA_LOG_INFO("Launching CUDA kernels");

	kernel_receive_udp_bw(rx_udp_stream, rx_gpu_exit_condition, &udp_queues,A,B);
	/* Launch stats proxy thread to report pipeline status */
	current_lcore = rte_get_next_lcore(current_lcore, true, false);
	if (rte_eal_remote_launch((void *)stats_core, NULL, current_lcore) != 0) {
		DOCA_LOG_ERR("Remote launch failed");
		goto exit;
	}


	DOCA_LOG_INFO("Waiting for termination");
	/* This loop keeps busy main thread until force_quit is set to 1 (e.g. typing ctrl+c) */
	while (DOCA_GPUNETIO_VOLATILE(force_quit) == false) {
		doca_pe_progress(pe);
		nanosleep(&ts, &ts);
	}

	DOCA_GPUNETIO_VOLATILE(*rx_cpu_exit_condition) = 1;
	cudaStreamSynchronize(rx_udp_stream);
	cudaStreamDestroy(rx_udp_stream);
	cudaStreamDestroy(compute_stream);

	float *A_CPU = (float *)malloc(size);
	memset(A_CPU, 0, size);
	cudaMemcpy(A_CPU, A, size, cudaMemcpyDeviceToHost);
	for (int i = 0; i<5;i++)
	{
		printf("A[%d]=%f\n",i,A_CPU[i]);
	}
	// struct eth_ip_udp_hdr *resp_hdr_on_cpu = (struct eth_ip_udp_hdr *) malloc(TX_BUF_MAX_SZ);
	// cudaMemcpy(resp_hdr_on_cpu,resp_hdr,42,cudaMemcpyDeviceToHost);
	// traverse the hdr

	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	cudaFree(resp_hdr);

	// also to free the A_CPU
	free(A_CPU);

	doca_gpu_mem_free(gpu_dev, rx_gpu_exit_condition);

	DOCA_LOG_INFO("GPU work ended");

	current_lcore = 0;
	RTE_LCORE_FOREACH_WORKER(current_lcore)
	{
		if (rte_eal_wait_lcore(current_lcore) < 0) {
			DOCA_LOG_ERR("Bad exit for coreid: %d", current_lcore);
			break;
		}
	}






exit:
	result = destroy_flow_udp_only_queue(df_port, &udp_queues);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function finalize_doca_flow returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	result = doca_gpu_destroy(gpu_dev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to destroy GPU: %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	result = doca_pe_destroy(pe);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function doca_pe_destroy returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	doca_dev_close(ddev);

	DOCA_LOG_INFO("Application finished successfully");
    cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	cudaFree(resp_hdr);
	return EXIT_SUCCESS;
}
