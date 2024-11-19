/*
 * Copyright (c) 2023-2024 NVIDIA CORPORATION AND AFFILIATES.
 * [License text preserved...]
 */

#include <stdlib.h>
#include <string.h>
#include <rte_ethdev.h>

#include "common.h"
#include "dpdk_tcp/tcp_session_table.h"

#define SLEEP_IN_NANOS (10 * 1000) /* Sample the PE every 10 microseconds */

DOCA_LOG_REGISTER(GPU_PACKET_PROCESSING);

bool force_quit;
static struct doca_gpu *gpu_dev;
static struct app_gpu_cfg app_cfg = {0};
static struct doca_dev *ddev;
static uint16_t dpdk_dev_port_id;
static struct tcp_bw_queues tcp_bw_queues;
static struct doca_flow_port *df_port;
static struct doca_pe *pe;

/*
 * DOCA PE callback for packet send errors
 */
void error_send_packet_cb(struct doca_eth_txq_gpu_event_error_send_packet *event_error, union doca_data event_user_data)
{
    uint16_t packet_index;

    doca_eth_txq_gpu_event_error_send_packet_get_position(event_error, &packet_index);
    DOCA_LOG_INFO("Error in send queue %ld, packet %d. Gracefully killing the app",
                 event_user_data.u64,
                 packet_index);
    DOCA_GPUNETIO_VOLATILE(force_quit) = true;
}
static uint64_t tcp_last_packet_time = 0;  // 记录上一个包的时间戳

void debug_send_packet_tcp_bw_cb(struct doca_eth_txq_gpu_event_notify_send_packet *event_notify,
                                union doca_data event_user_data)
{
    uint16_t packet_index;
    uint64_t packet_timestamp;
    uint64_t ts_diff = 0;

    // 获取包的位置信息
    doca_eth_txq_gpu_event_notify_send_packet_get_position(event_notify, &packet_index);

    // 获取包的时间戳
    doca_eth_txq_gpu_event_notify_send_packet_get_timestamp(event_notify, &packet_timestamp);

    // 计算与上一个包的时间差
    if (tcp_last_packet_time != 0) {
        ts_diff = packet_timestamp - tcp_last_packet_time;
    }

    // 记录调试信息
    DOCA_LOG_INFO("TCP BW debug event: Queue %ld packet %d sent at %ld, interval %.6f sec",
                  event_user_data.u64,
                  packet_index,
                  packet_timestamp,
                  (double)((ts_diff > 0 ? ((double)ts_diff) / 1000000000.0 : 0)));

    // 更新最后一个包的时间戳
    tcp_last_packet_time = packet_timestamp;
}
/*
 * Signal handler for graceful shutdown
 */
static void signal_handler(int signum)
{
    if (signum == SIGINT || signum == SIGTERM) {
        DOCA_LOG_INFO("Signal %d received, preparing to exit!", signum);
        DOCA_GPUNETIO_VOLATILE(force_quit) = true;
    }
}

int main(int argc, char **argv)
{
    doca_error_t result;
    int cuda_id;
    cudaError_t cuda_ret;
    struct doca_log_backend *sdk_log;
    struct timespec ts = {
        .tv_sec = 0,
        .tv_nsec = SLEEP_IN_NANOS,
    };

    /* Register logger backends */
    result = doca_log_backend_create_standard();
    if (result != DOCA_SUCCESS)
        return EXIT_FAILURE;

    result = doca_log_backend_create_with_file_sdk(stderr, &sdk_log);
    if (result != DOCA_SUCCESS)
        return EXIT_FAILURE;

    result = doca_log_backend_set_sdk_level(sdk_log, DOCA_LOG_LEVEL_WARNING);
    if (result != DOCA_SUCCESS)
        return EXIT_FAILURE;

    DOCA_LOG_INFO("===========================================================");
    DOCA_LOG_INFO("DOCA version: %s", doca_version());
    DOCA_LOG_INFO("===========================================================");

    /* Parse application arguments */
    result = doca_argp_init("doca_tcp_bw_processing", &app_cfg);
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

    DOCA_LOG_INFO("Options enabled:\n\tGPU %s\n\tNIC %s\n\tGPU Rx queues %d",
                  app_cfg.gpu_pcie_addr,
                  app_cfg.nic_pcie_addr,
                  app_cfg.queue_num);

    /* Initialize GPU */
    cuda_ret = cudaDeviceGetByPCIBusId(&cuda_id, app_cfg.gpu_pcie_addr);
    if (cuda_ret != cudaSuccess) {
        DOCA_LOG_ERR("Invalid GPU bus id provided %s", app_cfg.gpu_pcie_addr);
        return DOCA_ERROR_INVALID_VALUE;
    }

    cudaFree(0);
    cudaSetDevice(cuda_id);

    /* Initialize DOCA devices */
    result = init_doca_device(app_cfg.nic_pcie_addr, &ddev, &dpdk_dev_port_id);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to initialize DOCA device: %s", doca_error_get_descr(result));
        return EXIT_FAILURE;
    }

    result = doca_gpu_create(app_cfg.gpu_pcie_addr, &gpu_dev);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create GPU device: %s", doca_error_get_descr(result));
        return EXIT_FAILURE;
    }

    /* Initialize DOCA flow */
    df_port = init_doca_flow(dpdk_dev_port_id, app_cfg.queue_num);
    if (df_port == NULL) {
        DOCA_LOG_ERR("Failed to initialize DOCA flow");
        return EXIT_FAILURE;
    }

    /* Create Processing Engine */
    result = doca_pe_create(&pe);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create PE: %s", doca_error_get_descr(result));
        return EXIT_FAILURE;
    }
    /*doca_error_t create_tcp_bw_queues(struct tcp_bw_queues *tcp_ack_queues,
                                  struct doca_flow_port *df_port,
                                  struct doca_gpu *gpu_dev,
                                  struct doca_dev *ddev,
                                  uint32_t queue_num,
                                  struct doca_pe *pe,
                                  doca_eth_txq_gpu_event_error_send_packet_cb_t event_error_send_packet_cb,
                                  doca_eth_txq_gpu_event_notify_send_packet_cb_t event_notify_send_packet_cb)*/
    /* Create TCP bandwidth queues */
    result = create_tcp_bw_queues(&tcp_bw_queues, df_port, gpu_dev, ddev,
                                 app_cfg.queue_num, pe, &error_send_packet_cb,  &debug_send_packet_tcp_bw_cb);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create TCP BW queues: %s", doca_error_get_descr(result));
        return EXIT_FAILURE;
    }

    /* Create root pipe for TCP processing */
    result = create_tcp_bw_root_pipe(&tcp_bw_queues, df_port);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create TCP BW root pipe: %s", doca_error_get_descr(result));
        return EXIT_FAILURE;
    }

    /* Set up signal handlers */
    DOCA_GPUNETIO_VOLATILE(force_quit) = false;
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    /* Create and initialize CUDA streams */
    cudaStream_t rx_stream;
    uint32_t *cpu_exit_condition;
    uint32_t *gpu_exit_condition;

    if (cudaStreamCreateWithFlags(&rx_stream, cudaStreamNonBlocking) != cudaSuccess) {
        DOCA_LOG_ERR("Failed to create CUDA stream");
        return EXIT_FAILURE;
    }

    /* Allocate exit condition memory */
    result = doca_gpu_mem_alloc(gpu_dev, sizeof(uint32_t), 4096,
                               DOCA_GPU_MEM_TYPE_GPU_CPU,
                               (void **)&gpu_exit_condition,
                               (void **)&cpu_exit_condition);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to allocate GPU memory: %s", doca_error_get_descr(result));
        return EXIT_FAILURE;
    }
    *cpu_exit_condition = 0;

    /* Launch main processing kernel */
    DOCA_LOG_INFO("Launching TCP BW processing kernel");
    kernel_tcp_bw_test(rx_stream, gpu_exit_condition, &tcp_bw_queues);

    /* Main processing loop */
    while (DOCA_GPUNETIO_VOLATILE(force_quit) == false) {
        doca_pe_progress(pe);
        nanosleep(&ts, &ts);
    }

    /* Cleanup */
    DOCA_GPUNETIO_VOLATILE(*cpu_exit_condition) = 1;
    cudaStreamSynchronize(rx_stream);
    cudaStreamDestroy(rx_stream);
    doca_gpu_mem_free(gpu_dev, gpu_exit_condition);

    DOCA_LOG_INFO("GPU work ended");

    /* Clean up DOCA resources */
    result = destroy_tcp_bw_queues(&tcp_bw_queues);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to destroy TCP BW queues: %s", doca_error_get_descr(result));
        return EXIT_FAILURE;
    }

    result = doca_gpu_destroy(gpu_dev);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to destroy GPU: %s", doca_error_get_descr(result));
        return EXIT_FAILURE;
    }

    result = doca_pe_destroy(pe);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to destroy PE: %s", doca_error_get_descr(result));
        return EXIT_FAILURE;
    }

    doca_dev_close(ddev);

    DOCA_LOG_INFO("Application finished successfully");
    return EXIT_SUCCESS;
}