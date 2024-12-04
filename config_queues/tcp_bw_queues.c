//
// Created by yiwei on 24-11-19.
//
#include <stdlib.h>
#include <string.h>
#include <rte_ethdev.h>
#include "common.h"

DOCA_LOG_REGISTER(GPU_PACKET_PROCESSING_TCP_ACK);

doca_error_t create_tcp_bw_queues(struct tcp_bw_queues* tcp_ack_queues,
                                  struct doca_flow_port* df_port,
                                  struct doca_gpu* gpu_dev,
                                  struct doca_dev* ddev,
                                  uint32_t queue_num,
                                  struct doca_pe* pe,
                                  doca_eth_txq_gpu_event_error_send_packet_cb_t event_error_send_packet_cb,
                                  doca_eth_txq_gpu_event_notify_send_packet_cb_t event_notify_send_packet_cb)
{
    uint32_t cyclic_buffer_size = 0;
    doca_error_t result;
    union doca_data event_user_data[MAX_QUEUES] = {0};

    if (tcp_ack_queues == NULL || df_port == NULL || gpu_dev == NULL || ddev == NULL || queue_num == 0)
    {
        DOCA_LOG_ERR("Can't create TCP ACK queues, invalid input");
        return DOCA_ERROR_INVALID_VALUE;
    }

    tcp_ack_queues->gpu_dev = gpu_dev;
    tcp_ack_queues->ddev = ddev;
    tcp_ack_queues->port = df_port;
    tcp_ack_queues->numq = queue_num;
    tcp_ack_queues->numq_cpu_rss = queue_num;

    for (uint32_t idx = 0; idx < queue_num; idx++)
    {
        DOCA_LOG_INFO("Creating TCP ACK Eth Rxq %d", idx);

        // Create receive queue
        result = doca_eth_rxq_create(tcp_ack_queues->ddev,
                                     MAX_PKT_NUM,
                                     MAX_PKT_SIZE,
                                     &(tcp_ack_queues->eth_rxq_cpu[idx]));
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed doca_eth_rxq_create: %s", doca_error_get_descr(result));
            destroy_tcp_bw_queues(tcp_ack_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        // Set cyclic buffer type
        result = doca_eth_rxq_set_type(tcp_ack_queues->eth_rxq_cpu[idx], DOCA_ETH_RXQ_TYPE_CYCLIC);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed doca_eth_rxq_set_type: %s", doca_error_get_descr(result));
            destroy_tcp_bw_queues(tcp_ack_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        // Calculate buffer size
        result = doca_eth_rxq_estimate_packet_buf_size(DOCA_ETH_RXQ_TYPE_CYCLIC,
                                                       0,
                                                       0,
                                                       MAX_PKT_SIZE,
                                                       MAX_PKT_NUM,
                                                       0,
                                                       &cyclic_buffer_size);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed to get eth_rxq cyclic buffer size: %s", doca_error_get_descr(result));
            destroy_tcp_bw_queues(tcp_ack_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        // Create mmap for packet buffer
        result = doca_mmap_create(&tcp_ack_queues->pkt_buff_mmap[idx]);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed to create mmap: %s", doca_error_get_descr(result));
            destroy_tcp_bw_queues(tcp_ack_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_mmap_add_dev(tcp_ack_queues->pkt_buff_mmap[idx], tcp_ack_queues->ddev);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed to add dev to mmap: %s", doca_error_get_descr(result));
            destroy_tcp_bw_queues(tcp_ack_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        // Allocate GPU memory
        result = doca_gpu_mem_alloc(tcp_ack_queues->gpu_dev,
                                    cyclic_buffer_size,
                                    GPU_PAGE_SIZE,
                                    DOCA_GPU_MEM_TYPE_GPU,
                                    &tcp_ack_queues->gpu_pkt_addr[idx],
                                    NULL);
        if (result != DOCA_SUCCESS || tcp_ack_queues->gpu_pkt_addr[idx] == NULL)
        {
            DOCA_LOG_ERR("Failed to allocate gpu memory %s", doca_error_get_descr(result));
            destroy_tcp_bw_queues(tcp_ack_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        // Try DMABuf first
        // result = doca_gpu_dmabuf_fd(tcp_ack_queues->gpu_dev,
        //                           tcp_ack_queues->gpu_pkt_addr[idx],
        //                           cyclic_buffer_size,
        //                           &(tcp_ack_queues->dmabuf_fd[idx]));
        //
        // if (result == DOCA_SUCCESS) {
        //     DOCA_LOG_INFO("Mapping receive queue buffer with dmabuf mode");
        //     result = doca_mmap_set_dmabuf_memrange(tcp_ack_queues->pkt_buff_mmap[idx],
        //                                          tcp_ack_queues->dmabuf_fd[idx],
        //                                          tcp_ack_queues->gpu_pkt_addr[idx],
        //                                          0,
        //                                          cyclic_buffer_size);
        // }

        // Fallback to nvidia-peermem if DMABuf fails

        DOCA_LOG_INFO("Mapping receive queue buffer with nvidia-peermem mode");
        result = doca_mmap_set_memrange(tcp_ack_queues->pkt_buff_mmap[idx],
                                        tcp_ack_queues->gpu_pkt_addr[idx],
                                        cyclic_buffer_size);


        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed to set memrange for mmap %s", doca_error_get_descr(result));
            destroy_tcp_bw_queues(tcp_ack_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        // Set mmap permissions
        result = doca_mmap_set_permissions(tcp_ack_queues->pkt_buff_mmap[idx],
                                           DOCA_ACCESS_FLAG_LOCAL_READ_WRITE |
                                           DOCA_ACCESS_FLAG_PCI_RELAXED_ORDERING);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed to set permissions for mmap");
            destroy_tcp_bw_queues(tcp_ack_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        // Start mmap
        result = doca_mmap_start(tcp_ack_queues->pkt_buff_mmap[idx]);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed to start mmap");
            destroy_tcp_bw_queues(tcp_ack_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        // Set packet buffer
        result = doca_eth_rxq_set_pkt_buf(tcp_ack_queues->eth_rxq_cpu[idx],
                                          tcp_ack_queues->pkt_buff_mmap[idx],
                                          0,
                                          cyclic_buffer_size);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed to set packet buffer");
            destroy_tcp_bw_queues(tcp_ack_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        // Get context and set GPU datapath
        tcp_ack_queues->eth_rxq_ctx[idx] = doca_eth_rxq_as_doca_ctx(tcp_ack_queues->eth_rxq_cpu[idx]);
        if (tcp_ack_queues->eth_rxq_ctx[idx] == NULL)
        {
            DOCA_LOG_ERR("Failed to get context");
            destroy_tcp_bw_queues(tcp_ack_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_ctx_set_datapath_on_gpu(tcp_ack_queues->eth_rxq_ctx[idx], tcp_ack_queues->gpu_dev);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed to set GPU datapath");
            destroy_tcp_bw_queues(tcp_ack_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_ctx_start(tcp_ack_queues->eth_rxq_ctx[idx]);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed to start context");
            destroy_tcp_bw_queues(tcp_ack_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        // Get GPU handle
        result = doca_eth_rxq_get_gpu_handle(tcp_ack_queues->eth_rxq_cpu[idx],
                                             &(tcp_ack_queues->eth_rxq_gpu[idx]));
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed to get GPU handle");
            destroy_tcp_bw_queues(tcp_ack_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        // Create transmit queue
        result = doca_eth_txq_create(tcp_ack_queues->ddev,
                                     MAX_SQ_DESCR_NUM,
                                     &(tcp_ack_queues->eth_txq_cpu[idx]));
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed to create transmit queue");
            destroy_tcp_bw_queues(tcp_ack_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        // Enable checksum offloads for TCP/IP
        result = doca_eth_txq_set_l3_chksum_offload(tcp_ack_queues->eth_txq_cpu[idx], 1);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed to set L3 checksum offload");
            destroy_tcp_bw_queues(tcp_ack_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_eth_txq_set_l4_chksum_offload(tcp_ack_queues->eth_txq_cpu[idx], 1);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed to set L4 checksum offload");
            destroy_tcp_bw_queues(tcp_ack_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        // Get transmit queue context
        tcp_ack_queues->eth_txq_ctx[idx] = doca_eth_txq_as_doca_ctx(tcp_ack_queues->eth_txq_cpu[idx]);
        if (tcp_ack_queues->eth_txq_ctx[idx] == NULL)
        {
            DOCA_LOG_ERR("Failed to get transmit queue context");
            destroy_tcp_bw_queues(tcp_ack_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        // Set transmit queue GPU datapath
        result = doca_ctx_set_datapath_on_gpu(tcp_ack_queues->eth_txq_ctx[idx], tcp_ack_queues->gpu_dev);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed to set transmit queue GPU datapath");
            destroy_tcp_bw_queues(tcp_ack_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        // Register callbacks if progress engine is provided
        if (pe != NULL)
        {
            event_user_data[idx].u64 = idx;
            result = doca_eth_txq_gpu_event_error_send_packet_register(tcp_ack_queues->eth_txq_cpu[idx],
                                                                       event_error_send_packet_cb,
                                                                       event_user_data[idx]);
            if (result != DOCA_SUCCESS)
            {
                DOCA_LOG_ERR("Failed to register error callback");
                destroy_tcp_bw_queues(tcp_ack_queues);
                return DOCA_ERROR_BAD_STATE;
            }

            result = doca_eth_txq_gpu_event_notify_send_packet_register(tcp_ack_queues->eth_txq_cpu[idx],
                                                                        event_notify_send_packet_cb,
                                                                        event_user_data[idx]);
            if (result != DOCA_SUCCESS)
            {
                DOCA_LOG_ERR("Failed to register notify callback");
                destroy_tcp_bw_queues(tcp_ack_queues);
                return DOCA_ERROR_BAD_STATE;
            }

            result = doca_pe_connect_ctx(pe, tcp_ack_queues->eth_txq_ctx[idx]);
            if (result != DOCA_SUCCESS)
            {
                DOCA_LOG_ERR("Failed to connect progress engine");
                destroy_tcp_bw_queues(tcp_ack_queues);
                return DOCA_ERROR_BAD_STATE;
            }
        }

        result = doca_ctx_start(tcp_ack_queues->eth_txq_ctx[idx]);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed to start transmit context");
            destroy_tcp_bw_queues(tcp_ack_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_eth_txq_get_gpu_handle(tcp_ack_queues->eth_txq_cpu[idx],
                                             &(tcp_ack_queues->eth_txq_gpu[idx]));
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed to get transmit GPU handle");
            destroy_tcp_bw_queues(tcp_ack_queues);
            return DOCA_ERROR_BAD_STATE;
        }
    }
    //allocate txbuf on the gpu
    result = create_tx_buf(&tcp_ack_queues->buf_response,
                           tcp_ack_queues->gpu_dev,
                           tcp_ack_queues->ddev,
                           TX_BUF_NUM,
                           TX_BUF_MAX_SZ);
    if (result != DOCA_SUCCESS)
    {
        DOCA_LOG_ERR("Failed create buf_page_index: %s", doca_error_get_descr(result));
        destroy_tcp_bw_queues(tcp_ack_queues);
        return DOCA_ERROR_BAD_STATE;
    }
    // copy the page content to the gpu

    result = prepare_tx_buf(&tcp_ack_queues->buf_response, HTTP_GET_INDEX);
    if (result != DOCA_SUCCESS)
    {
        DOCA_LOG_ERR("Failed prepare buf_page_index: %s", doca_error_get_descr(result));
        destroy_tcp_bw_queues(tcp_ack_queues);
        return DOCA_ERROR_BAD_STATE;
    }

    result = create_tcp_bw_pipe(tcp_ack_queues, df_port);
    if (result != DOCA_SUCCESS)
    {
        DOCA_LOG_ERR("Failed to create CPU pipe: %s", doca_error_get_descr(result));
        destroy_tcp_bw_queues(tcp_ack_queues);
        return DOCA_ERROR_BAD_STATE;
    }

    result = create_tcp_gpu_bw_pipe(tcp_ack_queues, df_port);
    if (result != DOCA_SUCCESS)
    {
        DOCA_LOG_ERR("Failed to create GPU pipe: %s", doca_error_get_descr(result));
        destroy_tcp_bw_queues(tcp_ack_queues);
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}

// Destroy function
doca_error_t destroy_tcp_bw_queues(struct tcp_bw_queues* tcp_ack_queues)
{
    doca_error_t result;

    if (tcp_ack_queues == NULL)
    {
        DOCA_LOG_ERR("Can't destroy TCP ACK queues, invalid input");
        return DOCA_ERROR_INVALID_VALUE;
    }

    // First destroy all flows and pipes to prevent queue references
    if (tcp_ack_queues->rxq_pipe_gpu)
    {
        doca_flow_pipe_destroy(tcp_ack_queues->rxq_pipe_gpu);
        tcp_ack_queues->rxq_pipe_gpu = NULL;
    }

    if (tcp_ack_queues->rxq_pipe_cpu)
    {
        doca_flow_pipe_destroy(tcp_ack_queues->rxq_pipe_cpu);
        tcp_ack_queues->rxq_pipe_cpu = NULL;
    }

    // Wait for all pending operations to complete
    result = doca_flow_entries_process(tcp_ack_queues->port, 0, NULL, NULL);
    if (result != DOCA_SUCCESS)
    {
        DOCA_LOG_ERR("Failed to process pending flow operations: %s", doca_error_get_descr(result));
        return DOCA_ERROR_BAD_STATE;
    }

    for (int idx = 0; idx < tcp_ack_queues->numq; idx++)
    {
        DOCA_LOG_INFO("Destroying TCP ACK queue %d", idx);

        // First stop the contexts before destroying queues
        if (tcp_ack_queues->eth_txq_ctx[idx])
        {
            result = doca_ctx_stop(tcp_ack_queues->eth_txq_ctx[idx]);
            if (result != DOCA_SUCCESS)
            {
                DOCA_LOG_ERR("Failed to stop transmit context: %s", doca_error_get_descr(result));
                return DOCA_ERROR_BAD_STATE;
            }
            tcp_ack_queues->eth_txq_ctx[idx] = NULL;
        }

        if (tcp_ack_queues->eth_rxq_ctx[idx])
        {
            result = doca_ctx_stop(tcp_ack_queues->eth_rxq_ctx[idx]);
            if (result != DOCA_SUCCESS)
            {
                DOCA_LOG_ERR("Failed to stop receive context: %s", doca_error_get_descr(result));
                return DOCA_ERROR_BAD_STATE;
            }
            tcp_ack_queues->eth_rxq_ctx[idx] = NULL;
        }

        // Then destroy the queues
        if (tcp_ack_queues->eth_txq_cpu[idx])
        {
            result = doca_eth_txq_destroy(tcp_ack_queues->eth_txq_cpu[idx]);
            if (result != DOCA_SUCCESS)
            {
                DOCA_LOG_ERR("Failed to destroy transmit queue: %s", doca_error_get_descr(result));
                return DOCA_ERROR_BAD_STATE;
            }
            tcp_ack_queues->eth_txq_cpu[idx] = NULL;
        }

        if (tcp_ack_queues->eth_rxq_cpu[idx])
        {
            result = doca_eth_rxq_destroy(tcp_ack_queues->eth_rxq_cpu[idx]);
            if (result != DOCA_SUCCESS)
            {
                DOCA_LOG_ERR("Failed to destroy receive queue: %s", doca_error_get_descr(result));
                return DOCA_ERROR_BAD_STATE;
            }
            tcp_ack_queues->eth_rxq_cpu[idx] = NULL;
        }

        // Then cleanup memory mappings
        if (tcp_ack_queues->pkt_buff_mmap[idx])
        {
            result = doca_mmap_stop(tcp_ack_queues->pkt_buff_mmap[idx]);
            if (result != DOCA_SUCCESS)
            {
                DOCA_LOG_ERR("Failed to stop mmap: %s", doca_error_get_descr(result));
                return DOCA_ERROR_BAD_STATE;
            }

            result = doca_mmap_destroy(tcp_ack_queues->pkt_buff_mmap[idx]);
            if (result != DOCA_SUCCESS)
            {
                DOCA_LOG_ERR("Failed to destroy mmap: %s", doca_error_get_descr(result));
                return DOCA_ERROR_BAD_STATE;
            }
            tcp_ack_queues->pkt_buff_mmap[idx] = NULL;
        }

        // Finally free GPU memory
        if (tcp_ack_queues->gpu_pkt_addr[idx])
        {
            result = doca_gpu_mem_free(tcp_ack_queues->gpu_dev, tcp_ack_queues->gpu_pkt_addr[idx]);
            if (result != DOCA_SUCCESS)
            {
                DOCA_LOG_ERR("Failed to free GPU memory: %s", doca_error_get_descr(result));
                return DOCA_ERROR_BAD_STATE;
            }
            tcp_ack_queues->gpu_pkt_addr[idx] = NULL;
        }
    }
    result = destroy_tx_buf(&tcp_ack_queues->buf_response);
    if (result != DOCA_SUCCESS)
    {
        DOCA_LOG_ERR("Failed create buf_page_index: %s", doca_error_get_descr(result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}
