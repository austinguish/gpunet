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
#include "matmul/mat_message.h"

DOCA_LOG_REGISTER(GPU_PACKET_PROCESSING_UDP_BW);

doca_error_t create_udp_bw_queues(struct rxq_udp_bw_queues* udp_queues,
                                  struct doca_flow_port* df_port,
                                  struct doca_gpu* gpu_dev,
                                  struct doca_dev* ddev,
                                  struct doca_pe* pe,
                                  uint32_t queue_num,
                                  uint32_t sem_num,
                                  doca_eth_txq_gpu_event_error_send_packet_cb_t event_error_send_packet_cb,
                                  doca_eth_txq_gpu_event_notify_send_packet_cb_t
                                  event_notify_send_packet_cb)
{
    doca_error_t result;
    uint32_t cyclic_buffer_size = 0;
    union doca_data event_user_data[MAX_QUEUES] = {0};

    if (udp_queues == NULL || df_port == NULL || gpu_dev == NULL || ddev == NULL || queue_num == 0 ||
        sem_num == 0)
    {
        DOCA_LOG_ERR("Can't create UDP queues, invalid input");
        return DOCA_ERROR_INVALID_VALUE;
    }

    udp_queues->gpu_dev = gpu_dev;
    udp_queues->ddev = ddev;
    udp_queues->port = df_port;
    udp_queues->numq = queue_num;
    udp_queues->nums = sem_num;

    for (uint32_t idx = 0; idx < queue_num; idx++)
    {
        DOCA_LOG_INFO("Creating UDP Eth Rxq %d", idx);

        result = doca_eth_rxq_create(udp_queues->ddev,
                                     MAX_PKT_NUM,
                                     MAX_PKT_SIZE,
                                     &(udp_queues->eth_rxq_cpu[idx]));
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed doca_eth_rxq_create: %s", doca_error_get_descr(result));
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_eth_rxq_set_type(udp_queues->eth_rxq_cpu[idx], DOCA_ETH_RXQ_TYPE_CYCLIC);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed doca_eth_rxq_set_type: %s", doca_error_get_descr(result));
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_eth_rxq_estimate_packet_buf_size(DOCA_ETH_RXQ_TYPE_CYCLIC,
                                                       0,
                                                       0,
                                                       MAX_PKT_SIZE,
                                                       MAX_PKT_NUM,
                                                       0,
                                                       &cyclic_buffer_size);
        DOCA_LOG_INFO("the cyclic buffer size is %d", cyclic_buffer_size);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed to get eth_rxq cyclic buffer size: %s", doca_error_get_descr(result));
            destroy_udp_bw_queues(udp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_mmap_create(&udp_queues->pkt_buff_mmap[idx]);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed to create mmap: %s", doca_error_get_descr(result));
            destroy_udp_bw_queues(udp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_mmap_add_dev(udp_queues->pkt_buff_mmap[idx], udp_queues->ddev);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed to add dev to mmap: %s", doca_error_get_descr(result));
            destroy_udp_bw_queues(udp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_gpu_mem_alloc(udp_queues->gpu_dev,
                                    cyclic_buffer_size,
                                    GPU_PAGE_SIZE,
                                    DOCA_GPU_MEM_TYPE_GPU,
                                    &udp_queues->gpu_pkt_addr[idx],
                                    NULL);
        if (result != DOCA_SUCCESS || udp_queues->gpu_pkt_addr[idx] == NULL)
        {
            DOCA_LOG_ERR("Failed to allocate gpu memory %s", doca_error_get_descr(result));
            destroy_udp_bw_queues(udp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        /* Map GPU memory buffer used to receive packets with DMABuf */
        // result = doca_gpu_dmabuf_fd(udp_queues->gpu_dev,
        // 			    udp_queues->gpu_pkt_addr[idx],
        // 			    cyclic_buffer_size,
        // 			    &(udp_queues->dmabuf_fd[idx]));
        //
        // if (result == DOCA_SUCCESS) {
        // 	DOCA_LOG_INFO("Mapping receive queue buffer (0x%p size %dB dmabuf fd %d) with dmabuf mode",
        // 		      udp_queues->gpu_pkt_addr[idx],
        // 		      cyclic_buffer_size,
        // 		      udp_queues->dmabuf_fd[idx]);
        //
        // 	result = doca_mmap_set_dmabuf_memrange(udp_queues->pkt_buff_mmap[idx],
        // 					       udp_queues->dmabuf_fd[idx],
        // 					       udp_queues->gpu_pkt_addr[idx],
        // 					       0,
        // 					       cyclic_buffer_size);
        // }


        DOCA_LOG_INFO("Mapping receive queue buffer (0x%p size %dB) with nvidia-peermem mode",
                      udp_queues->gpu_pkt_addr[idx],
                      cyclic_buffer_size);

        /* If failed, use nvidia-peermem legacy method */
        result = doca_mmap_set_memrange(udp_queues->pkt_buff_mmap[idx],
                                        udp_queues->gpu_pkt_addr[idx],
                                        cyclic_buffer_size);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed to set memrange for mmap %s", doca_error_get_descr(result));
            destroy_udp_bw_queues(udp_queues);
            return DOCA_ERROR_BAD_STATE;
        }


        result = doca_mmap_set_permissions(udp_queues->pkt_buff_mmap[idx],
                                           DOCA_ACCESS_FLAG_LOCAL_READ_WRITE |
                                           DOCA_ACCESS_FLAG_PCI_RELAXED_ORDERING);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed to set permissions for mmap %s", doca_error_get_descr(result));
            destroy_udp_bw_queues(udp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_mmap_start(udp_queues->pkt_buff_mmap[idx]);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed to start mmap %s", doca_error_get_descr(result));
            destroy_udp_bw_queues(udp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_eth_rxq_set_pkt_buf(udp_queues->eth_rxq_cpu[idx],
                                          udp_queues->pkt_buff_mmap[idx],
                                          0,
                                          cyclic_buffer_size);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed to set cyclic buffer  %s", doca_error_get_descr(result));
            destroy_udp_bw_queues(udp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        udp_queues->eth_rxq_ctx[idx] = doca_eth_rxq_as_doca_ctx(udp_queues->eth_rxq_cpu[idx]);
        if (udp_queues->eth_rxq_ctx[idx] == NULL)
        {
            DOCA_LOG_ERR("Failed doca_eth_rxq_as_doca_ctx: %s", doca_error_get_descr(result));
            destroy_udp_bw_queues(udp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_ctx_set_datapath_on_gpu(udp_queues->eth_rxq_ctx[idx], udp_queues->gpu_dev);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed doca_ctx_set_datapath_on_gpu: %s", doca_error_get_descr(result));
            destroy_udp_bw_queues(udp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_ctx_start(udp_queues->eth_rxq_ctx[idx]);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed doca_ctx_start: %s", doca_error_get_descr(result));
            destroy_udp_bw_queues(udp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_eth_rxq_get_gpu_handle(udp_queues->eth_rxq_cpu[idx], &(udp_queues->eth_rxq_gpu[idx]));
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed doca_eth_rxq_get_gpu_handle: %s", doca_error_get_descr(result));
            destroy_udp_bw_queues(udp_queues);
            return DOCA_ERROR_BAD_STATE;
        }
        // create tx queue

        result = doca_eth_txq_create(udp_queues->ddev, MAX_SQ_DESCR_NUM, &(udp_queues->eth_txq_cpu[idx]));
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed doca_eth_txq_create: %s", doca_error_get_descr(result));
            destroy_udp_bw_queues(udp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_eth_txq_set_l3_chksum_offload(udp_queues->eth_txq_cpu[idx], 1);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed to set eth_txq l3 offloads: %s", doca_error_get_descr(result));
            destroy_udp_bw_queues(udp_queues);
            return DOCA_ERROR_BAD_STATE;
        }
        result = doca_eth_txq_set_l4_chksum_offload(udp_queues->eth_txq_cpu[idx], 1);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed to set l4 udp offloads: %s", doca_error_get_descr(result));
            destroy_udp_bw_queues(udp_queues);
            return DOCA_ERROR_BAD_STATE;
        }


        udp_queues->eth_txq_ctx[idx] = doca_eth_txq_as_doca_ctx(udp_queues->eth_txq_cpu[idx]);
        if (udp_queues->eth_txq_ctx[idx] == NULL)
        {
            DOCA_LOG_ERR("Failed doca_eth_txq_as_doca_ctx: %s", doca_error_get_descr(result));
            destroy_udp_bw_queues(udp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_ctx_set_datapath_on_gpu(udp_queues->eth_txq_ctx[idx], udp_queues->gpu_dev);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed doca_ctx_set_datapath_on_gpu: %s", doca_error_get_descr(result));
            destroy_udp_bw_queues(udp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        if (pe != NULL)
        {
            event_user_data[idx].u64 = idx;
            result = doca_eth_txq_gpu_event_error_send_packet_register(udp_queues->eth_txq_cpu[idx],
                                                                       event_error_send_packet_cb,
                                                                       event_user_data[idx]);
            if (result != DOCA_SUCCESS)
            {
                DOCA_LOG_ERR("Unable to set DOCA progress engine callback: %s",
                             doca_error_get_descr(result));
                destroy_udp_bw_queues(udp_queues);
                return DOCA_ERROR_BAD_STATE;
            }
            result = doca_eth_txq_gpu_event_notify_send_packet_register(udp_queues->eth_txq_cpu[idx],
                                                                        event_notify_send_packet_cb,
                                                                        event_user_data[idx]);
            if (result != DOCA_SUCCESS)
            {
                DOCA_LOG_ERR("Failed to register notify callback");
                destroy_udp_bw_queues(udp_queues);
                return DOCA_ERROR_BAD_STATE;
            }


            result = doca_pe_connect_ctx(pe, udp_queues->eth_txq_ctx[idx]);
            if (result != DOCA_SUCCESS)
            {
                DOCA_LOG_ERR("Unable to set DOCA progress engine to DOCA Eth Txq: %s",
                             doca_error_get_descr(result));
                destroy_udp_bw_queues(udp_queues);
                return DOCA_ERROR_BAD_STATE;
            }
        }

        result = doca_ctx_start(udp_queues->eth_txq_ctx[idx]);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed doca_ctx_start: %s", doca_error_get_descr(result));
            destroy_udp_bw_queues(udp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_eth_txq_get_gpu_handle(udp_queues->eth_txq_cpu[idx], &(udp_queues->eth_txq_gpu[idx]));
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed doca_eth_txq_get_gpu_handle: %s", doca_error_get_descr(result));
            destroy_udp_bw_queues(udp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_gpu_semaphore_create(udp_queues->gpu_dev, &(udp_queues->sem_cpu[idx]));
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed doca_gpu_semaphore_create: %s", doca_error_get_descr(result));
            destroy_udp_bw_queues(udp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        /*
         * Semaphore memory reside on CPU visible from GPU.
         * CPU will poll in busy wait on this semaphore (multiple reads)
         * while GPU access each item only once to update values.
         */
        result = doca_gpu_semaphore_set_memory_type(udp_queues->sem_cpu[idx], DOCA_GPU_MEM_TYPE_CPU_GPU);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed doca_gpu_semaphore_set_memory_type: %s", doca_error_get_descr(result));
            destroy_udp_bw_queues(udp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_gpu_semaphore_set_items_num(udp_queues->sem_cpu[idx], udp_queues->nums);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed doca_gpu_semaphore_set_items_num: %s", doca_error_get_descr(result));
            destroy_udp_bw_queues(udp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        /*
         * Semaphore memory reside on CPU visible from GPU.
         * The CPU reads packets info from this structure.
         * The GPU access each item only once to update values.
         */
        result = doca_gpu_semaphore_set_custom_info(udp_queues->sem_cpu[idx],
                                                    sizeof(struct MatrixCompletionInfo),
                                                    DOCA_GPU_MEM_TYPE_CPU_GPU);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed doca_gpu_semaphore_set_custom_info: %s", doca_error_get_descr(result));
            destroy_udp_bw_queues(udp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_gpu_semaphore_start(udp_queues->sem_cpu[idx]);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed doca_gpu_semaphore_start: %s", doca_error_get_descr(result));
            destroy_udp_bw_queues(udp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_gpu_semaphore_get_gpu_handle(udp_queues->sem_cpu[idx], &(udp_queues->sem_gpu[idx]));
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed doca_gpu_semaphore_get_gpu_handle: %s", doca_error_get_descr(result));
            destroy_udp_bw_queues(udp_queues);
            return DOCA_ERROR_BAD_STATE;
        }
    }


    result = create_tx_buf(&udp_queues->buf_response,
                           udp_queues->gpu_dev,
                           udp_queues->ddev,
                           TX_BUF_NUM,
                           TX_BUF_MAX_SZ);
    if (result != DOCA_SUCCESS)
    {
        DOCA_LOG_ERR("Failed create buf_page_contacts: %s", doca_error_get_descr(result));
        destroy_udp_bw_queues(udp_queues);
        return DOCA_ERROR_BAD_STATE;
    }
    result = prepare_resp_tx_buf(&udp_queues->buf_response);
    if (result != DOCA_SUCCESS)
    {
        DOCA_LOG_ERR("Failed prepare the response buf: %s", doca_error_get_descr(result));
        destroy_udp_bw_queues(udp_queues);
        return DOCA_ERROR_BAD_STATE;
    }

    /* Create UDP based flow pipe */
    result = create_udp_bw_pipe(udp_queues, df_port);
    if (result != DOCA_SUCCESS)
    {
        DOCA_LOG_ERR("Function build_rxq_pipe returned %s", doca_error_get_descr(result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}

doca_error_t destroy_udp_bw_queues(struct rxq_udp_bw_queues* udp_queues)
{
    doca_error_t result;

    if (udp_queues == NULL)
    {
        DOCA_LOG_ERR("Can't destroy UDP queues, invalid input");
        return DOCA_ERROR_INVALID_VALUE;
    }

    for (int idx = 0; idx < udp_queues->numq; idx++)
    {
        DOCA_LOG_INFO("Destroying UDP queue %d", idx);

        if (udp_queues->sem_cpu[idx])
        {
            result = doca_gpu_semaphore_stop(udp_queues->sem_cpu[idx]);
            if (result != DOCA_SUCCESS)
            {
                DOCA_LOG_ERR("Failed doca_gpu_semaphore_start: %s", doca_error_get_descr(result));
                return DOCA_ERROR_BAD_STATE;
            }

            result = doca_gpu_semaphore_destroy(udp_queues->sem_cpu[idx]);
            if (result != DOCA_SUCCESS)
            {
                DOCA_LOG_ERR("Failed doca_gpu_semaphore_destroy: %s", doca_error_get_descr(result));
                return DOCA_ERROR_BAD_STATE;
            }
        }

        result = doca_ctx_stop(udp_queues->eth_rxq_ctx[idx]);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed doca_ctx_stop: %s", doca_error_get_descr(result));
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_eth_rxq_destroy(udp_queues->eth_rxq_cpu[idx]);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed doca_eth_rxq_destroy: %s", doca_error_get_descr(result));
            return DOCA_ERROR_BAD_STATE;
        }


        result = doca_mmap_destroy(udp_queues->pkt_buff_mmap[idx]);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed to destroy mmap: %s", doca_error_get_descr(result));
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_gpu_mem_free(udp_queues->gpu_dev, udp_queues->gpu_pkt_addr[idx]);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed to free gpu memory: %s", doca_error_get_descr(result));
            return DOCA_ERROR_BAD_STATE;
        }
        result = doca_ctx_stop(udp_queues->eth_txq_ctx[idx]);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed doca_ctx_stop: %s", doca_error_get_descr(result));
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_eth_txq_destroy(udp_queues->eth_txq_cpu[idx]);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR("Failed doca_eth_rxq_destroy: %s", doca_error_get_descr(result));
            return DOCA_ERROR_BAD_STATE;
        }
    }
    result = destroy_tx_buf(&udp_queues->buf_response);
    if (result != DOCA_SUCCESS)
    {
        DOCA_LOG_ERR("Failed create buf_page_contacts: %s", doca_error_get_descr(result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}
