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

#ifndef DOCA_GPU_PACKET_PROCESSING_TCP_RSS_H
#define DOCA_GPU_PACKET_PROCESSING_TCP_RSS_H

#include <stdbool.h>
#include <arpa/inet.h>

#include <rte_ethdev.h>
#include <rte_malloc.h>

#include <doca_flow.h>
#include <doca_log.h>

#include <common.h>

#define TCP_PACKET_MAX_BURST_SIZE 4096

/*
 * Launch CPU thread to manage TCP 3way handshake
 *
 * @args [in]: thread input args
 * @return: 0 on success and 1 otherwise
 */
int tcp_cpu_rss_func(void *args);

/*
 * Extract the address of the IPv4 TCP header contained in the
 * raw ethernet frame packet buffer if present; otherwise null.
 *
 * @packet [in]: Packet to extract TCP hdr
 * @return: ptr on success and NULL otherwise
 */
const struct rte_tcp_hdr *extract_tcp_hdr(const struct rte_mbuf *packet);

/*
 * Create TCP session
 *
 * @queue_id [in]: DPDK queue id for TCP control packets
 * @pkt [in]: pkt triggering the TCP session creation
 * @port [in]: DOCA Flow port
 * @gpu_rss_pipe [in]: DOCA Flow GPU RSS pipe
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t create_tcp_session(const uint16_t queue_id,
				const struct rte_mbuf *pkt,
				struct doca_flow_port *port,
				struct doca_flow_pipe *gpu_rss_pipe);

/*
 * Destroy TCP session
 *
 * @queue_id [in]: DPDK queue id for TCP control packets
 * @pkt [in]: pkt triggering the TCP session destruction
 * @port [in]: DOCA Flow port
 */
void destroy_tcp_session(const uint16_t queue_id, const struct rte_mbuf *pkt, struct doca_flow_port *port);

/*
 * Log TCP flags
 *
 * @packet [in]: Packet to report TCP flags
 * @flags [in]: TCP Flags
 */
void log_tcp_flag(const struct rte_mbuf *packet, const char *flags);

/*
 * Create TCP ACK packet
 *
 * @src_packet [in]: Src packet to use to create ACK packet
 * @tcp_ack_pkt_pool [in]: DPDK mempool to create ACK packets
 * @return: ptr on success and NULL otherwise
 */
struct rte_mbuf *create_ack_packet(const struct rte_mbuf *src_packet, struct rte_mempool *tcp_ack_pkt_pool);

/*
 * Extract TCP session key
 *
 * @packet [in]: Extract session key from this packet
 * @return: object
 */
struct tcp_session_key extract_session_key(const struct rte_mbuf *packet);

#endif
