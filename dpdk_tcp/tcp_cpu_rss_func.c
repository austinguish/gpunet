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

#include "tcp_cpu_rss_func.h"
#include "tcp_session_table.h"

DOCA_LOG_REGISTER(TCP_CPU_RSS);

int tcp_cpu_rss_func(void *lcore_args)
{
	struct rte_mbuf **rx_packets;
	struct rte_mbuf **tx_packets;
	uint32_t num_tx_packets = 0;
	uint16_t port_id = DPDK_DEFAULT_PORT;
	const struct rxq_tcp_queues *tcp_queues = lcore_args;
	struct rte_mbuf *ack;
	int num_sent;
	doca_error_t result;
	uint16_t queue_id;

	if (tcp_queues == NULL) {
		DOCA_LOG_ERR("%s: 'tcp_queues argument cannot be NULL", __func__);
		DOCA_GPUNETIO_VOLATILE(force_quit) = true;
		return -1;
	}
	if (tcp_queues->port == NULL) {
		DOCA_LOG_ERR("%s: 'tcp_queues->port argument cannot be NULL", __func__);
		DOCA_GPUNETIO_VOLATILE(force_quit) = true;
		return -1;
	}
	if (tcp_queues->rxq_pipe_gpu == NULL) {
		DOCA_LOG_ERR("%s: 'tcp_queues->rxq_pipe_gpu argument cannot be NULL", __func__);
		DOCA_GPUNETIO_VOLATILE(force_quit) = true;
		return -1;
	}

	queue_id = rte_lcore_index(rte_lcore_id()) - tcp_queues->lcore_idx_start;

	rx_packets = (struct rte_mbuf **)calloc(TCP_PACKET_MAX_BURST_SIZE, sizeof(struct rte_mbuf *));
	if (rx_packets == NULL) {
		DOCA_LOG_ERR("No memory available to allocate DPDK rx packets");
		return -1;
	}

	tx_packets = (struct rte_mbuf **)calloc(TCP_PACKET_MAX_BURST_SIZE, sizeof(struct rte_mbuf *));
	if (tx_packets == NULL) {
		free(rx_packets);
		DOCA_LOG_ERR("No memory available to allocate DPDK tx packets");
		return -1;
	}

	DOCA_LOG_INFO("Core %u is performing TCP SYN/FIN processing on queue %u", rte_lcore_id(), queue_id);

	/* read global force_quit */
	while (DOCA_GPUNETIO_VOLATILE(force_quit) == false) {
		int num_rx_packets = rte_eth_rx_burst(port_id, queue_id, rx_packets, TCP_PACKET_MAX_BURST_SIZE);

		for (int i = 0; i < num_rx_packets; i++) {
			const struct rte_mbuf *pkt = rx_packets[i];
			const struct rte_tcp_hdr *tcp_hdr = extract_tcp_hdr(pkt);

			if (!tcp_hdr) {
				DOCA_LOG_WARN("Not a TCP packet");
				continue;
			}

			if (!tcp_hdr->syn && !tcp_hdr->fin && !tcp_hdr->rst) {
				DOCA_LOG_WARN("Unexpected TCP packet flags: 0x%x, expected SYN/RST/FIN",
					      tcp_hdr->tcp_flags);
				continue;
			}

			if (tcp_hdr->rst) {
				log_tcp_flag(pkt, "RST");
				destroy_tcp_session(queue_id, pkt, tcp_queues->port);
				continue; // Do not bother to ack
			} else if (tcp_hdr->fin) {
				log_tcp_flag(pkt, "FIN");
				destroy_tcp_session(queue_id, pkt, tcp_queues->port);
			} else if (tcp_hdr->syn) {
				log_tcp_flag(pkt, "SYN");
				result = create_tcp_session(queue_id, pkt, tcp_queues->port, tcp_queues->rxq_pipe_gpu);
				if (result != DOCA_SUCCESS)
					goto error;
			} else {
				DOCA_LOG_WARN("Unexpected TCP packet flags: 0x%x, expected SYN/RST/FIN",
					      tcp_hdr->tcp_flags);
				continue;
			}

			ack = create_ack_packet(pkt, tcp_queues->tcp_ack_pkt_pool);
			if (ack)
				tx_packets[num_tx_packets++] = ack;
		}

		while (num_tx_packets > 0) {
			num_sent = rte_eth_tx_burst(port_id, queue_id, tx_packets, num_tx_packets);
			DOCA_LOG_DBG("DPDK tx_burst sent %d packets", num_sent);
			num_tx_packets -= num_sent;
		}

		for (int i = 0; i < num_rx_packets; i++)
			rte_pktmbuf_free(rx_packets[i]);
	}

	free(rx_packets);
	free(tx_packets);

	return 0;
error:

	free(rx_packets);
	free(tx_packets);

	return -1;
}

const struct rte_tcp_hdr *extract_tcp_hdr(const struct rte_mbuf *packet)
{
	const struct rte_ether_hdr *eth_hdr = rte_pktmbuf_mtod(packet, struct rte_ether_hdr *);

	if (((uint16_t)htons(eth_hdr->ether_type)) != RTE_ETHER_TYPE_IPV4) {
		DOCA_LOG_ERR("Expected ether_type 0x%x, got 0x%x",
			     RTE_ETHER_TYPE_IPV4,
			     ((uint16_t)htons(eth_hdr->ether_type)));
		return NULL;
	}

	const struct rte_ipv4_hdr *ipv4_hdr = (struct rte_ipv4_hdr *)&eth_hdr[1];

	if (ipv4_hdr->next_proto_id != IPPROTO_TCP) {
		DOCA_LOG_ERR("Expected next_proto_id %d, got %d", IPPROTO_TCP, ipv4_hdr->next_proto_id);
		return NULL;
	}

	const struct rte_tcp_hdr *tcp_hdr = (struct rte_tcp_hdr *)&ipv4_hdr[1];

	return tcp_hdr;
}

doca_error_t create_tcp_session(const uint16_t queue_id,
				const struct rte_mbuf *pkt,
				struct doca_flow_port *port,
				struct doca_flow_pipe *gpu_rss_pipe)
{
	int ret;
	struct tcp_session_entry *session_entry;

	session_entry = rte_zmalloc("tcp_session", sizeof(struct tcp_session_entry), 0);
	if (!session_entry) {
		DOCA_LOG_ERR("Failed to allocate TCP session object");
		return DOCA_ERROR_NO_MEMORY;
	}
	session_entry->key = extract_session_key(pkt);
	enable_tcp_gpu_offload(port, queue_id, gpu_rss_pipe, session_entry);

	ret = rte_hash_add_key_data(tcp_session_table, &session_entry->key, session_entry);
	if (ret != 0) {
		DOCA_LOG_ERR("Couldn't add new has key data err %d", ret);
		return DOCA_ERROR_DRIVER;
	}

	return DOCA_SUCCESS;
}

void destroy_tcp_session(const uint16_t queue_id, const struct rte_mbuf *pkt, struct doca_flow_port *port)
{
	const struct tcp_session_key key = extract_session_key(pkt);
	struct tcp_session_entry *session_entry = NULL;

	if (rte_hash_lookup_data(tcp_session_table, &key, (void **)&session_entry) < 0 || !session_entry)
		return;

	disable_tcp_gpu_offload(port, queue_id, session_entry);

	rte_hash_del_key(tcp_session_table, &key);
	rte_free(session_entry);
}

void log_tcp_flag(const struct rte_mbuf *packet, const char *flags)
{
	const struct rte_ether_hdr *eth_hdr = rte_pktmbuf_mtod(packet, struct rte_ether_hdr *);
	const struct rte_ipv4_hdr *ipv4_hdr = (struct rte_ipv4_hdr *)&eth_hdr[1];
	const struct rte_tcp_hdr *tcp_hdr = (struct rte_tcp_hdr *)&ipv4_hdr[1];
	char src_addr[INET_ADDRSTRLEN];
	char dst_addr[INET_ADDRSTRLEN];

	inet_ntop(AF_INET, &ipv4_hdr->src_addr, src_addr, INET_ADDRSTRLEN);
	inet_ntop(AF_INET, &ipv4_hdr->dst_addr, dst_addr, INET_ADDRSTRLEN);
	DOCA_LOG_INFO("Received %s for TCP %s:%d>%s:%d",
		      flags,
		      src_addr,
		      htons(tcp_hdr->src_port),
		      dst_addr,
		      htons(tcp_hdr->dst_port));
}

struct rte_mbuf *create_ack_packet(const struct rte_mbuf *src_packet, struct rte_mempool *tcp_ack_pkt_pool)
{
	uint32_t RTE_TCP_OPT_NOP_bytes = 1;
	uint32_t RTE_TCP_OPT_MSS_nbytes = 4;
	uint32_t RTE_TCP_OPT_WND_SCALE_nbytes = 3;
	uint32_t RTE_TCP_OPT_SACK_PERMITTED_nbytes = 2;
	uint32_t RTE_TCP_OPT_TIMESTAMP_nbytes = 10;
	uint16_t mss = 8192; /* pick something */
	size_t tcp_option_array_len = RTE_TCP_OPT_MSS_nbytes + RTE_TCP_OPT_SACK_PERMITTED_nbytes +
				      RTE_TCP_OPT_TIMESTAMP_nbytes + RTE_TCP_OPT_NOP_bytes +
				      RTE_TCP_OPT_WND_SCALE_nbytes;

	struct rte_ether_hdr *dst_eth_hdr;
	struct rte_ipv4_hdr *dst_ipv4_hdr;
	struct rte_tcp_hdr *dst_tcp_hdr;
	uint8_t *dst_tcp_opts;
	struct rte_mbuf *dst_packet;
	const struct rte_ether_hdr *src_eth_hdr = rte_pktmbuf_mtod(src_packet, struct rte_ether_hdr *);
	const struct rte_ipv4_hdr *src_ipv4_hdr = (struct rte_ipv4_hdr *)&src_eth_hdr[1];
	const struct rte_tcp_hdr *src_tcp_hdr = (struct rte_tcp_hdr *)&src_ipv4_hdr[1];

	if (!src_tcp_hdr->syn) {
		/* Do not bother with TCP options unless responding to SYN */
		tcp_option_array_len = 0;
	}

	dst_packet = rte_pktmbuf_alloc(tcp_ack_pkt_pool);
	if (!dst_packet) {
		DOCA_LOG_ERR("Failed to allocate TCP ACK packet");
		return NULL;
	}

	dst_eth_hdr =
		(struct rte_ether_hdr *)rte_pktmbuf_append(dst_packet,
							   sizeof(struct rte_ether_hdr) + sizeof(struct rte_ipv4_hdr) +
								   sizeof(struct rte_tcp_hdr) + tcp_option_array_len);
	if (dst_eth_hdr == NULL)
		goto release_dst;

	dst_ipv4_hdr = (struct rte_ipv4_hdr *)&dst_eth_hdr[1];
	dst_tcp_hdr = (struct rte_tcp_hdr *)&dst_ipv4_hdr[1];
	dst_tcp_opts = (uint8_t *)&dst_tcp_hdr[1];

	dst_eth_hdr->src_addr = src_eth_hdr->dst_addr;
	dst_eth_hdr->dst_addr = src_eth_hdr->src_addr;
	dst_eth_hdr->ether_type = src_eth_hdr->ether_type;

	/* Reminder: double-check remaining ack fields */
	dst_ipv4_hdr->version = 4;
	dst_ipv4_hdr->ihl = 5;
	dst_ipv4_hdr->src_addr = src_ipv4_hdr->dst_addr;
	dst_ipv4_hdr->dst_addr = src_ipv4_hdr->src_addr;
	dst_ipv4_hdr->total_length =
		RTE_BE16(sizeof(struct rte_ipv4_hdr) + sizeof(struct rte_tcp_hdr) + tcp_option_array_len);
	dst_ipv4_hdr->fragment_offset = htons(RTE_IPV4_HDR_DF_FLAG);
	dst_ipv4_hdr->time_to_live = 64;
	dst_ipv4_hdr->next_proto_id = IPPROTO_TCP;

	dst_tcp_hdr->src_port = src_tcp_hdr->dst_port;
	dst_tcp_hdr->dst_port = src_tcp_hdr->src_port;
	dst_tcp_hdr->recv_ack = RTE_BE32(RTE_BE32(src_tcp_hdr->sent_seq) + 1);
	dst_tcp_hdr->sent_seq = src_tcp_hdr->syn ? RTE_BE32(1000) : src_tcp_hdr->recv_ack;
	dst_tcp_hdr->rx_win = RTE_BE16(60000);
	dst_tcp_hdr->dt_off = 5 + tcp_option_array_len / 4;

	if (!src_tcp_hdr->ack) {
		dst_tcp_hdr->syn = src_tcp_hdr->syn;
		dst_tcp_hdr->fin = src_tcp_hdr->fin;
	}
	dst_tcp_hdr->ack = 1;

	if (tcp_option_array_len) {
		uint8_t *mss_opt = dst_tcp_opts;
		uint8_t *sack_ok_opt = dst_tcp_opts + RTE_TCP_OPT_MSS_nbytes;
		uint8_t *ts_opt = sack_ok_opt + RTE_TCP_OPT_SACK_PERMITTED_nbytes;
		uint8_t *nop_opt = ts_opt + RTE_TCP_OPT_TIMESTAMP_nbytes;
		uint8_t *ws_opt = nop_opt + 1;
		time_t seconds = htonl(time(NULL));

		mss_opt[0] = RTE_TCP_OPT_MSS;
		mss_opt[1] = RTE_TCP_OPT_MSS_nbytes;
		mss_opt[2] = (uint8_t)(mss >> 8);
		mss_opt[3] = (uint8_t)mss;

		sack_ok_opt[0] = RTE_TCP_OPT_SACK_PERMITTED;
		sack_ok_opt[1] = RTE_TCP_OPT_SACK_PERMITTED_nbytes;

		ts_opt[0] = RTE_TCP_OPT_TIMESTAMP;
		ts_opt[1] = RTE_TCP_OPT_TIMESTAMP_nbytes;
		memcpy(ts_opt + 2, &seconds, 4);
		// ts_opt+6 (ECR) set below

		nop_opt[0] = RTE_TCP_OPT_NOP;

		ws_opt[0] = RTE_TCP_OPT_WND_SCALE;
		ws_opt[1] = RTE_TCP_OPT_WND_SCALE_nbytes;
		ws_opt[2] = 7; // pick a scale

		const uint8_t *src_tcp_option = (uint8_t *)&src_tcp_hdr[1];
		const uint8_t *src_tcp_options_end = src_tcp_option + 4 * src_tcp_hdr->data_off;
		uint32_t opt_len = 0;

		while (src_tcp_option < src_tcp_options_end) {
			DOCA_LOG_DBG("Processing TCP Option 0x%x", *src_tcp_option);
			switch (*src_tcp_option) {
			case RTE_TCP_OPT_END:
				src_tcp_option = src_tcp_options_end; // end loop
				break;
			case RTE_TCP_OPT_NOP:
				++src_tcp_option;
				break;
			case RTE_TCP_OPT_MSS:
				opt_len = *src_tcp_option;
				src_tcp_option += 4; // don't care
				break;
			case RTE_TCP_OPT_WND_SCALE:
				src_tcp_option += 3; // don't care
				break;
			case RTE_TCP_OPT_SACK_PERMITTED:
				src_tcp_option += 2; // don't care
				break;
			case RTE_TCP_OPT_SACK:
				opt_len = *src_tcp_option; // variable length; don't care
				src_tcp_option += opt_len;
				break;
			case RTE_TCP_OPT_TIMESTAMP: {
				const uint8_t *src_tsval = src_tcp_option + 2;
				uint8_t *dst_tsecr = ts_opt + 6;

				memcpy(dst_tsecr, src_tsval, 4);
				src_tcp_option += 10;
				break;
			}
			}
		}
	} /* tcp options */

	/* Use offloaded checksum operations */
	dst_packet->ol_flags |= RTE_MBUF_F_TX_IPV4 | RTE_MBUF_F_TX_IP_CKSUM | RTE_MBUF_F_TX_TCP_CKSUM;

	return dst_packet;

release_dst:
	rte_pktmbuf_free(dst_packet);
	return NULL;
}

struct tcp_session_key extract_session_key(const struct rte_mbuf *packet)
{
	const struct rte_ether_hdr *eth_hdr = rte_pktmbuf_mtod(packet, struct rte_ether_hdr *);
	const struct rte_ipv4_hdr *ipv4_hdr = (struct rte_ipv4_hdr *)&eth_hdr[1];
	const struct rte_tcp_hdr *tcp_hdr = (struct rte_tcp_hdr *)&ipv4_hdr[1];

	struct tcp_session_key key = {
		.src_addr = ipv4_hdr->src_addr,
		.dst_addr = ipv4_hdr->dst_addr,
		.src_port = tcp_hdr->src_port,
		.dst_port = tcp_hdr->dst_port,
	};

	return key;
}
