/*
 * Copyright (c) 2024 NVIDIA CORPORATION AND AFFILIATES.  All rights reserved.
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

#ifndef COMCH_UTILS_H_
#define COMCH_UTILS_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

#include <doca_comch.h>

struct comch_cfg;

/*
 * Set up a new comch channel for control messages
 *
 * This should only be used when a single client is connecting to a single server
 * Attempts to handle multiple connections to a server will fail or have undefined behavior
 * Success occurs when a server has a client connected, or a client is successfully connected to a server
 *
 * @server_name [in]: name for server to use or client to connect to
 * @pci_addr [in]: PCI address of device to use
 * @rep_pci_addr [in]: Repr address to use (server/DPU side only)
 * @user_data [in]: app user data that can be returned from a connection event
 * @client_recv_event_cb [in]: callback for new messages on client (client side only)
 * @server_recv_event_cb [in]: callback for new messages on server (server side only)
 * @comch_cfg [out]: comch utils configuration object
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t comch_utils_init(const char *server_name,
			      const char *pci_addr,
			      const char *rep_pci_addr,
			      void *user_data,
			      doca_comch_event_msg_recv_cb_t client_recv_event_cb,
			      doca_comch_event_msg_recv_cb_t server_recv_event_cb,
			      struct comch_cfg **comch_cfg);

/*
 * Set up a new comch channel for control messages and fast path
 *
 * This should only be used when a single client is connecting to a single server
 * Attempts to handle multiple connections to a server will fail or have undefined behavior
 * Success occurs when a server has a client connected, or a client is successfully connected to a server
 *
 * This function extends comch_utils_init() to add callbacks for new consumers generated on the opposite end of the
 * single doca_comch_connection that is created.
 *
 * @server_name [in]: name for server to use or client to connect to
 * @pci_addr [in]: PCI address of device to use
 * @rep_pci_addr [in]: Repr address to use (server/DPU side only)
 * @user_data [in]: app user data that can be returned from a connection event
 * @client_recv_event_cb [in]: callback for new messages on client (client side only)
 * @server_recv_event_cb [in]: callback for new messages on server (server side only)
 * @new_consumer_event_cb [in]: callback for new consumers created across the connection (client and server side)
 * @expired_consumer_event_cb [in]: callback for expired consumers on the connection (client and server side)
 * @comch_cfg [out]: comch utils configuration object
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t comch_utils_fast_path_init(const char *server_name,
					const char *pci_addr,
					const char *rep_pci_addr,
					void *user_data,
					doca_comch_event_msg_recv_cb_t client_recv_event_cb,
					doca_comch_event_msg_recv_cb_t server_recv_event_cb,
					doca_comch_event_consumer_cb_t new_consumer_event_cb,
					doca_comch_event_consumer_cb_t expired_consumer_event_cb,
					struct comch_cfg **comch_cfg);

/*
 * Tear down the comch created with init_comch
 *
 * The server side waits until the client has complete before exiting
 *
 * @comch_cfg [in]: pointer to comch utils configuration object to destroy
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t comch_utils_destroy(struct comch_cfg *comch_cfg);

/*
 * Send a message across a given connection
 *
 * The connection must have been created via comch_utils_init.
 * Connection is returned with the receive event callback.
 * Connection can be extracted from configuration object using comch_util_get_connection().
 * Caller must ensure there is a free task for success.
 *
 * @connection [in]: connection to send message across
 * @msg [in]: message to send on connection channel
 * @len [in]: length of message to send
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t comch_utils_send(struct doca_comch_connection *connection, const void *msg, uint32_t len);

/*
 * Return the user_data passed to comch_utils_init from a connection object
 *
 * @connection [in]: pointer to connection object
 * @return: pointer to the user_data
 */
void *comch_utils_get_user_data(struct doca_comch_connection *connection);

/*
 * Call progress on the client/server associated with a given connection
 *
 * @connection [in]: pointer to connection object
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t comch_utils_progress_connection(struct doca_comch_connection *connection);

/*
 * Get the connection associated with a comch utils configuration object
 *
 * Comch utils only supports one connection per server
 *
 * @comch_cfg [in]: pointer to comch utils configuration object
 * @return: pointer to associated connection
 */
struct doca_comch_connection *comch_util_get_connection(struct comch_cfg *comch_cfg);

/*
 * Get the maximum buffer size the configured comch client/server supports
 *
 * This max size is configured to the max supported.
 *
 * @comch_cfg [in]: pointer to comch utils configuration object
 * @return: maximum buffer size supported
 */
uint32_t comch_utils_get_max_buffer_size(struct comch_cfg *comch_cfg);

#endif /* COMCH_UTILS_H_ */
