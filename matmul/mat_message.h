#ifndef MAT_MESSAGE_H
#define MAT_MESSAGE_H

#include <stdint.h>
#include <stddef.h>


struct NetInfo
{
    uint8_t eth_src_addr_bytes[6]; /* Source addr bytes in tx order */
    uint8_t eth_dst_addr_bytes[6]; /* Destination addr bytes in tx order */
    uint32_t ip_src_addr; /* IP source address */
    uint32_t ip_dst_addr; /* IP destination address */
};

struct MatrixCompletionInfo
{
    uint32_t total_chunks_a; // A chunks
    uint32_t total_chunks_b; // B chunks
    uint32_t received_a_elems; // received chunks size
    uint32_t received_b_elems; //received chunks size
    uint32_t received_chunk_num;
    struct NetInfo net_info;
};

// set for matrix receive completionUDP
#define MATRIX_A_COMPLETE 0x1
#define MATRIX_B_COMPLETE 0x2
#define ALL_MATRICES_COMPLETE (MATRIX_A_COMPLETE | MATRIX_B_COMPLETE)

// add the requirement for alignment
#define MEMORY_ALIGNMENT 16


#define MAX_MATRIX_DIMENSION 16384
#define MAX_MATRIX_ELEMENTS (MAX_MATRIX_DIMENSION * MAX_MATRIX_DIMENSION)

struct MatrixPacketHeader
{
    // uint8_t padding[2];
    uint32_t matrix_id; // 0: Matrix A, 1: Matrix B
    uint32_t chunk_id;
    uint32_t total_chunks;
    uint32_t chunk_size; // actual number of floats in payload
}__attribute__((__packed__));
struct MatrixMessage
{
    struct MatrixPacketHeader header;
    float payload[];
}__attribute__((__packed__));



#define UDP_MTU 1500
#define UDP_HEADER_SIZE 42           // IP(20) + UDP(8) +ETH(14)
#define MATRIX_HEADER_SIZE (sizeof(struct MatrixPacketHeader))
#define MAX_PAYLOAD (UDP_MTU - UDP_HEADER_SIZE - MATRIX_HEADER_SIZE)
#define MAX_FLOATS_PER_PACKET ((MAX_PAYLOAD / sizeof(float)) & ~3)  // 确保是4的倍数

// 添加验证宏
#define IS_VALID_MATRIX_ID(id) ((id) == 0 || (id) == 1)
#define IS_VALID_CHUNK_SIZE(size) ((size) > 0 && (size) <= MAX_FLOATS_PER_PACKET)

// 内联函数
inline size_t calcMaxFloatsPerPacket()
{
    return MAX_FLOATS_PER_PACKET;
}

inline uint32_t calcRequiredChunks(size_t total_floats)
{
    if (total_floats > MAX_MATRIX_ELEMENTS)
    {
        return 0; // 错误标志
    }
    return (total_floats + MAX_FLOATS_PER_PACKET - 1) / MAX_FLOATS_PER_PACKET;
}

// 添加辅助函数
inline bool isValidMatrixPacket(const struct MatrixPacketHeader* header)
{
    return header != NULL &&
        IS_VALID_MATRIX_ID(header->matrix_id) &&
        IS_VALID_CHUNK_SIZE(header->chunk_size) &&
        header->chunk_id < header->total_chunks;
}

#endif //MAT_MESSAGE_H
