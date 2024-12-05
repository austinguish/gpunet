//
// Created by yiwei on 24-12-4.
//
#ifndef MAT_MESSAGE_H
#define MAT_MESSAGE_H

#include <stdint.h>
#include <stddef.h>

// 添加对齐要求
#define MEMORY_ALIGNMENT 16

#pragma pack(push, MEMORY_ALIGNMENT)  // 使用16字节对齐而不是1字节

// 添加更多的安全检查和限制
#define MAX_MATRIX_DIMENSION 1024  // 最大矩阵维度
#define MAX_MATRIX_ELEMENTS (MAX_MATRIX_DIMENSION * MAX_MATRIX_DIMENSION)

struct __align__(4) MatrixPacketHeader
{
    uint32_t matrix_id;      // 0: Matrix A, 1: Matrix B
    uint32_t chunk_id;
    uint32_t total_chunks;
    uint32_t chunk_size;     // actual number of floats in payload
};


struct __align__(4) MatrixMessage {
    struct MatrixPacketHeader header;
    float payload[];  // 确保payload对齐
};

#pragma pack(pop)

// 网络相关常量
#define UDP_MTU 1500
#define UDP_HEADER_SIZE 42  /* ethernet(14) + IP(20) + UDP(8) */
#define MATRIX_HEADER_SIZE (sizeof(struct MatrixPacketHeader))
#define MAX_PAYLOAD (UDP_MTU - UDP_HEADER_SIZE - MATRIX_HEADER_SIZE)
#define MAX_FLOATS_PER_PACKET ((MAX_PAYLOAD / sizeof(float)) & ~3)  // 确保是4的倍数

// 添加验证宏
#define IS_VALID_MATRIX_ID(id) ((id) == 0 || (id) == 1)
#define IS_VALID_CHUNK_SIZE(size) ((size) > 0 && (size) <= MAX_FLOATS_PER_PACKET)

// 内联函数
inline size_t calcMaxFloatsPerPacket() {
    return MAX_FLOATS_PER_PACKET;
}

inline uint32_t calcRequiredChunks(size_t total_floats) {
    if (total_floats > MAX_MATRIX_ELEMENTS) {
        return 0;  // 错误标志
    }
    return (total_floats + MAX_FLOATS_PER_PACKET - 1) / MAX_FLOATS_PER_PACKET;
}

// 添加辅助函数
inline bool isValidMatrixPacket(const struct MatrixPacketHeader* header) {
    return header != NULL &&
           IS_VALID_MATRIX_ID(header->matrix_id) &&
           IS_VALID_CHUNK_SIZE(header->chunk_size) &&
           header->chunk_id < header->total_chunks;
}

#endif //MAT_MESSAGE_H