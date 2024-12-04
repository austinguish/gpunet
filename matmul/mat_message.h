//
// Created by yiwei on 24-12-4.
//
#include <stdint.h>
#include <stddef.h>
#ifndef MAT_MESSAGE_H
#define MAT_MESSAGE_H
#pragma pack(push, 1)  // 确保内存紧凑对齐


struct MatrixPacketHeader {
    uint32_t matrix_id;      // 0: Matrix A, 1: Matrix B
    uint32_t chunk_id;
    uint32_t total_chunks;
    uint32_t chunk_size;
};


struct MatrixMessage {
    struct MatrixPacketHeader header;
    float payload[];
};

#pragma pack(pop)
#define UDP_MTU 1500
#define UDP_HEADER_SIZE 42  /*ethernet(14) + IP(20) + UDP(8) */
#define MATRIX_HEADER_SIZE (sizeof(struct MatrixPacketHeader))
#define MAX_PAYLOAD (UDP_MTU - UDP_HEADER_SIZE - MATRIX_HEADER_SIZE)
#define MAX_FLOATS_PER_PACKET (MAX_PAYLOAD / sizeof(float))

// calculate how many floats can be put in a packet
inline size_t calcMaxFloatsPerPacket() {
    return MAX_PAYLOAD / sizeof(float);
}

// given total number of floats(matrix elements),how many chunks are needed
inline uint32_t calcRequiredChunks(size_t total_floats) {
    return (total_floats + MAX_FLOATS_PER_PACKET - 1) / MAX_FLOATS_PER_PACKET;
}
#endif //MAT_MESSAGE_H
