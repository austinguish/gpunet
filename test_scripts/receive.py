import numpy as np
import socket
import struct
import sys
from typing import List, Optional
import torch
import time
import threading
from queue import Queue

class MatrixReceiver:
    def __init__(self, port: int = 5678):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('', port))
        self.received_chunks = {}
        self.result_ready = threading.Event()
        self.result_matrix: Optional[np.ndarray] = None
        self.start_time = None
        self.end_time = None

    def start_receiving(self):
        """Start receiving matrix chunks in a separate thread"""
        self.start_time = time.time()  # 记录开始接收的时间
        self.receive_thread = threading.Thread(target=self._receive_loop)
        self.receive_thread.daemon = True
        self.receive_thread.start()

    def _receive_loop(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(65535)  # Max UDP packet size
                if len(data) < 16:  # Minimum header size
                    continue

                header = struct.unpack('IIII', data[:16])
                matrix_id, chunk_id, total_chunks, chunk_size = header

                if matrix_id not in self.received_chunks:
                    self.received_chunks[matrix_id] = {}

                chunk_data = np.frombuffer(data[16:], dtype=np.float32)
                self.received_chunks[matrix_id][chunk_id] = chunk_data

                #print(f"Received chunk {chunk_id + 1}/{total_chunks} for matrix {matrix_id}")

                # Check if all chunks received
                if len(self.received_chunks[matrix_id]) == total_chunks:
                    # Reconstruct matrix
                    all_data = []
                    for i in range(total_chunks):
                        all_data.extend(self.received_chunks[matrix_id][i])

                    # Calculate matrix dimensions
                    total_elements = len(all_data)
                    matrix_size = int(np.sqrt(total_elements))

                    self.result_matrix = np.array(all_data).reshape(matrix_size, matrix_size).T
                    self.end_time = time.time()  # 记录接收完成的时间
                    self.result_ready.set()
                    break

            except Exception as e:
                print(f"Error receiving data: {e}")
                continue

    def get_total_time(self):
        """获取总接收时间"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    def wait_for_result(self, timeout=None) -> Optional[np.ndarray]:
        """Wait for the complete matrix to be received"""
        if self.result_ready.wait(timeout):
            return self.result_matrix
        return None

    def close(self):
        self.sock.close()

class MatrixSender:
    def __init__(self):
        self.target_ip = "10.134.11.66"
        self.target_port = 2574
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.UDP_MTU = 1500
        self.UDP_HEADER_SIZE = 42
        self.MATRIX_HEADER_SIZE = 16
        self.MAX_PAYLOAD = self.UDP_MTU - self.UDP_HEADER_SIZE - self.MATRIX_HEADER_SIZE
        self.MAX_FLOATS_PER_PACKET = (self.MAX_PAYLOAD // 4) & ~3
        self.start_time = None
        self.end_time = None

    def send_matrix(self, matrix: np.ndarray, matrix_id: int):
        """Send a matrix over UDP"""
        if self.start_time is None:
            self.start_time = time.time()  # 记录第一次发送的开始时间

        matrix_to_send = matrix
        flat_data = matrix_to_send.astype(np.float32).flatten()
        total_floats = len(flat_data)
        total_chunks = (total_floats + self.MAX_FLOATS_PER_PACKET - 1) // self.MAX_FLOATS_PER_PACKET

        for chunk_id in range(total_chunks):
            start_idx = chunk_id * self.MAX_FLOATS_PER_PACKET
            end_idx = min(start_idx + self.MAX_FLOATS_PER_PACKET, total_floats)
            chunk_size = end_idx - start_idx

            header = struct.pack('IIII', matrix_id, chunk_id, total_chunks, chunk_size)
            chunk_data = flat_data[start_idx:end_idx].tobytes()
            packet = header + chunk_data

            self.sock.sendto(packet, (self.target_ip, self.target_port))
            #print(f"Sent chunk {chunk_id+1}/{total_chunks} for matrix {matrix_id}")

        if matrix_id == 1:  # 记录最后一个矩阵发送完成的时间
            self.end_time = time.time()

    def get_total_time(self):
        """获取总发送时间"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    def close(self):
        self.sock.close()

def print_matrix_preview(name: str, matrix: np.ndarray):
    """Print first five elements of first column of a matrix"""
    first_column = matrix[:, 0]
    preview = first_column[:5]
    preview_str = ' '.join([f"{x:.6f}" for x in preview])
    print(f"First five elements of matrix {name} first column: {preview_str}")

def compute_gpu(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    """Perform matrix multiplication on GPU using PyTorch"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_a = torch.from_numpy(matrix_a).to(device)
    torch_b = torch.from_numpy(matrix_b).to(device)

    start_time = time.time()
    torch_result = torch.matmul(torch_a, torch_b)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()

    result = torch_result.cpu().numpy()
    print(f"GPU Computation time: {end_time - start_time:.4f} seconds")
    return result

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <matrix_size>")
        sys.exit(1)

    matrix_size = int(sys.argv[1])
    matrix_a = np.random.rand(matrix_size, matrix_size).astype(np.float32)
    matrix_b = np.random.rand(matrix_size, matrix_size).astype(np.float32)

    print_matrix_preview('A', matrix_a)
    print_matrix_preview('B', matrix_b)

    # Start receiver before sending matrices
    receiver = MatrixReceiver(port=5678)
    receiver.start_receiving()
    print("Started UDP receiver on port 5678")

    # 开始计时并发送矩阵
    total_start_time = time.time()

    sender = MatrixSender()
    try:
        print("\nSending Matrix A...")
        sender.send_matrix(matrix_a, matrix_id=0)
        print("Sending Matrix B...")
        sender.send_matrix(matrix_b, matrix_id=1)
    finally:
        sender.close()

    send_time = sender.get_total_time()
    print(f"Matrices sent successfully in {send_time:.4f} seconds")

    print("\nPerforming local GPU computation...")
    result = compute_gpu(matrix_a, matrix_b)
    print(f"GPU calculation completed. Result shape: {result.shape}")
    print_matrix_preview('C', result)

    print("\nWaiting for result matrix C from UDP...")
    received_result = receiver.wait_for_result(timeout=30)

    if received_result is not None:
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        receive_time = receiver.get_total_time()

        print("\nTiming Statistics:")
        print(f"Matrix sending time: {send_time:.4f} seconds")
        print(f"Matrix receiving time: {receive_time:.4f} seconds")
        print(f"Total round-trip time: {total_time:.4f} seconds")

        print("\nReceived result matrix C via UDP")
        print_matrix_preview('Received C', received_result)

        if np.allclose(result, received_result, rtol=1e-5, atol=1e-5):
            print("Local and received results match!")
        else:
            print("Warning: Results don't match!")
            if result.shape != received_result.shape:
                print(f"Shape mismatch: local {result.shape} vs received {received_result.shape}")
            else:
                diff = np.abs(result - received_result)
                max_diff = np.max(diff)
                avg_diff = np.mean(diff)
                print(f"Maximum difference: {max_diff}")
                print(f"Average difference: {avg_diff}")
    else:
        print("Timeout waiting for result matrix C")

    receiver.close()

if __name__ == "__main__":
    main()