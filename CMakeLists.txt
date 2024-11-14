cmake_minimum_required(VERSION 3.18)
project(gpu_packet_processing CUDA CXX C)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable required CPU features
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mssse3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mssse3 -march=native")

# 首先查找PkgConfig
find_package(PkgConfig REQUIRED)
if(NOT PKG_CONFIG_FOUND)
    message(FATAL_ERROR "pkg-config not found!")
endif()

# 设置DOCA和DPDK路径
set(DOCA_ROOT_DIR "/opt/mellanox/doca")
set(DOCA_LIB_DIR "${DOCA_ROOT_DIR}/lib/x86_64-linux-gnu")
set(DPDK_ROOT_DIR "/opt/mellanox/dpdk")
set(DPDK_LIB_DIR "${DPDK_ROOT_DIR}/lib/x86_64-linux-gnu")

# 设置PKG_CONFIG_PATH
set(ENV{PKG_CONFIG_PATH} "${DOCA_LIB_DIR}/pkgconfig:${DPDK_LIB_DIR}/pkgconfig:$ENV{PKG_CONFIG_PATH}")

# 查找DOCA包
pkg_check_modules(DOCA REQUIRED
    doca-common
    doca-gpunetio
    doca-argp
    doca-flow
    doca-telemetry-exporter
    doca-dpdk-bridge
    doca-comch
    doca-eth
)

# 查找DPDK包
pkg_check_modules(DPDK REQUIRED libdpdk)

# CUDA Configuration
find_package(CUDA 12.1 REQUIRED)
if(CUDA_FOUND)
    message(STATUS "Found CUDA ${CUDA_VERSION_STRING}")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
        -gencode arch=compute_80,code=sm_80
        -gencode arch=compute_90,code=sm_90
        -rdc=true
        -forward-unknown-to-host-compiler
        --relocatable-device-code=true
    )
endif()

# Add compile definitions
add_definitions(-DGPU_SUPPORT)
add_definitions(-DDOCA_ALLOW_EXPERIMENTAL_API)

# 源文件定义
set(COMMON_SOURCES
    common/comch_utils.c
    common/dpdk_utils.c
    common/flow_parser.c
    common/flow_pipes_manager.c
    common/flow_skeleton.c
    common/pack.c
    common/telemetry_exporter.c
    common/utils.c
)

set(CONFIG_QUEUE_SOURCES
    config_queues/args.c
    config_queues/device.c
    config_queues/flow.c
    config_queues/http_txbuf.c
    config_queues/udp_queues.c
    config_queues/tcp_queues.c
    config_queues/icmp_queues.c
)

set(DPDK_TCP_SOURCES
    dpdk_tcp/tcp_cpu_rss_func.c
    dpdk_tcp/tcp_session_table.c
)

set(CUDA_SOURCES
    gpu_kernels/receive_tcp.cu
    gpu_kernels/receive_udp.cu
    gpu_kernels/receive_icmp.cu
    gpu_kernels/http_server.cu
        bw/measure_udp_bw.cu
)

set(ALL_SOURCES
    ${COMMON_SOURCES}
    ${CONFIG_QUEUE_SOURCES}
    ${DPDK_TCP_SOURCES}
    samples/common.c
    gpu_packet_processing.c
)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/config_queues
    ${CMAKE_SOURCE_DIR}/samples
    ${CMAKE_SOURCE_DIR}/dpdk_tcp
    ${DOCA_INCLUDE_DIRS}
    ${DPDK_ROOT_DIR}/include
)

# Link directories
link_directories(
    ${DOCA_LIBRARY_DIRS}
    ${DPDK_LIB_DIR}
    ${DOCA_LIB_DIR}
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

# Create CUDA object library
cuda_add_library(gpu_kernels STATIC ${CUDA_SOURCES})
target_compile_options(gpu_kernels PRIVATE ${CUDA_NVCC_FLAGS})
set_target_properties(gpu_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Create main executable
add_executable(doca_gpu_packet_processing ${ALL_SOURCES})
set_target_properties(doca_gpu_packet_processing PROPERTIES
    COMPILE_FLAGS "-O3"
)

# Link libraries
target_link_libraries(doca_gpu_packet_processing
    PRIVATE
        gpu_kernels
        resolv
        pthread
        -L${DOCA_LIB_DIR}
        -Wl,--whole-archive
        -ldoca_dpdk_bridge
        -ldoca_comch
        -ldoca_eth
        -ldoca_gpunetio
        -ldoca_gpunetio_device
        -Wl,--no-whole-archive
        ${DOCA_LIBRARIES}
        -L${DPDK_LIB_DIR}
        -Wl,--whole-archive
        -lrte_node
        -lrte_graph
        -lrte_flow_perf
        -lrte_pipeline
        -lrte_table
        -lrte_pdump
        -lrte_port
        -lrte_fib
        -lrte_ipsec
        -lrte_vhost
        -lrte_stack
        -lrte_security
        -lrte_sched
        -lrte_reorder
        -lrte_rib
        -lrte_dmadev
        -lrte_regexdev
        -lrte_rawdev
        -lrte_power
        -lrte_pcapng
        -lrte_member
        -lrte_lpm
        -lrte_latencystats
        -lrte_jobstats
        -lrte_gso
        -lrte_gro
        -lrte_gpudev
        -lrte_eventdev
        -lrte_efd
        -lrte_distributor
        -lrte_cryptodev
        -lrte_compressdev
        -lrte_cfgfile
        -lrte_bpf
        -lrte_bitratestats
        -lrte_bbdev
        -lrte_acl
        -lrte_timer
        -lrte_metrics
        -lrte_cmdline
        -lrte_pci
        -lrte_ethdev
        -lrte_ip_frag
        -lrte_hash
        -lrte_meter
        -lrte_net
        -lrte_mbuf
        -lrte_mempool
        -lrte_rcu
        -lrte_ring
        -lrte_eal
        -lrte_telemetry
        -lrte_kvargs
        -Wl,--no-whole-archive
        -lcuda
        -lcudart
)

# Installation
install(TARGETS doca_gpu_packet_processing
    RUNTIME DESTINATION bin
)
