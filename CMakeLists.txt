cmake_minimum_required(VERSION 3.10)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_C_COMPILER /usr/bin/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)

set(CMAKE_CUDA_ARCHITECTURES "75;80;86")

project(DOCA_APPLICATIONS CUDA CXX C)

enable_language(CUDA)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -rdc=true -Xcompiler=-Wall,-Winvalid-pch,-Wnon-virtual-dtor")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "/opt/mellanox/doca/lib/x86_64-linux-gnu:/opt/mellanox/dpdk/lib/x86_64-linux-gnu")

project(DOCA_APPLICATIONS CUDA CXX C)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "/opt/mellanox/doca/lib/x86_64-linux-gnu:/opt/mellanox/dpdk/lib/x86_64-linux-gnu")

# Set CUDA settings
set(CMAKE_CUDA_ARCHITECTURES "80;90")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -rdc=true -Xcompiler=-Wall,-Winvalid-pch,-Wnon-virtual-dtor")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Add compile definitions
add_definitions(
        -D_FILE_OFFSET_BITS=64
        -DDOCA_ALLOW_EXPERIMENTAL_API
        -DDOCA_ALLOW_DEPRECATED_API
        -DDOCA_USE_LIBBSD
        -DRTE_USE_LIBBSD
        -DALLOW_EXPERIMENTAL_API
        -DGPU_SUPPORT
)

# Set compile options
add_compile_options(
        -Wall
        -Winvalid-pch
        -Wno-format-zero-length
        -Wno-missing-braces
        -Wno-missing-field-initializers
        -march=corei7
)

add_compile_options(-include ${CMAKE_SOURCE_DIR}/dpdk_include_path.h)

# Create dpdk_include_path.h
file(WRITE ${CMAKE_SOURCE_DIR}/dpdk_include_path.h
        "#include <rte_config.h>")

# Include directories
include_directories(
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/matmul
        ${CMAKE_SOURCE_DIR}/common
        ${CMAKE_SOURCE_DIR}/config_queues
        /opt/mellanox/doca/include
        /usr/include/libnl3
        /opt/mellanox/dpdk/include/dpdk
        /opt/mellanox/dpdk/include/dpdk/x86_64-linux-gnu/dpdk
        /opt/mellanox/dpdk/include/x86_64-linux-gnu/dpdk
        /usr/local/cuda/include
)

# Find required packages
find_package(CUDA REQUIRED)

# Find DOCA libraries
find_library(DOCA_ARGP_LIB doca_argp PATHS /opt/mellanox/doca/lib/x86_64-linux-gnu REQUIRED)
find_library(DOCA_GPUNETIO_LIB doca_gpunetio PATHS /opt/mellanox/doca/lib/x86_64-linux-gnu REQUIRED)
find_library(DOCA_GPUNETIO_DEVICE_LIB doca_gpunetio_device PATHS /opt/mellanox/doca/lib/x86_64-linux-gnu REQUIRED)
find_library(DOCA_ETH_LIB doca_eth PATHS /opt/mellanox/doca/lib/x86_64-linux-gnu REQUIRED)
find_library(DOCA_DPDK_BRIDGE_LIB doca_dpdk_bridge PATHS /opt/mellanox/doca/lib/x86_64-linux-gnu REQUIRED)
find_library(DOCA_FLOW_LIB doca_flow PATHS /opt/mellanox/doca/lib/x86_64-linux-gnu REQUIRED)
find_library(DOCA_FLOW_CT_LIB doca_flow_ct PATHS /opt/mellanox/doca/lib/x86_64-linux-gnu REQUIRED)
find_library(DOCA_FLOW_TUNE_LIB doca_flow_tune_server PATHS /opt/mellanox/doca/lib/x86_64-linux-gnu REQUIRED)
find_library(DOCA_COMMON_LIB doca_common PATHS /opt/mellanox/doca/lib/x86_64-linux-gnu REQUIRED)

# 移除重复的源文件
set(COMMON_SOURCES
        ${CMAKE_SOURCE_DIR}/config_queues/device.c
        ${CMAKE_SOURCE_DIR}/config_queues/args.c
        ${CMAKE_SOURCE_DIR}/config_queues/udp.c
        ${CMAKE_SOURCE_DIR}/config_queues/flow.c
        ${CMAKE_SOURCE_DIR}/config_queues/http_txbuf.c
        ${CMAKE_SOURCE_DIR}/config_queues/tcp_queues.c
        ${CMAKE_SOURCE_DIR}/config_queues/udp_queues.c
        ${CMAKE_SOURCE_DIR}/config_queues/icmp_queues.c
        ${CMAKE_SOURCE_DIR}/dpdk_tcp/tcp_cpu_rss_func.c
        ${CMAKE_SOURCE_DIR}/dpdk_tcp/tcp_session_table.c
        ${CMAKE_SOURCE_DIR}/common/utils.c
        ${CMAKE_SOURCE_DIR}/samples/common.c
)

# Define specific sources for each target by combining common sources with target-specific files
set(C_SOURCES
        ${COMMON_SOURCES}
        ${CMAKE_SOURCE_DIR}/gpu_packet_processing.c
)

set(mm_SOURCES
        ${COMMON_SOURCES}
        ${CMAKE_SOURCE_DIR}/matrix_mult.c
)

set(BW_SOURCES
        ${COMMON_SOURCES}
        ${CMAKE_SOURCE_DIR}/config_queues/tcp_bw_queues.c
        ${CMAKE_SOURCE_DIR}/gpu_bw.c
)

# CUDA sources remain unchanged as they don't have overlap
set(CUDA_SOURCES
        ${CMAKE_SOURCE_DIR}/gpu_kernels/receive_tcp.cu
        ${CMAKE_SOURCE_DIR}/gpu_kernels/receive_udp.cu
        ${CMAKE_SOURCE_DIR}/gpu_kernels/receive_icmp.cu
        ${CMAKE_SOURCE_DIR}/gpu_kernels/http_server.cu
        ${CMAKE_SOURCE_DIR}/gpu_kernels/tcp_bw.cu
        ${CMAKE_SOURCE_DIR}/gpu_kernels/udp.cu
        ${CMAKE_SOURCE_DIR}/gpu_kernels/udp_send.cu
)
# Create CUDA library for device code
add_library(gpu_kernels STATIC ${CUDA_SOURCES})
set_target_properties(gpu_kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
)

# Create the executable
add_executable(doca_gpu_packet_processing ${C_SOURCES})

# Link libraries
target_link_libraries(doca_gpu_packet_processing
        gpu_kernels
        ${DOCA_ARGP_LIB}
        ${DOCA_GPUNETIO_LIB}
        ${DOCA_GPUNETIO_DEVICE_LIB}
        ${DOCA_ETH_LIB}
        ${DOCA_DPDK_BRIDGE_LIB}
        ${DOCA_FLOW_LIB}
        ${DOCA_FLOW_CT_LIB}
        ${DOCA_FLOW_TUNE_LIB}
        ${DOCA_COMMON_LIB}
        resolv
        cudart
        cuda
        pthread
        bsd
)

# Set DPDK libraries
set(DPDK_LIBS
        rte_node rte_graph rte_flow_perf rte_pipeline rte_table
        rte_pdump rte_port rte_fib rte_ipsec rte_vhost rte_stack
        rte_security rte_sched rte_reorder rte_rib rte_dmadev
        rte_regexdev rte_rawdev rte_power rte_pcapng rte_member
        rte_lpm rte_latencystats rte_jobstats rte_gso rte_gro
        rte_gpudev rte_eventdev rte_efd rte_distributor rte_cryptodev
        rte_compressdev rte_cfgfile rte_bpf rte_bitratestats rte_bbdev
        rte_acl rte_timer rte_metrics rte_cmdline rte_pci rte_ethdev
        rte_ip_frag rte_hash rte_meter rte_net rte_mbuf rte_mempool
        rte_rcu rte_ring rte_eal rte_telemetry rte_kvargs
)

# Link DPDK libraries
foreach(lib ${DPDK_LIBS})
    target_link_libraries(doca_gpu_packet_processing
            /opt/mellanox/dpdk/lib/x86_64-linux-gnu/lib${lib}.so
    )
endforeach()



# add a new executable gpu_bw  for measuring the bandwidth




add_executable(gpu_bw ${BW_SOURCES})
target_link_libraries(gpu_bw
        gpu_kernels
        ${DOCA_ARGP_LIB}
        ${DOCA_GPUNETIO_LIB}
        ${DOCA_GPUNETIO_DEVICE_LIB}
        ${DOCA_ETH_LIB}
        ${DOCA_DPDK_BRIDGE_LIB}
        ${DOCA_FLOW_LIB}
        ${DOCA_FLOW_CT_LIB}
        ${DOCA_FLOW_TUNE_LIB}
        ${DOCA_COMMON_LIB}
        resolv
        cudart
        cuda
        pthread
        bsd
)

# Link DPDK libraries
foreach(lib ${DPDK_LIBS})
    target_link_libraries(gpu_bw
            /opt/mellanox/dpdk/lib/x86_64-linux-gnu/lib${lib}.so
    )
endforeach()


#Set installation rules

add_executable(matrix_mult ${mm_SOURCES})

# Link libraries
target_link_libraries(matrix_mult
        gpu_kernels
        ${DOCA_ARGP_LIB}
        ${DOCA_GPUNETIO_LIB}
        ${DOCA_GPUNETIO_DEVICE_LIB}
        ${DOCA_ETH_LIB}
        ${DOCA_DPDK_BRIDGE_LIB}
        ${DOCA_FLOW_LIB}
        ${DOCA_FLOW_CT_LIB}
        ${DOCA_FLOW_TUNE_LIB}
        ${DOCA_COMMON_LIB}
        resolv
        cudart
        cuda
        cublas
        pthread
        bsd
)


# Link DPDK libraries
foreach(lib ${DPDK_LIBS})
    target_link_libraries(matrix_mult
            /opt/mellanox/dpdk/lib/x86_64-linux-gnu/lib${lib}.so
    )
endforeach()


# Add matrix UDP receiver executable
set(MATRIX_UDP_SOURCES
        ${CMAKE_SOURCE_DIR}/mat_mul_cpu.cu
)

add_executable(matrix_udp_receiver ${MATRIX_UDP_SOURCES})

# Link libraries
target_link_libraries(matrix_udp_receiver
        resolv
        pthread
        bsd
        cudart
        cublas
)

# Link DPDK libraries
foreach(lib ${DPDK_LIBS})
    target_link_libraries(matrix_udp_receiver
            /opt/mellanox/dpdk/lib/x86_64-linux-gnu/lib${lib}.so
    )
endforeach()

# Set CUDA compile options
set_target_properties(matrix_udp_receiver PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_definitions(matrix_udp_receiver PRIVATE
            -DRTE_LOG_LEVEL=RTE_LOG_DEBUG
    )
endif()





if(CMAKE_BUILD_TYPE MATCHES Debug)
    # Add debug symbols
    add_compile_options(-g)

    # Add CUDA debug options
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")

    # Add sanitizer options if needed
    if(USE_SANITIZER)
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address)
    endif()

    # Add DPDK debug options
    add_definitions(-DRTE_LOG_LEVEL=RTE_LOG_DEBUG)
endif()

set_target_properties(doca_gpu_packet_processing PROPERTIES
        COMPILE_FLAGS "-g"
        CUDA_FLAGS "-g -G"
)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0")