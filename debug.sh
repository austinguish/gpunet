#!/bin/bash

# 保存为 ~/doca_debug_wrapper.sh
# 设置DOCA和DPDK环境变量
export LD_LIBRARY_PATH=/opt/mellanox/doca/lib/x86_64-linux-gnu:/opt/mellanox/dpdk/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# 检查参数
if [ "$1" == "--version" ]; then
    sudo -n gdbserver --version
    exit $?
fi

# 检查是否提供了可执行文件参数
if [ -z "$1" ]; then
    echo "Usage: $0 <executable> [args...]"
    exit 1
fi

# 检查可执行文件是否存在
if [ ! -f "$1" ]; then
    echo "Error: Executable file '$1' not found"
    exit 1
fi

# 启动gdbserver
echo "Starting gdbserver..."
sudo -n gdbserver localhost:1234 "$@"
