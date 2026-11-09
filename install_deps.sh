#!/bin/bash
# ============================================================================
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

PIGZ_SKIPPED=false

set -euo pipefail

run_command() {
    local cmd="$*"
    echo "Executing command: $cmd"

    # 真实退出码只能在 else 分支首条语句处取（issue #199）：`if ! cmd` 的
    # then 分支内 $? 是 ! 反转后的状态（恒为 0）；而条件失败且无 else 时
    # if 复合命令整体返回 0，fi 之后的 $? 同样为 0——两者都会把依赖安装
    # 失败静默上报为 exit 0。else 分支入口处 $? 仍为条件中命令的真实状态
    if output=$("$@" 2>&1); then
        return 0
    else
        local exit_code=$?
        echo -e "\nCommand execution failed!"
        echo -e "\nFailed command: $cmd"
        echo -e "\nError output: $output"
        echo -e "\nExit code: $exit_code"
        exit $exit_code
    fi
}

version_ge() {
    # Version comparison, format: xx.xx.xx
    IFS='.' read -r -a curr_arr <<< "$1"
    IFS='.' read -r -a req_arr <<< "$2"

    for ((i=0; i<${#req_arr[@]}; i++)); do
        curr=${curr_arr[i]:-0}
        req=${req_arr[i]}
        if (( curr > req )); then
            return 0
        elif (( curr < req )); then
            return 1
        fi
    done
    return 0
}

detect_os() {
    if [[ "$(uname -s)" == "Linux" ]]; then
        if [[ -f /etc/debian_version ]]; then
            OS="debian"
            PKG_MANAGER="apt"
        elif [[ -f /etc/redhat-release ]]; then
            OS="rhel"
            if command -v dnf &> /dev/null; then
                PKG_MANAGER="dnf"
            else
                PKG_MANAGER="yum"
            fi
        elif grep -qE '^NAME="openEuler"$|^NAME="EulerOS"$' /etc/os-release 2>/dev/null; then
            OS="euler"
            PKG_MANAGER="dnf"
        else
            echo "Unsupported Linux distribution, please install manually"
            exit 1
        fi
    elif [[ "$(uname -s)" == "Darwin" ]]; then
        OS="macos"
        if ! command -v brew &> /dev/null; then
            echo "Please install Homebrew first"
            exit 1
        fi
        PKG_MANAGER="brew"
    else
        echo "Unsupported OS type, please install manually"
        exit 1
    fi
}

install_gawk() {
    echo -e "\n==== Checking gawk ===="
    if command -v gawk &> /dev/null; then
        echo "gawk has been installed"
        return
    fi
    echo "Installing gawk..."
    case "$OS" in
        debian)
            run_command sudo $PKG_MANAGER update
            run_command sudo $PKG_MANAGER install -y gawk
            ;;
        rhel|euler)
            run_command sudo $PKG_MANAGER install -y gawk
            ;;
        macos)
            run_command brew install gawk
            ;;
    esac
    if command -v gawk &> /dev/null; then
        echo "gawk installed successfully"
    else
        echo "gawk installation failed"
        exit 1
    fi
}

install_python() {
    echo -e "\n==== Checking Python ===="
    local req_ver="3.7.0"
    local curr_ver=""
    if command -v python3 &> /dev/null; then
        curr_ver=$(python3 --version 2>&1 | awk '{print $2}')
        echo "Current Python version: $curr_ver"
        if version_ge "$curr_ver" "$req_ver"; then
            echo "Python version meets requirements"
            return
        fi
    fi
    echo "Installing Python..."
    case "$OS" in
        debian)
            run_command sudo $PKG_MANAGER update
            run_command sudo $PKG_MANAGER install -y python3 python3-pip python3-dev
            ;;
        rhel)
            if grep -q "release 7" /etc/redhat-release; then
                run_command sudo $PKG_MANAGER install -y centos-release-scl
                run_command sudo $PKG_MANAGER install -y rh-python38 rh-python38-python-devel
                # enable 必须在当前 shell 生效（对齐 install_gcc 分支的 #356/#305 口径）：
                # SCL enable 脚本可能引用未导出的 MANPATH/INFOPATH，source 前临时关闭
                # nounset 并先判存在，避免 set -e/-u 下脚本中断（issue #365）。
                if [[ -f /opt/rh/rh-python38/enable ]]; then
                    set +u
                    source /opt/rh/rh-python38/enable
                    set -u
                else
                    echo "Warning: /opt/rh/rh-python38/enable not found, fallback to explicit python paths."
                fi
                # 前置 PATH 保证复查时 command -v python3 命中 rh-python38 版本
                export PATH=/opt/rh/rh-python38/root/usr/bin:$PATH
            else
                run_command sudo $PKG_MANAGER install -y python3 python3-pip python3-devel
            fi
            ;;
        macos)
            run_command brew install python@3.10
            echo 'export PATH="/usr/local/opt/python@3.10/bin:$PATH"' >> ~/.zshrc
            # 子 shell 中 source 不影响父脚本，且 ~/.zshrc 缺失时会导致安装中止（issue #305）：
            # 追加落盘供后续会话使用，并在当前 shell 直接 export 供脚本内后续步骤使用
            export PATH="/usr/local/opt/python@3.10/bin:$PATH"
            ;;
        euler)
            run_command sudo $PKG_MANAGER install -y python3 python3-pip python3-devel
            ;;
    esac
    if command -v python3 &> /dev/null; then
        curr_ver=$(python3 --version 2>&1 | awk '{print $2}')
        if version_ge "$curr_ver" "$req_ver"; then
            echo "Python installed successfully ($curr_ver)"
        else
            echo "Python version still doesn't meet requirements, please install manually"
            exit 1
        fi
    else
        echo "Python installation failed"
        exit 1
    fi
}

install_python_deps() {
    echo -e "\n==== Checking Python dependencies ===="
    local SCRIPT_DIR
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local REQ_FILE="${SCRIPT_DIR}/requirements.txt"
    if [[ ! -f "$REQ_FILE" ]]; then
        echo "requirements.txt not found, skipping"
        return
    fi
    echo "Installing Python dependencies from $REQ_FILE ..."
    run_command pip3 install -r "$REQ_FILE"
    echo "Python dependencies installed successfully"
}

install_gcc() {
    echo -e "\n==== Checking GCC ===="
    local req_ver="7.3.0"
    local curr_ver=""
    if command -v gcc &> /dev/null; then
        curr_ver=$(gcc --version | awk '/^gcc/ {print $NF}')
    elif command -v g++ &> /dev/null; then
        curr_ver=$(g++ --version | awk '/^g\+\+/ {print $NF}')
    else
        curr_ver="0.0.0"
    fi
    echo "Current GCC version: $curr_ver"
    if version_ge "$curr_ver" "$req_ver"; then
        echo "GCC version meets requirements ($curr_ver)"
        return
    fi
    echo "Installing GCC..."
    case "$OS" in
        debian)
            run_command sudo $PKG_MANAGER update
            run_command sudo $PKG_MANAGER install -y gcc g++ build-essential
            ;;
        rhel)
            if grep -q "release 7" /etc/redhat-release; then
                run_command sudo $PKG_MANAGER install -y centos-release-scl
                run_command sudo $PKG_MANAGER install -y devtoolset-9-gcc devtoolset-9-gcc-c++
                # enable 必须在当前 shell 生效：run_command 以命令替换子 shell 执行，
                # 导出的 PATH/CC/CXX 随子 shell 退出而丢失（issue #356）。
                # 对齐 macOS 分支的"当前 shell 直接生效"口径（issue #305），
                # 并前置 PATH 保证复查时 command -v gcc 命中 devtoolset 版本。
                # SCL enable 脚本可能引用未导出的 MANPATH/INFOPATH（无 ${VAR:-} 兜底），
                # source 前临时关闭 nounset 并先判存在，避免 set -u 下脚本中断（!193 检视意见）。
                if [[ -f /opt/rh/devtoolset-9/enable ]]; then
                    set +u
                    source /opt/rh/devtoolset-9/enable
                    set -u
                else
                    echo "Warning: /opt/rh/devtoolset-9/enable not found, fallback to explicit toolchain paths."
                fi
                export CC=/opt/rh/devtoolset-9/root/usr/bin/gcc
                export CXX=/opt/rh/devtoolset-9/root/usr/bin/g++
                export PATH=/opt/rh/devtoolset-9/root/usr/bin:$PATH
            else
                run_command sudo $PKG_MANAGER install -y gcc gcc-c++
            fi
            ;;
        macos)
            if ! xcode-select -p &> /dev/null; then
                xcode-select --install
            fi
            run_command brew install gcc@11
            echo 'export CC=/usr/local/bin/gcc-11' >> ~/.zshrc
            echo 'export CXX=/usr/local/bin/g++-11' >> ~/.zshrc
            # 同上（issue #305）：落盘 + 当前 shell 直接生效，不依赖子 shell source
            export CC=/usr/local/bin/gcc-11
            export CXX=/usr/local/bin/g++-11
            ;;
        euler)
            run_command sudo $PKG_MANAGER install -y gcc gcc-c++
            ;;
    esac
    if command -v gcc &> /dev/null; then
        curr_ver=$(gcc --version | awk '/^gcc/ {print $NF}')
        if version_ge "$curr_ver" "$req_ver"; then
            echo "GCC installed successfully ($curr_ver)"
        else
            echo "GCC version still doesn't meet requirements, please install manually."
            exit 1
        fi
    else
        echo "GCC installation failed"
        exit 1
    fi
}

install_cmake() {
    echo -e "\n==== Checking CMake ===="
    local req_ver="3.16.0"
    local curr_ver=""
    if command -v cmake &> /dev/null; then
        curr_ver=$(cmake --version | awk '/^cmake/ {print $3}')
        echo "Current CMake version: $curr_ver"
        if version_ge "$curr_ver" "$req_ver"; then
            echo "CMake meets requirements"
            return
        fi
    fi
    echo "Installing CMake..."
    case "$OS" in
        debian)
            if grep -q "Ubuntu 18.04" /etc/os-release; then
                run_command wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
                run_command echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ bionic main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
                run_command sudo apt update
                run_command sudo apt install -y cmake make
            else
                run_command sudo $PKG_MANAGER update
                run_command sudo $PKG_MANAGER install -y cmake make
            fi
            ;;
        rhel)
            if grep -q "release 7" /etc/redhat-release; then
                run_command sudo $PKG_MANAGER install -y epel-release
                run_command sudo $PKG_MANAGER install -y cmake3 make
                run_command sudo ln -s /usr/bin/cmake3 /usr/bin/cmake
            else
                run_command sudo $PKG_MANAGER install -y cmake make
            fi
            ;;
        macos)
            run_command brew install cmake
            ;;
        euler)
            run_command sudo $PKG_MANAGER install -y cmake make
            ;;
    esac
    if command -v cmake &> /dev/null; then
        curr_ver=$(cmake --version | awk '/^cmake/ {print $3}')
        if version_ge "$curr_ver" "$req_ver"; then
            echo "CMake installed successfully ($curr_ver)"
        else
            echo "CMake version still doesn't meet requirements, please install manually"
            exit 1
        fi
    else
        echo "CMake installation failed"
        exit 1
    fi
}

install_pigz() {
    echo -e "\n==== Checking pigz ===="
    local req_ver="2.4"
    local curr_ver=""
    if command -v pigz &> /dev/null; then
        curr_ver=$(pigz --version 2>&1 | awk '{print $2}')
        echo "Current pigz version: $curr_ver"
        if version_ge "$curr_ver" "$req_ver"; then
            echo "pigz meets requirements"
            return
        fi
    fi
    # 非交互 stdin（CI / 管道 / < /dev/null）下 read 遇 EOF 返回非零，会触发 errexit
    # 静默终止整个安装（issue #357）：仅 tty 询问，非交互默认安装（pigz 体积小、
    # 无破坏性，与同文件其余安装项的无条件口径一致）。
    if [[ -t 0 ]]; then
        read -p "Install pigz? [Y/n] " -n 1 -r
        echo
        if [[ -n "$REPLY" && ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Skipping pigz installation"
            PIGZ_SKIPPED=true
            return
        fi
    fi
    echo "Installing pigz..."
    case "$OS" in
        debian|rhel|euler)
            run_command sudo $PKG_MANAGER install -y pigz
            ;;
        macos)
            run_command brew install pigz
            ;;
    esac
    if command -v pigz &> /dev/null; then
        curr_ver=$(pigz --version 2>&1 | awk '{print $2}')
        echo "pigz installed successfully"
    else
        echo "pigz installation failed, can be ignored"
    fi
}

install_dos2unix() {
    echo -e "\n==== Checking dos2unix ===="
    if command -v dos2unix &> /dev/null; then
        echo "dos2unix has been installed"
        return
    fi
    echo "Installing dos2unix..."
    case "$OS" in
        debian|rhel|euler)
            run_command sudo $PKG_MANAGER install -y dos2unix
            ;;
        macos)
            run_command brew install dos2unix
            ;;
    esac
    if command -v dos2unix &> /dev/null; then
        echo "dos2unix installed successfully"
    else
        echo "dos2unix installation failed"
        exit 1
    fi
}

install_patch() {
    echo -e "\n==== Checking patch ===="
    if command -v patch &> /dev/null; then
        echo "patch has been installed"
        return
    fi
    echo "Installing patch..."
    case "$OS" in
        # euler 分支对齐同文件 install_pigz/install_dos2unix 的口径（issue #359）：
        # openEuler/EulerOS 由 dnf 安装，缺分支时未预装 patch 的环境直接失败退出。
        debian|rhel|euler)
            run_command sudo $PKG_MANAGER install -y patch
            ;;
        macos)
            run_command brew install patch
            ;;
    esac
    if command -v patch &> /dev/null; then
        echo "patch installed successfully"
    else
        echo "patch installation failed"
        exit 1
    fi
}

install_lapack() {
    echo -e "\n==== Checking LAPACK ===="
    # 复数 QR/最小二乘的 complex_qr 测试目标在 Ascend 950 分支无条件
    # find_package(LAPACK REQUIRED)（test/complex_qr/CMakeLists.txt），依赖缺失时
    # 测试工程配置阶段直接失败且此前无任何安装/提示（issue #373）。库存在性无法用
    # command -v 判断：Linux 走 ldconfig 缓存，macOS 走 brew 清单。
    if [[ "$(uname -s)" == "Linux" ]]; then
        if ldconfig -p 2>/dev/null | grep -q "liblapack\.so"; then
            echo "LAPACK has been installed"
            return
        fi
    else
        if brew list --formula 2>/dev/null | grep -qx "lapack"; then
            echo "LAPACK has been installed"
            return
        fi
    fi
    echo "Installing LAPACK..."
    case "$OS" in
        debian)
            run_command sudo $PKG_MANAGER install -y liblapack-dev
            ;;
        rhel|euler)
            run_command sudo $PKG_MANAGER install -y lapack-devel
            ;;
        macos)
            run_command brew install lapack
            ;;
    esac
    if [[ "$(uname -s)" == "Linux" ]] && ! ldconfig -p 2>/dev/null | grep -q "liblapack\.so"; then
        echo "LAPACK installation failed, install manually (required by test/complex_qr on Ascend 950)"
        exit 1
    fi
    echo "LAPACK installed successfully"
}

main() {
    echo "===================================================="
    echo "Starting ops-solver dependency installation"
    echo "===================================================="

    detect_os
    install_gawk
    install_python
    install_python_deps
    install_gcc
    install_cmake
    install_pigz
    install_dos2unix
    install_patch
    install_lapack

    echo -e "===================================================="
    if [[ "${PIGZ_SKIPPED:-false}" == "true" ]]; then
        local install_hint
        case "$OS" in
            macos)
                install_hint="brew install pigz"
                ;;
            *)
                install_hint="sudo $PKG_MANAGER install pigz"
                ;;
        esac
        echo "All mandatory dependencies installed successfully! (pigz skipped, run '$install_hint' to install manually)"
    else
        echo "All dependencies installed successfully!"
    fi
    echo "===================================================="
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
