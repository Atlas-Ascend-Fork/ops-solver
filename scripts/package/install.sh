#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# ops-solver run 包安装脚本，参考 ops-blas install.sh 实现

set -e

OPP_PLATFORM_DIR="ops_solver"
PARAM_INVALID="0x0002"
FILE_NOT_EXIST="0x0080"

# 默认安装路径：root 用户 /usr/local/Ascend，普通用户 ~/Ascend
if [ "$(id -u)" != "0" ]; then
  DEFAULT_INSTALL_PATH="${HOME}/Ascend"
else
  DEFAULT_INSTALL_PATH="/usr/local/Ascend"
fi

# 解压目录：脚本位于 share/info/ops_solver/install.sh，归档根目录为 ../../..
RUN_PKG_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TARGET_INSTALL_PATH="${DEFAULT_INSTALL_PATH}"
IS_INSTALL="n"
IS_UNINSTALL="n"
IS_QUIET="n"
PKG_VERSION_DIR="cann"

exitlog() {
  echo "[OpsSolver] [$(date '+%Y-%m-%d %H:%M:%S')] [INFO]: End Time: $(date '+%Y-%m-%d %H:%M:%S')"
}

logandprint() {
  echo "[OpsSolver] [$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# 检查安装路径合法性（禁止明显非法字符如空格、<>|&$等）
check_install_path_valid() {
  local path="$1"
  case "${path}" in
    *' '*|*'<'*|*'>'*|*'|'*|*'&'*|*'$'*|*';'*|*'`'*)
      return 1
      ;;
    *)
      return 0
      ;;
  esac
}

check_install_path() {
  TARGET_INSTALL_PATH="$1"
  if [ -z "${TARGET_INSTALL_PATH}" ]; then
    logandprint "[ERROR]: ERR_NO:${PARAM_INVALID}; --install-path cannot be empty."
    exitlog
    exit 1
  fi
  if [[ "${TARGET_INSTALL_PATH}" == *" "* ]]; then
    logandprint "[ERROR]: ERR_NO:${PARAM_INVALID}; --install-path cannot contain spaces."
    exitlog
    exit 1
  fi
  if ! check_install_path_valid "${TARGET_INSTALL_PATH}"; then
    logandprint "[ERROR]: ERR_NO:${PARAM_INVALID}; --install-path contains invalid characters."
    exitlog
    exit 1
  fi
  # 展开 ~ 必须先于绝对路径转换（issue #198）：若先转成 $(pwd)/<path>，
  # 结果恒以 / 开头，~ 分支永不成立，~/xxx 会被安装成字面 $PWD/~/xxx 目录
  if [[ "${TARGET_INSTALL_PATH}" == ~* ]]; then
    TARGET_INSTALL_PATH="${TARGET_INSTALL_PATH/#\~/$HOME}"
  fi
  # 转换为绝对路径
  if [[ "${TARGET_INSTALL_PATH}" != /* ]]; then
    TARGET_INSTALL_PATH="$(cd "${USER_PWD:-.}" && pwd)/${TARGET_INSTALL_PATH}"
  fi
  # 若路径为软链接，解析为实际路径
  if resolved=$(readlink -f "${TARGET_INSTALL_PATH}" 2>/dev/null) && [ -n "${resolved}" ]; then
    TARGET_INSTALL_PATH="${resolved}"
  fi
}

# 解析参数（makeself 将用户参数传给脚本：./xxx.run --install --install-path=/opt/ascend）
# makeself 可能将 run 文件路径作为第一个参数传入，需跳过
get_opts() {
  while [ $# -gt 0 ]; do
    # 跳过 makeself 传入的 run 文件路径与工作目录参数。makeself 2.5.0-patch1 会向启动脚本
    # 注入两个参数（见 makeself-header.sh 的 scriptargs）：--<run文件> 与 --<cwd绝对路径>
    # （如 --/usr/local/Ascend），后者此前未被跳过，导致 --install 在任意目录都被判为非法参数
    # 而必然失败（报错信息含 cwd，表现为"报错随目录变化"）。
    # 跳过条件必须锚定"以 .run 结尾"或"以路径前缀开头"，不能对任意位置做 *".run"* 子串
    # 匹配——否则形如 --install-path=/data/x.run_env 的键值参数（值恰含 .run 子串）会被整体
    # 吞掉，用户安装路径被静默丢弃回落默认路径（issue #163）。
    if [[ "$1" == *.run ]] || [[ "$1" == ./*.run ]] || [[ "$1" == --./*.run ]] || \
       [[ "$1" == --.*.run ]] || [[ "$1" == --/* ]] || [[ "$1" == /* ]]; then
      shift
      continue
    fi
    # makeself 注入的原始工作目录参数（issue #190）：形如 --/home/tmpdir，
    # 是 makeself 打包时把调用方 pwd 一并传给内部脚本所致，并非用户参数，
    # 直接跳过，避免参数解析提前失败导致后续 --install/--install-path 未被处理
    if [[ "$1" == --/* ]]; then
      shift
      continue
    fi
    case "$1" in
      --install)
        IS_INSTALL="y"
        shift
        ;;
      --uninstall)
        IS_UNINSTALL="y"
        shift
        ;;
      --install-path=*)
        check_install_path "${1#*=}"
        shift
        ;;
      --quiet)
        IS_QUIET="y"
        shift
        ;;
      --help|-h)
        show_help
        exit 0
        ;;
      -*)
        logandprint "[ERROR]: ERR_NO:${PARAM_INVALID}; Unsupported parameter: $1"
        exitlog
        exit 1
        ;;
      *)
        shift
        ;;
    esac
  done
}

show_help() {
  if [ -f "$(dirname "$0")/help.info" ]; then
    cat "$(dirname "$0")/help.info"
  else
    echo "Usage: $0 [options]"
    echo "  --install                 Install ops_solver libraries"
    echo "  --install-path=<path>     Specify installation path (default: ${DEFAULT_INSTALL_PATH})"
    echo "  --uninstall               Uninstall ops_solver"
    echo "  --quiet                   Quiet mode"
    echo "  --help                    Show this help"
  fi
}

# 获取目标安装目录（TARGET_INSTALL_PATH + PKG_VERSION_DIR）
get_target_version_dir() {
  echo "${TARGET_INSTALL_PATH}/${PKG_VERSION_DIR}"
}

# 临时授予目录写权限（参考 ops-nn RESET_MOD=750 方式）
# 记录被修改的目录到 DIRS_GRANTED_WRITE 数组，以便后续恢复
DIRS_GRANTED_WRITE=()

grant_write_if_needed() {
  local dir="$1"
  if [ -d "${dir}" ] && [ ! -w "${dir}" ]; then
    # chmod 必须放进 if 条件（issue #200）：脚本 set -e 下独立 chmod 失败会
    # 直接中止安装且告警分支永不可达（死代码）；写成 if 形式后单个目录授权
    # 失败仅告警并继续，保持"打印告警后继续尝试安装"的设计行为
    if chmod u+w "${dir}" 2>/dev/null; then
      DIRS_GRANTED_WRITE+=("${dir}")
    else
      logandprint "[WARNING]: Failed to grant write permission on ${dir}."
    fi
  fi
}

# 恢复之前临时授予写权限的目录
restore_permissions() {
  local i=${#DIRS_GRANTED_WRITE[@]}
  while [ $i -gt 0 ]; do
    i=$((i - 1))
    local dir="${DIRS_GRANTED_WRITE[$i]}"
    if [ -d "${dir}" ]; then
      # 恢复失败不应使已成功的安装被判定为失败（issue #200）
      chmod u-w "${dir}" 2>/dev/null || true
    fi
  done
  DIRS_GRANTED_WRITE=()
}

do_install() {
  local target_dir
  target_dir=$(get_target_version_dir)
  local target_lib64="${target_dir}/lib64"
  local target_include="${target_dir}/include"
  local target_share="${target_dir}/share"
  local target_share_info="${target_dir}/share/info"
  local target_info="${target_dir}/share/info/${OPP_PLATFORM_DIR}"

  local src_lib64="${RUN_PKG_ROOT}/lib64"
  local src_include="${RUN_PKG_ROOT}/include"
  local src_info="${RUN_PKG_ROOT}/share/info/${OPP_PLATFORM_DIR}"

  if [ ! -d "${src_lib64}" ] || [ ! -d "${src_include}" ]; then
    logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST}; Run package is incomplete."
    exit 1
  fi

  if [ "${IS_QUIET}" != "y" ]; then
    logandprint "[INFO]: Installing ops_solver to ${target_dir}"
    logandprint "[INFO]: Continue? [y/n]"
    read -r yn
    if [ "${yn}" != "y" ] && [ "${yn}" != "Y" ]; then
      logandprint "[INFO]: Installation cancelled."
      exit 0
    fi
  fi

  # lib64/include 可能是软链接（如 cann-9.0.0/lib64 -> x86_64-linux/lib64），
  # 需要解析到实际目录来检查权限
  local real_lib64="${target_lib64}"
  local real_include="${target_include}"
  if [ -L "${target_lib64}" ]; then
    real_lib64="$(readlink -f "${target_lib64}")"
  fi
  if [ -L "${target_include}" ]; then
    real_include="$(readlink -f "${target_include}")"
  fi

  # 临时授予写权限（参考 ops-nn 的 reset_mod_dirs 机制）
  grant_write_if_needed "${target_dir}"
  grant_write_if_needed "${real_lib64}"
  grant_write_if_needed "${real_include}"
  grant_write_if_needed "${target_share}"
  grant_write_if_needed "${target_share_info}"
  grant_write_if_needed "${target_info}"

  mkdir -p "${target_lib64}" "${target_include}" "${target_info}"

  # 复制 .so 文件，若未找到则报错
  so_count=0
  for f in "${src_lib64}"/*.so; do
    if [ -f "$f" ]; then
      cp -f "$f" "${real_lib64}/"
      so_count=$((so_count + 1))
    fi
  done
  if [ ${so_count} -eq 0 ]; then
    restore_permissions
    logandprint "[ERROR]: No .so files found in run package (expected in ${src_lib64})."
    exit 1
  fi

  # 复制头文件
  for f in "${src_include}"/*.h; do
    [ -f "$f" ] && cp -f "$f" "${real_include}/"
  done

  # 复制 version.info 等
  if [ -d "${src_info}" ]; then
    cp -rf "${src_info}"/* "${target_info}/" 2>/dev/null || true
  fi

  # 恢复被临时修改的目录权限
  restore_permissions

  logandprint "[INFO]: ops_solver installed successfully to ${target_dir}"
  logandprint "[INFO]: Libraries: ${target_lib64}/ (${so_count} .so files)"
  logandprint "[INFO]: Headers:   ${target_include}/"
}

remove_uninstall_paths() {
  local description="$1"
  shift
  if rm "$@"; then
    return 0
  fi
  restore_permissions
  logandprint "[ERROR]: Failed to remove ${description}; ops_solver uninstall is incomplete."
  return 1
}

do_uninstall() {
  local target_dir
  target_dir=$(get_target_version_dir)
  local target_lib64="${target_dir}/lib64"
  local target_include="${target_dir}/include"
  local target_info="${target_dir}/share/info/${OPP_PLATFORM_DIR}"

  if [ ! -d "${target_dir}" ]; then
    logandprint "[INFO]: No installation found at ${target_dir}"
    exit 0
  fi

  if [ "${IS_QUIET}" != "y" ]; then
    logandprint "[INFO]: Uninstall ops_solver from ${target_dir}? [y/n]"
    read -r yn
    if [ "${yn}" != "y" ] && [ "${yn}" != "Y" ]; then
      logandprint "[INFO]: Uninstall cancelled."
      exit 0
    fi
  fi

  local real_lib64="${target_lib64}"
  local real_include="${target_include}"
  if [ -L "${target_lib64}" ]; then
    real_lib64="$(readlink -f "${target_lib64}")"
  fi
  if [ -L "${target_include}" ]; then
    real_include="$(readlink -f "${target_include}")"
  fi

  # 临时授予写权限以便删除文件
  grant_write_if_needed "${real_lib64}"
  grant_write_if_needed "${real_include}"
  grant_write_if_needed "${target_dir}/share/info"
  grant_write_if_needed "${target_info}"

  if [ -d "${real_lib64}" ]; then
    remove_uninstall_paths "solver libraries" -f "${real_lib64}"/libops_solver*.so
  fi
  if [ -d "${real_include}" ]; then
    remove_uninstall_paths "solver headers" -f "${real_include}"/cann_ops_solver*.h
  fi
  if [ -d "${target_info}" ]; then
    rm -rf "${target_info}"
  fi

  restore_permissions

  logandprint "[INFO]: ops_solver uninstalled from ${target_dir}"
}

main() {
  logandprint "[INFO]: Start ops_solver run package."
  get_opts "$@"

  if [ "${IS_INSTALL}" = "y" ] && [ "${IS_UNINSTALL}" = "y" ]; then
    logandprint "[ERROR]: Cannot use --install and --uninstall together."
    exit 1
  fi
  if [ "${IS_INSTALL}" != "y" ] && [ "${IS_UNINSTALL}" != "y" ]; then
    show_help
    exit 0
  fi

  if [ "${IS_INSTALL}" = "y" ]; then
    do_install
  else
    do_uninstall
  fi
  exitlog
}

main "$@"
