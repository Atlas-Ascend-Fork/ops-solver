#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Run the existing compliance scanner with a complete JDK accepted by JGit.
# Keep runtime preparation separate from the scanner's policy and report handling.
set -e

select_jdk() {
    local current_home candidate version major compiler_major
    current_home=$(java -XshowSettings:properties -version 2>&1 |
        sed -n 's/^[[:space:]]*java.home = //p' || true)
    for candidate in "${JAVA_HOME:-}" "$current_home" /usr/lib/jvm/* /usr/java/* \
        /Library/Java/JavaVirtualMachines/*/Contents/Home \
        /opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
        /usr/local/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home; do
        [ -n "$candidate" ] || continue
        [ -x "$candidate/bin/java" ] && [ -x "$candidate/bin/javac" ] || continue
        version=$("$candidate/bin/java" -version 2>&1 | head -n 1)
        major=$(printf '%s\n' "$version" | sed -n 's/.*version "\([0-9][0-9]*\).*/\1/p')
        compiler_major=$("$candidate/bin/javac" -version 2>&1 |
            sed -n 's/^javac \([0-9][0-9]*\).*/\1/p')
        if [ -n "$major" ] && [ "$major" -ge 17 ] &&
            [ -n "$compiler_major" ] && [ "$compiler_major" -ge 17 ]; then
            export JAVA_HOME="$candidate"
            export PATH="$JAVA_HOME/bin:$PATH"
            return 0
        fi
    done
    return 1
}

if ! select_jdk; then
    admin=()
    if [ "$(id -u)" -ne 0 ]; then
        admin=(sudo)
    fi
    if command -v apt-get >/dev/null 2>&1; then
        "${admin[@]}" apt-get update -qq
        "${admin[@]}" apt-get install -y openjdk-17-jdk
    elif command -v yum >/dev/null 2>&1; then
        "${admin[@]}" yum install -y java-17-openjdk-devel
    elif command -v brew >/dev/null 2>&1; then
        brew install openjdk@17
    else
        echo "[OAT] A complete JDK 17+ is required; install it and set JAVA_HOME." >&2
        exit 1
    fi
    if ! select_jdk; then
        echo "[OAT] No usable JDK 17+ after installation; compliance scan not run." >&2
        exit 1
    fi
fi

echo "[OAT] Runtime selected: $JAVA_HOME"
java -version
exec bash "$(dirname "$0")/oat_check.sh" "$@"
