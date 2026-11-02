/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include "../../utils/complex_qr/arch35/complex_qr_transfer.h"
namespace ComplexQrTransfer
{
constexpr uint32_t TILE = 64;
constexpr uint32_t COMPLEX_PITCH = 2 * TILE + 8;
constexpr uint32_t REAL_PITCH = TILE + 8;
constexpr uint32_t PLANE = TILE * TILE;
constexpr uint32_t INPUT_PLANE = TILE * REAL_PITCH;

template <uint32_t RowTile>
__simd_vf__ inline void Pack(__ubuf__ float* input, __ubuf__ float* output, uint32_t rows, uint32_t cols,
                             bool rightLayout)
{
    constexpr uint32_t complexPitch = 2 * RowTile + 8, plane = RowTile * TILE;
    Reg::RegTensor<uint32_t> lane, offsets, bits, maskBits;
    Reg::RegTensor<float> real, imag, high, low;
    uint32_t count = cols;
    auto mask = Reg::UpdateMask<float>(count);
    uint32_t all = TILE;
    auto fullMask = Reg::UpdateMask<float>(all);
    Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lane), int32_t(0));
    // Initialize every gather address, including masked-off tail lanes.
    Reg::Muls(offsets, lane, rightLayout ? uint32_t(2) : complexPitch, fullMask);
    Reg::Duplicate(maskBits, 0xffffe000U);
    for (uint16_t row = 0; row < RowTile; ++row)
    {
        Reg::Duplicate(real, 0.0f);
        Reg::Duplicate(imag, 0.0f);
        if (row < rows && cols)
        {
            uint32_t base = rightLayout ? row * complexPitch : 2 * row;
            Reg::Gather(real, input + base, offsets, mask);
            Reg::Gather(imag, input + base + 1, offsets, mask);
            if (rightLayout)
            {
                Reg::Muls(imag, imag, -1.0f, mask);
            }
        }
        Reg::And(bits, reinterpret_cast<Reg::RegTensor<uint32_t>&>(real), maskBits, fullMask);
        high = reinterpret_cast<Reg::RegTensor<float>&>(bits);
        Reg::Sub(low, real, high, fullMask);
        Reg::StoreAlign(output + row * TILE, high, fullMask);
        Reg::StoreAlign(output + 2 * plane + row * TILE, low, fullMask);
        Reg::And(bits, reinterpret_cast<Reg::RegTensor<uint32_t>&>(imag), maskBits, fullMask);
        high = reinterpret_cast<Reg::RegTensor<float>&>(bits);
        Reg::Sub(low, imag, high, fullMask);
        Reg::StoreAlign(output + plane + row * TILE, high, fullMask);
        Reg::StoreAlign(output + 3 * plane + row * TILE, low, fullMask);
    }
}

// Column-contiguous realification for the transposed B operand of V^H C.
__simd_vf__ inline void PackTransposedTile(__ubuf__ float* input, __ubuf__ float* output, uint32_t rows,
                                           uint32_t columns)
{
    constexpr uint32_t rowTile = 256, columnTile = 32, plane = rowTile * columnTile;
    Reg::RegTensor<uint32_t> lane, offsets, bits, maskBits;
    Reg::RegTensor<float> real, imag, high, low;
    uint32_t all = 64;
    auto full = Reg::UpdateMask<float>(all);
    Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lane), int32_t(0));
    Reg::Muls(offsets, lane, uint32_t(2), full);
    Reg::Duplicate(maskBits, 0xffffe000U);
    for (uint16_t column = 0; column < columnTile; ++column)
    {
        for (uint16_t row = 0; row < rowTile; row += 64)
        {
            uint32_t count = column < columns && row < rows ? (rows - row < 64 ? rows - row : 64) : 0;
            auto valid = Reg::UpdateMask<float>(count);
            Reg::Duplicate(real, 0.0f);
            Reg::Duplicate(imag, 0.0f);
            if (column < columns && row < rows)
            {
                Reg::Gather(real, input + column * 2 * rowTile + 2 * row, offsets, valid);
                Reg::Gather(imag, input + column * 2 * rowTile + 2 * row + 1, offsets, valid);
            }
            Reg::And(bits, reinterpret_cast<Reg::RegTensor<uint32_t>&>(real), maskBits, full);
            high = reinterpret_cast<Reg::RegTensor<float>&>(bits);
            Reg::Sub(low, real, high, full);
            Reg::StoreAlign(output + column * rowTile + row, high, full);
            Reg::StoreAlign(output + 2 * plane + column * rowTile + row, low, full);
            Reg::And(bits, reinterpret_cast<Reg::RegTensor<uint32_t>&>(imag), maskBits, full);
            high = reinterpret_cast<Reg::RegTensor<float>&>(bits);
            Reg::Sub(low, imag, high, full);
            Reg::StoreAlign(output + plane + column * rowTile + row, high, full);
            Reg::StoreAlign(output + 3 * plane + column * rowTile + row, low, full);
        }
    }
}

__aicore__ inline void PackTransposedDma(__gm__ const float* a, __gm__ float* high, __gm__ float* low, uint64_t lda,
                                         uint32_t first, uint32_t columnStart, uint32_t rows, uint32_t columns,
                                         uint32_t packedRows, uint32_t packedColumns)
{
    constexpr uint32_t rowTile = 256, columnTile = 32, plane = rowTile * columnTile;
    TPipe pipe;
    TBuf<QuePosition::VECCALC> inputBuffer, outputBuffer;
    pipe.InitBuffer(inputBuffer, 2 * plane * sizeof(float));
    pipe.InitBuffer(outputBuffer, 4 * plane * sizeof(float));
    auto input = inputBuffer.Get<float>(), output = outputBuffer.Get<float>();
    GlobalTensor<float> source, gh, gl;
    source.SetGlobalBuffer(const_cast<__gm__ float*>(a));
    gh.SetGlobalBuffer(high);
    gl.SetGlobalBuffer(low);
    uint32_t columnTiles = packedColumns / columnTile;
    uint64_t tiles = uint64_t((packedRows + rowTile - 1) / rowTile) * columnTiles;
    for (uint64_t tile = GetBlockIdx(); tile < tiles; tile += GetBlockNum())
    {
        uint32_t row = tile / columnTiles * rowTile, column = tile % columnTiles * columnTile;
        uint32_t nr = row < rows ? (rows - row < rowTile ? rows - row : rowTile) : 0;
        uint32_t nc = column < columns ? (columns - column < columnTile ? columns - column : columnTile) : 0;
        uint32_t width = packedRows - row < rowTile ? packedRows - row : rowTile;
        Sync<HardEvent::V_MTE2>();
        if (nr && nc)
        {
            Read(input, source, 2 * (first + row + uint64_t(columnStart + column) * lda), nc, 2 * nr, 2 * lda,
                 2 * rowTile);
        }
        Sync<HardEvent::MTE2_V>();
        asc_vf_call<PackTransposedTile>(reinterpret_cast<__ubuf__ float*>(input.GetPhyAddr()),
                                        reinterpret_cast<__ubuf__ float*>(output.GetPhyAddr()), nr, nc);
        Sync<HardEvent::V_MTE3>();
        uint64_t offset = uint64_t(column) * 2 * packedRows + row;
        Write(gh, offset, output, columnTile, width, 2 * packedRows, rowTile);
        Write(gh, offset + packedRows, output[plane], columnTile, width, 2 * packedRows, rowTile);
        Write(gl, offset, output[2 * plane], columnTile, width, 2 * packedRows, rowTile);
        Write(gl, offset + packedRows, output[3 * plane], columnTile, width, 2 * packedRows, rowTile);
        Sync<HardEvent::MTE3_V>();
    }
}

__simd_vf__ inline void UpdateRight(__ubuf__ float* products, __ubuf__ float* original, __ubuf__ float* output,
                                    uint32_t rows, uint32_t cols)
{
    Reg::RegTensor<uint32_t> lane, offsets;
    Reg::RegTensor<float> real, imag, product, first, second;
    uint32_t count = cols, all = TILE;
    auto mask = Reg::UpdateMask<float>(count);
    auto full = Reg::UpdateMask<float>(all);
    Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lane), int32_t(0));
    Reg::Muls(offsets, lane, uint32_t(2), full);
    for (uint16_t row = 0; row < rows; ++row)
    {
        Reg::Gather(real, original + row * COMPLEX_PITCH, offsets, mask);
        Reg::Gather(imag, original + row * COMPLEX_PITCH + 1, offsets, mask);
        Reg::LoadAlign(product, products + row * REAL_PITCH);
        Reg::Sub(real, real, product, mask);
        Reg::LoadAlign(product, products + INPUT_PLANE + row * REAL_PITCH);
        // The blocked computation applies Q to C^H; write its conjugate back.
        Reg::Add(imag, imag, product, mask);
        Reg::Interleave(first, second, real, imag);
        Reg::StoreAlign(output + row * 2 * TILE, first, full);
        Reg::StoreAlign(output + row * 2 * TILE + TILE, second, full);
    }
}

template <uint32_t RowTile>
__simd_vf__ inline void Update(__ubuf__ float* products, __ubuf__ float* original, __ubuf__ float* output,
                               uint32_t rows, uint32_t cols)
{
    constexpr uint32_t complexPitch = 2 * RowTile + 8, inputPlane = RowTile * REAL_PITCH;
    Reg::RegTensor<uint32_t> lane, offsets, complexOffsets;
    Reg::RegTensor<float> real, imag, product, first, second;
    uint32_t full = TILE;
    auto storeMask = Reg::UpdateMask<float>(full);
    Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lane), int32_t(0));
    Reg::Muls(offsets, lane, REAL_PITCH, storeMask);
    Reg::Muls(complexOffsets, lane, uint32_t(2), storeMask);
    // Interleave produces two 64-float vectors; only 2*rows values are stored by DMA.
    for (uint16_t col = 0; col < cols; ++col)
    {
        for (uint16_t chunk = 0; chunk < RowTile / TILE; ++chunk)
        {
            uint32_t start = chunk * TILE;
            uint32_t count = rows > start ? (rows - start < TILE ? rows - start : TILE) : 0;
            auto mask = Reg::UpdateMask<float>(count);
            Reg::Gather(product, products + start * REAL_PITCH + col, offsets, mask);
            Reg::Gather(real, original + col * complexPitch + 2 * start, complexOffsets, mask);
            Reg::Sub(real, real, product, mask);
            Reg::Gather(product, products + inputPlane + start * REAL_PITCH + col, offsets, mask);
            Reg::Gather(imag, original + col * complexPitch + 2 * start + 1, complexOffsets, mask);
            Reg::Sub(imag, imag, product, mask);
            Reg::Interleave(first, second, real, imag);
            Reg::StoreAlign(output + col * 2 * RowTile + 2 * start, first, storeMask);
            Reg::StoreAlign(output + col * 2 * RowTile + 2 * start + TILE, second, storeMask);
        }
    }
}

template <uint32_t RowTile>
__aicore__ inline void PackDma(__gm__ const float* a, __gm__ float* high, __gm__ float* low, uint64_t lda,
                               uint32_t first, uint32_t end, uint32_t rows, uint32_t cols, uint32_t packedRows,
                               uint32_t packedCols, bool rightLayout)
{
    constexpr uint32_t complexPitch = 2 * RowTile + 8, plane = RowTile * TILE;
    TPipe pipe;
    TBuf<QuePosition::VECCALC> inputBuffer, outputBuffer;
    pipe.InitBuffer(inputBuffer, TILE * complexPitch * sizeof(float));
    pipe.InitBuffer(outputBuffer, 4 * plane * sizeof(float));
    auto input = inputBuffer.Get<float>(), output = outputBuffer.Get<float>();
    GlobalTensor<float> source, highGm, lowGm;
    source.SetGlobalBuffer(const_cast<__gm__ float*>(a));
    highGm.SetGlobalBuffer(high);
    lowGm.SetGlobalBuffer(low);
    uint32_t columnTiles = packedCols / TILE;
    for (uint64_t t = GetBlockIdx(); t < uint64_t(packedRows / RowTile) * columnTiles; t += GetBlockNum())
    {
        uint32_t row = t / columnTiles * RowTile, col = t % columnTiles * TILE;
        uint32_t nr = row >= rows ? 0 : rows - row < RowTile ? rows - row : RowTile;
        uint32_t nc = col >= cols ? 0 : cols - col < TILE ? cols - col : TILE;
        Sync<HardEvent::V_MTE2>();
        if (nr && nc)
        {
            if (rightLayout)
            {
                Read(input, source, 2 * (end + col + uint64_t(first + row) * lda), nr, 2 * nc, 2 * lda, complexPitch);
            }
            else
            {
                Read(input, source, 2 * (first + row + uint64_t(end + col) * lda), nc, 2 * nr, 2 * lda, complexPitch);
            }
        }
        Sync<HardEvent::MTE2_V>();
        asc_vf_call<Pack<RowTile>>(reinterpret_cast<__ubuf__ float*>(input.GetPhyAddr()),
                                   reinterpret_cast<__ubuf__ float*>(output.GetPhyAddr()), nr, nc, rightLayout);
        Sync<HardEvent::V_MTE3>();
        uint64_t offset = uint64_t(row) * packedCols + col, imag = uint64_t(packedRows) * packedCols;
        Write(highGm, offset, output, RowTile, TILE, packedCols, TILE);
        Write(highGm, offset + imag, output[plane], RowTile, TILE, packedCols, TILE);
        Write(lowGm, offset, output[2 * plane], RowTile, TILE, packedCols, TILE);
        Write(lowGm, offset + imag, output[3 * plane], RowTile, TILE, packedCols, TILE);
        Sync<HardEvent::MTE3_V>();
    }
}

template <uint32_t RowTile>
__aicore__ inline void UpdateDma(__gm__ float* a, __gm__ const float* product, uint64_t lda, uint32_t first,
                                 uint32_t end, uint32_t rows, uint32_t cols, uint32_t packedRows, uint32_t packedCols,
                                 bool rightLayout)
{
    constexpr uint32_t complexPitch = 2 * RowTile + 8, inputPlane = RowTile * REAL_PITCH, plane = RowTile * TILE;
    TPipe pipe;
    TBuf<QuePosition::VECCALC> productBuffer, originalBuffer, outputBuffer;
    pipe.InitBuffer(productBuffer, 2 * inputPlane * sizeof(float));
    pipe.InitBuffer(originalBuffer, TILE * complexPitch * sizeof(float));
    pipe.InitBuffer(outputBuffer, 2 * plane * sizeof(float));
    auto products = productBuffer.Get<float>(), original = originalBuffer.Get<float>(),
         output = outputBuffer.Get<float>();
    GlobalTensor<float> target, productGm;
    target.SetGlobalBuffer(a);
    productGm.SetGlobalBuffer(const_cast<__gm__ float*>(product));
    uint32_t columnTiles = (cols + TILE - 1) / TILE;
    for (uint64_t t = GetBlockIdx(); t < uint64_t((rows + RowTile - 1) / RowTile) * columnTiles; t += GetBlockNum())
    {
        uint32_t row = t / columnTiles * RowTile, col = t % columnTiles * TILE;
        uint32_t nr = rows - row < RowTile ? rows - row : RowTile, nc = cols - col < TILE ? cols - col : TILE;
        uint64_t offset = uint64_t(row) * packedCols + col, imag = uint64_t(packedRows) * packedCols;
        uint64_t targetOffset =
            rightLayout ? 2 * (end + col + uint64_t(first + row) * lda) : 2 * (first + row + uint64_t(end + col) * lda);
        Sync<HardEvent::V_MTE2>();
        Read(products, productGm, offset, nr, nc, packedCols, REAL_PITCH);
        Read(products[inputPlane], productGm, offset + imag, nr, nc, packedCols, REAL_PITCH);
        Read(original, target, targetOffset, rightLayout ? nr : nc, 2 * (rightLayout ? nc : nr), 2 * lda, complexPitch);
        Sync<HardEvent::MTE2_V>();
        if (rightLayout)
        {
            asc_vf_call<UpdateRight>(reinterpret_cast<__ubuf__ float*>(products.GetPhyAddr()),
                                     reinterpret_cast<__ubuf__ float*>(original.GetPhyAddr()),
                                     reinterpret_cast<__ubuf__ float*>(output.GetPhyAddr()), nr, nc);
        }
        else
        {
            asc_vf_call<Update<RowTile>>(reinterpret_cast<__ubuf__ float*>(products.GetPhyAddr()),
                                         reinterpret_cast<__ubuf__ float*>(original.GetPhyAddr()),
                                         reinterpret_cast<__ubuf__ float*>(output.GetPhyAddr()), nr, nc);
        }
        Sync<HardEvent::V_MTE3>();
        Write(target, targetOffset, output, rightLayout ? nr : nc, 2 * (rightLayout ? nc : nr), 2 * lda, 2 * RowTile);
        Sync<HardEvent::MTE3_V>();
    }
}

template <uint32_t Tile>
__simd_vf__ inline void PackReflectorTile(__ubuf__ float* input, __ubuf__ float* output, uint32_t rowBase,
                                          uint32_t colBase, uint32_t rows, uint32_t cols, uint32_t firstRow)
{
    constexpr uint32_t TILE = Tile, COMPLEX_PITCH = 2 * Tile + 8;
    constexpr uint32_t REAL_PITCH = Tile + 8, INPUT_PLANE = Tile * REAL_PITCH;
    Reg::RegTensor<uint32_t> lane, offsets, bits, maskBits, columnIndex, rowIndex;
    Reg::RegTensor<float> real, imag, high, low, negative, one, zero;
    Reg::MaskReg diagonal, rowValid, columnValid, valid, lower;
    uint32_t all = TILE;
    auto full = Reg::UpdateMask<float>(all);
    Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lane), int32_t(0));
    Reg::Muls(offsets, lane, COMPLEX_PITCH, full);
    Reg::Duplicate(maskBits, 0xffffe000U);
    Reg::Duplicate(one, 1.0f);
    Reg::Duplicate(zero, 0.0f);
    Reg::Adds(columnIndex, lane, colBase, full);
    Reg::Compares<uint32_t, CMPMODE::LT>(columnValid, lane, cols, full);
#pragma unroll
    for (uint16_t index = 0; index < 8; ++index)
    {
        uint32_t row = firstRow + index;
        uint32_t globalRow = rowBase + row;
        Reg::Duplicate(rowIndex, row);
        Reg::Compares<uint32_t, CMPMODE::LT>(rowValid, rowIndex, rows, full);
        Reg::And(valid, rowValid, columnValid, full);
        Reg::Compares<uint32_t, CMPMODE::LT>(lower, columnIndex, globalRow, full);
        Reg::And(lower, lower, valid, full);
        Reg::Gather(real, input + 2 * row, offsets, lower);
        Reg::Gather(imag, input + 2 * row + 1, offsets, lower);
        Reg::Select(real, real, zero, lower);
        Reg::Select(imag, imag, zero, lower);
        Reg::Compares<uint32_t, CMPMODE::EQ>(diagonal, columnIndex, globalRow, full);
        Reg::And(diagonal, diagonal, valid, full);
        Reg::Select(real, one, real, diagonal);
        Reg::And(bits, reinterpret_cast<Reg::RegTensor<uint32_t>&>(real), maskBits, full);
        high = reinterpret_cast<Reg::RegTensor<float>&>(bits);
        Reg::Sub(low, real, high, full);
        Reg::StoreAlign(output + row * REAL_PITCH, high, full);
        Reg::StoreAlign(output + 2 * INPUT_PLANE + row * REAL_PITCH, low, full);
        Reg::And(bits, reinterpret_cast<Reg::RegTensor<uint32_t>&>(imag), maskBits, full);
        high = reinterpret_cast<Reg::RegTensor<float>&>(bits);
        Reg::Sub(low, imag, high, full);
        Reg::StoreAlign(output + INPUT_PLANE + row * REAL_PITCH, high, full);
        Reg::StoreAlign(output + 3 * INPUT_PLANE + row * REAL_PITCH, low, full);
        Reg::Muls(negative, high, -1.0f, full);
        Reg::StoreAlign(output + 4 * INPUT_PLANE + row * REAL_PITCH, negative, full);
        Reg::Muls(negative, low, -1.0f, full);
        Reg::StoreAlign(output + 5 * INPUT_PLANE + row * REAL_PITCH, negative, full);
    }
}

template <uint32_t Tile, uint32_t Columns>
__simd_vf__ inline void TransposeReflectorTile(__ubuf__ float* input, __ubuf__ float* output, uint32_t firstColumn)
{
    constexpr uint32_t TILE = Tile, REAL_PITCH = Tile + 8;
    constexpr uint32_t INPUT_PLANE = Tile * REAL_PITCH, PLANE = Columns * Tile;
    Reg::RegTensor<uint32_t> lane, offsets;
    Reg::RegTensor<float> value;
    uint32_t all = TILE;
    auto mask = Reg::UpdateMask<float>(all);
    Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lane), int32_t(0));
    Reg::Muls(offsets, lane, REAL_PITCH, mask);
#pragma unroll
    for (uint16_t plane = 0; plane < 6; ++plane)
    {
#pragma unroll
        for (uint16_t column = 0; column < Columns; ++column)
        {
            Reg::Gather(value, input + plane * INPUT_PLANE + firstColumn + column, offsets, mask);
            Reg::StoreAlign(output + plane * PLANE + column * TILE, value, mask);
        }
    }
}

__aicore__ inline void StoreAdjointReflectors(LocalTensor<float> planar, LocalTensor<float> transposed,
                                              GlobalTensor<float> ah, GlobalTensor<float> al, uint32_t row,
                                              uint32_t col, uint32_t rows, uint32_t panel)
{
    constexpr uint32_t TILE = 64, TRANSPOSE_COLUMNS = 8, PLANE = TRANSPOSE_COLUMNS * TILE;
    uint64_t top, bottom;
    for (uint32_t firstColumn = 0; firstColumn < TILE; firstColumn += TRANSPOSE_COLUMNS)
    {
        asc_vf_call<TransposeReflectorTile<TILE, TRANSPOSE_COLUMNS>>(
            reinterpret_cast<__ubuf__ float*>(planar.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(transposed.GetPhyAddr()), firstColumn);
        Sync<HardEvent::V_MTE3>();
        top = uint64_t(col + firstColumn) * 2 * rows + row;
        bottom = uint64_t(col + firstColumn + panel) * 2 * rows + row;
        Write(ah, top, transposed, TRANSPOSE_COLUMNS, TILE, 2 * rows, TILE);
        Write(ah, top + rows, transposed[PLANE], TRANSPOSE_COLUMNS, TILE, 2 * rows, TILE);
        Write(ah, bottom, transposed[4 * PLANE], TRANSPOSE_COLUMNS, TILE, 2 * rows, TILE);
        Write(ah, bottom + rows, transposed, TRANSPOSE_COLUMNS, TILE, 2 * rows, TILE);
        Write(al, top, transposed[2 * PLANE], TRANSPOSE_COLUMNS, TILE, 2 * rows, TILE);
        Write(al, top + rows, transposed[3 * PLANE], TRANSPOSE_COLUMNS, TILE, 2 * rows, TILE);
        Write(al, bottom, transposed[5 * PLANE], TRANSPOSE_COLUMNS, TILE, 2 * rows, TILE);
        Write(al, bottom + rows, transposed[2 * PLANE], TRANSPOSE_COLUMNS, TILE, 2 * rows, TILE);
        Sync<HardEvent::MTE3_V>();
    }
}

__aicore__ inline void ReflectorsDma(__gm__ const float* a, __gm__ float* adjHigh, __gm__ float* adjLow,
                                     __gm__ float* vHigh, __gm__ float* vLow, uint64_t lda, uint32_t first,
                                     uint32_t rows, uint32_t panel, uint32_t actualRows, uint32_t actualPanel)
{
    constexpr uint32_t TILE = 64, COMPLEX_PITCH = 2 * TILE + 8, REAL_PITCH = TILE + 8;
    constexpr uint32_t INPUT_PLANE = TILE * REAL_PITCH;
    constexpr uint32_t TRANSPOSE_COLUMNS = 8, PLANE = TRANSPOSE_COLUMNS * TILE;
    static_assert(6 * PLANE <= TILE * COMPLEX_PITCH, "transpose scratch must fit the consumed input buffer");
    TPipe pipe;
    TBuf<QuePosition::VECCALC> inputBuffer, planarBuffer;
    pipe.InitBuffer(inputBuffer, TILE * COMPLEX_PITCH * sizeof(float));
    pipe.InitBuffer(planarBuffer, 6 * INPUT_PLANE * sizeof(float));
    auto input = inputBuffer.Get<float>(), planar = planarBuffer.Get<float>();
    auto transposed = input;
    GlobalTensor<float> source, ah, al, vh, vl;
    source.SetGlobalBuffer(const_cast<__gm__ float*>(a));
    ah.SetGlobalBuffer(adjHigh);
    al.SetGlobalBuffer(adjLow);
    vh.SetGlobalBuffer(vHigh);
    vl.SetGlobalBuffer(vLow);
    const uint32_t columnTiles = panel / TILE;
    for (uint64_t tile = GetBlockIdx(); tile < uint64_t(rows / TILE) * columnTiles; tile += GetBlockNum())
    {
        uint32_t row = tile / columnTiles * TILE, col = tile % columnTiles * TILE;
        uint32_t nr = row >= actualRows ? 0 : actualRows - row < TILE ? actualRows - row : TILE;
        uint32_t nc = col >= actualPanel ? 0 : actualPanel - col < TILE ? actualPanel - col : TILE;
        Sync<HardEvent::V_MTE2>();
        if (nr && nc)
        {
            Read(input, source, 2 * (first + row + uint64_t(first + col) * lda), nc, 2 * nr, 2 * lda, COMPLEX_PITCH);
        }
        Sync<HardEvent::MTE2_V>();
        for (uint32_t firstRow = 0; firstRow < TILE; firstRow += 8)
        {
            asc_vf_call<PackReflectorTile<TILE>>(reinterpret_cast<__ubuf__ float*>(input.GetPhyAddr()),
                                                 reinterpret_cast<__ubuf__ float*>(planar.GetPhyAddr()), row, col, nr,
                                                 nc, firstRow);
            PipeBarrier<PIPE_V>();
        }
        Sync<HardEvent::V_MTE3>();
        uint64_t top = uint64_t(row) * 2 * panel + col, bottom = uint64_t(row + rows) * 2 * panel + col;
        Write(vh, top, planar, TILE, TILE, 2 * panel, REAL_PITCH);
        Write(vh, top + panel, planar[4 * INPUT_PLANE], TILE, TILE, 2 * panel, REAL_PITCH);
        Write(vh, bottom, planar[INPUT_PLANE], TILE, TILE, 2 * panel, REAL_PITCH);
        Write(vh, bottom + panel, planar, TILE, TILE, 2 * panel, REAL_PITCH);
        Write(vl, top, planar[2 * INPUT_PLANE], TILE, TILE, 2 * panel, REAL_PITCH);
        Write(vl, top + panel, planar[5 * INPUT_PLANE], TILE, TILE, 2 * panel, REAL_PITCH);
        Write(vl, bottom, planar[3 * INPUT_PLANE], TILE, TILE, 2 * panel, REAL_PITCH);
        Write(vl, bottom + panel, planar[2 * INPUT_PLANE], TILE, TILE, 2 * panel, REAL_PITCH);
        StoreAdjointReflectors(planar, transposed, ah, al, row, col, rows, panel);
    }
}
}  // namespace ComplexQrTransfer
