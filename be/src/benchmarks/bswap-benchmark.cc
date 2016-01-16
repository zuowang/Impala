// Copyright 2015 Cloudera Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <stdlib.h>
#include <immintrin.h>
#include <iostream>

#include "exec/parquet-common.h"
#include "runtime/decimal-value.h"
#include "util/benchmark.h"
#include "util/cpu-info.h"
#include "util/bit-util.h"

#include "common/names.h"

using std::numeric_limits;
using namespace impala;

// Machine Info: Intel(R) Xeon(R) CPU E5-2695 v3 @ 2.30GHz
// Decimal4Value encode: Function     Rate (iters/ms)          Comparison
// ----------------------------------------------------------------------
//                         Impala               724.3                  1X
//                          Bswap                1649              2.277X
//                            SSE                6949              9.593X
//                           AVX2           1.337e+04              18.46X
//
//                         Impala               719.7                  1X
//                          Bswap                1626               2.26X
//                            SSE                6897              9.583X
//                           AVX2           1.328e+04              18.45X
//
//                         Impala                 717                  1X
//                          Bswap                1624              2.265X
//                            SSE                6944              9.684X
//                           AVX2           1.336e+04              18.64X
//
//                         Impala               716.2                  1X
//                          Bswap                1639              2.289X
//                            SSE                7039              9.828X
//                           AVX2           1.355e+04              18.92X
// Decimal8Value encode: Function     Rate (iters/ms)          Comparison
// ----------------------------------------------------------------------
//                         Impala               653.8                  1X
//                          Bswap                1363              2.084X
//                            SSE                3717              5.686X
//                           AVX2                7079              10.83X
//
//                         Impala               656.4                  1X
//                          Bswap                1360              2.072X
//                            SSE                3687              5.617X
//                           AVX2                7032              10.71X
//
//                         Impala               648.5                  1X
//                          Bswap                1361              2.099X
//                            SSE                3758              5.795X
//                           AVX2                7089              10.93X
//
//                         Impala               653.2                  1X
//                          Bswap                1372                2.1X
//                            SSE                3650              5.588X
//                           AVX2                7049              10.79X
// Decimal16Value encode:Function     Rate (iters/ms)          Comparison
// ----------------------------------------------------------------------
//                         Impala               92.83                  1X
//                          Bswap               768.6               8.28X
//                            SSE                1894               20.4X
//                           AVX2                3549              38.23X
//
//                         Impala               93.05                  1X
//                          Bswap               735.3              7.903X
//                            SSE                1883              20.24X
//                           AVX2                3523              37.86X
//
//                         Impala               92.08                  1X
//                          Bswap               760.9              8.264X
//                            SSE                1859              20.19X
//                           AVX2                3528              38.32X
//
//                         Impala               91.58                  1X
//                          Bswap               758.8              8.286X
//                            SSE                1869              20.41X
//                           AVX2                3535               38.6X
//
//                         Impala               91.84                  1X
//                          Bswap               768.3              8.366X
//                            SSE                1882              20.49X
//                           AVX2                3493              38.03X
//
//                         Impala               93.38                  1X
//                          Bswap               772.5              8.273X
//                            SSE                1893              20.27X
//                           AVX2                3532              37.83X
//
//                         Impala               91.75                  1X
//                          Bswap               760.3              8.286X
//                            SSE                1863              20.31X
//                           AVX2                3512              38.27X
//
//                         Impala               92.08                  1X
//                          Bswap               768.3              8.343X
//                            SSE                1891              20.54X
//                           AVX2                3523              38.26X

struct TestData {
  int32_t num_values;
  int fixed_len_size;
  uint8_t* buffer;
  vector<Decimal4Value> d4_values;
  vector<Decimal8Value> d8_values;
  vector<Decimal16Value> d16_values;
};

template <int num_bytes, int fixed_len_size>
__attribute__((target("avx2")))
inline void ByteSwapAVX2(void* dest, const void* source, int len) {
  uint8_t* dst = reinterpret_cast<uint8_t*>(dest);
  __m256i mask;
  __m256i idx;
  int j;
  if (num_bytes == 4) {
    const Decimal4Value* src = reinterpret_cast<const Decimal4Value*>(source);
    switch (fixed_len_size) {
      case 1:
        mask = _mm256_set_epi8(0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
            0x80, 0x80, 0x80, 0x80, 12, 8, 4, 0,
            0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
            0x80, 0x80, 0x80, 0x80, 12, 8, 4, 0);
        idx = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 4, 0);
        j = 0;
        for (; j <= len - 8; j += 8) {
          _mm256_storeu_si256((__m256i *)(dst),
              _mm256_permutevar8x32_epi32(
                  _mm256_shuffle_epi8(
                      _mm256_loadu_si256((__m256i *)(src + j)), mask), idx));
          dst += 8;
        }
        for (; j < len; ++j) {
          BitUtil::ByteSwap(dst, src + j, fixed_len_size);
          ++dst;
        }
        return;
      case 2:
        mask = _mm256_set_epi8(
            0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 12, 13, 8, 9, 4, 5, 0, 1,
            0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 12, 13, 8, 9, 4, 5, 0, 1);
        idx = _mm256_set_epi32(0, 0, 0, 0, 5, 4, 1, 0);
        j = 0;
        for (; j <= len - 8; j += 8) {
          _mm256_storeu_si256((__m256i *)(dst),
              _mm256_permutevar8x32_epi32(
                  _mm256_shuffle_epi8(
                      _mm256_loadu_si256((__m256i *)(src + j)), mask), idx));
          dst += 16;
        }
        for (; j < len; ++j) {
          BitUtil::ByteSwap(dst, src + j, fixed_len_size);
          dst += 2;
        }
        return;
      case 3:
        mask = _mm256_set_epi8(
            0x80, 0x80, 0x80, 0x80, 12, 13, 14, 8, 9, 10, 4, 5, 6, 0, 1, 2,
            0x80, 0x80, 0x80, 0x80, 12, 13, 14, 8, 9, 10, 4, 5, 6, 0, 1, 2);
        idx = _mm256_set_epi32(0, 0, 6, 5, 4, 2, 1, 0);
        j = 0;
        for (; j <= len - 8; j += 8) {
          _mm256_storeu_si256((__m256i *)(dst),
              _mm256_permutevar8x32_epi32(
                  _mm256_shuffle_epi8(
                      _mm256_loadu_si256((__m256i *)(src + j)), mask), idx));
          dst += 24;
        }
        for (; j < len; ++j) {
          BitUtil::ByteSwap(dst, src + j, fixed_len_size);
          dst += 3;
        }
        return;
      case 4:
        mask = _mm256_set_epi8(12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3,
            12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);
        j = 0;
        for (; j <= len - 8; j += 8) {
          _mm256_storeu_si256((__m256i *)(dst),
              _mm256_shuffle_epi8(_mm256_loadu_si256((__m256i *)(src + j)), mask));
          dst += 32;
        }
        for (; j < len; ++j) {
          BitUtil::ByteSwap(dst, src + j, fixed_len_size);
          dst += 4;
        }
        return;
    }
  } else if (num_bytes == 8) {
    const Decimal8Value* src = reinterpret_cast<const Decimal8Value*>(source);
    switch (fixed_len_size) {
      case 5:
        mask = _mm256_set_epi8(
            0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 3, 4, 8, 9, 10, 11, 12, 0, 1, 2,
            0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4);
        idx = _mm256_set_epi32(0, 0, 6, 5, 4, 2, 1, 0);
        j = 0;
        for (; j <= len - 4; j += 4) {
          _mm256_storeu_si256((__m256i *)(dst),
              _mm256_permutevar8x32_epi32(
                  _mm256_shuffle_epi8(
                      _mm256_loadu_si256((__m256i *)(src + j)), mask), idx));
          *reinterpret_cast<int16_t*>(dst + 10) = *reinterpret_cast<int16_t*>(dst + 20);
          dst += 20;
        }
        for (; j < len; ++j) {
          BitUtil::ByteSwap(dst, src + j, fixed_len_size);
          dst += 5;
        }
        return;
      case 6:
        mask = _mm256_set_epi8(
            0x80, 0x80, 0x80, 0x80, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5,
            0x80, 0x80, 0x80, 0x80, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5);
        idx = _mm256_set_epi32(0, 0, 6, 5, 4, 2, 1, 0);
        j = 0;
        for (; j <= len - 4; j += 4) {
          _mm256_storeu_si256((__m256i *)(dst),
              _mm256_permutevar8x32_epi32(
                  _mm256_shuffle_epi8(
                      _mm256_loadu_si256((__m256i *)(src + j)), mask), idx));
          dst += 24;
        }
        for (; j < len; ++j) {
          BitUtil::ByteSwap(dst, src + j, fixed_len_size);
          dst += 6;
        }
        return;
      case 7:
        mask = _mm256_set_epi8(
            0x80, 0x80, 5, 6, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4,
            0x80, 0x80, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6);
        idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        j = 0;
        for (; j <= len - 4; j += 4) {
          _mm256_storeu_si256((__m256i *)(dst),
              _mm256_permutevar8x32_epi32(
                  _mm256_shuffle_epi8(
                      _mm256_loadu_si256((__m256i *)(src + j)), mask), idx));
          *reinterpret_cast<int16_t*>(dst + 14) = *reinterpret_cast<int16_t*>(dst + 28);
          dst += 28;
        }
        for (; j < len; ++j) {
          BitUtil::ByteSwap(dst, src + j, fixed_len_size);
          dst += 7;
        }
        return;
      case 8:
        mask = _mm256_set_epi8(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);
        j = 0;
        for (; j <= len - 4; j += 4) {
          _mm256_storeu_si256((__m256i *)(dst),
              _mm256_shuffle_epi8(_mm256_loadu_si256((__m256i *)(src + j)), mask));
          dst += 32;
        }
        for (; j < len; ++j) {
          BitUtil::ByteSwap(dst, src + j, fixed_len_size);
          dst += 8;
        }
        return;
    }
  } else if (num_bytes == 16) {
    const Decimal16Value* src = reinterpret_cast<const Decimal16Value*>(source);
    switch (fixed_len_size) {
      case 9:
        mask = _mm256_set_epi8(
            0x80, 0x80, 0x80, 0x80, 0x80, 6, 7, 8, 0x80, 0x80, 0, 1, 2, 3, 4, 5,
            0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0, 1, 2, 3, 4, 5, 6, 7, 8);
        idx = _mm256_set_epi32(0, 0, 6, 5, 4, 2, 1, 0);
        j = 0;
        for (; j <= len - 2; j += 2) {
          _mm256_storeu_si256((__m256i *)(dst),
              _mm256_permutevar8x32_epi32(
                  _mm256_shuffle_epi8(
                      _mm256_loadu_si256((__m256i *)(src + j)), mask), idx));
          *(dst + 9) = *(dst + 20);
          *reinterpret_cast<int16_t*>(dst + 10) = *reinterpret_cast<int16_t*>(dst + 21);
          dst += 18;
        }
        for (; j < len; ++j) {
          BitUtil::ByteSwap(dst, src + j, fixed_len_size);
          dst += 9;
        }
        return;
      case 10:
        mask = _mm256_set_epi8(
            0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7,
            0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        idx = _mm256_set_epi32(0, 0, 6, 5, 4, 2, 1, 0);
        j = 0;
        for (; j <= len - 2; j += 2) {
          _mm256_storeu_si256((__m256i *)(dst),
              _mm256_permutevar8x32_epi32(
                  _mm256_shuffle_epi8(
                      _mm256_loadu_si256((__m256i *)(src + j)), mask), idx));
          *reinterpret_cast<int16_t*>(dst + 10) = *reinterpret_cast<int16_t*>(dst + 20);
          dst += 20;
        }
        for (; j < len; ++j) {
          BitUtil::ByteSwap(dst, src + j, fixed_len_size);
          dst += 10;
        }
        return;
      case 11:
        mask = _mm256_set_epi8(
            0x80, 0x80, 0x80, 0x80, 0x80, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            0x80, 0x80, 0x80, 0x80, 0x80, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        idx = _mm256_set_epi32(0, 0, 6, 5, 4, 2, 1, 0);
        j = 0;
        for (; j <= len - 2; j += 2) {
          _mm256_storeu_si256((__m256i *)(dst),
              _mm256_permutevar8x32_epi32(
                  _mm256_shuffle_epi8(
                      _mm256_loadu_si256((__m256i *)(src + j)), mask), idx));
          *(dst + 11) = *reinterpret_cast<int16_t*>(dst + 22);
          dst += 22;
        }
        for (; j < len; ++j) {
          BitUtil::ByteSwap(dst, src + j, fixed_len_size);
          dst += 11;
        }
        return;
      case 12:
        mask = _mm256_set_epi8(
            0x80, 0x80, 0x80, 0x80, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            0x80, 0x80, 0x80, 0x80, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
        idx = _mm256_set_epi32(0, 0, 6, 5, 4, 2, 1, 0);
        j = 0;
        for (; j <= len - 2; j += 2) {
          _mm256_storeu_si256((__m256i *)(dst),
              _mm256_permutevar8x32_epi32(
                  _mm256_shuffle_epi8(
                      _mm256_loadu_si256((__m256i *)(src + j)), mask), idx));
          dst += 24;
        }
        for (; j < len; ++j) {
          BitUtil::ByteSwap(dst, src + j, fixed_len_size);
          dst += 12;
        }
        return;
      case 13:
        mask = _mm256_set_epi8(
            0x80, 0x80, 0x80, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            0x80, 0x80, 0x80, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
        j = 0;
        for (; j <= len - 2; j += 2) {
          _mm256_storeu_si256((__m256i *)(dst),
                  _mm256_shuffle_epi8(_mm256_loadu_si256((__m256i *)(src + j)), mask));
          *(dst + 13) = *(dst + 26);
          *reinterpret_cast<int16_t*>(dst + 14) = *reinterpret_cast<int16_t*>(dst + 27);
          dst += 26;
        }
        for (; j < len; ++j) {
          BitUtil::ByteSwap(dst, src + j, fixed_len_size);
          dst += 13;
        }
        return;
      case 14:
        mask = _mm256_set_epi8(
            0x80, 0x80, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            0x80, 0x80, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);
        j = 0;
        for (; j <= len - 2; j += 2) {
          _mm256_storeu_si256((__m256i *)(dst),
                  _mm256_shuffle_epi8(_mm256_loadu_si256((__m256i *)(src + j)), mask));
          *reinterpret_cast<int16_t*>(dst + 14) = *reinterpret_cast<int16_t*>(dst + 28);
          dst += 28;
        }
        for (; j < len; ++j) {
          BitUtil::ByteSwap(dst, src + j, fixed_len_size);
          dst += 14;
        }
        return;
      case 15:
        mask = _mm256_set_epi8(
            0x80, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
            0x80, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
        j = 0;
        for (; j <= len - 2; j += 2) {
          _mm256_storeu_si256((__m256i *)(dst),
                  _mm256_shuffle_epi8(_mm256_loadu_si256((__m256i *)(src + j)), mask));
          *(dst + 15) = *(dst + 30);
          dst += 30;
        }
        for (; j < len; ++j) {
          BitUtil::ByteSwap(dst, src + j, fixed_len_size);
          dst += 15;
        }
        return;
      case 16:
        mask = _mm256_set_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        j = 0;
        for (; j <= len - 2; j += 2) {
          _mm256_storeu_si256((__m256i *)(dst),
              _mm256_shuffle_epi8(_mm256_loadu_si256((__m256i *)(src + j)), mask));
          dst += 32;
        }
        return;
    }
  }
}

template <int num_bytes>
__attribute__((target("avx2")))
void TestAVX2(int batch_size, void* d) {
  TestData* data = reinterpret_cast<TestData*>(d);
  for (int i = 0; i < batch_size; ++i) {
    if (num_bytes == 4) {
      switch (data->fixed_len_size) {
        case 1:
          ByteSwapAVX2<4, 1>(data->buffer, &data->d4_values[0], data->num_values);
          break;
        case 2:
          ByteSwapAVX2<4, 2>(data->buffer, &data->d4_values[0], data->num_values);
          break;
        case 3:
          ByteSwapAVX2<4, 3>(data->buffer, &data->d4_values[0], data->num_values);
          break;
        case 4:
          ByteSwapAVX2<4, 4>(data->buffer, &data->d4_values[0], data->num_values);
          break;
      }
    } else if (num_bytes == 8) {
      switch (data->fixed_len_size) {
        case 5:
          ByteSwapAVX2<8, 5>(data->buffer, &data->d8_values[0], data->num_values);
          break;
        case 6:
          ByteSwapAVX2<8, 6>(data->buffer, &data->d8_values[0], data->num_values);
          break;
        case 7:
          ByteSwapAVX2<8, 7>(data->buffer, &data->d8_values[0], data->num_values);
          break;
        case 8:
          ByteSwapAVX2<8, 8>(data->buffer, &data->d8_values[0], data->num_values);
          break;
      }
    } else if (num_bytes == 16) {
      switch (data->fixed_len_size) {
        case 9:
          ByteSwapAVX2<16, 9>(data->buffer, &data->d16_values[0], data->num_values);
          break;
        case 10:
          ByteSwapAVX2<16, 10>(data->buffer, &data->d16_values[0], data->num_values);
          break;
        case 11:
          ByteSwapAVX2<16, 11>(data->buffer, &data->d16_values[0], data->num_values);
          break;
        case 12:
          ByteSwapAVX2<16, 12>(data->buffer, &data->d16_values[0], data->num_values);
          break;
        case 13:
          ByteSwapAVX2<16, 13>(data->buffer, &data->d16_values[0], data->num_values);
          break;
        case 14:
          ByteSwapAVX2<16, 14>(data->buffer, &data->d16_values[0], data->num_values);
          break;
        case 15:
          ByteSwapAVX2<16, 15>(data->buffer, &data->d16_values[0], data->num_values);
          break;
        case 16:
          ByteSwapAVX2<16, 16>(data->buffer, &data->d16_values[0], data->num_values);
          break;
      }
    }
  }
}

template <int num_bytes, int fixed_len_size>
__attribute__((target("sse4.2")))
inline void ByteSwapSSE(void* dest, const void* source, int len) {
  uint8_t* dst = reinterpret_cast<uint8_t*>(dest);
  __m128i mask;
  int j;
  if (num_bytes == 4) {
    const Decimal4Value* src = reinterpret_cast<const Decimal4Value*>(source);
    switch (fixed_len_size) {
      case 1:
        mask = _mm_set_epi8(0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
            0x80, 0x80, 0x80, 0x80, 0x80, 12, 8, 4, 0);
        j = 0;
        for (; j <= len - 4; j += 4) {
          _mm_storeu_si128((__m128i *)(dst),
              _mm_shuffle_epi8(_mm_loadu_si128((__m128i *)(src + j)), mask));
          dst += 4;
        }
        for (; j < len; ++j) {
          BitUtil::ByteSwap(dst, src + j, fixed_len_size);
          ++dst;
        }
        return;
      case 2:
        mask = _mm_set_epi8(0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
            12, 13, 8, 9, 4, 5, 0, 1);
        j = 0;
        for (; j <= len - 4; j += 4) {
          _mm_storeu_si128((__m128i *)(dst),
              _mm_shuffle_epi8(_mm_loadu_si128((__m128i *)(src + j)), mask));
          dst += 8;
        }
        for (; j < len; ++j) {
          BitUtil::ByteSwap(dst, src + j, fixed_len_size);
          dst += 2;
        }
        return;
      case 3:
        mask = _mm_set_epi8(0x80, 0x80, 0x80, 0x80, 12, 13, 14, 8,
            9, 10, 4, 5, 6, 0, 1, 2);
        j = 0;
        for (; j <= len - 4; j += 4) {
          _mm_storeu_si128((__m128i *)(dst),
              _mm_shuffle_epi8(_mm_loadu_si128((__m128i *)(src + j)), mask));
          dst += 12;
        }
        for (; j < len; ++j) {
          BitUtil::ByteSwap(dst, src + j, fixed_len_size);
          dst += 3;
        }
        return;
      case 4:
        mask = _mm_set_epi8(12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);
        j = 0;
        for (; j <= len - 4; j += 4) {
          _mm_storeu_si128((__m128i *)(dst),
              _mm_shuffle_epi8(_mm_loadu_si128((__m128i *)(src + j)), mask));
          dst += 16;
        }
        for (; j < len; ++j) {
          BitUtil::ByteSwap(dst, src + j, fixed_len_size);
          dst += 4;
        }
        return;
    }
  } else if (num_bytes == 8) {
    const Decimal8Value* src = reinterpret_cast<const Decimal8Value*>(source);
    switch (fixed_len_size) {
      case 5:
        mask = _mm_set_epi8(0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 8, 9,
            10, 11, 12, 0, 1, 2, 3, 4);
        j = 0;
        for (; j <= len - 2; j += 2) {
          _mm_storeu_si128((__m128i *)(dst),
              _mm_shuffle_epi8(_mm_loadu_si128((__m128i *)(src + j)), mask));
          dst += 10;
        }
        for (; j < len; ++j) {
          BitUtil::ByteSwap(dst, src + j, fixed_len_size);
          dst += 5;
        }
        return;
      case 6:
        mask = _mm_set_epi8(0x80, 0x80, 0x80, 0x80, 8, 9, 10, 11,
            12, 13, 0, 1, 2, 3, 4, 5);
        j = 0;
        for (; j <= len - 2; j += 2) {
          _mm_storeu_si128((__m128i *)(dst),
              _mm_shuffle_epi8(_mm_loadu_si128((__m128i *)(src + j)), mask));
          dst += 12;
        }
        for (; j < len; ++j) {
          BitUtil::ByteSwap(dst, src + j, fixed_len_size);
          dst += 6;
        }
        return;
      case 7:
        mask = _mm_set_epi8(0x80, 0x80, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6);
        j = 0;
        for (; j <= len - 2; j += 2) {
          _mm_storeu_si128((__m128i *)(dst),
              _mm_shuffle_epi8(_mm_loadu_si128((__m128i *)(src + j)), mask));
          dst += 14;
        }
        for (; j < len; ++j) {
          BitUtil::ByteSwap(dst, src + j, fixed_len_size);
          dst += 7;
        }
        return;
      case 8:
        mask = _mm_set_epi8(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);
        j = 0;
        for (; j <= len - 2; j += 2) {
          _mm_storeu_si128((__m128i *)(dst),
              _mm_shuffle_epi8(_mm_loadu_si128((__m128i *)(src + j)), mask));
          dst += 16;
        }
        for (; j < len; ++j) {
          BitUtil::ByteSwap(dst, src + j, fixed_len_size);
          dst += 8;
        }
        return;
    }
  } else if (num_bytes == 16) {
    const Decimal16Value* src = reinterpret_cast<const Decimal16Value*>(source);
    switch (fixed_len_size) {
      case 9:
        mask = _mm_set_epi8(0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0,
            1, 2, 3, 4, 5, 6, 7, 8);
        j = 0;
        for (; j < len; ++j) {
          _mm_storeu_si128((__m128i *)(dst),
              _mm_shuffle_epi8(_mm_loadu_si128((__m128i *)(src + j)), mask));
          dst += 9;
        }
        return;
      case 10:
        mask = _mm_set_epi8(0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0, 1,
            2, 3, 4, 5, 6, 7, 8, 9);
        j = 0;
        for (; j < len; ++j) {
          _mm_storeu_si128((__m128i *)(dst),
              _mm_shuffle_epi8(_mm_loadu_si128((__m128i *)(src + j)), mask));
          dst += 10;
        }
        return;
      case 11:
        mask = _mm_set_epi8(0x80, 0x80, 0x80, 0x80, 0x80, 0, 1, 2,
            3, 4, 5, 6, 7, 8, 9, 10);
        j = 0;
        for (; j < len; ++j) {
          _mm_storeu_si128((__m128i *)(dst),
              _mm_shuffle_epi8(_mm_loadu_si128((__m128i *)(src + j)), mask));
          dst += 11;
        }
        return;
      case 12:
        mask = _mm_set_epi8(0x80, 0x80, 0x80, 0x80, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
        j = 0;
        for (; j < len; ++j) {
          _mm_storeu_si128((__m128i *)(dst),
              _mm_shuffle_epi8(_mm_loadu_si128((__m128i *)(src + j)), mask));
          dst += 12;
        }
        return;
      case 13:
        mask = _mm_set_epi8(0x80, 0x80, 0x80, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
        j = 0;
        for (; j < len; ++j) {
          _mm_storeu_si128((__m128i *)(dst),
              _mm_shuffle_epi8(_mm_loadu_si128((__m128i *)(src + j)), mask));
          dst += 13;
        }
        return;
      case 14:
        mask = _mm_set_epi8(0x80, 0x80, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);
        j = 0;
        for (; j < len; ++j) {
          _mm_storeu_si128((__m128i *)(dst),
              _mm_shuffle_epi8(_mm_loadu_si128((__m128i *)(src + j)), mask));
          dst += 14;
        }
        return;
      case 15:
        mask = _mm_set_epi8(0x80, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
        j = 0;
        for (; j < len; ++j) {
          _mm_storeu_si128((__m128i *)(dst),
              _mm_shuffle_epi8(_mm_loadu_si128((__m128i *)(src + j)), mask));
          dst += 15;
        }
        return;
      case 16:
        mask = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        j = 0;
        for (; j < len; ++j) {
          _mm_storeu_si128((__m128i *)(dst),
              _mm_shuffle_epi8(_mm_loadu_si128((__m128i *)(src + j)), mask));
          dst += 16;
        }
        return;
    }
  }
}

template <int num_bytes>
__attribute__((target("sse4.2")))
void TestSSE(int batch_size, void* d) {
  TestData* data = reinterpret_cast<TestData*>(d);
  for (int i = 0; i < batch_size; ++i) {
    if (num_bytes == 4) {
      switch (data->fixed_len_size) {
        case 1:
          ByteSwapSSE<4, 1>(data->buffer, &data->d4_values[0], data->num_values);
          break;
        case 2:
          ByteSwapSSE<4, 2>(data->buffer, &data->d4_values[0], data->num_values);
          break;
        case 3:
          ByteSwapSSE<4, 3>(data->buffer, &data->d4_values[0], data->num_values);
          break;
        case 4:
          ByteSwapSSE<4, 4>(data->buffer, &data->d4_values[0], data->num_values);
          break;
      }
    } else if (num_bytes == 8) {
      switch (data->fixed_len_size) {
        case 5:
          ByteSwapSSE<8, 5>(data->buffer, &data->d8_values[0], data->num_values);
          break;
        case 6:
          ByteSwapSSE<8, 6>(data->buffer, &data->d8_values[0], data->num_values);
          break;
        case 7:
          ByteSwapSSE<8, 7>(data->buffer, &data->d8_values[0], data->num_values);
          break;
        case 8:
          ByteSwapSSE<8, 8>(data->buffer, &data->d8_values[0], data->num_values);
          break;
      }
    } else if (num_bytes == 16) {
      switch (data->fixed_len_size) {
        case 9:
          ByteSwapSSE<16, 9>(data->buffer, &data->d16_values[0], data->num_values);
          break;
        case 10:
          ByteSwapSSE<16, 10>(data->buffer, &data->d16_values[0], data->num_values);
          break;
        case 11:
          ByteSwapSSE<16, 11>(data->buffer, &data->d16_values[0], data->num_values);
          break;
        case 12:
          ByteSwapSSE<16, 12>(data->buffer, &data->d16_values[0], data->num_values);
          break;
        case 13:
          ByteSwapSSE<16, 13>(data->buffer, &data->d16_values[0], data->num_values);
          break;
        case 14:
          ByteSwapSSE<16, 14>(data->buffer, &data->d16_values[0], data->num_values);
          break;
        case 15:
          ByteSwapSSE<16, 15>(data->buffer, &data->d16_values[0], data->num_values);
          break;
        case 16:
          ByteSwapSSE<16, 16>(data->buffer, &data->d16_values[0], data->num_values);
          break;
      }
    }
  }
}

template <int fixed_len_size>
inline void ByteSwap(void* dest, const void* source) {
  uint8_t* dst = reinterpret_cast<uint8_t*>(dest);
  const uint8_t* src = reinterpret_cast<const uint8_t*>(source);
  switch (fixed_len_size) {
    case 1:
      *reinterpret_cast<int8_t*>(dst) = *reinterpret_cast<const int8_t*>(src);
      return;
    case 2:
      *reinterpret_cast<int16_t*>(dst) =
          BitUtil::ByteSwap(*reinterpret_cast<const int16_t*>(src));
      return;
    case 3:
      *reinterpret_cast<int16_t*>(dst + 1) =
          BitUtil::ByteSwap(*reinterpret_cast<const int16_t*>(src));
      *reinterpret_cast<int8_t*>(dst) = *reinterpret_cast<const int8_t*>(src + 2);
      return;
    case 4:
      *reinterpret_cast<int32_t*>(dst) =
          BitUtil::ByteSwap(*reinterpret_cast<const int32_t*>(src));
      return;
    case 5:
      *reinterpret_cast<int32_t*>(dst + 1) =
          BitUtil::ByteSwap(*reinterpret_cast<const int32_t*>(src));
      *reinterpret_cast<int8_t*>(dst) = *reinterpret_cast<const int8_t*>(src + 4);
      return;
    case 6:
      *reinterpret_cast<int32_t*>(dst + 2) =
          BitUtil::ByteSwap(*reinterpret_cast<const int32_t*>(src));
      *reinterpret_cast<int16_t*>(dst) =
          BitUtil::ByteSwap(*reinterpret_cast<const int16_t*>(src + 4));
      return;
    case 7:
      *reinterpret_cast<int32_t*>(dst + 3) =
          BitUtil::ByteSwap(*reinterpret_cast<const int32_t*>(src));
      *reinterpret_cast<int16_t*>(dst + 1) =
          BitUtil::ByteSwap(*reinterpret_cast<const int16_t*>(src + 4));
      *reinterpret_cast<int8_t*>(dst) = *reinterpret_cast<const int8_t*>(src + 6);
      return;
    case 8:
      *reinterpret_cast<int64_t*>(dst) =
          BitUtil::ByteSwap(*reinterpret_cast<const int64_t*>(src));
      return;
    case 9:
      *reinterpret_cast<int64_t*>(dst + 1) =
          BitUtil::ByteSwap(*reinterpret_cast<const int64_t*>(src));
      *reinterpret_cast<int8_t*>(dst) = *reinterpret_cast<const int8_t*>(src + 8);
      return;
    case 10:
      *reinterpret_cast<int64_t*>(dst + 2) =
          BitUtil::ByteSwap(*reinterpret_cast<const int64_t*>(src));
      *reinterpret_cast<int16_t*>(dst) =
          BitUtil::ByteSwap(*reinterpret_cast<const int16_t*>(src + 8));
      return;
    case 11:
      *reinterpret_cast<int64_t*>(dst + 3) =
          BitUtil::ByteSwap(*reinterpret_cast<const int64_t*>(src));
      *reinterpret_cast<int16_t*>(dst + 1) =
          BitUtil::ByteSwap(*reinterpret_cast<const int16_t*>(src + 8));
      *reinterpret_cast<int8_t*>(dst) = *reinterpret_cast<const int8_t*>(src + 10);
      return;
    case 12:
      *reinterpret_cast<int64_t*>(dst + 4) =
          BitUtil::ByteSwap(*reinterpret_cast<const int64_t*>(src));
      *reinterpret_cast<int32_t*>(dst) =
          BitUtil::ByteSwap(*reinterpret_cast<const int32_t*>(src + 8));
      return;
    case 13:
      *reinterpret_cast<int64_t*>(dst + 5) =
          BitUtil::ByteSwap(*reinterpret_cast<const int64_t*>(src));
      *reinterpret_cast<int32_t*>(dst + 1) =
          BitUtil::ByteSwap(*reinterpret_cast<const int32_t*>(src + 8));
      *reinterpret_cast<int8_t*>(dst) = *reinterpret_cast<const int8_t*>(src + 12);
      return;
    case 14:
      *reinterpret_cast<int64_t*>(dst + 6) =
          BitUtil::ByteSwap(*reinterpret_cast<const int64_t*>(src));
      *reinterpret_cast<int32_t*>(dst + 2) =
          BitUtil::ByteSwap(*reinterpret_cast<const int32_t*>(src + 8));
      *reinterpret_cast<int16_t*>(dst) =
          BitUtil::ByteSwap(*reinterpret_cast<const int16_t*>(src + 12));
      return;
    case 15:
      *reinterpret_cast<int64_t*>(dst + 7) =
          BitUtil::ByteSwap(*reinterpret_cast<const int64_t*>(src));
      *reinterpret_cast<int32_t*>(dst + 3) =
          BitUtil::ByteSwap(*reinterpret_cast<const int32_t*>(src + 8));
      *reinterpret_cast<int16_t*>(dst + 1) =
          BitUtil::ByteSwap(*reinterpret_cast<const int16_t*>(src + 12));
      *reinterpret_cast<int8_t*>(dst) = *reinterpret_cast<const int8_t*>(src + 14);
      return;
    case 16:
      *reinterpret_cast<int64_t*>(dst + 8) =
          BitUtil::ByteSwap(*reinterpret_cast<const int64_t*>(src));
      *reinterpret_cast<int64_t*>(dst) =
          BitUtil::ByteSwap(*reinterpret_cast<const int64_t*>(src + 8));
      return;
    default: break;
  }
}

template<int num_bytes>
void TestBswap(int batch_size, void* d) {
  TestData* data = reinterpret_cast<TestData*>(d);
  for (int i = 0; i < batch_size; ++i) {
    uint8_t* buffer = data->buffer;
    if (num_bytes == 4) {
      switch(data->fixed_len_size) {
        case 1:
          for (int j = 0; j < data->num_values; ++j) {
            ByteSwap<1>(buffer, &data->d4_values[j]);
          }
        case 2:
          for (int j = 0; j < data->num_values; ++j) {
            ByteSwap<2>(buffer, &data->d4_values[j]);
          }
        case 3:
          for (int j = 0; j < data->num_values; ++j) {
            ByteSwap<3>(buffer, &data->d4_values[j]);
          }
        case 4:
          for (int j = 0; j < data->num_values; ++j) {
            ByteSwap<4>(buffer, &data->d4_values[j]);
          }
        }
    } else if (num_bytes == 8) {
      switch(data->fixed_len_size) {
        case 5:
          for (int j = 0; j < data->num_values; ++j) {
            ByteSwap<5>(buffer, &data->d4_values[j]);
          }
        case 6:
          for (int j = 0; j < data->num_values; ++j) {
            ByteSwap<6>(buffer, &data->d4_values[j]);
          }
        case 7:
          for (int j = 0; j < data->num_values; ++j) {
            ByteSwap<7>(buffer, &data->d4_values[j]);
          }
        case 8:
          for (int j = 0; j < data->num_values; ++j) {
            ByteSwap<8>(buffer, &data->d4_values[j]);
          }
      }
    } else if (num_bytes == 16) {
      switch(data->fixed_len_size) {
        case 9:
          for (int j = 0; j < data->num_values; ++j) {
            ByteSwap<9>(buffer, &data->d4_values[j]);
          }
        case 10:
          for (int j = 0; j < data->num_values; ++j) {
            ByteSwap<10>(buffer, &data->d4_values[j]);
          }
        case 11:
          for (int j = 0; j < data->num_values; ++j) {
            ByteSwap<11>(buffer, &data->d4_values[j]);
          }
        case 12:
          for (int j = 0; j < data->num_values; ++j) {
            ByteSwap<12>(buffer, &data->d4_values[j]);
          }
        case 13:
          for (int j = 0; j < data->num_values; ++j) {
            ByteSwap<13>(buffer, &data->d4_values[j]);
          }
        case 14:
          for (int j = 0; j < data->num_values; ++j) {
            ByteSwap<14>(buffer, &data->d4_values[j]);
          }
        case 15:
          for (int j = 0; j < data->num_values; ++j) {
            ByteSwap<15>(buffer, &data->d4_values[j]);
          }
        case 16:
          for (int j = 0; j < data->num_values; ++j) {
            ByteSwap<16>(buffer, &data->d4_values[j]);
          }
      }
    }
  }
}

template<int num_bytes>
void TestImpala(int batch_size, void* d) {
  TestData* data = reinterpret_cast<TestData*>(d);
  for (int i = 0; i < batch_size; ++i) {
    uint8_t* buffer = data->buffer;
    for (int j = 0; j < data->num_values; ++j) {
      if (num_bytes == 4) {
        buffer += ParquetPlainEncoder::Encode(buffer, data->fixed_len_size,
            data->d4_values[j]);
      } else if (num_bytes == 8) {
        buffer += ParquetPlainEncoder::Encode(buffer, data->fixed_len_size,
            data->d8_values[j]);
      } else if (num_bytes == 16) {
        buffer += ParquetPlainEncoder::Encode(buffer, data->fixed_len_size,
            data->d16_values[j]);
      }
    }
  }
}

double Rand() {
  return rand() / static_cast<double>(RAND_MAX);
}

int main(int argc, char **argv) {
  CpuInfo::Init();
  cout << Benchmark::GetMachineInfo() << endl;

  int num_values = 1024;
  TestData data;
  data.num_values = num_values;
  data.buffer = new uint8_t[num_values * 16];
  data.d4_values.resize(num_values);
  data.d8_values.resize(num_values);
  data.d16_values.resize(num_values);

  uint8_t* base_buf;
  Benchmark suite_d4("Decimal4Value encode");
  for (int fixed_len_size = 1; fixed_len_size <= 4; ++fixed_len_size) {
    data.fixed_len_size = fixed_len_size;
    uint32_t mask = (1ll << (fixed_len_size * 8)) - 1;
    for (int i = 0; i < num_values; ++i) {
      uint32_t tmp = numeric_limits<uint32_t>::max() * Rand();
      data.d4_values[i] = tmp & mask;
      if (Rand() < 0) data.d4_values[i] = -data.d4_values[i];
    }
    TestImpala<4>(1, &data);
    base_buf = data.buffer;
    data.buffer = new uint8_t[num_values * 16];
    TestBswap<4>(1, &data);
    DCHECK_EQ(memcmp(base_buf, data.buffer, num_values * data.fixed_len_size), 0);
    TestSSE<4>(1, &data);
    DCHECK_EQ(memcmp(base_buf, data.buffer, num_values * data.fixed_len_size), 0);
    TestAVX2<4>(1, &data);
    DCHECK_EQ(memcmp(base_buf, data.buffer, num_values * data.fixed_len_size), 0);

    int baseline = suite_d4.AddBenchmark("Impala", TestImpala<4>, &data, -1);
    suite_d4.AddBenchmark("Bswap", TestBswap<4>, &data, baseline);
    suite_d4.AddBenchmark("SSE", TestSSE<4>, &data, baseline);
    suite_d4.AddBenchmark("AVX2", TestAVX2<4>, &data, baseline);
  }
  cout << suite_d4.Measure();

  Benchmark suite_d8("Decimal8Value encode");
  for (int fixed_len_size = 5; fixed_len_size <= 8; ++fixed_len_size) {
    data.fixed_len_size = fixed_len_size;
    uint64_t mask = (1ll << (fixed_len_size * 8)) - 1;
    for (int i = 0; i < num_values; ++i) {
      uint64_t tmp = numeric_limits<uint64_t>::max() * Rand();
      data.d8_values[i] = tmp & mask;
      if (Rand() < 0) data.d8_values[i] = -data.d8_values[i];
    }
    TestImpala<8>(1, &data);
    base_buf = data.buffer;
    data.buffer = new uint8_t[num_values * 16];
    TestBswap<8>(1, &data);
    DCHECK_EQ(memcmp(base_buf, data.buffer, num_values * data.fixed_len_size), 0);
    TestSSE<8>(1, &data);
    DCHECK_EQ(memcmp(base_buf, data.buffer, num_values * data.fixed_len_size), 0);
    TestAVX2<8>(1, &data);
    DCHECK_EQ(memcmp(base_buf, data.buffer, num_values * data.fixed_len_size), 0);

    int baseline = suite_d8.AddBenchmark("Impala", TestImpala<8>, &data, -1);
    suite_d8.AddBenchmark("Bswap", TestBswap<8>, &data, baseline);
    suite_d8.AddBenchmark("SSE", TestSSE<8>, &data, baseline);
    suite_d8.AddBenchmark("AVX2", TestAVX2<8>, &data, baseline);
  }
  cout << suite_d8.Measure();

  Benchmark suite_d16("Decimal16Value encode");
  for (int fixed_len_size = 9; fixed_len_size <= 16; ++fixed_len_size) {
    data.fixed_len_size = fixed_len_size;
    uint64_t mask = (1ll << ((fixed_len_size - 8) * 8)) - 1;
    for (int i = 0; i < num_values; ++i) {
      int128_t tmp = numeric_limits<uint64_t>::max() * Rand();
      tmp &= mask;
      tmp <<= 64;
      uint64_t lo = numeric_limits<uint64_t>::max() * Rand();
      tmp |= lo;
      data.d16_values[i] = tmp;
      if (Rand() < 0.5) data.d16_values[i] = -data.d16_values[i];
    }
    TestImpala<16>(1, &data);
    base_buf = data.buffer;
    data.buffer = new uint8_t[num_values * 16];
    TestBswap<16>(1, &data);
    DCHECK_EQ(memcmp(base_buf, data.buffer, num_values * data.fixed_len_size), 0);
    TestSSE<16>(1, &data);
    DCHECK_EQ(memcmp(base_buf, data.buffer, num_values * data.fixed_len_size), 0);
    TestAVX2<16>(1, &data);
    DCHECK_EQ(memcmp(base_buf, data.buffer, num_values * data.fixed_len_size), 0);

    int baseline = suite_d16.AddBenchmark("Impala", TestImpala<16>, &data, -1);
    suite_d16.AddBenchmark("Bswap", TestBswap<16>, &data, baseline);
    suite_d16.AddBenchmark("SSE", TestSSE<16>, &data, baseline);
    suite_d16.AddBenchmark("AVX2", TestAVX2<16>, &data, baseline);
  }
  cout << suite_d16.Measure();

  return 0;
}
