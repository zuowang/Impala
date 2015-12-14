// Copyright 2016 Cloudera Inc.
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

#ifndef IMPALA_UTIL_MEM_UTIL_H
#define IMPALA_UTIL_MEM_UTIL_H

// BSD LICENSE
//
// Copyright(c) 2010-2014 Intel Corporation. All rights reserved.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//   * Redistributions of source code must retain the above copyright
//     notice, this list of conditions and the following disclaimer.
//   * Redistributions in binary form must reproduce the above copyright
//     notice, this list of conditions and the following disclaimer in
//     the documentation and/or other materials provided with the
//     distribution.
//   * Neither the name of Intel Corporation nor the names of its
//     contributors may be used to endorse or promote products derived
//     from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <cpuid.h>
#include <immintrin.h>
#include <stdint.h>

#include <cstring>

#include "codegen/impala-ir.h"

#define TARGET_AVX2 __attribute__((target("avx2")))

namespace impala {

// This class contains AVX2 implementation of memcpy().
class MemUtil {
 public:
#ifdef IR_COMPILE
  static void* IR_ALWAYS_INLINE memcpy(void* dst, const void* src, size_t n) {
    return std::memcpy(dst, src, n);
  }
#else
  __attribute__((target("default")))
  static inline void* memcpy(void* dst, const void* src, size_t n) {
    return std::memcpy(dst, src, n);
  }

  // Copy bytes from one location to another. The locations must not overlap.
  static inline void* TARGET_AVX2 memcpy(void* dst, const void* src, size_t n) {
    void* ret = dst;
    int dstofss;
    int bits;

    // Copy less than 16 bytes
    if (n < 16) {
      if (n & 0x01) {
        *(uint8_t*)dst = *(const uint8_t*)src;
        src = (const uint8_t*)src + 1;
        dst = (uint8_t*)dst + 1;
      }
      if (n & 0x02) {
        *(uint16_t*)dst = *(const uint16_t*)src;
        src = (const uint16_t*)src + 1;
        dst = (uint16_t*)dst + 1;
      }
      if (n & 0x04) {
        *(uint32_t*)dst = *(const uint32_t*)src;
        src = (const uint32_t*)src + 1;
        dst = (uint32_t*)dst + 1;
      }
      if (n & 0x08) *(uint64_t*)dst = *(const uint64_t*)src;
      return ret;
    }

    // Fast way when copy size doesn't exceed 512 bytes
    if (n <= 32) {
      mov16((uint8_t*)dst, (const uint8_t*)src);
      mov16((uint8_t*)dst - 16 + n, (const uint8_t*)src - 16 + n);
      return ret;
    }
    if (n <= 64) {
      mov32_avx2((uint8_t*)dst, (const uint8_t*)src);
      mov32_avx2((uint8_t*)dst - 32 + n, (const uint8_t*)src - 32 + n);
      return ret;
    }

    if (n <= 512) {
      if (n >= 256) {
        n -= 256;
        mov256_avx2((uint8_t*)dst, (const uint8_t*)src);
        src = (const uint8_t*)src + 256;
        dst = (uint8_t*)dst + 256;
      }
      if (n >= 128) {
        n -= 128;
        mov128_avx2((uint8_t*)dst, (const uint8_t*)src);
        src = (const uint8_t*)src + 128;
        dst = (uint8_t*)dst + 128;
      }
      if (n >= 64) {
        n -= 64;
        mov64_avx2((uint8_t*)dst, (const uint8_t*)src);
        src = (const uint8_t*)src + 64;
        dst = (uint8_t*)dst + 64;
      }
  COPY_BLOCK_64_BACK31:
      if (n > 32) {
        mov32_avx2((uint8_t*)dst, (const uint8_t*)src);
        mov32_avx2((uint8_t*)dst - 32 + n, (const uint8_t*)src - 32 + n);
        return ret;
      }
      if (n > 0) mov32_avx2((uint8_t*)dst - 32 + n, (const uint8_t*)src - 32 + n);
      return ret;
    }

    // Make store aligned when copy size exceeds 512 bytes
    dstofss = 32 - (int)((long long)(void*)dst & 0x1F);
    n -= dstofss;
    mov32_avx2((uint8_t*)dst, (const uint8_t*)src);
    src = (const uint8_t*)src + dstofss;
    dst = (uint8_t*)dst + dstofss;

    // Copy 256-byte blocks. Use copy block function for better instruction order control,
    // which is important when load is unaligned.
    mov256blocks_avx2((uint8_t*)dst, (const uint8_t*)src, n);
    bits = n;
    n = n & 255;
    bits -= n;
    src = (const uint8_t*)src + bits;
    dst = (uint8_t*)dst + bits;

    // Copy 64-byte blocks. Use copy block function for better instruction order control,
    // which is important when load is unaligned.
    if (n >= 64) {
      mov64blocks_avx2((uint8_t*)dst, (const uint8_t*)src, n);
      bits = n;
      n = n & 63;
      bits -= n;
      src = (const uint8_t*)src + bits;
      dst = (uint8_t*)dst + bits;
    }

    // Copy whatever left
    goto COPY_BLOCK_64_BACK31;
  }

 private:
  // Copy 16 bytes from one location to another, locations should not overlap.
  static inline void TARGET_AVX2 mov16(uint8_t* dst, const uint8_t* src) {
    __m128i xmm0 = _mm_loadu_si128((const __m128i*)src);
    _mm_storeu_si128((__m128i*)dst, xmm0);
  }

  // Copy 32 bytes from one location to another, locations should not overlap.
  static inline void TARGET_AVX2 mov32_avx2(uint8_t* dst, const uint8_t* src) {
    __m256i ymm0 = _mm256_loadu_si256((const __m256i *)src);
    _mm256_storeu_si256((__m256i *)dst, ymm0);
  }

  // Copy 64 bytes from one location to another, locations should not overlap.
  static inline void TARGET_AVX2 mov64_avx2(uint8_t* dst, const uint8_t* src) {
    mov32_avx2(dst + 0 * 32, src + 0 * 32);
    mov32_avx2(dst + 1 * 32, src + 1 * 32);
  }

  // Copy 128 bytes from one location to another, locations should not overlap.
  static inline void TARGET_AVX2 mov128_avx2(uint8_t* dst, const uint8_t* src) {
    mov32_avx2(dst + 0 * 32, src + 0 * 32);
    mov32_avx2(dst + 1 * 32, src + 1 * 32);
    mov32_avx2(dst + 2 * 32, src + 2 * 32);
    mov32_avx2(dst + 3 * 32, src + 3 * 32);
  }

  // Copy 256 bytes from one location to another, locations should not overlap.
  static inline void TARGET_AVX2 mov256_avx2(uint8_t* dst, const uint8_t* src) {
    mov32_avx2(dst + 0 * 32, src + 0 * 32);
    mov32_avx2(dst + 1 * 32, src + 1 * 32);
    mov32_avx2(dst + 2 * 32, src + 2 * 32);
    mov32_avx2(dst + 3 * 32, src + 3 * 32);
    mov32_avx2(dst + 4 * 32, src + 4 * 32);
    mov32_avx2(dst + 5 * 32, src + 5 * 32);
    mov32_avx2(dst + 6 * 32, src + 6 * 32);
    mov32_avx2(dst + 7 * 32, src + 7 * 32);
  }

  // Copy 64-byte blocks from one location to another, locations should not overlap.
  static inline void TARGET_AVX2
  mov64blocks_avx2(uint8_t* dst, const uint8_t* src, size_t n) {
    __m256i ymm0, ymm1;

    while (n >= 64) {
      ymm0 = _mm256_loadu_si256((const __m256i *)(src + 0 * 32));
      n -= 64;
      ymm1 = _mm256_loadu_si256((const __m256i *)(src + 1 * 32));
      src = src + 64;
      _mm256_storeu_si256((__m256i *)(dst + 0 * 32), ymm0);
      _mm256_storeu_si256((__m256i *)(dst + 1 * 32), ymm1);
      dst = dst + 64;
    }
  }

  // Copy 256-byte blocks from one location to another, locations should not overlap.
  static inline void TARGET_AVX2
  mov256blocks_avx2(uint8_t* dst, const uint8_t* src, size_t n) {
    __m256i ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;

    while (n >= 256) {
      ymm0 = _mm256_loadu_si256((const __m256i *)(src + 0 * 32));
      n -= 256;
      ymm1 = _mm256_loadu_si256((const __m256i *)(src + 1 * 32));
      ymm2 = _mm256_loadu_si256((const __m256i *)(src + 2 * 32));
      ymm3 = _mm256_loadu_si256((const __m256i *)(src + 3 * 32));
      ymm4 = _mm256_loadu_si256((const __m256i *)(src + 4 * 32));
      ymm5 = _mm256_loadu_si256((const __m256i *)(src + 5 * 32));
      ymm6 = _mm256_loadu_si256((const __m256i *)(src + 6 * 32));
      ymm7 = _mm256_loadu_si256((const __m256i *)(src + 7 * 32));
      src = src + 256;
      _mm256_storeu_si256((__m256i *)(dst + 0 * 32), ymm0);
      _mm256_storeu_si256((__m256i *)(dst + 1 * 32), ymm1);
      _mm256_storeu_si256((__m256i *)(dst + 2 * 32), ymm2);
      _mm256_storeu_si256((__m256i *)(dst + 3 * 32), ymm3);
      _mm256_storeu_si256((__m256i *)(dst + 4 * 32), ymm4);
      _mm256_storeu_si256((__m256i *)(dst + 5 * 32), ymm5);
      _mm256_storeu_si256((__m256i *)(dst + 6 * 32), ymm6);
      _mm256_storeu_si256((__m256i *)(dst + 7 * 32), ymm7);
      dst = dst + 256;
    }
  }

#endif

};

}

#endif
