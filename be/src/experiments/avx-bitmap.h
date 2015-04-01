// Copyright 2012 Cloudera Inc.
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


#ifndef IMPALA_UTIL_AVX_BITMAP_H
#define IMPALA_UTIL_AVX_BITMAP_H

//#include <vector>
//#include <memory.h>
//#include <stdint.h>
//#include <stdio.h>
#include <immintrin.h>

#include "util/bit-util.h"

using namespace std;

namespace impala {

//
// Implement function Set(), Get(), And(), Or() in original bitmap with AVX intrinsics.
//
class AVXBitmap {
 public:
  AVXBitmap(int64_t num_bits) {
    buf_size_ = BitUtil::RoundUpNumi64(num_bits);
    buffer_ = new int64_t[buf_size_];
    size_ = num_bits;
    msize_ = _mm256_set1_ps((float)size_);
    mand_ = _mm256_set1_ps(63.0);
    //mzero_ = _mm256_set1_ps(0.0);
    mone_ = _mm256_set1_epi32(1);
  }

  virtual ~AVXBitmap() {
    if (buffer_) delete [] buffer_;
  }

  // Sets the bit at 'bit_index' to v. If mod is true, this
  // function will first mod the bit_index by the bitmap size.
  template<bool mod>
  inline void Set(int64_t bit_index, bool v) {
    if (mod) bit_index %= size();
    int64_t word_index = bit_index >> 8;
    bit_index &= 63;
    if (v) {
      buffer_[word_index] |= (1 << bit_index);
    } else {
      buffer_[word_index] &= ~(1 << bit_index);
    }
  }

  template<bool mod>
  inline void Set(__m256 bit_index, __m256i mask) {
    if (mod) {
      __m256 f = _mm256_div_ps(bit_index, msize_);//19~21 latency, 14 throughtput
      __m256 m = _mm256_mul_ps(f, msize_);//5 latency, 1 throughtput
      bit_index = _mm256_sub_ps(bit_index, m);//3 latency, 1 throughtput
    }
    __m256i word_index = _mm256_srli_si256(_mm256_cvtps_epi32(bit_index), 8);//2 latency, 2 throughput
    __m256 a = _mm256_and_ps(bit_index, mand_);//1 latency, 1 throughput
    __m256i di = _mm256_sllv_epi64(mone_, _mm256_cvtps_epi32(a));//1 latency, 1 throughput
    __m256 c2;
    __m256 c1 = _mm256_mask_i32gather_ps(c2, (float const*)buffer_, word_index,
        _mm256_cvtepi32_ps(mask), 4);
    __m256 e1 = _mm256_or_ps(c1, _mm256_cvtepi32_ps(di));
    _mm256_maskstore_ps((float*)buffer_, mask, e1);
    __m256 e2 = _mm256_andnot_ps(c2, _mm256_cvtepi32_ps(di));
    _mm256_maskstore_ps((float*)buffer_, mask, e2);
  }

  // Returns true if the bit at 'bit_index' is set. If mod is true, this
  // function will first mod the bit_index by the bitmap size.
  template<bool mod>
  inline bool Get(int64_t bit_index) const {
    if (mod) bit_index %= size();
    int64_t word_index = bit_index >> 8;
    bit_index &= 63;
    return (buffer_[word_index] & (1 << bit_index)) != 0;
  }

  // Returns true if the bit at 'bit_index' is set. If mod is true, this
  // function will first mod the bit_index by the bitmap size.
  template<bool mod>
  inline __m256i Get(__m256 bit_index) const {
    if (mod) {
      __m256 f = _mm256_div_ps(bit_index, msize_);//19~21 latency, 14 throughtput
      __m256 m = _mm256_mul_ps(f, msize_);//5 latency, 1 throughtput
      bit_index = _mm256_sub_ps(bit_index, m);//3 latency, 1 throughtput
    }
    __m256i word_index = _mm256_srli_si256(_mm256_cvtps_epi32(bit_index), 8);//2 latency, 2 throughput
    __m256 a = _mm256_and_ps(bit_index, mand_);//1 latency, 1 throughput
    __m256i ci = _mm256_i64gather_epi64((int const*)buffer_, word_index, 8);//7 throughput
    __m256i ai = _mm256_cvtps_epi32(a);//3 latency, 1 throughput
    __m256i di = _mm256_sllv_epi64(mone_, ai);//1 latency, 1 throughput
    return _mm256_and_si256(ci, di);//1 latency, 1 throughput
    //return _mm256_cmp_ps(e, mzero_, _CMP_EQ_OQ);
  }

  // Bitwise ANDs the src bitmap into this one.
  void And(const AVXBitmap* src) {
    DCHECK_EQ(size(), src->size());
    if (buf_size_ < 4) {
      for (int i = 0; i < buf_size_; ++i) {
        buffer_[i] |= src->buffer_[i];
      }
      return;
    }
    int i = 0;
    for (; i < buf_size_; i+=4) {
      __m256 t1 = _mm256_loadu_ps((float const*)(src->buffer_ + i));//3
      __m256 t2 = _mm256_loadu_ps((float const*)(buffer_ + i));//3
      t1 =  _mm256_and_ps(t1, t2);//1
      _mm256_storeu_ps((float*)(buffer_ + i), t1);//4
    }
    for (i = i - 4; i < buf_size_; ++i) {
      buffer_[i] |= src->buffer_[i];
    }
  }

  // Bitwise ORs the src bitmap into this one.
  void Or(const AVXBitmap* src) {
    DCHECK_EQ(size(), src->size());
    if (buf_size_ < 4) {
      for (int i = 0; i < buf_size_; ++i) {
        buffer_[i] |= src->buffer_[i];
      }
      return;
    }
    int i = 0;
    for (; i < buf_size_; i+=4) {
      __m256 t1 = _mm256_loadu_ps((float const*)(src->buffer_ + i));//3
      __m256 t2 = _mm256_loadu_ps((float const*)(buffer_ + i));//3
      t1 =  _mm256_or_ps(t1, t2);//1
      _mm256_storeu_ps((float*)(buffer_ + i), t1);//4
    }
    for (i = i - 4; i < buf_size_; ++i) {
      buffer_[i] |= src->buffer_[i];
    }
  }

  void SetAllBits(bool b) {
    memset(&buffer_[0], 255 * b, buf_size_ * sizeof(int64_t));
  }

  int64_t size() const { return size_; }

 private:
  int64_t* buffer_;
  int64_t size_;
  int64_t buf_size_;
  __m256 msize_;
  __m256 mand_;
  //__m256 mzero_;
  __m256i mone_;
};

}

#endif
