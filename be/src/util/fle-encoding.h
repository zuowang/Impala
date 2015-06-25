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

#ifndef IMPALA_FLE_ENCODING_H
#define IMPALA_FLE_ENCODING_H

#include <immintrin.h>
#include <math.h>

#include "common/compiler-util.h"
#include "util/bit-stream-utils.inline.h"
#include "util/bit-util.h"

namespace impala {

class FleDecoder {
 public:
  Init(uint8_t* buffer, int buffer_len, int bit_width, int num_vals) {
    DCHECK_GE(bit_width_, 0);
    DCHECK_LE(bit_width_, 64);
    DCHECK_GE(buffer_len - num_vals * bit_width / 8, 0);
    num_vals_ = num_vals;
    bit_width_ = bit_width;
    buffer_ = reinterpret_cast<uint64_t*>(buffer);
    buffer_end_ = buffer_;
    count_ = 64;

    clr_mask = _mm256_set1_epi64x(0x0102040810204080);
    for (int i = 0; i < 8; ++i) {
      bitv[i] = _mm256_set1_epi8(0x01 << i);
    }
    shf0_mask = _mm256_setr_epi64x(0x0707070707070707, 0x0606060606060606,
        0x0505050505050505, 0x0404040404040404);
    shf1_mask = _mm256_setr_epi64x(0x0303030303030303, 0x0202020202020202,
        0x0101010101010101, 0x0000000000000000);
    shf2_mask = _mm256_setr_epi64x(0x0f0f0f0f0f0f0f0f, 0x0e0e0e0e0e0e0e0e,
        0x0d0d0d0d0d0d0d0d, 0x0c0c0c0c0c0c0c0c);
    shf3_mask = _mm256_setr_epi64x(0x0b0b0b0b0b0b0b0b, 0x0a0a0a0a0a0a0a0a,
        0x0909090909090909, 0x0808080808080808);
    shf_ret_mask = _mm256_setr_epi64x(0x8003800280018000, 0x8007800680058004,
        0x800b800a80098008, 0x800f800e800d800c);
    shf_ret1_mask = _mm256_setr_epi64x(0x0380028001800080, 0x0780068005800480,
        0x0b800a8009800880, 0x0f800e800d800c80);
  }

  FleDecoder* NewFleDecoder(uint8_t* buffer, int buffer_len, int bit_width,
      int num_vals) {
    FleDecoder* fle_decoder = NULL;
    switch(bit_width) {
      case 1:
        fle_decoder = new FleDecoder_1();
      case 2:
        fle_decoder = new FleDecoder_2();
      case 3:
        fle_decoder = new FleDecoder_3();
      case 4:
        fle_decoder = new FleDecoder_4();
      case 5:
        fle_decoder = new FleDecoder_5();
      case 6:
        fle_decoder = new FleDecoder_6();
      case 7:
        fle_decoder = new FleDecoder_7();
      case 8:
        fle_decoder = new FleDecoder_8();
      case 9:
        fle_decoder = new FleDecoder_9();
      case 10:
        fle_decoder = new FleDecoder_10();
      case 11:
        fle_decoder = new FleDecoder_11();
      case 12:
        fle_decoder = new FleDecoder_12();
      case 13:
        fle_decoder = new FleDecoder_13();
      case 14:
        fle_decoder = new FleDecoder_14();
      case 15:
        fle_decoder = new FleDecoder_15();
      case 16:
        fle_decoder = new FleDecoder_16();
      case 17:
        fle_decoder = new FleDecoder_17();
      case 18:
        fle_decoder = new FleDecoder_18();
      case 19:
        fle_decoder = new FleDecoder_19();
      case 20:
        fle_decoder = new FleDecoder_20();
      case 21:
        fle_decoder = new FleDecoder_21();
      case 22:
        fle_decoder = new FleDecoder_22();
      case 23:
        fle_decoder = new FleDecoder_23();
      case 24:
        fle_decoder = new FleDecoder_24();
      case 25:
        fle_decoder = new FleDecoder_25();
      case 26:
        fle_decoder = new FleDecoder_26();
      case 27:
        fle_decoder = new FleDecoder_27();
      case 28:
        fle_decoder = new FleDecoder_28();
      case 29:
        fle_decoder = new FleDecoder_29();
      case 30:
        fle_decoder = new FleDecoder_30();
      case 31:
        fle_decoder = new FleDecoder_31();
      case 32:
        fle_decoder = new FleDecoder_32();
      default:
        DCHECK(false);
    }

    fle_decoder->Init(buffer, buffer_len, bit_width, num_vals);
  }

  template<typename T>
  virtual bool Get(T* val) = 0;

 private:
  __m256i clr_mask;
  __m256i bitv[8];
  __m256i shf0_mask;
  __m256i shf1_mask;
  __m256i shf2_mask;
  __m256i shf3_mask;
  __m256i shf_ret_mask;
  __m256i shf_ret1_mask;
  uint64_t* buffer_;
  uint64_t* buffer_end_;
  int num_vals_;
  int bit_width_;
  int count_;
};

class FleDecoder_1 : public FleDecoder {
 public:
  inline bool Get(T* val) {
    if (UNLIKELY(count_ == 64 || (num_vals_ < 0 && count_ == num_vals_ + 64))) {
      if (num_vals_ <= 0) return false;
      count_ = 0;
      __m256i buf = _mm256_loadu_si256((__m256i const*)buffer_end_);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      __m256i bit, ret_mask, ret;

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret = _mm256_and_si256(bitv[0], ret_mask);
      _mm256_storeu_si256((__m256i*)(current_value_), ret);
      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret = _mm256_and_si256(bitv[0], ret_mask);
      _mm256_storeu_si256((__m256i*)(current_value_ + 32), ret);
      buffer_end_ += bit_width_;
      num_vals_ -= 64;
    }
    *val = current_value_[count_];
    ++count_;
  }

 private:
  uint8_t current_value_[64];
};

class FleDecoder_2 : public FleDecoder {
 public:
  inline bool Get(T* val) {
    if (UNLIKELY(count_ == 64 || (num_vals_ < 0 && count_ == num_vals_ + 64))) {
      if (num_vals_ <= 0) return false;
      count_ = 0;
      __m256i buf = _mm256_loadu_si256((__m256i const*)buffer_end_);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      __m256i bit, ret_mask, tmp_ret, ret;

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      _mm256_storeu_si256((__m256i*)(current_value_), ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      _mm256_storeu_si256((__m256i*)(current_value_ + 32), ret);
      buffer_end_ += bit_width_;
      num_vals_ -= 64;
    }
    *val = current_value_[count_];
    ++count_;
  }

 private:
  uint8_t current_value_[64];
};

class FleDecoder_3 : public FleDecoder {
 public:
  inline bool Get(T* val) {
    if (UNLIKELY(count_ == 64 || (num_vals_ < 0 && count_ == num_vals_ + 64))) {
      if (num_vals_ <= 0) return false;
      count_ = 0;
      __m256i buf = _mm256_loadu_si256((__m256i const*)buffer_end_);
      __m256i buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      __m256i bit, ret_mask, tmp_ret, ret;

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      _mm256_storeu_si256((__m256i*)(current_value_), ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      _mm256_storeu_si256((__m256i*)(current_value_ + 32), ret);
      buffer_end_ += bit_width_;
      num_vals_ -= 64;
    }
    *val = current_value_[count_];
    ++count_;
  }

 private:
  uint8_t current_value_[64];
};

class FleDecoder_4 : public FleDecoder {
 public:
  inline bool Get(T* val) {
    if (UNLIKELY(count_ == 64 || (num_vals_ < 0 && count_ == num_vals_ + 64))) {
      if (num_vals_ <= 0) return false;
      count_ = 0;
      __m256i buf = _mm256_loadu_si256((__m256i const*)buffer_end_);
      __m256i buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      __m256i bit, ret_mask, tmp_ret, ret;

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      _mm256_storeu_si256((__m256i*)(current_value_), ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      _mm256_storeu_si256((__m256i*)(current_value_ + 32), ret);
      buffer_end_ += bit_width_;
      num_vals_ -= 64;
    }
    *val = current_value_[count_];
    ++count_;
  }

 private:
  uint8_t current_value_[64];
};

class FleDecoder_5 : public FleDecoder {
 public:
  inline bool Get(T* val) {
    if (UNLIKELY(count_ == 64 || (num_vals_ < 0 && count_ == num_vals_ + 64))) {
      if (num_vals_ <= 0) return false;
      count_ = 0;
      __m256i buf = _mm256_loadu_si256((__m256i const*)buffer_end_);
      __m256i buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      __m256i buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 4));
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);
      __m256i bit, ret_mask, tmp_ret, ret;

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      _mm256_storeu_si256((__m256i*)(current_value_), ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      _mm256_storeu_si256((__m256i*)(current_value_ + 32), ret);
      buffer_end_ += bit_width_;
      num_vals_ -= 64;
    }
    *val = current_value_[count_];
    ++count_;
  }

 private:
  uint8_t current_value_[64];
};

class FleDecoder_6 : public FleDecoder {
 public:
  inline bool Get(T* val) {
    if (UNLIKELY(count_ == 64 || (num_vals_ < 0 && count_ == num_vals_ + 64))) {
      if (num_vals_ <= 0) return false;
      count_ = 0;
      __m256i buf = _mm256_loadu_si256((__m256i const*)buffer_end_);
      __m256i buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      __m256i buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 4));
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);
      __m256i bit, ret_mask, tmp_ret, ret;

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      _mm256_storeu_si256((__m256i*)(current_value_), ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      _mm256_storeu_si256((__m256i*)(current_value_ + 32), ret);
      buffer_end_ += bit_width_;
      num_vals_ -= 64;
    }
    *val = current_value_[count_];
    ++count_;
  }

 private:
  uint8_t current_value_[64];
};

class FleDecoder_7 : public FleDecoder {
 public:
  inline bool Get(T* val) {
    if (UNLIKELY(count_ == 64 || (num_vals_ < 0 && count_ == num_vals_ + 64))) {
      if (num_vals_ <= 0) return false;
      count_ = 0;
      __m256i buf = _mm256_loadu_si256((__m256i const*)buffer_end_);
      __m256i buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      __m256i buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 4));
      __m256i buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);
      __m256i bit, ret_mask, tmp_ret, ret;

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      _mm256_storeu_si256((__m256i*)(current_value_), ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      _mm256_storeu_si256((__m256i*)(current_value_ + 32), ret);
      buffer_end_ += bit_width_;
      num_vals_ -= 64;
    }
    *val = current_value_[count_];
    ++count_;
  }

 private:
  uint8_t current_value_[64];
};

class FleDecoder_8 : public FleDecoder {
 public:
  inline bool Get(T* val) {
    if (UNLIKELY(count_ == 64 || (num_vals_ < 0 && count_ == num_vals_ + 64))) {
      if (num_vals_ <= 0) return false;
      count_ = 0;
      __m256i buf = _mm256_loadu_si256((__m256i const*)buffer_end_);
      __m256i buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      __m256i buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 4));
      __m256i buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);
      __m256i bit, ret_mask, tmp_ret, ret;

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret = _mm256_and_si256(bitv[0], ret_mask);

      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      _mm256_storeu_si256((__m256i*)(current_value_), ret);


      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret = _mm256_and_si256(bitv[0], ret_mask);

      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      _mm256_storeu_si256((__m256i*)(current_value_ + 32), ret);
      buffer_end_ += bit_width_;
      num_vals_ -= 64;
    }
    *val = current_value_[count_];
    ++count_;
  }

 private:
  uint8_t current_value_[64];
};

class FleDecoder_9 : public FleDecoder {
 public:
  inline bool Get(T* val) {
    if (UNLIKELY(count_ == 64 || (num_vals_ < 0 && count_ == num_vals_ + 64))) {
      if (num_vals_ <= 0) return false;
      count_ = 0;
      __m256i buf = _mm256_loadu_si256((__m256i const*)buffer_end_);
      __m256i buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      __m256i buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 4));
      __m256i buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);
      __m256i buf4 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 8));
      buf4 = _mm256_permute4x64_epi64(buf4, 0x44);
      __m256i bit, ret_mask, tmp_ret, tmp_ret1, ret, ret1;

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret = _mm256_and_si256(bitv[0], ret_mask);

      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf4, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret1 = _mm256_and_si256(bitv[0], ret_mask);

      tmp_ret = _mm256_unpacklo_epi8(ret, ret1);
      tmp_ret1 = _mm256_unpackhi_epi8(ret, ret1);
      ret = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret1 = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_), ret);
      _mm256_storeu_si256((__m256i*)(current_value_ + 16), ret1);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret = _mm256_and_si256(bitv[0], ret_mask);

      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf4, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret1 = _mm256_and_si256(bitv[0], ret_mask);

      tmp_ret = _mm256_unpacklo_epi8(ret, ret1);
      tmp_ret1 = _mm256_unpackhi_epi8(ret, ret1);
      ret = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret1 = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 32), ret);
      _mm256_storeu_si256((__m256i*)(current_value_ + 48), ret1);
      buffer_end_ += bit_width_;
      num_vals_ -= 64;
    }
    *val = current_value_[count_];
    ++count_;
  }

 private:
  uint16_t current_value_[64];
};

class FleDecoder_10 : public FleDecoder {
 public:
  inline bool Get(T* val) {
    if (UNLIKELY(count_ == 64 || (num_vals_ < 0 && count_ == num_vals_ + 64))) {
      if (num_vals_ <= 0) return false;
      count_ = 0;
      __m256i buf = _mm256_loadu_si256((__m256i const*)buffer_end_);
      __m256i buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      __m256i buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 4));
      __m256i buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);
      __m256i buf4 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 8));
      buf4 = _mm256_permute4x64_epi64(buf4, 0x44);
      __m256i bit, ret_mask, tmp_ret, tmp_ret1, ret, ret1;

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf4, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret1 = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf4, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);

      tmp_ret = _mm256_unpacklo_epi8(ret, ret1);
      tmp_ret1 = _mm256_unpackhi_epi8(ret, ret1);
      ret = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret1 = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_), ret);
      _mm256_storeu_si256((__m256i*)(current_value_ + 16), ret1);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf4, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret1 = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf4, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);

      tmp_ret = _mm256_unpacklo_epi8(ret, ret1);
      tmp_ret1 = _mm256_unpackhi_epi8(ret, ret1);
      ret = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret1 = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 32), ret);
      _mm256_storeu_si256((__m256i*)(current_value_ + 48), ret1);
      buffer_end_ += bit_width_;
      num_vals_ -= 64;
    }
    *val = current_value_[count_];
    ++count_;
  }

 private:
  uint16_t current_value_[64];
};

class FleDecoder_11 : public FleDecoder {
 public:
  inline bool Get(T* val) {
    if (UNLIKELY(count_ == 64 || (num_vals_ < 0 && count_ == num_vals_ + 64))) {
      if (num_vals_ <= 0) return false;
      count_ = 0;
      __m256i buf = _mm256_loadu_si256((__m256i const*)buffer_end_);
      __m256i buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      __m256i buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 4));
      __m256i buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);
      __m256i buf4 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 8));
      __m256i buf5 = _mm256_permute4x64_epi64(buf4, 0xee);
      buf4 = _mm256_permute4x64_epi64(buf4, 0x44);

      __m256i bit, ret_mask, tmp_ret, tmp_ret1, ret, ret1;

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf4, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret1 = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf4, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);
      bit = _mm256_shuffle_epi8(buf5, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);

      tmp_ret = _mm256_unpacklo_epi8(ret, ret1);
      tmp_ret1 = _mm256_unpackhi_epi8(ret, ret1);
      ret = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret1 = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_), ret);
      _mm256_storeu_si256((__m256i*)(current_value_ + 16), ret1);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf4, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret1 = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf4, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);
      bit = _mm256_shuffle_epi8(buf5, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);

      tmp_ret = _mm256_unpacklo_epi8(ret, ret1);
      tmp_ret1 = _mm256_unpackhi_epi8(ret, ret1);
      ret = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret1 = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 32), ret);
      _mm256_storeu_si256((__m256i*)(current_value_ + 48), ret1);
      buffer_end_ += bit_width_;
      num_vals_ -= 64;
    }
    *val = current_value_[count_];
    ++count_;
  }

 private:
  uint16_t current_value_[64];
};

class FleDecoder_12 : public FleDecoder {
 public:
  inline bool Get(T* val) {
    if (UNLIKELY(count_ == 64 || (num_vals_ < 0 && count_ == num_vals_ + 64))) {
      if (num_vals_ <= 0) return false;
      count_ = 0;
      __m256i buf = _mm256_loadu_si256((__m256i const*)buffer_end_);
      __m256i buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      __m256i buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 4));
      __m256i buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);
      __m256i buf4 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 8));
      __m256i buf5 = _mm256_permute4x64_epi64(buf4, 0xee);
      buf4 = _mm256_permute4x64_epi64(buf4, 0x44);

      __m256i bit, ret_mask, tmp_ret, tmp_ret1, ret, ret1;

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf4, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret1 = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf4, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);
      bit = _mm256_shuffle_epi8(buf5, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);
      bit = _mm256_shuffle_epi8(buf5, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);

      tmp_ret = _mm256_unpacklo_epi8(ret, ret1);
      tmp_ret1 = _mm256_unpackhi_epi8(ret, ret1);
      ret = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret1 = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_), ret);
      _mm256_storeu_si256((__m256i*)(current_value_ + 16), ret1);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf4, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret1 = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf4, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);
      bit = _mm256_shuffle_epi8(buf5, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);
      bit = _mm256_shuffle_epi8(buf5, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);

      tmp_ret = _mm256_unpacklo_epi8(ret, ret1);
      tmp_ret1 = _mm256_unpackhi_epi8(ret, ret1);
      ret = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret1 = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 32), ret);
      _mm256_storeu_si256((__m256i*)(current_value_ + 48), ret1);
      buffer_end_ += bit_width_;
      num_vals_ -= 64;
    }
    *val = current_value_[count_];
    ++count_;
  }

 private:
  uint16_t current_value_[64];
};

class FleDecoder_13 : public FleDecoder {
 public:
  inline bool Get(T* val) {
    if (UNLIKELY(count_ == 64 || (num_vals_ < 0 && count_ == num_vals_ + 64))) {
      if (num_vals_ <= 0) return false;
      count_ = 0;
      __m256i buf = _mm256_loadu_si256((__m256i const*)buffer_end_);
      __m256i buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      __m256i buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 4));
      __m256i buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);
      __m256i buf4 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 8));
      __m256i buf5 = _mm256_permute4x64_epi64(buf4, 0xee);
      buf4 = _mm256_permute4x64_epi64(buf4, 0x44);
      __m256i buf6 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 12));
      buf6 = _mm256_permute4x64_epi64(buf6, 0x44);

      __m256i bit, ret_mask, tmp_ret, tmp_ret1, ret, ret1;

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf4, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret1 = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf4, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);
      bit = _mm256_shuffle_epi8(buf5, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);
      bit = _mm256_shuffle_epi8(buf5, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);
      bit = _mm256_shuffle_epi8(buf6, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);

      tmp_ret = _mm256_unpacklo_epi8(ret, ret1);
      tmp_ret1 = _mm256_unpackhi_epi8(ret, ret1);
      ret = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret1 = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_), ret);
      _mm256_storeu_si256((__m256i*)(current_value_ + 16), ret1);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf4, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret1 = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf4, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);
      bit = _mm256_shuffle_epi8(buf5, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);
      bit = _mm256_shuffle_epi8(buf5, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);
      bit = _mm256_shuffle_epi8(buf6, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);

      tmp_ret = _mm256_unpacklo_epi8(ret, ret1);
      tmp_ret1 = _mm256_unpackhi_epi8(ret, ret1);
      ret = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret1 = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 32), ret);
      _mm256_storeu_si256((__m256i*)(current_value_ + 48), ret1);
      buffer_end_ += bit_width_;
      num_vals_ -= 64;
    }
    *val = current_value_[count_];
    ++count_;
  }

 private:
  uint16_t current_value_[64];
};

class FleDecoder_14 : public FleDecoder {
 public:
  inline bool Get(T* val) {
    if (UNLIKELY(count_ == 64 || (num_vals_ < 0 && count_ == num_vals_ + 64))) {
      if (num_vals_ <= 0) return false;
      count_ = 0;
      __m256i buf = _mm256_loadu_si256((__m256i const*)buffer_end_);
      __m256i buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      __m256i buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 4));
      __m256i buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);
      __m256i buf4 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 8));
      __m256i buf5 = _mm256_permute4x64_epi64(buf4, 0xee);
      buf4 = _mm256_permute4x64_epi64(buf4, 0x44);
      __m256i buf6 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 12));
      buf6 = _mm256_permute4x64_epi64(buf6, 0x44);

      __m256i bit, ret_mask, tmp_ret, tmp_ret1, ret, ret1;

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf4, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret1 = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf4, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);
      bit = _mm256_shuffle_epi8(buf5, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);
      bit = _mm256_shuffle_epi8(buf5, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);
      bit = _mm256_shuffle_epi8(buf6, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);
      bit = _mm256_shuffle_epi8(buf6, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);

      tmp_ret = _mm256_unpacklo_epi8(ret, ret1);
      tmp_ret1 = _mm256_unpackhi_epi8(ret, ret1);
      ret = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret1 = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_), ret);
      _mm256_storeu_si256((__m256i*)(current_value_ + 16), ret1);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf4, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret1 = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf4, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);
      bit = _mm256_shuffle_epi8(buf5, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);
      bit = _mm256_shuffle_epi8(buf5, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);
      bit = _mm256_shuffle_epi8(buf6, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);
      bit = _mm256_shuffle_epi8(buf6, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);

      tmp_ret = _mm256_unpacklo_epi8(ret, ret1);
      tmp_ret1 = _mm256_unpackhi_epi8(ret, ret1);
      ret = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret1 = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 32), ret);
      _mm256_storeu_si256((__m256i*)(current_value_ + 48), ret1);
      buffer_end_ += bit_width_;
      num_vals_ -= 64;
    }
    *val = current_value_[count_];
    ++count_;
  }

 private:
  uint16_t current_value_[64];
};

class FleDecoder_15 : public FleDecoder {
 public:
  inline bool Get(T* val) {
    if (UNLIKELY(count_ == 64 || (num_vals_ < 0 && count_ == num_vals_ + 64))) {
      if (num_vals_ <= 0) return false;
      count_ = 0;
      __m256i buf = _mm256_loadu_si256((__m256i const*)buffer_end_);
      __m256i buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      __m256i buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 4));
      __m256i buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);
      __m256i buf4 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 8));
      __m256i buf5 = _mm256_permute4x64_epi64(buf4, 0xee);
      buf4 = _mm256_permute4x64_epi64(buf4, 0x44);
      __m256i buf6 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 12));
      __m256i buf7 = _mm256_permute4x64_epi64(buf6, 0xee);
      buf6 = _mm256_permute4x64_epi64(buf6, 0x44);

      __m256i bit, ret_mask, tmp_ret, tmp_ret1, ret, ret1;

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf4, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret1 = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf4, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);
      bit = _mm256_shuffle_epi8(buf5, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);
      bit = _mm256_shuffle_epi8(buf5, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);
      bit = _mm256_shuffle_epi8(buf6, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);
      bit = _mm256_shuffle_epi8(buf6, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);
      bit = _mm256_shuffle_epi8(buf7, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);

      tmp_ret = _mm256_unpacklo_epi8(ret, ret1);
      tmp_ret1 = _mm256_unpackhi_epi8(ret, ret1);
      ret = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret1 = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_), ret);
      _mm256_storeu_si256((__m256i*)(current_value_ + 16), ret1);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret = _mm256_or_si256(ret, tmp_ret);

      bit = _mm256_shuffle_epi8(buf4, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret1 = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf4, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);
      bit = _mm256_shuffle_epi8(buf5, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);
      bit = _mm256_shuffle_epi8(buf5, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);
      bit = _mm256_shuffle_epi8(buf6, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);
      bit = _mm256_shuffle_epi8(buf6, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);
      bit = _mm256_shuffle_epi8(buf7, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret1 = _mm256_or_si256(ret1, tmp_ret);

      tmp_ret = _mm256_unpacklo_epi8(ret, ret1);
      tmp_ret1 = _mm256_unpackhi_epi8(ret, ret1);
      ret = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret1 = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 32), ret);
      _mm256_storeu_si256((__m256i*)(current_value_ + 48), ret1);
      buffer_end_ += bit_width_;
      num_vals_ -= 64;
    }
    *val = current_value_[count_];
    ++count_;
  }

 private:
  uint16_t current_value_[64];
};

class FleDecoder_16 : public FleDecoder {
 public:
  inline bool Get(T* val) {
    if (UNLIKELY(count_ == 64 || (num_vals_ < 0 && count_ == num_vals_ + 64))) {
      if (num_vals_ <= 0) return false;
      count_ = 0;
      __m256i ret[4], buf, buf1, buf2, buf3;
      buf = _mm256_loadu_si256((__m256i const*)buffer_end_);
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 4));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      __m256i bit, ret_mask, tmp_ret, tmp_ret1;

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[0] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[1] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 8));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 12));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[2] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[3] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);

      tmp_ret = _mm256_unpacklo_epi8(ret[0], ret[2]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[0], ret[2]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_), ret[0]);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 16), ret[2]);

      tmp_ret = _mm256_unpacklo_epi8(ret[1], ret[3]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[1], ret[3]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 32), ret[1]);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 48), ret[3]);
      buffer_end_ += bit_width_;
      num_vals_ -= 64;
    }
    *val = current_value_[count_];
    ++count_;
  }

 private:
  uint16_t current_value_[64];
};

class FleDecoder_17 : public FleDecoder {
 public:
  inline bool Get(T* val) {
    if (UNLIKELY(count_ == 64 || (num_vals_ < 0 && count_ == num_vals_ + 64))) {
      if (num_vals_ <= 0) return false;
      count_ = 0;
      __m256i ret[8], buf, buf1, buf2, buf3;
      buf = _mm256_loadu_si256((__m256i const*)buffer_end_);
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 4));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      __m256i bit, ret_mask, tmp_ret, tmp_ret1;

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[0] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[1] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 8));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 12));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[2] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[3] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 16));
      buf = _mm256_permute4x64_epi64(buf, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[4] = _mm256_and_si256(bitv[0], ret_mask);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[5] = _mm256_and_si256(bitv[0], ret_mask);

      ret[6] = _mm256_setzero_si256();
      ret[7] = _mm256_setzero_si256();

      tmp_ret = _mm256_unpacklo_epi8(ret[0], ret[2]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[0], ret[2]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[1], ret[3]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[1], ret[3]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[4], ret[6]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[4], ret[6]);
      ret[4] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[6] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[5], ret[7]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[5], ret[7]);
      ret[5] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[7] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);

      tmp_ret = _mm256_unpacklo_epi16(ret[0], ret[4]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[0], ret[4]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_), ret[0]);
      ret[4] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 8), ret[4]);
      tmp_ret = _mm256_unpacklo_epi16(ret[2], ret[6]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[2], ret[6]);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 16), ret[2]);
      ret[6] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 24), ret[6]);
      tmp_ret = _mm256_unpacklo_epi16(ret[1], ret[5]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[1], ret[5]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 32), ret[1]);
      ret[5] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 40), ret[5]);
      tmp_ret = _mm256_unpacklo_epi16(ret[3], ret[7]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[3], ret[7]);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 48), ret[3]);
      ret[7] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 56), ret[7]);
      buffer_end_ += bit_width_;
      num_vals_ -= 64;
    }
    *val = current_value_[count_];
    ++count_;
  }

 private:
  uint32_t current_value_[64];
};

class FleDecoder_18 : public FleDecoder {
 public:
  inline bool Get(T* val) {
    if (UNLIKELY(count_ == 64 || (num_vals_ < 0 && count_ == num_vals_ + 64))) {
      if (num_vals_ <= 0) return false;
      count_ = 0;
      __m256i ret[8], buf, buf1, buf2, buf3;
      buf = _mm256_loadu_si256((__m256i const*)buffer_end_);
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 4));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      __m256i bit, ret_mask, tmp_ret, tmp_ret1;

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[0] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[1] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 8));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 12));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[2] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[3] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 16));
      buf = _mm256_permute4x64_epi64(buf, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[4] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[5] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);

      ret[6] = _mm256_setzero_si256();
      ret[7] = _mm256_setzero_si256();

      tmp_ret = _mm256_unpacklo_epi8(ret[0], ret[2]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[0], ret[2]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[1], ret[3]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[1], ret[3]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[4], ret[6]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[4], ret[6]);
      ret[4] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[6] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[5], ret[7]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[5], ret[7]);
      ret[5] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[7] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);

      tmp_ret = _mm256_unpacklo_epi16(ret[0], ret[4]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[0], ret[4]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_), ret[0]);
      ret[4] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 8), ret[4]);
      tmp_ret = _mm256_unpacklo_epi16(ret[2], ret[6]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[2], ret[6]);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 16), ret[2]);
      ret[6] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 24), ret[6]);
      tmp_ret = _mm256_unpacklo_epi16(ret[1], ret[5]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[1], ret[5]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 32), ret[1]);
      ret[5] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 40), ret[5]);
      tmp_ret = _mm256_unpacklo_epi16(ret[3], ret[7]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[3], ret[7]);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 48), ret[3]);
      ret[7] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 56), ret[7]);
      buffer_end_ += bit_width_;
      num_vals_ -= 64;
    }
    *val = current_value_[count_];
    ++count_;
  }

 private:
  uint32_t current_value_[64];
};

class FleDecoder_19 : public FleDecoder {
 public:
  inline bool Get(T* val) {
    if (UNLIKELY(count_ == 64 || (num_vals_ < 0 && count_ == num_vals_ + 64))) {
      if (num_vals_ <= 0) return false;
      count_ = 0;
      __m256i ret[8], buf, buf1, buf2, buf3;
      buf = _mm256_loadu_si256((__m256i const*)buffer_end_);
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 4));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      __m256i bit, ret_mask, tmp_ret, tmp_ret1;

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[0] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[1] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 8));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 12));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[2] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[3] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 16));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[4] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[5] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);

      ret[6] = _mm256_setzero_si256();
      ret[7] = _mm256_setzero_si256();

      tmp_ret = _mm256_unpacklo_epi8(ret[0], ret[2]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[0], ret[2]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[1], ret[3]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[1], ret[3]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[4], ret[6]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[4], ret[6]);
      ret[4] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[6] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[5], ret[7]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[5], ret[7]);
      ret[5] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[7] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);

      tmp_ret = _mm256_unpacklo_epi16(ret[0], ret[4]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[0], ret[4]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_), ret[0]);
      ret[4] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 8), ret[4]);
      tmp_ret = _mm256_unpacklo_epi16(ret[2], ret[6]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[2], ret[6]);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 16), ret[2]);
      ret[6] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 24), ret[6]);
      tmp_ret = _mm256_unpacklo_epi16(ret[1], ret[5]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[1], ret[5]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 32), ret[1]);
      ret[5] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 40), ret[5]);
      tmp_ret = _mm256_unpacklo_epi16(ret[3], ret[7]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[3], ret[7]);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 48), ret[3]);
      ret[7] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 56), ret[7]);
      buffer_end_ += bit_width_;
      num_vals_ -= 64;
    }
    *val = current_value_[count_];
    ++count_;
  }

 private:
  uint32_t current_value_[64];
};

class FleDecoder_20 : public FleDecoder {
 public:
  inline bool Get(T* val) {
    if (UNLIKELY(count_ == 64 || (num_vals_ < 0 && count_ == num_vals_ + 64))) {
      if (num_vals_ <= 0) return false;
      count_ = 0;
      __m256i ret[8], buf, buf1, buf2, buf3;
      buf = _mm256_loadu_si256((__m256i const*)buffer_end_);
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 4));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      __m256i bit, ret_mask, tmp_ret, tmp_ret1;

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[0] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[1] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 8));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 12));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[2] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[3] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 16));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[4] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[5] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);

      ret[6] = _mm256_setzero_si256();
      ret[7] = _mm256_setzero_si256();

      tmp_ret = _mm256_unpacklo_epi8(ret[0], ret[2]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[0], ret[2]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[1], ret[3]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[1], ret[3]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[4], ret[6]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[4], ret[6]);
      ret[4] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[6] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[5], ret[7]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[5], ret[7]);
      ret[5] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[7] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);

      tmp_ret = _mm256_unpacklo_epi16(ret[0], ret[4]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[0], ret[4]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_), ret[0]);
      ret[4] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 8), ret[4]);
      tmp_ret = _mm256_unpacklo_epi16(ret[2], ret[6]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[2], ret[6]);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 16), ret[2]);
      ret[6] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 24), ret[6]);
      tmp_ret = _mm256_unpacklo_epi16(ret[1], ret[5]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[1], ret[5]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 32), ret[1]);
      ret[5] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 40), ret[5]);
      tmp_ret = _mm256_unpacklo_epi16(ret[3], ret[7]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[3], ret[7]);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 48), ret[3]);
      ret[7] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 56), ret[7]);
      buffer_end_ += bit_width_;
      num_vals_ -= 64;
    }
    *val = current_value_[count_];
    ++count_;
  }

 private:
  uint32_t current_value_[64];
};

class FleDecoder_21 : public FleDecoder {
 public:
  inline bool Get(T* val) {
    if (UNLIKELY(count_ == 64 || (num_vals_ < 0 && count_ == num_vals_ + 64))) {
      if (num_vals_ <= 0) return false;
      count_ = 0;
      __m256i ret[8], buf, buf1, buf2, buf3;
      buf = _mm256_loadu_si256((__m256i const*)buffer_end_);
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 4));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      __m256i bit, ret_mask, tmp_ret, tmp_ret1;

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[0] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[1] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 8));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 12));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[2] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[3] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 16));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 20));
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[4] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[5] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);

      ret[6] = _mm256_setzero_si256();
      ret[7] = _mm256_setzero_si256();

      tmp_ret = _mm256_unpacklo_epi8(ret[0], ret[2]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[0], ret[2]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[1], ret[3]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[1], ret[3]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[4], ret[6]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[4], ret[6]);
      ret[4] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[6] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[5], ret[7]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[5], ret[7]);
      ret[5] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[7] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);

      tmp_ret = _mm256_unpacklo_epi16(ret[0], ret[4]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[0], ret[4]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_), ret[0]);
      ret[4] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 8), ret[4]);
      tmp_ret = _mm256_unpacklo_epi16(ret[2], ret[6]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[2], ret[6]);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 16), ret[2]);
      ret[6] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 24), ret[6]);
      tmp_ret = _mm256_unpacklo_epi16(ret[1], ret[5]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[1], ret[5]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 32), ret[1]);
      ret[5] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 40), ret[5]);
      tmp_ret = _mm256_unpacklo_epi16(ret[3], ret[7]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[3], ret[7]);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 48), ret[3]);
      ret[7] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 56), ret[7]);
      buffer_end_ += bit_width_;
      num_vals_ -= 64;
    }
    *val = current_value_[count_];
    ++count_;
  }

 private:
  uint32_t current_value_[64];
};

class FleDecoder_22 : public FleDecoder {
 public:
  inline bool Get(T* val) {
    if (UNLIKELY(count_ == 64 || (num_vals_ < 0 && count_ == num_vals_ + 64))) {
      if (num_vals_ <= 0) return false;
      count_ = 0;
      __m256i ret[8], buf, buf1, buf2, buf3;
      buf = _mm256_loadu_si256((__m256i const*)buffer_end_);
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 4));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      __m256i bit, ret_mask, tmp_ret, tmp_ret1;

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[0] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[1] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 8));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 12));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[2] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[3] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 16));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 20));
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[4] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[5] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);

      ret[6] = _mm256_setzero_si256();
      ret[7] = _mm256_setzero_si256();

      tmp_ret = _mm256_unpacklo_epi8(ret[0], ret[2]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[0], ret[2]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[1], ret[3]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[1], ret[3]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[4], ret[6]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[4], ret[6]);
      ret[4] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[6] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[5], ret[7]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[5], ret[7]);
      ret[5] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[7] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);

      tmp_ret = _mm256_unpacklo_epi16(ret[0], ret[4]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[0], ret[4]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_), ret[0]);
      ret[4] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 8), ret[4]);
      tmp_ret = _mm256_unpacklo_epi16(ret[2], ret[6]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[2], ret[6]);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 16), ret[2]);
      ret[6] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 24), ret[6]);
      tmp_ret = _mm256_unpacklo_epi16(ret[1], ret[5]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[1], ret[5]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 32), ret[1]);
      ret[5] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 40), ret[5]);
      tmp_ret = _mm256_unpacklo_epi16(ret[3], ret[7]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[3], ret[7]);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 48), ret[3]);
      ret[7] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 56), ret[7]);
      buffer_end_ += bit_width_;
      num_vals_ -= 64;
    }
    *val = current_value_[count_];
    ++count_;
  }

 private:
  uint32_t current_value_[64];
};

class FleDecoder_23 : public FleDecoder {
 public:
  inline bool Get(T* val) {
    if (UNLIKELY(count_ == 64 || (num_vals_ < 0 && count_ == num_vals_ + 64))) {
      if (num_vals_ <= 0) return false;
      count_ = 0;
      __m256i ret[8], buf, buf1, buf2, buf3;
      buf = _mm256_loadu_si256((__m256i const*)buffer_end_);
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 4));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      __m256i bit, ret_mask, tmp_ret, tmp_ret1;

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[0] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[1] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 8));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 12));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[2] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[3] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 16));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 20));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[4] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[5] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);

      ret[6] = _mm256_setzero_si256();
      ret[7] = _mm256_setzero_si256();

      tmp_ret = _mm256_unpacklo_epi8(ret[0], ret[2]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[0], ret[2]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[1], ret[3]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[1], ret[3]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[4], ret[6]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[4], ret[6]);
      ret[4] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[6] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[5], ret[7]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[5], ret[7]);
      ret[5] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[7] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);

      tmp_ret = _mm256_unpacklo_epi16(ret[0], ret[4]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[0], ret[4]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_), ret[0]);
      ret[4] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 8), ret[4]);
      tmp_ret = _mm256_unpacklo_epi16(ret[2], ret[6]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[2], ret[6]);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 16), ret[2]);
      ret[6] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 24), ret[6]);
      tmp_ret = _mm256_unpacklo_epi16(ret[1], ret[5]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[1], ret[5]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 32), ret[1]);
      ret[5] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 40), ret[5]);
      tmp_ret = _mm256_unpacklo_epi16(ret[3], ret[7]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[3], ret[7]);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 48), ret[3]);
      ret[7] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 56), ret[7]);
      buffer_end_ += bit_width_;
      num_vals_ -= 64;
    }
    *val = current_value_[count_];
    ++count_;
  }

 private:
  uint32_t current_value_[64];
};

class FleDecoder_24 : public FleDecoder {
 public:
  inline bool Get(T* val) {
    if (UNLIKELY(count_ == 64 || (num_vals_ < 0 && count_ == num_vals_ + 64))) {
      if (num_vals_ <= 0) return false;
      count_ = 0;
      __m256i ret[8], buf, buf1, buf2, buf3;
      buf = _mm256_loadu_si256((__m256i const*)buffer_end_);
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 4));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      __m256i bit, ret_mask, tmp_ret, tmp_ret1;

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[0] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[1] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 8));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 12));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[2] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[3] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 16));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 20));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[4] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[5] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);

      ret[6] = _mm256_setzero_si256();
      ret[7] = _mm256_setzero_si256();

      tmp_ret = _mm256_unpacklo_epi8(ret[0], ret[2]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[0], ret[2]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[1], ret[3]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[1], ret[3]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[4], ret[6]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[4], ret[6]);
      ret[4] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[6] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[5], ret[7]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[5], ret[7]);
      ret[5] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[7] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);

      tmp_ret = _mm256_unpacklo_epi16(ret[0], ret[4]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[0], ret[4]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_), ret[0]);
      ret[4] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 8), ret[4]);
      tmp_ret = _mm256_unpacklo_epi16(ret[2], ret[6]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[2], ret[6]);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 16), ret[2]);
      ret[6] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 24), ret[6]);
      tmp_ret = _mm256_unpacklo_epi16(ret[1], ret[5]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[1], ret[5]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 32), ret[1]);
      ret[5] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 40), ret[5]);
      tmp_ret = _mm256_unpacklo_epi16(ret[3], ret[7]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[3], ret[7]);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 48), ret[3]);
      ret[7] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 56), ret[7]);
      buffer_end_ += bit_width_;
      num_vals_ -= 64;
    }
    *val = current_value_[count_];
    ++count_;
  }

 private:
  uint32_t current_value_[64];
};

class FleDecoder_25 : public FleDecoder {
 public:
  inline bool Get(T* val) {
    if (UNLIKELY(count_ == 64 || (num_vals_ < 0 && count_ == num_vals_ + 64))) {
      if (num_vals_ <= 0) return false;
      count_ = 0;
      __m256i ret[8], buf, buf1, buf2, buf3;
      buf = _mm256_loadu_si256((__m256i const*)buffer_end_);
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 4));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      __m256i bit, ret_mask, tmp_ret, tmp_ret1;

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[0] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[1] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 8));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 12));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[2] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[3] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 16));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 20));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[4] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[5] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 24));
      buf = _mm256_permute4x64_epi64(buf, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[6] = _mm256_and_si256(bitv[0], ret_mask);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[7] = _mm256_and_si256(bitv[0], ret_mask);

      tmp_ret = _mm256_unpacklo_epi8(ret[0], ret[2]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[0], ret[2]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[1], ret[3]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[1], ret[3]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[4], ret[6]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[4], ret[6]);
      ret[4] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[6] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[5], ret[7]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[5], ret[7]);
      ret[5] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[7] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);

      tmp_ret = _mm256_unpacklo_epi16(ret[0], ret[4]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[0], ret[4]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_), ret[0]);
      ret[4] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 8), ret[4]);
      tmp_ret = _mm256_unpacklo_epi16(ret[2], ret[6]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[2], ret[6]);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 16), ret[2]);
      ret[6] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 24), ret[6]);
      tmp_ret = _mm256_unpacklo_epi16(ret[1], ret[5]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[1], ret[5]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 32), ret[1]);
      ret[5] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 40), ret[5]);
      tmp_ret = _mm256_unpacklo_epi16(ret[3], ret[7]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[3], ret[7]);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 48), ret[3]);
      ret[7] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 56), ret[7]);
      buffer_end_ += bit_width_;
      num_vals_ -= 64;
    }
    *val = current_value_[count_];
    ++count_;
  }

 private:
  uint32_t current_value_[64];
};

class FleDecoder_26 : public FleDecoder {
 public:
  inline bool Get(T* val) {
    if (UNLIKELY(count_ == 64 || (num_vals_ < 0 && count_ == num_vals_ + 64))) {
      if (num_vals_ <= 0) return false;
      count_ = 0;
      __m256i ret[8], buf, buf1, buf2, buf3;
      buf = _mm256_loadu_si256((__m256i const*)buffer_end_);
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 4));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      __m256i bit, ret_mask, tmp_ret, tmp_ret1;

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[0] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[1] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 8));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 12));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[2] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[3] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 16));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 20));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[4] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[5] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 24));
      buf = _mm256_permute4x64_epi64(buf, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[6] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[6] = _mm256_or_si256(ret[6], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[7] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[7] = _mm256_or_si256(ret[7], tmp_ret);

      tmp_ret = _mm256_unpacklo_epi8(ret[0], ret[2]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[0], ret[2]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[1], ret[3]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[1], ret[3]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[4], ret[6]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[4], ret[6]);
      ret[4] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[6] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[5], ret[7]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[5], ret[7]);
      ret[5] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[7] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);

      tmp_ret = _mm256_unpacklo_epi16(ret[0], ret[4]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[0], ret[4]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_), ret[0]);
      ret[4] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 8), ret[4]);
      tmp_ret = _mm256_unpacklo_epi16(ret[2], ret[6]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[2], ret[6]);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 16), ret[2]);
      ret[6] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 24), ret[6]);
      tmp_ret = _mm256_unpacklo_epi16(ret[1], ret[5]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[1], ret[5]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 32), ret[1]);
      ret[5] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 40), ret[5]);
      tmp_ret = _mm256_unpacklo_epi16(ret[3], ret[7]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[3], ret[7]);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 48), ret[3]);
      ret[7] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 56), ret[7]);
      buffer_end_ += bit_width_;
      num_vals_ -= 64;
    }
    *val = current_value_[count_];
    ++count_;
  }

 private:
  uint32_t current_value_[64];
};

class FleDecoder_27 : public FleDecoder {
 public:
  inline bool Get(T* val) {
    if (UNLIKELY(count_ == 64 || (num_vals_ < 0 && count_ == num_vals_ + 64))) {
      if (num_vals_ <= 0) return false;
      count_ = 0;
      __m256i ret[8], buf, buf1, buf2, buf3;
      buf = _mm256_loadu_si256((__m256i const*)buffer_end_);
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 4));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      __m256i bit, ret_mask, tmp_ret, tmp_ret1;

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[0] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[1] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 8));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 12));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[2] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[3] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 16));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 20));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[4] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[5] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 24));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[6] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[6] = _mm256_or_si256(ret[6], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[6] = _mm256_or_si256(ret[6], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[7] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[7] = _mm256_or_si256(ret[7], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[7] = _mm256_or_si256(ret[7], tmp_ret);

      tmp_ret = _mm256_unpacklo_epi8(ret[0], ret[2]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[0], ret[2]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[1], ret[3]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[1], ret[3]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[4], ret[6]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[4], ret[6]);
      ret[4] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[6] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[5], ret[7]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[5], ret[7]);
      ret[5] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[7] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);

      tmp_ret = _mm256_unpacklo_epi16(ret[0], ret[4]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[0], ret[4]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_), ret[0]);
      ret[4] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 8), ret[4]);
      tmp_ret = _mm256_unpacklo_epi16(ret[2], ret[6]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[2], ret[6]);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 16), ret[2]);
      ret[6] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 24), ret[6]);
      tmp_ret = _mm256_unpacklo_epi16(ret[1], ret[5]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[1], ret[5]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 32), ret[1]);
      ret[5] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 40), ret[5]);
      tmp_ret = _mm256_unpacklo_epi16(ret[3], ret[7]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[3], ret[7]);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 48), ret[3]);
      ret[7] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 56), ret[7]);
      buffer_end_ += bit_width_;
      num_vals_ -= 64;
    }
    *val = current_value_[count_];
    ++count_;
  }

 private:
  uint32_t current_value_[64];
};

class FleDecoder_28 : public FleDecoder {
 public:
  inline bool Get(T* val) {
    if (UNLIKELY(count_ == 64 || (num_vals_ < 0 && count_ == num_vals_ + 64))) {
      if (num_vals_ <= 0) return false;
      count_ = 0;
      __m256i ret[8], buf, buf1, buf2, buf3;
      buf = _mm256_loadu_si256((__m256i const*)buffer_end_);
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 4));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      __m256i bit, ret_mask, tmp_ret, tmp_ret1;

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[0] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[1] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 8));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 12));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[2] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[3] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 16));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 20));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[4] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[5] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 24));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[6] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[6] = _mm256_or_si256(ret[6], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[6] = _mm256_or_si256(ret[6], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[6] = _mm256_or_si256(ret[6], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[7] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[7] = _mm256_or_si256(ret[7], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[7] = _mm256_or_si256(ret[7], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[7] = _mm256_or_si256(ret[7], tmp_ret);

      tmp_ret = _mm256_unpacklo_epi8(ret[0], ret[2]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[0], ret[2]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[1], ret[3]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[1], ret[3]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[4], ret[6]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[4], ret[6]);
      ret[4] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[6] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[5], ret[7]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[5], ret[7]);
      ret[5] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[7] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);

      tmp_ret = _mm256_unpacklo_epi16(ret[0], ret[4]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[0], ret[4]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_), ret[0]);
      ret[4] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 8), ret[4]);
      tmp_ret = _mm256_unpacklo_epi16(ret[2], ret[6]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[2], ret[6]);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 16), ret[2]);
      ret[6] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 24), ret[6]);
      tmp_ret = _mm256_unpacklo_epi16(ret[1], ret[5]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[1], ret[5]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 32), ret[1]);
      ret[5] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 40), ret[5]);
      tmp_ret = _mm256_unpacklo_epi16(ret[3], ret[7]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[3], ret[7]);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 48), ret[3]);
      ret[7] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 56), ret[7]);
      buffer_end_ += bit_width_;
      num_vals_ -= 64;
    }
    *val = current_value_[count_];
    ++count_;
  }

 private:
  uint32_t current_value_[64];
};

class FleDecoder_29 : public FleDecoder {
 public:
  inline bool Get(T* val) {
    if (UNLIKELY(count_ == 64 || (num_vals_ < 0 && count_ == num_vals_ + 64))) {
      if (num_vals_ <= 0) return false;
      count_ = 0;
      __m256i ret[8], buf, buf1, buf2, buf3;
      buf = _mm256_loadu_si256((__m256i const*)buffer_end_);
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 4));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      __m256i bit, ret_mask, tmp_ret, tmp_ret1;

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[0] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[1] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 8));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 12));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[2] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[3] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 16));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 20));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[4] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[5] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 24));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 28));
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[6] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[6] = _mm256_or_si256(ret[6], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[6] = _mm256_or_si256(ret[6], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[6] = _mm256_or_si256(ret[6], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[6] = _mm256_or_si256(ret[6], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[7] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[7] = _mm256_or_si256(ret[7], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[7] = _mm256_or_si256(ret[7], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[7] = _mm256_or_si256(ret[7], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[7] = _mm256_or_si256(ret[7], tmp_ret);

      tmp_ret = _mm256_unpacklo_epi8(ret[0], ret[2]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[0], ret[2]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[1], ret[3]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[1], ret[3]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[4], ret[6]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[4], ret[6]);
      ret[4] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[6] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[5], ret[7]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[5], ret[7]);
      ret[5] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[7] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);

      tmp_ret = _mm256_unpacklo_epi16(ret[0], ret[4]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[0], ret[4]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_), ret[0]);
      ret[4] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 8), ret[4]);
      tmp_ret = _mm256_unpacklo_epi16(ret[2], ret[6]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[2], ret[6]);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 16), ret[2]);
      ret[6] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 24), ret[6]);
      tmp_ret = _mm256_unpacklo_epi16(ret[1], ret[5]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[1], ret[5]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 32), ret[1]);
      ret[5] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 40), ret[5]);
      tmp_ret = _mm256_unpacklo_epi16(ret[3], ret[7]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[3], ret[7]);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 48), ret[3]);
      ret[7] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 56), ret[7]);
      buffer_end_ += bit_width_;
      num_vals_ -= 64;
    }
    *val = current_value_[count_];
    ++count_;
  }

 private:
  uint32_t current_value_[64];
};

class FleDecoder_30 : public FleDecoder {
 public:
  inline bool Get(T* val) {
    if (UNLIKELY(count_ == 64 || (num_vals_ < 0 && count_ == num_vals_ + 64))) {
      if (num_vals_ <= 0) return false;
      count_ = 0;
      __m256i ret[8], buf, buf1, buf2, buf3;
      buf = _mm256_loadu_si256((__m256i const*)buffer_end_);
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 4));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      __m256i bit, ret_mask, tmp_ret, tmp_ret1;

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[0] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[1] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 8));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 12));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[2] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[3] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 16));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 20));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[4] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[5] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 24));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 28));
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[6] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[6] = _mm256_or_si256(ret[6], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[6] = _mm256_or_si256(ret[6], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[6] = _mm256_or_si256(ret[6], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[6] = _mm256_or_si256(ret[6], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[6] = _mm256_or_si256(ret[6], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[7] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[7] = _mm256_or_si256(ret[7], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[7] = _mm256_or_si256(ret[7], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[7] = _mm256_or_si256(ret[7], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[7] = _mm256_or_si256(ret[7], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[7] = _mm256_or_si256(ret[7], tmp_ret);

      tmp_ret = _mm256_unpacklo_epi8(ret[0], ret[2]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[0], ret[2]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[1], ret[3]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[1], ret[3]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[4], ret[6]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[4], ret[6]);
      ret[4] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[6] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[5], ret[7]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[5], ret[7]);
      ret[5] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[7] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);

      tmp_ret = _mm256_unpacklo_epi16(ret[0], ret[4]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[0], ret[4]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_), ret[0]);
      ret[4] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 8), ret[4]);
      tmp_ret = _mm256_unpacklo_epi16(ret[2], ret[6]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[2], ret[6]);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 16), ret[2]);
      ret[6] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 24), ret[6]);
      tmp_ret = _mm256_unpacklo_epi16(ret[1], ret[5]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[1], ret[5]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 32), ret[1]);
      ret[5] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 40), ret[5]);
      tmp_ret = _mm256_unpacklo_epi16(ret[3], ret[7]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[3], ret[7]);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 48), ret[3]);
      ret[7] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 56), ret[7]);
      buffer_end_ += bit_width_;
      num_vals_ -= 64;
    }
    *val = current_value_[count_];
    ++count_;
  }

 private:
  uint32_t current_value_[64];
};

class FleDecoder_31 : public FleDecoder {
 public:
  inline bool Get(T* val) {
    if (UNLIKELY(count_ == 64 || (num_vals_ < 0 && count_ == num_vals_ + 64))) {
      if (num_vals_ <= 0) return false;
      count_ = 0;
      __m256i ret[8], buf, buf1, buf2, buf3;
      buf = _mm256_loadu_si256((__m256i const*)buffer_end_);
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 4));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      __m256i bit, ret_mask, tmp_ret, tmp_ret1;

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[0] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[1] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 8));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 12));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[2] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[3] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 16));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 20));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[4] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[5] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 24));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 28));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[6] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[6] = _mm256_or_si256(ret[6], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[6] = _mm256_or_si256(ret[6], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[6] = _mm256_or_si256(ret[6], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[6] = _mm256_or_si256(ret[6], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[6] = _mm256_or_si256(ret[6], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[6] = _mm256_or_si256(ret[6], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[7] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[7] = _mm256_or_si256(ret[7], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[7] = _mm256_or_si256(ret[7], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[7] = _mm256_or_si256(ret[7], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[7] = _mm256_or_si256(ret[7], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[7] = _mm256_or_si256(ret[7], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[7] = _mm256_or_si256(ret[7], tmp_ret);

      tmp_ret = _mm256_unpacklo_epi8(ret[0], ret[2]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[0], ret[2]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[1], ret[3]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[1], ret[3]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[4], ret[6]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[4], ret[6]);
      ret[4] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[6] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[5], ret[7]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[5], ret[7]);
      ret[5] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[7] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);

      tmp_ret = _mm256_unpacklo_epi16(ret[0], ret[4]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[0], ret[4]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_), ret[0]);
      ret[4] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 8), ret[4]);
      tmp_ret = _mm256_unpacklo_epi16(ret[2], ret[6]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[2], ret[6]);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 16), ret[2]);
      ret[6] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 24), ret[6]);
      tmp_ret = _mm256_unpacklo_epi16(ret[1], ret[5]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[1], ret[5]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 32), ret[1]);
      ret[5] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 40), ret[5]);
      tmp_ret = _mm256_unpacklo_epi16(ret[3], ret[7]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[3], ret[7]);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 48), ret[3]);
      ret[7] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 56), ret[7]);
      buffer_end_ += bit_width_;
      num_vals_ -= 64;
    }
    *val = current_value_[count_];
    ++count_;
  }

 private:
  uint32_t current_value_[64];
};

class FleDecoder_32 : public FleDecoder {
 public:
  inline bool Get(T* val) {
    if (UNLIKELY(count_ == 64 || (num_vals_ < 0 && count_ == num_vals_ + 64))) {
      if (num_vals_ <= 0) return false;
      count_ = 0;
      __m256i ret[8], buf, buf1, buf2, buf3;
      buf = _mm256_loadu_si256((__m256i const*)buffer_end_);
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 4));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      __m256i bit, ret_mask, tmp_ret, tmp_ret1;

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[0] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[0] = _mm256_or_si256(ret[0], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[1] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[1] = _mm256_or_si256(ret[1], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 8));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 12));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[2] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[2] = _mm256_or_si256(ret[2], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[3] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[3] = _mm256_or_si256(ret[3], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 16));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 20));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[4] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[4] = _mm256_or_si256(ret[4], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[5] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[5] = _mm256_or_si256(ret[5], tmp_ret);

      buf = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 24));
      buf1 = _mm256_permute4x64_epi64(buf, 0xee);
      buf = _mm256_permute4x64_epi64(buf, 0x44);
      buf2 = _mm256_loadu_si256((__m256i const*)(buffer_end_ + 28));
      buf3 = _mm256_permute4x64_epi64(buf2, 0xee);
      buf2 = _mm256_permute4x64_epi64(buf2, 0x44);

      bit = _mm256_shuffle_epi8(buf, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[6] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[6] = _mm256_or_si256(ret[6], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[6] = _mm256_or_si256(ret[6], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[6] = _mm256_or_si256(ret[6], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[6] = _mm256_or_si256(ret[6], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[6] = _mm256_or_si256(ret[6], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf0_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[6] = _mm256_or_si256(ret[6], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf2_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[6] = _mm256_or_si256(ret[6], tmp_ret);

      bit = _mm256_shuffle_epi8(buf, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      ret[7] = _mm256_and_si256(bitv[0], ret_mask);
      bit = _mm256_shuffle_epi8(buf, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[1], ret_mask);
      ret[7] = _mm256_or_si256(ret[7], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[2], ret_mask);
      ret[7] = _mm256_or_si256(ret[7], tmp_ret);
      bit = _mm256_shuffle_epi8(buf1, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[3], ret_mask);
      ret[7] = _mm256_or_si256(ret[7], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[4], ret_mask);
      ret[7] = _mm256_or_si256(ret[7], tmp_ret);
      bit = _mm256_shuffle_epi8(buf2, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[5], ret_mask);
      ret[7] = _mm256_or_si256(ret[7], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf1_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[6], ret_mask);
      ret[7] = _mm256_or_si256(ret[7], tmp_ret);
      bit = _mm256_shuffle_epi8(buf3, shf3_mask);
      bit = _mm256_and_si256(bit, clr_mask);
      ret_mask = _mm256_cmpeq_epi8(bit, clr_mask);
      tmp_ret = _mm256_and_si256(bitv[7], ret_mask);
      ret[7] = _mm256_or_si256(ret[7], tmp_ret);

      tmp_ret = _mm256_unpacklo_epi8(ret[0], ret[2]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[0], ret[2]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[1], ret[3]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[1], ret[3]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[4], ret[6]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[4], ret[6]);
      ret[4] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[6] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      tmp_ret = _mm256_unpacklo_epi8(ret[5], ret[7]);
      tmp_ret1 = _mm256_unpackhi_epi8(ret[5], ret[7]);
      ret[5] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      ret[7] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);

      tmp_ret = _mm256_unpacklo_epi16(ret[0], ret[4]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[0], ret[4]);
      ret[0] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_), ret[0]);
      ret[4] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 8), ret[4]);
      tmp_ret = _mm256_unpacklo_epi16(ret[2], ret[6]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[2], ret[6]);
      ret[2] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 16), ret[2]);
      ret[6] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 24), ret[6]);
      tmp_ret = _mm256_unpacklo_epi16(ret[1], ret[5]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[1], ret[5]);
      ret[1] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 32), ret[1]);
      ret[5] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 40), ret[5]);
      tmp_ret = _mm256_unpacklo_epi16(ret[3], ret[7]);
      tmp_ret1 = _mm256_unpackhi_epi16(ret[3], ret[7]);
      ret[3] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x20);
      _mm256_storeu_si256((__m256i*)(current_value_ + 48), ret[3]);
      ret[7] = _mm256_permute2x128_si256(tmp_ret, tmp_ret1, 0x31);
      _mm256_storeu_si256((__m256i*)(current_value_ + 56), ret[7]);
      buffer_end_ += bit_width_;
      num_vals_ -= 64;
    }
    *val = current_value_[count_];
    ++count_;
  }

  inline void Eq(uint64_t value, int64_t num_rows,
      vector<int64_t>& skip_bit_vec, int flag) {
    uint64_t j = 0;
    while (num_rows > 0) {
      uint64_t Meq = ~0x0;
      for (int i = 0; i < bit_width_; ++i) {
        uint64_t C;
        if (value & bitv[i] == bitv[i]) {
          C = ~0x0;
        } else {
          C = 0x0;
        }
        Meq = Meq & ~(buffer_end_[i] ^ C);
      }
      switch() {
        case 0:
          skip_bit_vec[j] = Meq;
          break;
        case 1:
          skip_bit_vec[j] &= Meq;
          break;
        case 2:
          skip_bit_vec[j] |= Meq;
          break;
      }
      ++j;
      num_rows -= 64;
    }
  }

  inline void Lt(uint64_t value, int64_t num_rows,
      vector<int64_t>& skip_bit_vec, int flag) {
    uint64_t j = 0;
    while (num_rows > 0) {
      uint64_t Mlt = 0x0;
      uint64_t Meq = ~0x0;
      for (int i = 0; i < bit_width_; ++i) {
        uint64_t C;
        if (value & bitv[i] == bitv[i]) {
          C = ~0x0;
        } else {
          C = 0x0;
        }
        Mlt = Mlt | (Meq & C & ~buffer_end_[i]);
        Meq = Meq & ~(buffer_end_[i] ^ C);
      }
      switch() {
        case 0:
          skip_bit_vec[j] = Mlt;
          break;
        case 1:
          skip_bit_vec[j] &= Mlt;
          break;
        case 2:
          skip_bit_vec[j] |= Mlt;
          break;
      }
      ++j;
      num_rows -= 64;
    }
  }

  inline void FleDecoder::Le(uint64_t value, int64_t num_rows,
      vector<int64_t>& skip_bit_vec, int flag) {
    uint64_t j = 0;
    while (num_rows > 0) {
      __m256i Mlt = 0x0;
      __m256i Meq = ~0x0;
      for (int i = 0; i < bit_width_; ++i) {
        uint64_t C;
        if (value & bitv[i] == bitv[i]) {
          C = ~0x0;
        } else {
          C = 0x0;
        }
        Mlt = Mlt | (Meq & C & ~buffer_end_[i]);
        Meq = Meq & ~(buffer_end_[i] ^ C);
      }
      switch() {
        case 0:
          skip_bit_vec[j] = Mle;
          break;
        case 1:
          skip_bit_vec[j] &= Mle;
          break;
        case 2:
          skip_bit_vec[j] |= Mle;
          break;
      }
      ++j;
      num_rows -= 64;
    }
  }

  inline void FleDecoder::Gt(uint64_t value, int64_t num_rows,
      vector<int64_t>& skip_bit_vec, int flag) {
    uint64_t i = 0;
    while (num_rows > 0) {
      __m256i Mgt = 0x0;
      __m256i Meq = ~0x0;
      for (int i = 0; i < bit_width_; ++i) {
        uint64_t C;
        if (value & bitv[i] == bitv[i]) {
          C = ~0x0;
        } else {
          C = 0x0;
        }
        Mgt = Mgt | (Meq & ~C & buffer_end_[i]);
        Meq = Meq & ~(buffer_end_[i] ^ C);
      }
      switch() {
        case 0:
          skip_bit_vec[j] = MGt;
          break;
        case 1:
          skip_bit_vec[j] &= MGt;
          break;
        case 2:
          skip_bit_vec[j] |= MGt;
          break;
      }
      ++j;
      num_rows -= 64;
    }
  }

  inline void FleDecoder::Ge(uint64_t value, int64_t num_rows,
      vector<int64_t>& skip_bit_vec, int flag) {
    uint64_t j = 0;
    while (num_rows > 0) {
      __m256i Mgt = 0x0;
      __m256i Meq = ~0x0;
      for (int i = 0; i < bit_width_; ++i) {
        uint64_t C;
        if (value & bitv[i] == bitv[i]) {
          C = ~0x0;
        } else {
          C = 0x0;
        }
        Mgt = Mgt | (Meq & ~C & buffer_end_[i]);
        Meq = Meq & ~(buffer_end_[i] ^ C);
      }
      switch() {
        case 0:
          skip_bit_vec[j] = MGe;
          break;
        case 1:
          skip_bit_vec[j] &= MGe;
          break;
        case 2:
          skip_bit_vec[j] |= MGe;
          break;
      }
      ++j;
      num_rows -= 64;
    }
  }

 private:
  uint32_t current_value_[64];
};

class FleEncoder {
 public:
  FleEncoder(uint8_t* buffer, int buffer_len, int bit_width)
    : bit_width_(bit_width) {
    DCHECK_GE(bit_width_, 1);
    DCHECK_LE(bit_width_, 64);
    DCHECK_GE(buffer_len % 8, 0);
    buffer_ = reinterpret_cast<uint64_t*>(buffer);
    buffer_end_ = buffer_;
    buffer_end_guard_ = buffer_ + buffer_len / 8 - bit_width;
    count_ = 0;
    pcurrent_value8_ = reinterpret_cast<uint8_t*>(current_value_);
    pcurrent_value16_ = reinterpret_cast<uint16_t*>(current_value_);

    fa[1] = &FleEncoder::Pack_1;
    fa[2] = &FleEncoder::Pack_2;
    fa[3] = &FleEncoder::Pack_3;
    fa[4] = &FleEncoder::Pack_4;
    fa[5] = &FleEncoder::Pack_5;
    fa[6] = &FleEncoder::Pack_6;
    fa[7] = &FleEncoder::Pack_7;
    fa[8] = &FleEncoder::Pack_8;
    fa[9] = &FleEncoder::Pack_9;
    fa[10] = &FleEncoder::Pack_10;
    fa[11] = &FleEncoder::Pack_11;
    fa[12] = &FleEncoder::Pack_12;
    fa[13] = &FleEncoder::Pack_13;
    fa[14] = &FleEncoder::Pack_14;
    fa[15] = &FleEncoder::Pack_15;
    fa[16] = &FleEncoder::Pack_16;

    fa[17] = &FleEncoder::Pack_16_32;
    fa[18] = &FleEncoder::Pack_16_32;
    fa[19] = &FleEncoder::Pack_16_32;
    fa[20] = &FleEncoder::Pack_16_32;
    fa[21] = &FleEncoder::Pack_16_32;
    fa[22] = &FleEncoder::Pack_16_32;
    fa[23] = &FleEncoder::Pack_16_32;
    fa[24] = &FleEncoder::Pack_16_32;
    fa[25] = &FleEncoder::Pack_16_32;
    fa[26] = &FleEncoder::Pack_16_32;
    fa[27] = &FleEncoder::Pack_16_32;
    fa[28] = &FleEncoder::Pack_16_32;
    fa[29] = &FleEncoder::Pack_16_32;
    fa[30] = &FleEncoder::Pack_16_32;
    fa[31] = &FleEncoder::Pack_16_32;
    fa[32] = &FleEncoder::Pack_16_32;

    for (int i = 0; i < 8; ++i) {
      bitv[i] = _mm256_set1_epi8(0x01 << i);
    }

    clr_mask = _mm256_set1_epi64x(0x00ff00ff00ff00ff);
    clr32_mask = _mm256_set1_epi64x(0x0000ffff0000ffff);
  }

  bool Put(uint64_t value);

  int Flush();

  uint8_t* buffer() { return reinterpret_cast<uint8_t*>(buffer_); }
  int len() { return (buffer_end_ - buffer_) * 8;};

  void Pack_1();
  void Pack_2();
  void Pack_3();
  void Pack_4();
  void Pack_5();
  void Pack_6();
  void Pack_7();
  void Pack_8();
  void Pack_9();
  void Pack_10();
  void Pack_11();
  void Pack_12();
  void Pack_13();
  void Pack_14();
  void Pack_15();
  void Pack_16();
  void Pack_16_32();

 private:
  typedef void(impala::FleEncoder::*PPACK)();
  PPACK fa[33];
  __m256i bitv[8];
  __m256i clr_mask;
  __m256i clr32_mask;
  uint64_t* buffer_;
  uint64_t* buffer_end_;
  uint64_t* buffer_end_guard_;
  uint32_t current_value_[64];
  uint8_t* pcurrent_value8_;
  uint16_t* pcurrent_value16_;
  int bit_width_;
  int count_;
};

inline void FleDecoder::Lt(uint64_t value, int64_t num_rows,
    vector<int64_t>& skip_bit_vec, int flag) {
  uint64_t i = 0;
  while (num_rows > 0) {
    __m256i Mlt = _mm256_set1_epi32(0x0);
    __m256i Meq = _mm256_set1_epi32(~0x0);
    for (int i = 0; i < bit_width_; ++i) {
      __m256i C;
      if (value & bitv[i] == bitv[i]) {
        C = _mm256_set1_epi32(~0x0);
      } else {
        C = _mm256_set1_epi32(0x0);
      }
      __m256i v  = _mm256_loadu_si256((__m256i const*)buffer_end_);
      Mlt = _mm256_or_si256(Mlt, _mm256_and_si256(Meq, _mm256_andnot_si256(C, v)));
      Meq = _mm256_andnot_si256(Meq, _mm256_xor_si256(v, C));
    }
    _mm256_storeu_si256((__m256i*)(skip_bit_vec[i]), Mlt);
    i += 4;
    num_rows -= 256;
  }
}

inline void FleDecoder::Le(uint64_t value, int64_t num_rows,
    vector<int64_t>& skip_bit_vec, int flag) {
  uint64_t i = 0;
  while (num_rows > 0) {
    __m256i Mlt = _mm256_set1_epi32(0x0);
    __m256i Meq = _mm256_set1_epi32(~0x0);
    for (int i = 0; i < bit_width_; ++i) {
      __m256i C;
      if (value & bitv[i] == bitv[i]) {
        C = _mm256_set1_epi32(~0x0);
      } else {
        C = _mm256_set1_epi32(0x0);
      }
      __m256i v  = _mm256_loadu_si256((__m256i const*)buffer_end_);
      Mlt = _mm256_or_si256(Mlt, _mm256_and_si256(Meq, _mm256_andnot_si256(C, v)));
      Meq = _mm256_andnot_si256(Meq, _mm256_xor_si256(v, C));
    }
    _mm256_storeu_si256((__m256i*)(skip_bit_vec[i]), _mm256_and_si256(Mlt, Meq));
    i += 4;
    num_rows -= 256;
  }
}

inline void FleDecoder::Gt(uint64_t value, int64_t num_rows,
    vector<int64_t>& skip_bit_vec, int flag) {
  uint64_t i = 0;
  while (num_rows > 0) {
    __m256i Mgt = _mm256_set1_epi32(0x0);
    __m256i Meq = _mm256_set1_epi32(~0x0);
    for (int i = 0; i < bit_width_; ++i) {
      __m256i C;
      if (value & bitv[i] == bitv[i]) {
        C = _mm256_set1_epi32(~0x0);
      } else {
        C = _mm256_set1_epi32(0x0);
      }
      __m256i v  = _mm256_loadu_si256((__m256i const*)buffer_end_);
      Mgt = _mm256_or_si256(Mgt, _mm256_and_si256(_mm256_andnot_si256(Meq, C), v));
      Meq = _mm256_andnot_si256(Meq, _mm256_xor_si256(v, C));
    }
    _mm256_storeu_si256((__m256i*)(skip_bit_vec[i]), Mgt);
    i += 4;
    num_rows -= 256;
  }
}

inline void FleDecoder::Ge(uint64_t value, int64_t num_rows,
    vector<int64_t>& skip_bit_vec, int flag) {
  uint64_t i = 0;
  while (num_rows > 0) {
    __m256i Mgt = _mm256_set1_epi32(0x0);
    __m256i Meq = _mm256_set1_epi32(~0x0);
    for (int i = 0; i < bit_width_; ++i) {
      __m256i C;
      if (value & bitv[i] == bitv[i]) {
        C = _mm256_set1_epi32(~0x0);
      } else {
        C = _mm256_set1_epi32(0x0);
      }
      __m256i v  = _mm256_loadu_si256((__m256i const*)buffer_end_);
      Mgt = _mm256_or_si256(Mgt, _mm256_and_si256(_mm256_andnot_si256(Meq, C), v));
      Meq = _mm256_andnot_si256(Meq, _mm256_xor_si256(v, C));
    }
    _mm256_storeu_si256((__m256i*)(skip_bit_vec[i]), _mm256_and_si256(Mgt, Meq));
    i += 4;
    num_rows -= 256;
  }
}

inline bool FleEncoder::Put(uint64_t value) {
  DCHECK(bit_width_ == 64 || value < (1LL << bit_width_));
  if (UNLIKELY(count_ == 64)) {
    if (buffer_end_ > buffer_end_guard_) return false;

//    for (int i = 0; i < bit_width_; ++i) {
//      buffer_end_[i] = 0;
//    }

    (this->*fa[bit_width_])();

    buffer_end_ += bit_width_;
    count_ = 0;
  }

//  uint64_t set1 = 0x01ULL << (63 - count_);
//  uint64_t index = _tzcnt_u64(value);
//  while (index != 64) {
//    buffer_end_[index] |= set1;
//    value = _blsr_u64(value);
//    index = _tzcnt_u64(value);
//  }

//  for (int l = 0; l < bit_width_; ++l) {
//    buffer_end_[l] |= ((value >> l) & 0x01ULL) << (63 - count_);
//  }

//  uint64_t set1 = 0x01ULL << (63 - count_);
//  for (int l = 0; l < bit_width_; ++l) {
//    if (value & (0x01ULL << l)) buffer_end_[l] |= set1;
//  }


//  for (int i = 0; i < 64 -  _lzcnt_u64(value); ++i) {
//  for (int i = 0; i < bit_width_; ++i) {
//    *current_value_[i] |= ((value >> i) & 0x01) << (63 - count_);
//  }


  if (bit_width_ <= 8) {
    pcurrent_value8_[63 - count_] = value;
  } else if (bit_width_ <= 16) {
    pcurrent_value16_[63 - count_] = value;
  } else {
    current_value_[63 - count_] = value;
//    current_value_[count_] = value;
  }

  ++count_;
  return true;
}

inline void FleEncoder::Pack_1() {
  __m256i dat, ret;
  dat = _mm256_loadu_si256((__m256i const*)pcurrent_value8_);
  ret = _mm256_cmpeq_epi8(dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[0] = _mm256_movemask_epi8(ret);
  dat = _mm256_loadu_si256((__m256i const*)(pcurrent_value8_ + 32));
  ret = _mm256_cmpeq_epi8(dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[1] = _mm256_movemask_epi8(ret);
}

inline void FleEncoder::Pack_2() {
  __m256i dat, dat1, tmp_dat, ret;
  dat = _mm256_loadu_si256((__m256i const*)pcurrent_value8_);
  dat1 = _mm256_loadu_si256((__m256i const*)(pcurrent_value8_ + 32));

  tmp_dat = _mm256_and_si256(dat, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[0] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[1] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[2] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[3] = _mm256_movemask_epi8(ret);
}

inline void FleEncoder::Pack_3() {
  __m256i dat, dat1, tmp_dat, ret;
  dat = _mm256_loadu_si256((__m256i const*)pcurrent_value8_);
  dat1 = _mm256_loadu_si256((__m256i const*)(pcurrent_value8_ + 32));

  tmp_dat = _mm256_and_si256(dat, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[0] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[1] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[2] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[3] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[4] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[5] = _mm256_movemask_epi8(ret);
}

inline void FleEncoder::Pack_4() {
  __m256i dat, dat1, tmp_dat, ret;
  dat = _mm256_loadu_si256((__m256i const*)pcurrent_value8_);
  dat1 = _mm256_loadu_si256((__m256i const*)(pcurrent_value8_ + 32));

  tmp_dat = _mm256_and_si256(dat, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[0] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[1] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[2] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[3] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[4] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[5] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[6] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[7] = _mm256_movemask_epi8(ret);
}

inline void FleEncoder::Pack_5() {
  __m256i dat, dat1, tmp_dat, ret;
  dat = _mm256_loadu_si256((__m256i const*)pcurrent_value8_);
  dat1 = _mm256_loadu_si256((__m256i const*)(pcurrent_value8_ + 32));

  tmp_dat = _mm256_and_si256(dat, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[0] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[1] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[2] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[3] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[4] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[5] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[6] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[7] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat, bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[8] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[9] = _mm256_movemask_epi8(ret);
}

inline void FleEncoder::Pack_6() {
  __m256i dat, dat1, tmp_dat, ret;
  dat = _mm256_loadu_si256((__m256i const*)pcurrent_value8_);
  dat1 = _mm256_loadu_si256((__m256i const*)(pcurrent_value8_ + 32));

  tmp_dat = _mm256_and_si256(dat, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[0] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[1] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[2] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[3] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[4] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[5] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[6] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[7] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat, bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[8] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[9] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat, bitv[5]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
  reinterpret_cast<uint32_t*>(buffer_end_)[10] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[5]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
  reinterpret_cast<uint32_t*>(buffer_end_)[11] = _mm256_movemask_epi8(ret);
}

inline void FleEncoder::Pack_7() {
  __m256i dat, dat1, tmp_dat, ret;
  dat = _mm256_loadu_si256((__m256i const*)pcurrent_value8_);
  dat1 = _mm256_loadu_si256((__m256i const*)(pcurrent_value8_ + 32));

  tmp_dat = _mm256_and_si256(dat, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[0] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[1] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[2] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[3] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[4] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[5] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[6] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[7] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat, bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[8] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[9] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat, bitv[5]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
  reinterpret_cast<uint32_t*>(buffer_end_)[10] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[5]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
  reinterpret_cast<uint32_t*>(buffer_end_)[11] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat, bitv[6]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[6]);
  reinterpret_cast<uint32_t*>(buffer_end_)[12] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[6]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[6]);
  reinterpret_cast<uint32_t*>(buffer_end_)[13] = _mm256_movemask_epi8(ret);
}

inline void FleEncoder::Pack_8() {
  __m256i dat, dat1, tmp_dat, ret;
  dat = _mm256_loadu_si256((__m256i const*)pcurrent_value8_);
  dat1 = _mm256_loadu_si256((__m256i const*)(pcurrent_value8_ + 32));

  tmp_dat = _mm256_and_si256(dat, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[0] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[1] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[2] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[3] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[4] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[5] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[6] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[7] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat, bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[8] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[9] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat, bitv[5]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
  reinterpret_cast<uint32_t*>(buffer_end_)[10] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[5]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
  reinterpret_cast<uint32_t*>(buffer_end_)[11] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat, bitv[6]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[6]);
  reinterpret_cast<uint32_t*>(buffer_end_)[12] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[6]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[6]);
  reinterpret_cast<uint32_t*>(buffer_end_)[13] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat, bitv[7]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[7]);
  reinterpret_cast<uint32_t*>(buffer_end_)[14] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat1, bitv[7]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[7]);
  reinterpret_cast<uint32_t*>(buffer_end_)[15] = _mm256_movemask_epi8(ret);
}

inline void FleEncoder::Pack_9() {
  __m256i dat, dat1, dat2, dat3, pack_dat, pack_dat1, tmp_dat, tmp_dat1, ret;
  dat = _mm256_loadu_si256((__m256i const*)(pcurrent_value16_));
  dat1 = _mm256_loadu_si256((__m256i const*)(pcurrent_value16_ + 16));
  tmp_dat = _mm256_and_si256(dat, clr_mask);
  tmp_dat1 = _mm256_and_si256(dat1, clr_mask);
  pack_dat = _mm256_packus_epi16(tmp_dat, tmp_dat1);
  pack_dat = _mm256_permute4x64_epi64(pack_dat, 0xd8);
  dat2 = _mm256_loadu_si256((__m256i const*)(pcurrent_value16_ + 32));
  dat3 = _mm256_loadu_si256((__m256i const*)(pcurrent_value16_ + 48));
  tmp_dat = _mm256_and_si256(dat2, clr_mask);
  tmp_dat1 = _mm256_and_si256(dat3, clr_mask);
  pack_dat1 = _mm256_packus_epi16(tmp_dat, tmp_dat1);
  pack_dat1 = _mm256_permute4x64_epi64(pack_dat1, 0xd8);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[0] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[1] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[2] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[3] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[4] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[5] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[6] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[7] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[8] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[9] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[5]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
  reinterpret_cast<uint32_t*>(buffer_end_)[10] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[5]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
  reinterpret_cast<uint32_t*>(buffer_end_)[11] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[6]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[6]);
  reinterpret_cast<uint32_t*>(buffer_end_)[12] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[6]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[6]);
  reinterpret_cast<uint32_t*>(buffer_end_)[13] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[7]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[7]);
  reinterpret_cast<uint32_t*>(buffer_end_)[14] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[7]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[7]);
  reinterpret_cast<uint32_t*>(buffer_end_)[15] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_srli_si256(dat, 1);
  tmp_dat = _mm256_and_si256(tmp_dat, clr_mask);
  tmp_dat1 = _mm256_srli_si256(dat1, 1);
  tmp_dat1 = _mm256_and_si256(tmp_dat1, clr_mask);
  pack_dat = _mm256_packus_epi16(tmp_dat, tmp_dat1);
  pack_dat = _mm256_permute4x64_epi64(pack_dat, 0xd8);
  tmp_dat = _mm256_srli_si256(dat2, 1);
  tmp_dat = _mm256_and_si256(tmp_dat, clr_mask);
  tmp_dat1 = _mm256_srli_si256(dat3, 1);
  tmp_dat1 = _mm256_and_si256(tmp_dat1, clr_mask);
  pack_dat1 = _mm256_packus_epi16(tmp_dat, tmp_dat1);
  pack_dat1 = _mm256_permute4x64_epi64(pack_dat1, 0xd8);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[16] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[17] = _mm256_movemask_epi8(ret);
}


inline void FleEncoder::Pack_10() {
  __m256i dat, dat1, dat2, dat3, pack_dat, pack_dat1, tmp_dat, tmp_dat1, ret;
  dat = _mm256_loadu_si256((__m256i const*)(pcurrent_value16_));
  dat1 = _mm256_loadu_si256((__m256i const*)(pcurrent_value16_ + 16));
  tmp_dat = _mm256_and_si256(dat, clr_mask);
  tmp_dat1 = _mm256_and_si256(dat1, clr_mask);
  pack_dat = _mm256_packus_epi16(tmp_dat, tmp_dat1);
  pack_dat = _mm256_permute4x64_epi64(pack_dat, 0xd8);
  dat2 = _mm256_loadu_si256((__m256i const*)(pcurrent_value16_ + 32));
  dat3 = _mm256_loadu_si256((__m256i const*)(pcurrent_value16_ + 48));
  tmp_dat = _mm256_and_si256(dat2, clr_mask);
  tmp_dat1 = _mm256_and_si256(dat3, clr_mask);
  pack_dat1 = _mm256_packus_epi16(tmp_dat, tmp_dat1);
  pack_dat1 = _mm256_permute4x64_epi64(pack_dat1, 0xd8);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[0] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[1] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[2] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[3] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[4] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[5] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[6] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[7] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[8] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[9] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[5]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
  reinterpret_cast<uint32_t*>(buffer_end_)[10] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[5]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
  reinterpret_cast<uint32_t*>(buffer_end_)[11] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[6]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[6]);
  reinterpret_cast<uint32_t*>(buffer_end_)[12] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[6]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[6]);
  reinterpret_cast<uint32_t*>(buffer_end_)[13] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[7]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[7]);
  reinterpret_cast<uint32_t*>(buffer_end_)[14] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[7]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[7]);
  reinterpret_cast<uint32_t*>(buffer_end_)[15] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_srli_si256(dat, 1);
  tmp_dat = _mm256_and_si256(tmp_dat, clr_mask);
  tmp_dat1 = _mm256_srli_si256(dat1, 1);
  tmp_dat1 = _mm256_and_si256(tmp_dat1, clr_mask);
  pack_dat = _mm256_packus_epi16(tmp_dat, tmp_dat1);
  pack_dat = _mm256_permute4x64_epi64(pack_dat, 0xd8);
  tmp_dat = _mm256_srli_si256(dat2, 1);
  tmp_dat = _mm256_and_si256(tmp_dat, clr_mask);
  tmp_dat1 = _mm256_srli_si256(dat3, 1);
  tmp_dat1 = _mm256_and_si256(tmp_dat1, clr_mask);
  pack_dat1 = _mm256_packus_epi16(tmp_dat, tmp_dat1);
  pack_dat1 = _mm256_permute4x64_epi64(pack_dat1, 0xd8);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[16] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[17] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[18] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[19] = _mm256_movemask_epi8(ret);
}

inline void FleEncoder::Pack_11() {
  __m256i dat, dat1, dat2, dat3, tmp_dat, tmp_dat1, pack_dat, pack_dat1, ret;
  dat = _mm256_loadu_si256((__m256i const*)pcurrent_value16_);
  dat1 = _mm256_loadu_si256((__m256i const*)(pcurrent_value16_ + 16));
  tmp_dat = _mm256_and_si256(dat, clr_mask);
  tmp_dat1 = _mm256_and_si256(dat1, clr_mask);
  pack_dat = _mm256_packus_epi16(tmp_dat, tmp_dat1);
  pack_dat = _mm256_permute4x64_epi64(pack_dat, 0xd8);
  dat2 = _mm256_loadu_si256((__m256i const*)(pcurrent_value16_ + 32));
  dat3 = _mm256_loadu_si256((__m256i const*)(pcurrent_value16_ + 48));
  tmp_dat = _mm256_and_si256(dat2, clr_mask);
  tmp_dat1 = _mm256_and_si256(dat3, clr_mask);
  pack_dat1 = _mm256_packus_epi16(tmp_dat, tmp_dat1);
  pack_dat1 = _mm256_permute4x64_epi64(pack_dat1, 0xd8);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[0] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[1] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[2] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[3] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[4] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[5] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[6] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[7] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[8] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[9] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[5]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
  reinterpret_cast<uint32_t*>(buffer_end_)[10] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[5]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
  reinterpret_cast<uint32_t*>(buffer_end_)[11] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[6]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[6]);
  reinterpret_cast<uint32_t*>(buffer_end_)[12] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[6]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[6]);
  reinterpret_cast<uint32_t*>(buffer_end_)[13] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[7]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[7]);
  reinterpret_cast<uint32_t*>(buffer_end_)[14] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[7]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[7]);
  reinterpret_cast<uint32_t*>(buffer_end_)[15] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_srli_si256(dat, 1);
  tmp_dat = _mm256_and_si256(tmp_dat, clr_mask);
  tmp_dat1 = _mm256_srli_si256(dat1, 1);
  tmp_dat1 = _mm256_and_si256(tmp_dat1, clr_mask);
  pack_dat = _mm256_packus_epi16(tmp_dat, tmp_dat1);
  pack_dat = _mm256_permute4x64_epi64(pack_dat, 0xd8);
  tmp_dat = _mm256_srli_si256(dat2, 1);
  tmp_dat = _mm256_and_si256(tmp_dat, clr_mask);
  tmp_dat1 = _mm256_srli_si256(dat3, 1);
  tmp_dat1 = _mm256_and_si256(tmp_dat1, clr_mask);
  pack_dat1 = _mm256_packus_epi16(tmp_dat, tmp_dat1);
  pack_dat1 = _mm256_permute4x64_epi64(pack_dat1, 0xd8);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[16] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[17] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[18] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[19] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[20] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[21] = _mm256_movemask_epi8(ret);
}


inline void FleEncoder::Pack_12() {
  __m256i dat, dat1, dat2, dat3, tmp_dat, tmp_dat1, pack_dat, pack_dat1, ret;
  dat = _mm256_loadu_si256((__m256i const*)pcurrent_value16_);
  dat1 = _mm256_loadu_si256((__m256i const*)(pcurrent_value16_ + 16));
  tmp_dat = _mm256_and_si256(dat, clr_mask);
  tmp_dat1 = _mm256_and_si256(dat1, clr_mask);
  pack_dat = _mm256_packus_epi16(tmp_dat, tmp_dat1);
  pack_dat = _mm256_permute4x64_epi64(pack_dat, 0xd8);
  dat2 = _mm256_loadu_si256((__m256i const*)(pcurrent_value16_ + 32));
  dat3 = _mm256_loadu_si256((__m256i const*)(pcurrent_value16_ + 48));
  tmp_dat = _mm256_and_si256(dat2, clr_mask);
  tmp_dat1 = _mm256_and_si256(dat3, clr_mask);
  pack_dat1 = _mm256_packus_epi16(tmp_dat, tmp_dat1);
  pack_dat1 = _mm256_permute4x64_epi64(pack_dat1, 0xd8);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[0] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[1] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[2] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[3] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[4] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[5] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[6] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[7] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[8] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[9] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[5]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
  reinterpret_cast<uint32_t*>(buffer_end_)[10] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[5]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
  reinterpret_cast<uint32_t*>(buffer_end_)[11] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[6]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[6]);
  reinterpret_cast<uint32_t*>(buffer_end_)[12] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[6]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[6]);
  reinterpret_cast<uint32_t*>(buffer_end_)[13] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[7]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[7]);
  reinterpret_cast<uint32_t*>(buffer_end_)[14] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[7]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[7]);
  reinterpret_cast<uint32_t*>(buffer_end_)[15] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_srli_si256(dat, 1);
  tmp_dat = _mm256_and_si256(tmp_dat, clr_mask);
  tmp_dat1 = _mm256_srli_si256(dat1, 1);
  tmp_dat1 = _mm256_and_si256(tmp_dat1, clr_mask);
  pack_dat = _mm256_packus_epi16(tmp_dat, tmp_dat1);
  pack_dat = _mm256_permute4x64_epi64(pack_dat, 0xd8);
  tmp_dat = _mm256_srli_si256(dat2, 1);
  tmp_dat = _mm256_and_si256(tmp_dat, clr_mask);
  tmp_dat1 = _mm256_srli_si256(dat3, 1);
  tmp_dat1 = _mm256_and_si256(tmp_dat1, clr_mask);
  pack_dat1 = _mm256_packus_epi16(tmp_dat, tmp_dat1);
  pack_dat1 = _mm256_permute4x64_epi64(pack_dat1, 0xd8);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[16] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[17] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[18] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[19] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[20] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[21] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[22] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[23] = _mm256_movemask_epi8(ret);
}

inline void FleEncoder::Pack_13() {
  __m256i dat, dat1, dat2, dat3, tmp_dat, tmp_dat1, pack_dat, pack_dat1, ret;
  dat = _mm256_loadu_si256((__m256i const*)pcurrent_value16_);
  dat1 = _mm256_loadu_si256((__m256i const*)(pcurrent_value16_ + 16));
  tmp_dat = _mm256_and_si256(dat, clr_mask);
  tmp_dat1 = _mm256_and_si256(dat1, clr_mask);
  pack_dat = _mm256_packus_epi16(tmp_dat, tmp_dat1);
  pack_dat = _mm256_permute4x64_epi64(pack_dat, 0xd8);
  dat2 = _mm256_loadu_si256((__m256i const*)(pcurrent_value16_ + 32));
  dat3 = _mm256_loadu_si256((__m256i const*)(pcurrent_value16_ + 48));
  tmp_dat = _mm256_and_si256(dat2, clr_mask);
  tmp_dat1 = _mm256_and_si256(dat3, clr_mask);
  pack_dat1 = _mm256_packus_epi16(tmp_dat, tmp_dat1);
  pack_dat1 = _mm256_permute4x64_epi64(pack_dat1, 0xd8);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[0] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[1] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[2] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[3] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[4] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[5] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[6] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[7] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[8] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[9] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[5]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
  reinterpret_cast<uint32_t*>(buffer_end_)[10] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[5]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
  reinterpret_cast<uint32_t*>(buffer_end_)[11] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[6]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[6]);
  reinterpret_cast<uint32_t*>(buffer_end_)[12] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[6]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[6]);
  reinterpret_cast<uint32_t*>(buffer_end_)[13] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[7]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[7]);
  reinterpret_cast<uint32_t*>(buffer_end_)[14] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[7]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[7]);
  reinterpret_cast<uint32_t*>(buffer_end_)[15] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_srli_si256(dat, 1);
  tmp_dat = _mm256_and_si256(tmp_dat, clr_mask);
  tmp_dat1 = _mm256_srli_si256(dat1, 1);
  tmp_dat1 = _mm256_and_si256(tmp_dat1, clr_mask);
  pack_dat = _mm256_packus_epi16(tmp_dat, tmp_dat1);
  pack_dat = _mm256_permute4x64_epi64(pack_dat, 0xd8);
  tmp_dat = _mm256_srli_si256(dat2, 1);
  tmp_dat = _mm256_and_si256(tmp_dat, clr_mask);
  tmp_dat1 = _mm256_srli_si256(dat3, 1);
  tmp_dat1 = _mm256_and_si256(tmp_dat1, clr_mask);
  pack_dat1 = _mm256_packus_epi16(tmp_dat, tmp_dat1);
  pack_dat1 = _mm256_permute4x64_epi64(pack_dat1, 0xd8);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[16] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[17] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[18] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[19] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[20] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[21] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[22] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[23] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[24] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[25] = _mm256_movemask_epi8(ret);
}

inline void FleEncoder::Pack_14() {
  __m256i dat, dat1, dat2, dat3, tmp_dat, tmp_dat1, pack_dat, pack_dat1, ret;
  dat = _mm256_loadu_si256((__m256i const*)pcurrent_value16_);
  dat1 = _mm256_loadu_si256((__m256i const*)(pcurrent_value16_ + 16));
  tmp_dat = _mm256_and_si256(dat, clr_mask);
  tmp_dat1 = _mm256_and_si256(dat1, clr_mask);
  pack_dat = _mm256_packus_epi16(tmp_dat, tmp_dat1);
  pack_dat = _mm256_permute4x64_epi64(pack_dat, 0xd8);
  dat2 = _mm256_loadu_si256((__m256i const*)(pcurrent_value16_ + 32));
  dat3 = _mm256_loadu_si256((__m256i const*)(pcurrent_value16_ + 48));
  tmp_dat = _mm256_and_si256(dat2, clr_mask);
  tmp_dat1 = _mm256_and_si256(dat3, clr_mask);
  pack_dat1 = _mm256_packus_epi16(tmp_dat, tmp_dat1);
  pack_dat1 = _mm256_permute4x64_epi64(pack_dat1, 0xd8);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[0] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[1] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[2] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[3] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[4] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[5] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[6] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[7] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[8] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[9] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[5]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
  reinterpret_cast<uint32_t*>(buffer_end_)[10] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[5]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
  reinterpret_cast<uint32_t*>(buffer_end_)[11] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[6]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[6]);
  reinterpret_cast<uint32_t*>(buffer_end_)[12] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[6]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[6]);
  reinterpret_cast<uint32_t*>(buffer_end_)[13] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[7]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[7]);
  reinterpret_cast<uint32_t*>(buffer_end_)[14] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[7]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[7]);
  reinterpret_cast<uint32_t*>(buffer_end_)[15] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_srli_si256(dat, 1);
  tmp_dat = _mm256_and_si256(tmp_dat, clr_mask);
  tmp_dat1 = _mm256_srli_si256(dat1, 1);
  tmp_dat1 = _mm256_and_si256(tmp_dat1, clr_mask);
  pack_dat = _mm256_packus_epi16(tmp_dat, tmp_dat1);
  pack_dat = _mm256_permute4x64_epi64(pack_dat, 0xd8);
  tmp_dat = _mm256_srli_si256(dat2, 1);
  tmp_dat = _mm256_and_si256(tmp_dat, clr_mask);
  tmp_dat1 = _mm256_srli_si256(dat3, 1);
  tmp_dat1 = _mm256_and_si256(tmp_dat1, clr_mask);
  pack_dat1 = _mm256_packus_epi16(tmp_dat, tmp_dat1);
  pack_dat1 = _mm256_permute4x64_epi64(pack_dat1, 0xd8);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[16] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[17] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[18] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[19] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[20] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[21] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[22] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[23] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[24] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[25] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[5]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
  reinterpret_cast<uint32_t*>(buffer_end_)[26] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[5]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
  reinterpret_cast<uint32_t*>(buffer_end_)[27] = _mm256_movemask_epi8(ret);
}

inline void FleEncoder::Pack_15() {
  __m256i dat, dat1, dat2, dat3, tmp_dat, tmp_dat1, pack_dat, pack_dat1, ret;
  dat = _mm256_loadu_si256((__m256i const*)pcurrent_value16_);
  dat1 = _mm256_loadu_si256((__m256i const*)(pcurrent_value16_ + 16));
  tmp_dat = _mm256_and_si256(dat, clr_mask);
  tmp_dat1 = _mm256_and_si256(dat1, clr_mask);
  pack_dat = _mm256_packus_epi16(tmp_dat, tmp_dat1);
  pack_dat = _mm256_permute4x64_epi64(pack_dat, 0xd8);
  dat2 = _mm256_loadu_si256((__m256i const*)(pcurrent_value16_ + 32));
  dat3 = _mm256_loadu_si256((__m256i const*)(pcurrent_value16_ + 48));
  tmp_dat = _mm256_and_si256(dat2, clr_mask);
  tmp_dat1 = _mm256_and_si256(dat3, clr_mask);
  pack_dat1 = _mm256_packus_epi16(tmp_dat, tmp_dat1);
  pack_dat1 = _mm256_permute4x64_epi64(pack_dat1, 0xd8);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[0] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[1] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[2] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[3] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[4] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[5] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[6] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[7] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[8] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[9] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[5]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
  reinterpret_cast<uint32_t*>(buffer_end_)[10] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[5]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
  reinterpret_cast<uint32_t*>(buffer_end_)[11] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[6]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[6]);
  reinterpret_cast<uint32_t*>(buffer_end_)[12] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[6]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[6]);
  reinterpret_cast<uint32_t*>(buffer_end_)[13] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[7]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[7]);
  reinterpret_cast<uint32_t*>(buffer_end_)[14] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[7]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[7]);
  reinterpret_cast<uint32_t*>(buffer_end_)[15] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_srli_si256(dat, 1);
  tmp_dat = _mm256_and_si256(tmp_dat, clr_mask);
  tmp_dat1 = _mm256_srli_si256(dat1, 1);
  tmp_dat1 = _mm256_and_si256(tmp_dat1, clr_mask);
  pack_dat = _mm256_packus_epi16(tmp_dat, tmp_dat1);
  pack_dat = _mm256_permute4x64_epi64(pack_dat, 0xd8);
  tmp_dat = _mm256_srli_si256(dat2, 1);
  tmp_dat = _mm256_and_si256(tmp_dat, clr_mask);
  tmp_dat1 = _mm256_srli_si256(dat3, 1);
  tmp_dat1 = _mm256_and_si256(tmp_dat1, clr_mask);
  pack_dat1 = _mm256_packus_epi16(tmp_dat, tmp_dat1);
  pack_dat1 = _mm256_permute4x64_epi64(pack_dat1, 0xd8);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[16] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[17] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[18] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[19] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[20] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[21] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[22] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[23] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[24] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[25] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[5]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
  reinterpret_cast<uint32_t*>(buffer_end_)[26] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[5]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
  reinterpret_cast<uint32_t*>(buffer_end_)[27] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[6]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[6]);
  reinterpret_cast<uint32_t*>(buffer_end_)[28] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[6]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[6]);
  reinterpret_cast<uint32_t*>(buffer_end_)[29] = _mm256_movemask_epi8(ret);
}

inline void FleEncoder::Pack_16() {
  __m256i dat, dat1, dat2, dat3, tmp_dat, tmp_dat1, pack_dat, pack_dat1, ret;
  dat = _mm256_loadu_si256((__m256i const*)pcurrent_value16_);
  dat1 = _mm256_loadu_si256((__m256i const*)(pcurrent_value16_ + 16));
  tmp_dat = _mm256_and_si256(dat, clr_mask);
  tmp_dat1 = _mm256_and_si256(dat1, clr_mask);
  pack_dat = _mm256_packus_epi16(tmp_dat, tmp_dat1);
  pack_dat = _mm256_permute4x64_epi64(pack_dat, 0xd8);
  dat2 = _mm256_loadu_si256((__m256i const*)(pcurrent_value16_ + 32));
  dat3 = _mm256_loadu_si256((__m256i const*)(pcurrent_value16_ + 48));
  tmp_dat = _mm256_and_si256(dat2, clr_mask);
  tmp_dat1 = _mm256_and_si256(dat3, clr_mask);
  pack_dat1 = _mm256_packus_epi16(tmp_dat, tmp_dat1);
  pack_dat1 = _mm256_permute4x64_epi64(pack_dat1, 0xd8);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[0] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[1] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[2] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[3] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[4] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[5] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[6] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[7] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[8] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[9] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[5]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
  reinterpret_cast<uint32_t*>(buffer_end_)[10] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[5]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
  reinterpret_cast<uint32_t*>(buffer_end_)[11] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[6]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[6]);
  reinterpret_cast<uint32_t*>(buffer_end_)[12] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[6]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[6]);
  reinterpret_cast<uint32_t*>(buffer_end_)[13] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[7]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[7]);
  reinterpret_cast<uint32_t*>(buffer_end_)[14] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[7]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[7]);
  reinterpret_cast<uint32_t*>(buffer_end_)[15] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_srli_si256(dat, 1);
  tmp_dat = _mm256_and_si256(tmp_dat, clr_mask);
  tmp_dat1 = _mm256_srli_si256(dat1, 1);
  tmp_dat1 = _mm256_and_si256(tmp_dat1, clr_mask);
  pack_dat = _mm256_packus_epi16(tmp_dat, tmp_dat1);
  pack_dat = _mm256_permute4x64_epi64(pack_dat, 0xd8);
  tmp_dat = _mm256_srli_si256(dat2, 1);
  tmp_dat = _mm256_and_si256(tmp_dat, clr_mask);
  tmp_dat1 = _mm256_srli_si256(dat3, 1);
  tmp_dat1 = _mm256_and_si256(tmp_dat1, clr_mask);
  pack_dat1 = _mm256_packus_epi16(tmp_dat, tmp_dat1);
  pack_dat1 = _mm256_permute4x64_epi64(pack_dat1, 0xd8);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[16] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[17] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[18] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[19] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[20] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[21] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[22] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[23] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[24] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[25] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[5]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
  reinterpret_cast<uint32_t*>(buffer_end_)[26] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[5]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
  reinterpret_cast<uint32_t*>(buffer_end_)[27] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[6]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[6]);
  reinterpret_cast<uint32_t*>(buffer_end_)[28] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[6]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[6]);
  reinterpret_cast<uint32_t*>(buffer_end_)[29] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(pack_dat, bitv[7]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[7]);
  reinterpret_cast<uint32_t*>(buffer_end_)[30] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(pack_dat1, bitv[7]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[7]);
  reinterpret_cast<uint32_t*>(buffer_end_)[31] = _mm256_movemask_epi8(ret);
}

inline void FleEncoder::Pack_16_32() {
//  memset(buffer_end_, 0, bit_width_ * 8);
//  for (int i = 0; i < count_; ++i) {
//    for (int l = 0; l < bit_width_; ++l) {
//      buffer_end_[l] |= ((current_value_[i] >> l) & 0x01ULL) << (63 - i);
//    }
//  }


  __m256i dat[8], tmp_dat, tmp_dat1, pack_dat[8], ret;
  for (int l = 0; l < 8; ++l) {
    dat[l] =  _mm256_loadu_si256((__m256i const*)(current_value_ + 8 * l));
  }
  for (int l = 0; l < 4; ++l) {
    tmp_dat = _mm256_and_si256(dat[2 * l], clr32_mask);
    tmp_dat1 = _mm256_and_si256(dat[2 * l + 1], clr32_mask);
    pack_dat[l] = _mm256_packus_epi32(tmp_dat, tmp_dat1);
    pack_dat[l] = _mm256_permute4x64_epi64(pack_dat[l], 0xd8);
    tmp_dat = _mm256_srli_si256(dat[2 * l], 2);
    tmp_dat = _mm256_and_si256(tmp_dat, clr32_mask);
    tmp_dat1 = _mm256_srli_si256(dat[2 * l + 1], 2);
    tmp_dat1 = _mm256_and_si256(tmp_dat1, clr32_mask);
    pack_dat[4 + l] = _mm256_packus_epi32(tmp_dat, tmp_dat1);
    pack_dat[4 + l] = _mm256_permute4x64_epi64(pack_dat[4 + l], 0xd8);
  }
  for (int l = 0; l < 4; ++l) {
    tmp_dat = _mm256_and_si256(pack_dat[2 * l], clr_mask);
    tmp_dat1 = _mm256_and_si256(pack_dat[2 * l + 1], clr_mask);
    dat[l] = _mm256_packus_epi16(tmp_dat, tmp_dat1);
    dat[l] = _mm256_permute4x64_epi64(dat[l], 0xd8);
    tmp_dat = _mm256_srli_si256(pack_dat[2 * l], 1);
    tmp_dat = _mm256_and_si256(tmp_dat, clr_mask);
    tmp_dat1 = _mm256_srli_si256(pack_dat[2 * l + 1], 1);
    tmp_dat1 = _mm256_and_si256(tmp_dat1, clr_mask);
    dat[4 + l] = _mm256_packus_epi16(tmp_dat, tmp_dat1);
    dat[4 + l] = _mm256_permute4x64_epi64(dat[4 + l], 0xd8);
  }

// bit 0-7
  tmp_dat = _mm256_and_si256(dat[0], bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[0] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat[1], bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[1] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(dat[0], bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[2] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat[1], bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[3] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(dat[0], bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[4] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat[1], bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[5] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(dat[0], bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[6] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat[1], bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[7] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(dat[0], bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[8] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat[1], bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[9] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(dat[0], bitv[5]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
  reinterpret_cast<uint32_t*>(buffer_end_)[10] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat[1], bitv[5]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
  reinterpret_cast<uint32_t*>(buffer_end_)[11] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(dat[0], bitv[6]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[6]);
  reinterpret_cast<uint32_t*>(buffer_end_)[12] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat[1], bitv[6]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[6]);
  reinterpret_cast<uint32_t*>(buffer_end_)[13] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(dat[0], bitv[7]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[7]);
  reinterpret_cast<uint32_t*>(buffer_end_)[14] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat[1], bitv[7]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[7]);
  reinterpret_cast<uint32_t*>(buffer_end_)[15] = _mm256_movemask_epi8(ret);

// bit 8-15
  tmp_dat = _mm256_and_si256(dat[4], bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[16] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat[5], bitv[0]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
  reinterpret_cast<uint32_t*>(buffer_end_)[17] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(dat[4], bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[18] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat[5], bitv[1]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
  reinterpret_cast<uint32_t*>(buffer_end_)[19] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(dat[4], bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[20] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat[5], bitv[2]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
  reinterpret_cast<uint32_t*>(buffer_end_)[21] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(dat[4], bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[22] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat[5], bitv[3]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
  reinterpret_cast<uint32_t*>(buffer_end_)[23] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(dat[4], bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[24] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat[5], bitv[4]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
  reinterpret_cast<uint32_t*>(buffer_end_)[25] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(dat[4], bitv[5]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
  reinterpret_cast<uint32_t*>(buffer_end_)[26] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat[5], bitv[5]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
  reinterpret_cast<uint32_t*>(buffer_end_)[27] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(dat[4], bitv[6]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[6]);
  reinterpret_cast<uint32_t*>(buffer_end_)[28] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat[5], bitv[6]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[6]);
  reinterpret_cast<uint32_t*>(buffer_end_)[29] = _mm256_movemask_epi8(ret);

  tmp_dat = _mm256_and_si256(dat[4], bitv[7]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[7]);
  reinterpret_cast<uint32_t*>(buffer_end_)[30] = _mm256_movemask_epi8(ret);
  tmp_dat = _mm256_and_si256(dat[5], bitv[7]);
  ret = _mm256_cmpeq_epi8(tmp_dat, bitv[7]);
  reinterpret_cast<uint32_t*>(buffer_end_)[31] = _mm256_movemask_epi8(ret);

// bit 16-23
  if (bit_width_ >= 24) {
    tmp_dat = _mm256_and_si256(dat[2], bitv[0]);
    ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
    reinterpret_cast<uint32_t*>(buffer_end_)[32] = _mm256_movemask_epi8(ret);
    tmp_dat = _mm256_and_si256(dat[3], bitv[0]);
    ret = _mm256_cmpeq_epi8(tmp_dat, bitv[0]);
    reinterpret_cast<uint32_t*>(buffer_end_)[33] = _mm256_movemask_epi8(ret);

    tmp_dat = _mm256_and_si256(dat[2], bitv[1]);
    ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
    reinterpret_cast<uint32_t*>(buffer_end_)[34] = _mm256_movemask_epi8(ret);
    tmp_dat = _mm256_and_si256(dat[3], bitv[1]);
    ret = _mm256_cmpeq_epi8(tmp_dat, bitv[1]);
    reinterpret_cast<uint32_t*>(buffer_end_)[35] = _mm256_movemask_epi8(ret);

    tmp_dat = _mm256_and_si256(dat[2], bitv[2]);
    ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
    reinterpret_cast<uint32_t*>(buffer_end_)[36] = _mm256_movemask_epi8(ret);
    tmp_dat = _mm256_and_si256(dat[3], bitv[2]);
    ret = _mm256_cmpeq_epi8(tmp_dat, bitv[2]);
    reinterpret_cast<uint32_t*>(buffer_end_)[37] = _mm256_movemask_epi8(ret);

    tmp_dat = _mm256_and_si256(dat[2], bitv[3]);
    ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
    reinterpret_cast<uint32_t*>(buffer_end_)[38] = _mm256_movemask_epi8(ret);
    tmp_dat = _mm256_and_si256(dat[3], bitv[3]);
    ret = _mm256_cmpeq_epi8(tmp_dat, bitv[3]);
    reinterpret_cast<uint32_t*>(buffer_end_)[39] = _mm256_movemask_epi8(ret);

    tmp_dat = _mm256_and_si256(dat[2], bitv[4]);
    ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
    reinterpret_cast<uint32_t*>(buffer_end_)[40] = _mm256_movemask_epi8(ret);
    tmp_dat = _mm256_and_si256(dat[3], bitv[4]);
    ret = _mm256_cmpeq_epi8(tmp_dat, bitv[4]);
    reinterpret_cast<uint32_t*>(buffer_end_)[41] = _mm256_movemask_epi8(ret);

    tmp_dat = _mm256_and_si256(dat[2], bitv[5]);
    ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
    reinterpret_cast<uint32_t*>(buffer_end_)[42] = _mm256_movemask_epi8(ret);
    tmp_dat = _mm256_and_si256(dat[3], bitv[5]);
    ret = _mm256_cmpeq_epi8(tmp_dat, bitv[5]);
    reinterpret_cast<uint32_t*>(buffer_end_)[43] = _mm256_movemask_epi8(ret);

    tmp_dat = _mm256_and_si256(dat[2], bitv[6]);
    ret = _mm256_cmpeq_epi8(tmp_dat, bitv[6]);
    reinterpret_cast<uint32_t*>(buffer_end_)[44] = _mm256_movemask_epi8(ret);
    tmp_dat = _mm256_and_si256(dat[3], bitv[6]);
    ret = _mm256_cmpeq_epi8(tmp_dat, bitv[6]);
    reinterpret_cast<uint32_t*>(buffer_end_)[45] = _mm256_movemask_epi8(ret);

    tmp_dat = _mm256_and_si256(dat[2], bitv[7]);
    ret = _mm256_cmpeq_epi8(tmp_dat, bitv[7]);
    reinterpret_cast<uint32_t*>(buffer_end_)[46] = _mm256_movemask_epi8(ret);
    tmp_dat = _mm256_and_si256(dat[3], bitv[7]);
    ret = _mm256_cmpeq_epi8(tmp_dat, bitv[7]);
    reinterpret_cast<uint32_t*>(buffer_end_)[47] = _mm256_movemask_epi8(ret);

    int idx = 48;
    for (int l = 0; l < bit_width_ - 24; ++l) {
      tmp_dat = _mm256_and_si256(dat[6], bitv[l]);
      ret = _mm256_cmpeq_epi8(tmp_dat, bitv[l]);
      reinterpret_cast<uint32_t*>(buffer_end_)[idx] = _mm256_movemask_epi8(ret);
      tmp_dat = _mm256_and_si256(dat[7], bitv[l]);
      ret = _mm256_cmpeq_epi8(tmp_dat, bitv[l]);
      reinterpret_cast<uint32_t*>(buffer_end_)[idx + 1] = _mm256_movemask_epi8(ret);
      idx += 2;
    }
  } else {
    int idx = 32;
    for (int l = 0; l < bit_width_ - 16; ++l) {
      tmp_dat = _mm256_and_si256(dat[2], bitv[l]);
      ret = _mm256_cmpeq_epi8(tmp_dat, bitv[l]);
      reinterpret_cast<uint32_t*>(buffer_end_)[idx] = _mm256_movemask_epi8(ret);
      tmp_dat = _mm256_and_si256(dat[3], bitv[l]);
      ret = _mm256_cmpeq_epi8(tmp_dat, bitv[l]);
      reinterpret_cast<uint32_t*>(buffer_end_)[idx + 1] = _mm256_movemask_epi8(ret);
      idx += 2;
    }
  }

}


inline int FleEncoder::Flush() {
  (this->*fa[bit_width_])();

  buffer_end_ += bit_width_;
  count_ = 0;
  return (buffer_end_ - buffer_) * 8;
}

}

#endif
