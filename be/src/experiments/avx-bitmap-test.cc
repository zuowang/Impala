// Copyright 2014 Cloudera Inc.
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

#include "avx-bitmap.h"
#include <stdio.h>
#include <time.h>

using namespace impala;

#define NUM_ROWS 1024000000
#define BITMAP_SIZE 20480
#define BIG_BITMAP_SIZE 2048000000

const int NUM_BITS = 10240 * 256;
const int NUM_VALUES = 1024000;

int main() {
struct TestData {
  int64_t* bit_index;
  bool* v;
  int num_values;
  int num_bits;
};

  TestData d;
  d.num_bits = NUM_BITS;
  d.num_values = NUM_VALUES;
  d.bit_index = new int64_t[NUM_VALUES];
  d.v = new bool[NUM_VALUES];
  TestData* data=&d;
/*
  for (int i = 0; i < 1000; ++i) {
    AVXBitmap avx_bitmap(data->num_bits);
    __m256i shufmask1 = _mm256_set1_epi32(4);
    // Unroll this to focus more on Put performance.
    for (int j = 0; j < data->num_values; j += 32) {
      __m256i v32, v4;
      v32 = _mm256_loadu_si256((__m256i const*)(data->v + j));
      int64_t* cur_bit_index = data->bit_index + j;
      __m256 bit_index4;
      bit_index4 = _mm256_loadu_ps((float const*)(cur_bit_index + 0));
      v4 = _mm256_shuffle_epi8(v32, shufmask1);
      avx_bitmap.Set<true>(bit_index4, v4);

      bit_index4 = _mm256_loadu_ps((float const*)(cur_bit_index + 4));
      v4 = _mm256_shuffle_epi8(v32, shufmask1);
      avx_bitmap.Set<true>(bit_index4, v4);

      bit_index4 = _mm256_loadu_ps((float const*)(cur_bit_index + 8));
      v4 = _mm256_shuffle_epi8(v32, shufmask1);
      avx_bitmap.Set<true>(bit_index4, v4);

      bit_index4 = _mm256_loadu_ps((float const*)(cur_bit_index + 12));
      v4 = _mm256_shuffle_epi8(v32, shufmask1);
      avx_bitmap.Set<true>(bit_index4, v4);

      bit_index4 = _mm256_loadu_ps((float const*)(cur_bit_index + 16));
      v4 = _mm256_shuffle_epi8(v32, shufmask1);
      avx_bitmap.Set<true>(bit_index4, v4);

      bit_index4 = _mm256_loadu_ps((float const*)(cur_bit_index + 20));
      v4 = _mm256_shuffle_epi8(v32, shufmask1);
      avx_bitmap.Set<true>(bit_index4, v4);

      bit_index4 = _mm256_loadu_ps((float const*)(cur_bit_index + 24));
      v4 = _mm256_shuffle_epi8(v32, shufmask1);
      avx_bitmap.Set<true>(bit_index4, v4);

      bit_index4 = _mm256_loadu_ps((float const*)(cur_bit_index + 28));
      v4 = _mm256_shuffle_epi8(v32, shufmask1);
      avx_bitmap.Set<true>(bit_index4, v4);
    }
  }
*/

  for (int i = 0; i < 1000; ++i) {
    AVXBitmap avx_bitmap(data->num_bits);
    // Unroll this to focus more on Put performance.
    for (int j = 0; j < data->num_values; j += 32) {
      __m256 bit_index4;
      __m256i v4, v32;

      int64_t* cur_bit_index = data->bit_index + j;
      bit_index4 = _mm256_loadu_ps((float const*)(cur_bit_index + 0));
      v32 = avx_bitmap.Get<true>(bit_index4);

      bit_index4 = _mm256_loadu_ps((float const*)(cur_bit_index + 4));
      v4 = _mm256_slli_si256(avx_bitmap.Get<true>(bit_index4), 8);
      v32 = _mm256_or_si256(v32, v4);

      bit_index4 = _mm256_loadu_ps((float const*)(cur_bit_index + 8));
      v4 = _mm256_slli_si256(avx_bitmap.Get<true>(bit_index4), 16);
      v32 = _mm256_or_si256(v32, v4);

      bit_index4 = _mm256_loadu_ps((float const*)(cur_bit_index + 12));
      v4 = _mm256_slli_si256(avx_bitmap.Get<true>(bit_index4), 24);
      v32 = _mm256_or_si256(v32, v4);

      bit_index4 = _mm256_loadu_ps((float const*)(cur_bit_index + 16));
      v4 = _mm256_slli_si256(avx_bitmap.Get<true>(bit_index4), 32);
      v32 = _mm256_or_si256(v32, v4);

      bit_index4 = _mm256_loadu_ps((float const*)(cur_bit_index + 20));
      v4 = _mm256_slli_si256(avx_bitmap.Get<true>(bit_index4), 40);
      v32 = _mm256_or_si256(v32, v4);

      bit_index4 = _mm256_loadu_ps((float const*)(cur_bit_index + 24));
      v4 = _mm256_slli_si256(avx_bitmap.Get<true>(bit_index4), 48);
      v32 = _mm256_or_si256(v32, v4);

      bit_index4 = _mm256_loadu_ps((float const*)(cur_bit_index + 28));
      v4 = _mm256_slli_si256(avx_bitmap.Get<true>(bit_index4), 56);
      v32 = _mm256_or_si256(v32, v4);

      _mm256_storeu_si256((__m256i *)(data->v + j), v32);
    }
  }


  delete [] d.bit_index;
  delete [] d.v;

/*
  AVXBitmap bm(BITMAP_SIZE);
  float* rows = new float[NUM_ROWS];
  clock_t start = clock();
  int rows_filted = 0;
  for (int i = 0; i < NUM_ROWS; i += 32) {
    __m256i v4, v32;
    float* cur_rows = rows + i;
    v32 = bm.Get<true>(_mm256_loadu_ps((float const*)(cur_rows + 0)));

    v4 = bm.Get<true>(_mm256_loadu_ps((float const*)(cur_rows + 4)));
    v32 = _mm256_or_si256(v32, _mm256_slli_si256(v4, 8));

    v4 = bm.Get<true>(_mm256_loadu_ps((float const*)(cur_rows + 8)));
    v32 = _mm256_or_si256(v32, _mm256_slli_si256(v4, 16));

    v4 = bm.Get<true>(_mm256_loadu_ps((float const*)(cur_rows + 12)));
    v32 = _mm256_or_si256(v32, _mm256_slli_si256(v4, 24));

    v4 = bm.Get<true>(_mm256_loadu_ps((float const*)(cur_rows + 16)));
    v32 = _mm256_or_si256(v32, _mm256_slli_si256(v4, 32));

    v4 = bm.Get<true>(_mm256_loadu_ps((float const*)(cur_rows + 20)));
    v32 = _mm256_or_si256(v32, _mm256_slli_si256(v4, 40));

    v4 = bm.Get<true>(_mm256_loadu_ps((float const*)(cur_rows + 24)));
    v32 = _mm256_or_si256(v32, _mm256_slli_si256(v4, 48));

    v4 = bm.Get<true>(_mm256_loadu_ps((float const*)(cur_rows + 28)));
    v32 = _mm256_or_si256(v32, _mm256_slli_si256(v4, 56));

    bool tmp[32];
    _mm256_storeu_si256((__m256i *)tmp, v32);
    for (int i = 0; i < 32; ++i) {
      if (tmp[i]) ++rows_filted;
    }
  }

  clock_t end = clock();
  printf("time for Get operation: %f rows filtered:%d\n", (double)(end - start),
      rows_filted);
  delete [] rows;

  AVXBitmap bm1(BIG_BITMAP_SIZE);
  AVXBitmap bm2(BIG_BITMAP_SIZE);
  start = clock();
  bm1.Or(&bm2);
  end = clock();
  printf("time for Or operation: %f\n", (double)(end - start));

  start = clock();
  bm1.And(&bm2);
  end = clock();
  printf("time for And operation: %f\n", (double)(end - start));

  __m256 bit_index;
  __m256i mask = _mm256_set1_epi32(1);
  start = clock();
  for (int i = 0; i < NUM_ROWS; i+=8) {
    bm.Set<true>(bit_index, mask);
  }
  end = clock();
  printf("time for Set operation: %f\n", (double)(end - start));
*/
  return 0;
}
