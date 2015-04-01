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

#include <iostream>
#include <sstream>
#include <immintrin.h>

#include "runtime/mem-pool.h"
#include "runtime/mem-tracker.h"
#include "experiments/avx-bitmap.h"
#include "util/benchmark.h"
#include "util/bitmap.h"
#include "util/cpu-info.h"

// Benchmark to measure how quickly we can do bitmap setting and getting.

// Op Set:               Function     Rate (iters/ms)          Comparison
// ----------------------------------------------------------------------
//                       "Bitmap"             0.03574                  1X
//                    "AVXBitmap"               126.9             0.9546X

// Op Get:               Function     Rate (iters/ms)          Comparison
// ----------------------------------------------------------------------
//                        "Bitmap"                           132.9                  1X
//                        "AVXBitmap"                        126.9             0.9546X

using namespace std;
using namespace impala;

const int NUM_BITS = 10240 * 256;
const int NUM_VALUES = 1024000;

struct TestData {
  int64_t* bit_index;
  bool* v;
  int num_values;
  int num_bits;
};

void TestBitmapSet(int batch_size, void* d) {
  TestData* data = reinterpret_cast<TestData*>(d);
  for (int i = 0; i < batch_size; ++i) {
    Bitmap bitmap(data->num_bits);
    // Unroll this to focus more on Put performance.
    for (int j = 0; j < data->num_values; j += 32) {
      bitmap.Set<true>(data->bit_index[j + 0], data->v[j + 0]);
      bitmap.Set<true>(data->bit_index[j + 1], data->v[j + 1]);
      bitmap.Set<true>(data->bit_index[j + 2], data->v[j + 2]);
      bitmap.Set<true>(data->bit_index[j + 3], data->v[j + 3]);
      bitmap.Set<true>(data->bit_index[j + 4], data->v[j + 4]);
      bitmap.Set<true>(data->bit_index[j + 5], data->v[j + 5]);
      bitmap.Set<true>(data->bit_index[j + 6], data->v[j + 6]);
      bitmap.Set<true>(data->bit_index[j + 7], data->v[j + 7]);
      bitmap.Set<true>(data->bit_index[j + 8], data->v[j + 8]);
      bitmap.Set<true>(data->bit_index[j + 9], data->v[j + 9]);
      bitmap.Set<true>(data->bit_index[j + 10], data->v[j + 10]);
      bitmap.Set<true>(data->bit_index[j + 11], data->v[j + 11]);
      bitmap.Set<true>(data->bit_index[j + 12], data->v[j + 12]);
      bitmap.Set<true>(data->bit_index[j + 13], data->v[j + 13]);
      bitmap.Set<true>(data->bit_index[j + 14], data->v[j + 14]);
      bitmap.Set<true>(data->bit_index[j + 15], data->v[j + 15]);
      bitmap.Set<true>(data->bit_index[j + 16], data->v[j + 16]);
      bitmap.Set<true>(data->bit_index[j + 17], data->v[j + 17]);
      bitmap.Set<true>(data->bit_index[j + 18], data->v[j + 18]);
      bitmap.Set<true>(data->bit_index[j + 19], data->v[j + 19]);
      bitmap.Set<true>(data->bit_index[j + 20], data->v[j + 20]);
      bitmap.Set<true>(data->bit_index[j + 21], data->v[j + 21]);
      bitmap.Set<true>(data->bit_index[j + 22], data->v[j + 22]);
      bitmap.Set<true>(data->bit_index[j + 23], data->v[j + 23]);
      bitmap.Set<true>(data->bit_index[j + 24], data->v[j + 24]);
      bitmap.Set<true>(data->bit_index[j + 25], data->v[j + 25]);
      bitmap.Set<true>(data->bit_index[j + 26], data->v[j + 26]);
      bitmap.Set<true>(data->bit_index[j + 27], data->v[j + 27]);
      bitmap.Set<true>(data->bit_index[j + 28], data->v[j + 28]);
      bitmap.Set<true>(data->bit_index[j + 29], data->v[j + 29]);
      bitmap.Set<true>(data->bit_index[j + 30], data->v[j + 30]);
      bitmap.Set<true>(data->bit_index[j + 31], data->v[j + 31]);
    }
  }
}

void TestAVXBitmapSet(int batch_size, void* d) {
  TestData* data = reinterpret_cast<TestData*>(d);
  __m256i shufmask1 = _mm256_set1_epi32(4);
  for (int i = 0; i < batch_size; ++i) {
    AVXBitmap avx_bitmap(data->num_bits);
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
}

void TestBitmapGet(int batch_size, void* d) {
  TestData* data = reinterpret_cast<TestData*>(d);
  for (int i = 0; i < batch_size; ++i) {
    Bitmap bitmap(data->num_bits);
    // Unroll this to focus more on Put performance.
    for (int j = 0; j < data->num_values; j += 32) {
      data->v[j + 0] = bitmap.Get<true>(data->bit_index[j + 0]);
      data->v[j + 1] = bitmap.Get<true>(data->bit_index[j + 1]);
      data->v[j + 2] = bitmap.Get<true>(data->bit_index[j + 2]);
      data->v[j + 3] = bitmap.Get<true>(data->bit_index[j + 3]);
      data->v[j + 4] = bitmap.Get<true>(data->bit_index[j + 4]);
      data->v[j + 5] = bitmap.Get<true>(data->bit_index[j + 5]);
      data->v[j + 6] = bitmap.Get<true>(data->bit_index[j + 6]);
      data->v[j + 7] = bitmap.Get<true>(data->bit_index[j + 7]);
      data->v[j + 8] = bitmap.Get<true>(data->bit_index[j + 8]);
      data->v[j + 9] = bitmap.Get<true>(data->bit_index[j + 9]);
      data->v[j + 10] = bitmap.Get<true>(data->bit_index[j + 10]);
      data->v[j + 11] = bitmap.Get<true>(data->bit_index[j + 11]);
      data->v[j + 12] = bitmap.Get<true>(data->bit_index[j + 12]);
      data->v[j + 13] = bitmap.Get<true>(data->bit_index[j + 13]);
      data->v[j + 14] = bitmap.Get<true>(data->bit_index[j + 14]);
      data->v[j + 15] = bitmap.Get<true>(data->bit_index[j + 15]);
      data->v[j + 16] = bitmap.Get<true>(data->bit_index[j + 16]);
      data->v[j + 17] = bitmap.Get<true>(data->bit_index[j + 17]);
      data->v[j + 18] = bitmap.Get<true>(data->bit_index[j + 18]);
      data->v[j + 19] = bitmap.Get<true>(data->bit_index[j + 19]);
      data->v[j + 20] = bitmap.Get<true>(data->bit_index[j + 20]);
      data->v[j + 21] = bitmap.Get<true>(data->bit_index[j + 21]);
      data->v[j + 22] = bitmap.Get<true>(data->bit_index[j + 22]);
      data->v[j + 23] = bitmap.Get<true>(data->bit_index[j + 23]);
      data->v[j + 24] = bitmap.Get<true>(data->bit_index[j + 24]);
      data->v[j + 25] = bitmap.Get<true>(data->bit_index[j + 25]);
      data->v[j + 26] = bitmap.Get<true>(data->bit_index[j + 26]);
      data->v[j + 27] = bitmap.Get<true>(data->bit_index[j + 27]);
      data->v[j + 28] = bitmap.Get<true>(data->bit_index[j + 28]);
      data->v[j + 29] = bitmap.Get<true>(data->bit_index[j + 29]);
      data->v[j + 30] = bitmap.Get<true>(data->bit_index[j + 30]);
      data->v[j + 31] = bitmap.Get<true>(data->bit_index[j + 31]);
    }
  }
}

void TestAVXBitmapGet(int batch_size, void* d) {
  TestData* data = reinterpret_cast<TestData*>(d);
  for (int i = 0; i < batch_size; ++i) {
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
}

int main(int argc, char** argv) {
  CpuInfo::Init();

  MemTracker tracker;
  MemPool pool(&tracker);

  Benchmark bmset_suite("Op Set");
  TestData data;
  data.num_bits = NUM_BITS;
  data.num_values = NUM_VALUES;
  data.bit_index = (int64_t*) pool.Allocate(NUM_VALUES * 8);
  data.v = (bool*) pool.Allocate(NUM_VALUES);
  int bmset_baseline = bmset_suite.AddBenchmark("Bitmap", TestBitmapSet, &data, -1);
  bmset_suite.AddBenchmark("AVXBitmap", TestAVXBitmapSet, &data, bmset_baseline);

  Benchmark bmget_suite("Op Get");
  int bmget_baseline = bmget_suite.AddBenchmark("Bitmap", TestBitmapGet, &data, -1);
  bmget_suite.AddBenchmark("AVXBitmap", TestAVXBitmapGet, &data, bmget_baseline);

  cout << Benchmark::GetMachineInfo() << endl;
  cout << bmset_suite.Measure() << endl;
  cout << bmget_suite.Measure() << endl;

  pool.FreeAll();
  return 0;
}
