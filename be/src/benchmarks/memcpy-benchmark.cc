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

#include "util/benchmark.h"
#include "util/cpu-info.h"
#include "util/mem-util.h"

#include "common/names.h"

using namespace impala;

// Machine Info: Intel(R) Xeon(R) CPU E5-2695 v3 @ 2.30GHz
// memcpy:               Function     Rate (iters/ms)          Comparison
// ----------------------------------------------------------------------
//                "Glibc 7 Bytes"           2.504e+05                  1X
//                  "RTE 7 Bytes"           2.985e+05              1.192X
//
//               "Glibc 15 Bytes"           2.631e+05                  1X
//                 "RTE 15 Bytes"           2.891e+05              1.099X
//
//               "Glibc 31 Bytes"           3.709e+05                  1X
//                 "RTE 31 Bytes"           3.324e+05             0.8963X
//
//               "Glibc 63 Bytes"           2.555e+05                  1X
//                 "RTE 63 Bytes"           2.956e+05              1.157X
//
//              "Glibc 127 Bytes"           2.466e+05                  1X
//                "RTE 127 Bytes"           2.045e+05             0.8292X
//
//              "Glibc 255 Bytes"           4.937e+04                  1X
//                "RTE 255 Bytes"           1.947e+05              3.943X
//
//              "Glibc 383 Bytes"           4.879e+04                  1X
//                "RTE 383 Bytes"           1.712e+05              3.508X
//
//              "Glibc 511 Bytes"           3.372e+04                  1X
//                "RTE 511 Bytes"           1.474e+05               4.37X
//
//              "Glibc 767 Bytes"            2.44e+04                  1X
//                "RTE 767 Bytes"           8.385e+04              3.437X
//
//             "Glibc 1023 Bytes"            2.03e+04                  1X
//               "RTE 1023 Bytes"           7.186e+04               3.54X
//
//             "Glibc 1535 Bytes"           1.561e+04                  1X
//               "RTE 1535 Bytes"           5.451e+04              3.492X
//
//             "Glibc 2047 Bytes"           1.268e+04                  1X
//               "RTE 2047 Bytes"           3.981e+04              3.141X
//
//             "Glibc 3071 Bytes"                9219                  1X
//               "RTE 3071 Bytes"           2.838e+04              3.079X
//
//             "Glibc 4095 Bytes"                7433                  1X
//               "RTE 4095 Bytes"            2.19e+04              2.946X
//
//             "Glibc 6143 Bytes"                5284                  1X
//               "RTE 6143 Bytes"           1.476e+04              2.793X
//
//             "Glibc 8191 Bytes"                4018                  1X
//               "RTE 8191 Bytes"           1.044e+04              2.598X
//
//            "Glibc 12287 Bytes"                2800                  1X
//              "RTE 12287 Bytes"                7306               2.61X
//
//            "Glibc 16383 Bytes"                2100                  1X
//              "RTE 16383 Bytes"                5295              2.522X
//
// memcpy constant bytes:Function     Rate (iters/ms)          Comparison
// ----------------------------------------------------------------------
//                  Glibc 7 Bytes           4.137e+05                  1X
//                    RTE 7 Bytes           3.036e+05             0.7337X
//
//                 Glibc 15 Bytes           3.552e+05                  1X
//                   RTE 15 Bytes           2.793e+05             0.7864X
//
//                 Glibc 31 Bytes           2.937e+05                  1X
//                   RTE 31 Bytes           3.052e+05              1.039X
//
//                 Glibc 63 Bytes           2.334e+05                  1X
//                   RTE 63 Bytes           3.028e+05              1.297X
//
//                Glibc 127 Bytes           9.353e+04                  1X
//                  RTE 127 Bytes           1.996e+05              2.134X
//
//                Glibc 255 Bytes           5.855e+04                  1X
//                  RTE 255 Bytes           1.947e+05              3.326X
//
//                Glibc 383 Bytes           4.359e+04                  1X
//                  RTE 383 Bytes           1.545e+05              3.545X
//
//                Glibc 511 Bytes           3.597e+04                  1X
//                  RTE 511 Bytes           1.357e+05              3.772X
//
//                Glibc 767 Bytes           3.221e+04                  1X
//                  RTE 767 Bytes           8.861e+04              2.751X
//
//               Glibc 1023 Bytes           2.632e+04                  1X
//                 RTE 1023 Bytes           7.302e+04              2.774X
//
//               Glibc 1535 Bytes           2.057e+04                  1X
//                 RTE 1535 Bytes           5.077e+04              2.468X
//
//               Glibc 2047 Bytes           1.702e+04                  1X
//                 RTE 2047 Bytes           4.056e+04              2.383X
//
//               Glibc 3071 Bytes           1.248e+04                  1X
//                 RTE 3071 Bytes           2.869e+04              2.298X
//
//               Glibc 4095 Bytes                9919                  1X
//                 RTE 4095 Bytes           1.991e+04              2.007X
//
//               Glibc 6143 Bytes                7244                  1X
//                 RTE 6143 Bytes            1.47e+04               2.03X
//
//               Glibc 8191 Bytes                5422                  1X
//                 RTE 8191 Bytes           1.064e+04              1.963X
//
//              Glibc 12287 Bytes                2806                  1X
//                RTE 12287 Bytes                7334              2.614X
//
//              Glibc 16383 Bytes                2088                  1X
//                RTE 16383 Bytes                5398              2.585X

struct TestData {
  uint8_t* dst;
  uint8_t* src;
  int size;
};

void TestGlibcMemcpy(int batch_size, void* d) {
  TestData* data = reinterpret_cast<TestData*>(d);
  for (int i = 0; i < batch_size; ++i) {
    memcpy(data->dst, data->src, data->size);
  }
}

void TestRteMemcpy(int batch_size, void* d) {
  TestData* data = reinterpret_cast<TestData*>(d);
  for (int i = 0; i < batch_size; ++i) {
    MemUtil::memcpy(data->dst, data->src, data->size);
  }
}

#define TEST(SIZE) \
  void TestGlibc##SIZE(int batch_size, void* d) { \
    TestData* data = reinterpret_cast<TestData*>(d); \
    for (int i = 0; i < batch_size; ++i) { \
      memcpy(data->dst, data->src, SIZE); \
    } \
  } \
  void TestRte##SIZE(int batch_size, void* d) { \
    TestData* data = reinterpret_cast<TestData*>(d); \
    for (int i = 0; i < batch_size; ++i) { \
      MemUtil::memcpy(data->dst, data->src, SIZE); \
    } \
  } \

TEST(7);
TEST(15);
TEST(31);
TEST(63);
TEST(127);
TEST(255);
TEST(383);
TEST(511);
TEST(767);
TEST(1023);
TEST(1535);
TEST(2047);
TEST(3071);
TEST(4095);
TEST(6143);
TEST(8191);
TEST(12287);
TEST(16383);

int main(int argc, char** argv) {
  CpuInfo::Init();
  cout << Benchmark::GetMachineInfo() << endl;
  int size[] = {7, 15, 31, 63, 127, 255, 383, 511, 767, 1023, 1535, 2047,
                  3071, 4095, 6143, 8191, 12287, 16383};
  int len = sizeof(size) / sizeof(int);
  TestData data_a[len];

  Benchmark memcpy_suite("memcpy");
  for (int i = 0; i < len; ++i) {
    data_a[i].dst = new uint8_t[size[i]];
    data_a[i].src = new uint8_t[size[i]];
    data_a[i].size = size[i];

    stringstream name;
    name << "\"Glibc " << size[i] << " Bytes\"";
    int baseline = memcpy_suite.AddBenchmark(name.str(), TestGlibcMemcpy, &data_a[i], -1);

    name.str("");
    name << "\"RTE " << size[i] << " Bytes\"";
    memcpy_suite.AddBenchmark(name.str(), TestRteMemcpy, &data_a[i], baseline);
  }
  cout << memcpy_suite.Measure() << endl;

  TestData data;
  data.dst = new uint8_t[20480];
  data.src = new uint8_t[20480];
  Benchmark constant_suite("memcpy constant bytes");

  int baseline = constant_suite.AddBenchmark("Glibc 7 Bytes", TestGlibc7, &data, -1);
  constant_suite.AddBenchmark("RTE 7 Bytes", TestRte7, &data, baseline);
  baseline = constant_suite.AddBenchmark("Glibc 15 Bytes", TestGlibc15, &data, -1);
  constant_suite.AddBenchmark("RTE 15 Bytes", TestRte15, &data, baseline);
  baseline = constant_suite.AddBenchmark("Glibc 31 Bytes", TestGlibc31, &data, -1);
  constant_suite.AddBenchmark("RTE 31 Bytes", TestRte31, &data, baseline);
  baseline = constant_suite.AddBenchmark("Glibc 63 Bytes", TestGlibc63, &data, -1);
  constant_suite.AddBenchmark("RTE 63 Bytes", TestRte63, &data, baseline);
  baseline = constant_suite.AddBenchmark("Glibc 127 Bytes", TestGlibc127, &data, -1);
  constant_suite.AddBenchmark("RTE 127 Bytes", TestRte127, &data, baseline);
  baseline = constant_suite.AddBenchmark("Glibc 255 Bytes", TestGlibc255, &data, -1);
  constant_suite.AddBenchmark("RTE 255 Bytes", TestRte255, &data, baseline);
  baseline = constant_suite.AddBenchmark("Glibc 383 Bytes", TestGlibc383, &data, -1);
  constant_suite.AddBenchmark("RTE 383 Bytes", TestRte383, &data, baseline);
  baseline = constant_suite.AddBenchmark("Glibc 511 Bytes", TestGlibc511, &data, -1);
  constant_suite.AddBenchmark("RTE 511 Bytes", TestRte511, &data, baseline);
  baseline = constant_suite.AddBenchmark("Glibc 767 Bytes", TestGlibc767, &data, -1);
  constant_suite.AddBenchmark("RTE 767 Bytes", TestRte767, &data, baseline);
  baseline = constant_suite.AddBenchmark("Glibc 1023 Bytes", TestGlibc1023, &data, -1);
  constant_suite.AddBenchmark("RTE 1023 Bytes", TestRte1023, &data, baseline);
  baseline = constant_suite.AddBenchmark("Glibc 1535 Bytes", TestGlibc1535, &data, -1);
  constant_suite.AddBenchmark("RTE 1535 Bytes", TestRte1535, &data, baseline);
  baseline = constant_suite.AddBenchmark("Glibc 2047 Bytes", TestGlibc2047, &data, -1);
  constant_suite.AddBenchmark("RTE 2047 Bytes", TestRte2047, &data, baseline);
  baseline = constant_suite.AddBenchmark("Glibc 3071 Bytes", TestGlibc3071, &data, -1);
  constant_suite.AddBenchmark("RTE 3071 Bytes", TestRte3071, &data, baseline);
  baseline = constant_suite.AddBenchmark("Glibc 4095 Bytes", TestGlibc4095, &data, -1);
  constant_suite.AddBenchmark("RTE 4095 Bytes", TestRte4095, &data, baseline);
  baseline = constant_suite.AddBenchmark("Glibc 6143 Bytes", TestGlibc6143, &data, -1);
  constant_suite.AddBenchmark("RTE 6143 Bytes", TestRte6143, &data, baseline);
  baseline = constant_suite.AddBenchmark("Glibc 8191 Bytes", TestGlibc8191, &data, -1);
  constant_suite.AddBenchmark("RTE 8191 Bytes", TestRte8191, &data, baseline);
  baseline = constant_suite.AddBenchmark("Glibc 12287 Bytes", TestGlibc12287, &data, -1);
  constant_suite.AddBenchmark("RTE 12287 Bytes", TestRte12287, &data, baseline);
  baseline = constant_suite.AddBenchmark("Glibc 16383 Bytes", TestGlibc16383, &data, -1);
  constant_suite.AddBenchmark("RTE 16383 Bytes", TestRte16383, &data, baseline);

  cout << constant_suite.Measure() << endl;
  return 0;
}
