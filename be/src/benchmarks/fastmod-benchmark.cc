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

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <sstream>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include "util/benchmark.h"
#include "util/bitmap.h"
#include "util/cpu-info.h"

#include "common/names.h"

using namespace impala;

// Machine Info: Intel(R) Xeon(R) CPU E5-2695 v3 @ 2.30GHz
// mod prime:            Function     Rate (iters/ms)          Comparison
// ----------------------------------------------------------------------
//                              %               222.3                  1X
//                       fast mod                 504              2.267X
//
// mod mersenne prime:   Function     Rate (iters/ms)          Comparison
// ----------------------------------------------------------------------
//                              %               222.1                  1X
//                       fast mod                 501              2.255X
//             mod mersenne prime               563.5              2.537X
//
// mod 2^n:              Function     Rate (iters/ms)          Comparison
// ----------------------------------------------------------------------
//                          % 2^8               221.7                  1X
//                    & (2^8 - 1)               916.8              4.136X
//                   fast mod 2^8               494.6              2.231X
//
//                         % 2^16               219.2                  1X
//                   & (2^16 - 1)               902.9              4.119X
//                  fast mod 2^16                 489              2.231X
//
//                         % 2^25               217.3                  1X
//                   & (2^25 - 1)               898.8              4.136X
//                  fast mod 2^25               484.8              2.231X

struct TestData {
//  TestData() : divisor_class(0) {}
  Divisor divisor_class;
  uint32_t divisor;
  uint32_t bitmask;
  vector<uint32_t> dividends;
  vector<uint32_t> remainders;
};

void TestMod(int batch_size, void* d) {
  TestData* data = reinterpret_cast<TestData*>(d);
  for (int j = 0; j < batch_size; ++j) {
    for (int i = 0; i < data->dividends.size(); ++i) {
      data->remainders[i] = data->dividends[i] % data->divisor;
    }
  }
}

void TestModAnd(int batch_size, void* d) {
  TestData* data = reinterpret_cast<TestData*>(d);
  for (int j = 0; j < batch_size; ++j) {
    for (int i = 0; i < data->dividends.size(); ++i) {
      data->remainders[i] = data->dividends[i] & data->bitmask;
    }
  }
}

void TestFastMod(int batch_size, void* d) {
  TestData* data = reinterpret_cast<TestData*>(d);
  for (int j = 0; j < batch_size; ++j) {
    for (int i = 0; i < data->dividends.size(); ++i) {
      data->remainders[i] = data->divisor_class.GetMod(data->dividends[i]);
    }
  }
}

void TestModMersennePrime(int batch_size, void* d) {
  TestData* data = reinterpret_cast<TestData*>(d);
  for (int j = 0; j < batch_size; ++j) {
    for (int i = 0; i < data->dividends.size(); ++i) {
      // Mersenne prime 131071 = 2^12
      const uint32_t tmp = (data->dividends[i] & 131071) + (data->dividends[i] >> 12);
      data->remainders[i] = (tmp >= 131071) ? tmp - 131071 : tmp;
    }
  }
}

void Init(TestData* data, uint32_t divisor) {
  data->divisor = divisor;
  data->divisor_class = Divisor(divisor);
  int test_size = 1024;
  data->dividends.resize(test_size);
  for (uint32_t i = 0; i < test_size; ++i) {
    uint32_t result = (rand() >> 8) & 0xffff;
    result <<= 16;
    result |= (rand() >> 8) & 0xffff;
    data->dividends[i] = result;
  }
  data->remainders.resize(test_size);
  data->bitmask = divisor - 1;
}

int main(int argc, char **argv) {
  CpuInfo::Init();
  cout << Benchmark::GetMachineInfo() << endl;

  TestData data = {Divisor(0)};
  Init(&data, 32213);

  Benchmark suite("mod prime");
  suite.AddBenchmark("%", TestMod, &data);
  suite.AddBenchmark("fast mod", TestFastMod, &data);
  cout << suite.Measure() << endl;

  Init(&data, 131071);

  Benchmark mersenne_suite("mod mersenne prime");
  mersenne_suite.AddBenchmark("%", TestMod, &data);
  mersenne_suite.AddBenchmark("fast mod", TestFastMod, &data);
  mersenne_suite.AddBenchmark("mod mersenne prime", TestModMersennePrime, &data);
  cout << mersenne_suite.Measure() << endl;

  // Benchmarking when divisor is a power of 2.
  Benchmark suite_power2("mod 2^n");
  int size[] = {8, 16, 25};
  char name[128];
  for (int i = 0; i < sizeof(size) / sizeof(int); ++i) {
    Init(&data, 1L << size[i]);
    snprintf(name, sizeof(name), "%% 2^%d", size[i]);
    int baseline = suite_power2.AddBenchmark(name, TestMod, &data, -1);
    snprintf(name, sizeof(name), "& (2^%d - 1)", size[i]);
    suite_power2.AddBenchmark(name, TestModAnd, &data, baseline);
    snprintf(name, sizeof(name), "fast mod 2^%d", size[i]);
    suite_power2.AddBenchmark(name, TestFastMod, &data, baseline);
  }
  cout << suite_power2.Measure() << endl;

  return 0;
}
