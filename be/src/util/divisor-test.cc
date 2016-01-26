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
#include <limits.h>

#include <boost/utility.hpp>
#include <gtest/gtest.h>
#include "util/cpu-info.h"
#include "util/divisor.h"

#include "common/names.h"

namespace impala {

TEST(Divisor, SetupTest) {
  // multiplier = floor(2^N * (2^l - d) / d) + 1, where N = 32, l = ceil(log2(d))
  Divisor d1(0);
  EXPECT_EQ(d1.multiplier(), 0);
  Divisor d2(1);
  EXPECT_EQ(d2.multiplier(), 1);
  Divisor d3(63);
  EXPECT_EQ(d3.multiplier(), 68174085);
  Divisor d4(64);
  EXPECT_EQ(d4.multiplier(), 1);
}

TEST(Divisor, GetModTest) {
  for (uint32_t i = 1; i < 256; ++i) {
    Divisor d(i);
    for (uint32_t dividend = 0; dividend < 1024; ++dividend) {
      EXPECT_EQ(d.GetMod(dividend), dividend % i);
    }
  }
  for (int i = 0; i < 1024; ++i) {
    uint32_t divisor = rand();
    uint32_t dividend = rand();
    if (divisor != 0) {
      Divisor d(divisor);
      EXPECT_EQ(d.GetMod(dividend), dividend % divisor);
    }
  }
}

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  impala::CpuInfo::Init();
  return RUN_ALL_TESTS();
}
