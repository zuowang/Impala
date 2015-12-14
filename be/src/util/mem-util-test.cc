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

#include <gtest/gtest.h>
#include "util/cpu-info.h"
#include "util/mem-util.h"

#include "common/names.h"

namespace impala {

TEST(MemUtil, memcpy) {
  for (int i = 0; i < 16384; ++i) {
    uint8_t dst[i];
    uint8_t src[i];
    for (int j = 0; j < i; ++j) {
      src[j] = rand() & 0xff;
    }
    MemUtil::memcpy(dst, src, i);
    EXPECT_TRUE(memcmp(dst, src, i) == 0);
  }
}

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  impala::CpuInfo::Init();
  return RUN_ALL_TESTS();
}
