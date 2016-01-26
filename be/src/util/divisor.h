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


#ifndef IMPALA_UTIL_DIVISOR_H
#define IMPALA_UTIL_DIVISOR_H

#include <stdint.h>

namespace impala {

class Divisor;

/// Fast division algorithm when denominator is a constant, roughly twice as fast as %.
/// Here is how it works:
/// Suppose m, d, l are nonnegative integers such that d != 0 and
/// 2^(N+l) <= m * d <= 2^(N+l) + 2^l.
/// Then floor(n / d) = floor(m * n / 2^(N+l)) for every integer n with 0 <= n < 2^N.
/// Let l = ceil(log2(d)), m = floor(2^(N+l) / d) + 1 Then
/// q = floor(m * n / 2^(N+l)) = floor((m - 2^N) * n / 2^(N+l) + n / 2^l)
/// Since m - 2^N = floor(2^(N+l) / d) - 2^N + 1 = floor(2^N * (2^l - d) / d) + 1
/// we can pre-caculate l and m - 2^N to acccelate the calculation of q.
/// We choose N = 32, so the divisor is of type uint32_t.
/// Please see the paper for detail.
/// T. Granlund and P. L. Montgomery, Division by invariant integers using
/// multiplication, https://gmplib.org/~tege/divcnst-pldi94.pdf.
class Divisor {
 public:
  Divisor(uint32_t divisor) {
    divisor_ = divisor;
    switch (divisor) {
      case 0:
        ceil_log_divisor_ = 0;
        multiplier_ = 0;
        break;
      case 1:
        ceil_log_divisor_ = 0;
        multiplier_ = 1;
        break;
      case 2:
        ceil_log_divisor_ = 1;
        multiplier_ = 1;
        break;
      default:
        /// Calculate ceil(log2(divisor))
        /// Let 2^(k-1) < divisor <= 2^k, Then ceil(log2(divisor)) = k.
        /// 2^(k-1) <= (divisor - 1) < 2^k, (divisor - 1) got (64 - k) leading zeros,
        /// hence 64 - __builtin_clzl(divisor - 1) = 64 - (64 - k) = k
        ceil_log_divisor_ = 64 - __builtin_clzl(divisor - 1);
        /// Calculate 2^ceil_log_divisor_, overflow to 0 if ceil_log_divisor_ = 32
        const uint32_t tmp = ceil_log_divisor_ < 32 ? 1 << ceil_log_divisor_ : 0;
        multiplier_ = 1 + (static_cast<uint64_t>(tmp - divisor) << 32) / divisor;
    }
  }

  uint32_t GetMod(uint32_t dividend) const {
    const uint64_t m = (static_cast<uint64_t>(dividend) * multiplier_) >> 32;
    const uint64_t q = (m + dividend) >> ceil_log_divisor_;
    return dividend - q * divisor_;
  }

  /// This is only used for testing.
  uint32_t multiplier() const { return multiplier_; }

 private:
  uint32_t divisor_;
  uint32_t multiplier_;
  uint32_t ceil_log_divisor_;
};

}

#endif
