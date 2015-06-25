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

#include "exprs/simple-predicates.h"

bool AndOperate::GetBitVec(int64_t num_rows, vector<int64_t>& skip_bit_vec) {
  DCHECK(child0_);
  DCHECK(child1_);
  vector<int64_t> bit_vec0(skip_bit_vec.size());
  if (!child0_->GetBitVec(num_rows, bit_vec0)) return false;
  vector<int64_t> bit_vec1(skip_bit_vec.size());
  if (!child1_->GetBitVec(num_rows, bit_vec1)) return false;
  for (int i = 0; i < skip_bit_vec.size(); ++i) {
    skip_bit_vec[i] = bit_vec0[i] & bit_vec1[i];
  }
  return true;
}

void OrOperate::GetBitVec(int64_t num_rows, vector<int64_t>& skip_bit_vec) {
  DCHECK(child0_);
  DCHECK(child1_);
  vector<int64_t> bit_vec0(skip_bit_vec.size());
  if (!child0_->GetBitVec(num_rows, bit_vec0)) return false;
  vector<int64_t> bit_vec1(skip_bit_vec.size());
  if (!child1_->GetBitVec(num_rows, bit_vec1)) return false;
  for (int i = 0; i < skip_bit_vec.size(); ++i) {
    skip_bit_vec[i] = bit_vec0[i] | bit_vec1[i];
  }
  return true;
}

bool EqOperate::GetBitVec(int64_t num_rows, vector<int64_t>& skip_bit_vec) {
  DCHECK(column_reader_);
  return column_reader_->dict_decoder()->Eq(num_rows, skip_bit_vec, val_);
}

bool LtOperate::GetBitVec(int64_t num_rows, vector<int64_t>& skip_bit_vec) {
  DCHECK(column_reader_);
  return column_reader_->dict_decoder()->Lt(num_rows, skip_bit_vec, val_);
}

bool LeOperate::GetBitVec(int64_t num_rows, vector<int64_t>& skip_bit_vec) {
  DCHECK(column_reader_);
  return column_reader_->dict_decoder()->Le(num_rows, skip_bit_vec, val_);
}

bool GtOperate::GetBitVec(int64_t num_rows, vector<int64_t>& skip_bit_vec) {
  DCHECK(column_reader_);
  return column_reader_->dict_decoder()->Gt(num_rows, skip_bit_vec, val_);
}

bool GeOperate::GetBitVec(int64_t num_rows, vector<int64_t>& skip_bit_vec) {
  DCHECK(column_reader_);
  return column_reader_->dict_decoder()->Ge(num_rows, skip_bit_vec, val_);
}

