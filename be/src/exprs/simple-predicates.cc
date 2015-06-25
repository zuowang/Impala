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

using namespace impala;

namespace impala {

void AndOperate::GetBitset(int64_t num_rows, boost::dynamic_bitset<>& skip_bitset) {
  DCHECK(child0_);
  DCHECK(child1_);
  child0_->GetBitset(num_rows, skip_bitset);
  boost::dynamic_bitset<> tmp_bitset;
  child1_->GetBitset(num_rows, tmp_bitset);
  skip_bitset &= tmp_bitset;
}

void OrOperate::GetBitset(int64_t num_rows, boost::dynamic_bitset<>& skip_bitset) {
  DCHECK(child0_);
  DCHECK(child1_);
  child0_->GetBitset(num_rows, skip_bitset);
  boost::dynamic_bitset<> tmp_bitset;
  child1_->GetBitset(num_rows, tmp_bitset);
  skip_bitset |= tmp_bitset;
}

template <typename T>
void EqOperate<T>::GetBitset(int64_t num_rows, boost::dynamic_bitset<>& skip_bitset) {
  DCHECK(column_reader_);
  column_reader_->Eq(num_rows, skip_bitset, val_);
}

template <typename T>
void LtOperate<T>::GetBitset(int64_t num_rows, boost::dynamic_bitset<>& skip_bitset) {
  DCHECK(column_reader_);
  column_reader_->Lt(num_rows, skip_bitset, val_);
}

template <typename T>
void LeOperate<T>::GetBitset(int64_t num_rows, boost::dynamic_bitset<>& skip_bitset) {
  DCHECK(column_reader_);
  column_reader_->Le(num_rows, skip_bitset, val_);
}

template <typename T>
void GtOperate<T>::GetBitset(int64_t num_rows, boost::dynamic_bitset<>& skip_bitset) {
  DCHECK(column_reader_);
  column_reader_->Gt(num_rows, skip_bitset, val_);
}

template <typename T>
void GeOperate<T>::GetBitset(int64_t num_rows, boost::dynamic_bitset<>& skip_bitset) {
  DCHECK(column_reader_);
  column_reader_->Ge(num_rows, skip_bitset, val_);
}

//template class EqOperate<int8_t>;
//template class EqOperate<int16_t>;
//template class EqOperate<int32_t>;
//template class EqOperate<int64_t>;
//template class EqOperate<float>;
//template class EqOperate<double>;
//template class EqOperate<TimestampValue>;
//template class EqOperate<StringValue>;
//template class EqOperate<Decimal4Value>;
//template class EqOperate<Decimal8Value>;
//template class EqOperate<Decimal16Value>;

//template void EqOperate<int8_t>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void EqOperate<int16_t>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void EqOperate<int32_t>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void EqOperate<int64_t>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void EqOperate<float>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void EqOperate<double>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void EqOperate<TimestampValue>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void EqOperate<StringValue>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void EqOperate<Decimal4Value>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void EqOperate<Decimal4Value>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void EqOperate<Decimal4Value>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void EqOperate<Decimal4Value>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//
//template void LtOperate<int8_t>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void LtOperate<int16_t>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void LtOperate<int32_t>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void LtOperate<int64_t>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void LtOperate<float>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void LtOperate<double>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void LtOperate<TimestampValue>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void LtOperate<StringValue>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void LtOperate<Decimal4Value>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void LtOperate<Decimal4Value>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void LtOperate<Decimal4Value>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void LtOperate<Decimal4Value>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//
//template void LeOperate<int8_t>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void LeOperate<int16_t>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void LeOperate<int32_t>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void LeOperate<int64_t>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void LeOperate<float>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void LeOperate<double>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void LeOperate<TimestampValue>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void LeOperate<StringValue>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void LeOperate<Decimal4Value>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void LeOperate<Decimal4Value>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void LeOperate<Decimal4Value>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void LeOperate<Decimal4Value>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//
//template void GtOperate<int8_t>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void GtOperate<int16_t>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void GtOperate<int32_t>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void GtOperate<int64_t>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void GtOperate<float>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void GtOperate<double>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void GtOperate<TimestampValue>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void GtOperate<StringValue>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void GtOperate<Decimal4Value>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void GtOperate<Decimal4Value>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void GtOperate<Decimal4Value>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void GtOperate<Decimal4Value>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//
//template void GeOperate<int8_t>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void GeOperate<int16_t>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void GeOperate<int32_t>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void GeOperate<int64_t>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void GeOperate<float>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void GeOperate<double>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void GeOperate<TimestampValue>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void GeOperate<StringValue>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void GeOperate<Decimal4Value>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void GeOperate<Decimal4Value>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void GeOperate<Decimal4Value>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);
//template void GeOperate<Decimal4Value>::GetBitset(int64_t num_rows,
//    boost::dynamic_bitset<>& skip_bitset);

}
