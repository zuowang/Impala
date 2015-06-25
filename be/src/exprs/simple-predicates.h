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


#ifndef IMPALA_EXPRS_SIMPLE_PREDICATES_H_
#define IMPALA_EXPRS_SIMPLE_PREDICATES_H_

#include <string>
#include "util/fle-encoding.h"

using namespace impala_udf;

namespace impala {

class SimplePredicate {
 public:
  virtual bool GetBitVec(num_rows, skip_bit_vec, flag) = 0;
};

class AndOperate: public SimplePredicate {
 public:
  virtual bool GetBitVec(num_rows, skip_bit_vec, flag);

 protected:
  AndOperate(SimplePredicate* child0, SimplePredicate* child1) : child0_(child0), child1_(child1) { }

 private:
  SimplePredicate* child0_;
  SimplePredicate* child1_;
};

class OrOperate: public SimplePredicate {
 public:
  virtual bool GetBitVec(num_rows, skip_bit_vec, flag);

 protected:
  OrOperate(SimplePredicate* child0, SimplePredicate* child1) : child0_(child0), child1_(child1) { }

 private:
  SimplePredicate* child0_;
  SimplePredicate* child1_;
};

template <typename T>
class EqOperate: public SimplePredicate {
 public:
  virtual bool GetBitVec(num_rows, skip_bit_vec, flag);

 protected:
  EqOperate(ColumnReader* column_reader, T val) : column_reader_(column_reader), val_(val) { }

 private:
  ColumnReader* column_reader_;
  T val_;
};

template <typename T>
class LtOperate: public SimplePredicate {
 public:
  virtual bool GetBitVec(num_rows, skip_bit_vec, flag);

 protected:
  LtOperate(ColumnReader* column_reader, T val) : column_reader_(column_reader), val_(val) { }

 private:
  ColumnReader* column_reader_;
  T val_;
};

template <typename T>
class LeOperate: public SimplePredicate {
 public:
  virtual bool GetBitVec(num_rows, skip_bit_vec, flag);

 protected:
  LeOperate(ColumnReader* column_reader, T val) : column_reader_(column_reader), val_(val) { }

 private:
  ColumnReader* column_reader_;
  T val_;
};

template <typename T>
class GtOperate: public SimplePredicate {
 public:
  virtual bool GetBitVec(num_rows, skip_bit_vec, flag);

 protected:
  GtOperate(ColumnReader* column_reader, T val) : column_reader_(column_reader), val_(val) { }

 private:
  ColumnReader* column_reader_;
  T val_;
};

template <typename T>
class GeOperate: public SimplePredicate {
 public:
  virtual bool GetBitVec(num_rows, skip_bit_vec, flag);

 protected:
  GeOperate(ColumnReader* column_reader, T val) : column_reader_(column_reader), val_(val) { }

 private:
  ColumnReader* column_reader_;
  T val_;
};

}

#endif
