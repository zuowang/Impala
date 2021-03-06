# Copyright 2012 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


cmake_minimum_required(VERSION 2.6)

# where to put generated libraries
set(LIBRARY_OUTPUT_PATH "${BUILD_OUTPUT_ROOT_DIRECTORY}/exec")

# where to put generated binaries
set(EXECUTABLE_OUTPUT_PATH "${BUILD_OUTPUT_ROOT_DIRECTORY}/exec")

add_library(Exec
  aggregation-node.cc
  aggregation-node-ir.cc
  analytic-eval-node.cc
  base-sequence-scanner.cc
  blocking-join-node.cc
  catalog-op-executor.cc
  data-sink.cc
  data-source-scan-node.cc
  delimited-text-parser.cc
  empty-set-node.cc
  exec-node.cc
  exchange-node.cc
  external-data-source-executor.cc
  filter-context.cc
  hash-join-node.cc
  hash-join-node-ir.cc
  old-hash-table.cc
  hash-table.cc
  hbase-table-sink.cc
  hbase-table-writer.cc
  hdfs-scan-node.cc
  hdfs-scanner.cc
  hdfs-scanner-ir.cc
  hdfs-table-sink.cc
  hdfs-table-writer.cc
  hdfs-rcfile-scanner.cc
  hdfs-sequence-scanner.cc
  hdfs-avro-scanner.cc
  hdfs-avro-table-writer.cc
  hdfs-avro-scanner-ir.cc
  hdfs-text-scanner.cc
  hdfs-lzo-text-scanner.cc
  hdfs-text-table-writer.cc
  hdfs-sequence-table-writer.cc
  hdfs-parquet-scanner.cc
  hdfs-parquet-table-writer.cc
  hbase-scan-node.cc
  hbase-table-scanner.cc
  incr-stats-util.cc
  nested-loop-join-node.cc
  partitioned-aggregation-node.cc
  partitioned-aggregation-node-ir.cc
  partitioned-hash-join-node.cc
  partitioned-hash-join-node-ir.cc
  kudu-scanner.cc
  kudu-scan-node.cc
  kudu-table-sink.cc
  kudu-util.cc
  read-write-util.cc
  scan-node.cc
  scanner-context.cc
  select-node.cc
  singular-row-src-node.cc
  sort-exec-exprs.cc
  sort-node.cc
  subplan-node.cc
  text-converter.cc
  topn-node.cc
  union-node.cc
  unnest-node.cc
)

# Some of the macros defined in gutil (COMPILE_ASSERT, DISALLOW_COPY_AND_ASSIGN)
# are redefined in Kudu's kudu/client/stubs.h and because "-Werror" is set, compilation
# fails on macro redefinition warnings. The ideal workaround would be to just disable
# that particular warning (which is possible in recent LLVM releases with
# "-Wno-macro-redefined") but gcc doesn't have a flag to do it, so we simply disable
# "-Werror" for these particular files.
set_source_files_properties(kudu-scanner.cc PROPERTIES COMPILE_FLAGS
  "${CXX_COMMON_FLAGS} -Wno-error")
set_source_files_properties(kudu-scan-node.cc PROPERTIES COMPILE_FLAGS
  "${CXX_COMMON_FLAGS} -Wno-error")
set_source_files_properties(kudu-table-sink.cc PROPERTIES COMPILE_FLAGS
  "${CXX_COMMON_FLAGS} -Wno-error")
set_source_files_properties(kudu-util.cc PROPERTIES COMPILE_FLAGS
  "${CXX_COMMON_FLAGS} -Wno-error")

add_dependencies(Exec thrift-deps)

ADD_BE_TEST(zigzag-test)
ADD_BE_TEST(old-hash-table-test)
ADD_BE_TEST(hash-table-test)
ADD_BE_TEST(delimited-text-parser-test)
ADD_BE_TEST(read-write-util-test)
ADD_BE_TEST(parquet-plain-test)
ADD_BE_TEST(parquet-version-test)
ADD_BE_TEST(row-batch-list-test)
ADD_BE_TEST(incr-stats-util-test)
if($ENV{KUDU_IS_SUPPORTED} STREQUAL true)
  ADD_BE_TEST(kudu-scan-node-test)
  ADD_BE_TEST(kudu-table-sink-test)
  SET_TARGET_PROPERTIES(kudu-scan-node-test PROPERTIES COMPILE_FLAGS
    "${CXX_COMMON_FLAGS} -Wno-error")
  SET_TARGET_PROPERTIES(kudu-table-sink-test PROPERTIES COMPILE_FLAGS
    "${CXX_COMMON_FLAGS} -Wno-error")
endif()
