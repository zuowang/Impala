// Copyright 2012 Cloudera Inc.
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


#ifndef IMPALA_EXEC_HDFS_SEQUENCE_SCANNER_H
#define IMPALA_EXEC_HDFS_SEQUENCE_SCANNER_H

// This scanner parses Sequence file located in HDFS, and writes the
// content as tuples in the Impala in-memory representation of data, e.g.
// (tuples, rows, row batches).
//
// TODO: Make the various sequence file formats behave more similarly.  They should
// all have a structure similar to block compressed operating in batches rather than
// row at a time.  
//
// org.apache.hadoop.io.SequenceFile is the original SequenceFile implementation
// and should be viewed as the canonical definition of this format. If
// anything is unclear in this file you should consult the code in
// org.apache.hadoop.io.SequenceFile.
//
// The following is a pseudo-BNF grammar for SequenceFile. Comments are prefixed
// with dashes:
//
// seqfile ::=
//   <file-header>
//   <record-block>+
//
// record-block ::=
//   <record>+
//   <file-sync-hash>
//
// file-header ::=
//   <file-version-header>
//   <file-key-class-name>
//   <file-value-class-name>
//   <file-is-compressed>
//   <file-is-block-compressed>
//   [<file-compression-codec-class>]
//   <file-header-metadata>
//   <file-sync-field>
//
// file-version-header ::= Byte[4] {'S', 'E', 'Q', 6}
//
// -- The name of the Java class responsible for reading the key buffer
//
// file-key-class-name ::=
//   Text {"org.apache.hadoop.io.BytesWritable"}
//
// -- The name of the Java class responsible for reading the value buffer
//
// -- We don't care what this is.
// file-value-class-name ::=
//
// -- Boolean variable indicating whether or not the file uses compression
// -- for key/values in this file
//
// file-is-compressed ::= Byte[1]
//
// -- A boolean field indicating whether or not the file is block compressed.
//
// file-is-block-compressed ::= Byte[1] {false}
//
// -- The Java class name of the compression codec iff <file-is-compressed>
// -- is true. The named class must implement
// -- org.apache.hadoop.io.compress.CompressionCodec.
// -- The expected value is org.apache.hadoop.io.compress.GzipCodec.
//
// file-compression-codec-class ::= Text
//
// -- A collection of key-value pairs defining metadata values for the
// -- file. The Map is serialized using standard JDK serialization, i.e.
// -- an Int corresponding to the number of key-value pairs, followed by
// -- Text key and value pairs.
//
// file-header-metadata ::= Map<Text, Text>
//
// -- A 16 byte marker that is generated by the writer. This marker appears
// -- at regular intervals at the beginning of records or record blocks
// -- intended to enable readers to skip to a random part of the file
// -- the sync hash is preceeded by a length of -1, refered to as the sync marker
//
// file-sync-hash ::= Byte[16]
//
// -- Records are all of one type as determined by the compression bits in the header
//
// record ::=
//   <uncompressed-record>     |
//   <block-compressed-record> |
//   <record-compressed-record>
//
// uncompressed-record ::=
//   <record-length>
//   <key-length>
//   <key>
//   <value>
//
// record-compressed-record ::=
//   <record-length>
//   <key-length>
//   <key>
//   <compressed-value>
//
// block-compressed-record ::=
//   <file-sync-field>
//   <key-lengths-block-size>
//   <key-lengths-block>
//   <keys-block-size>
//   <keys-block>
//   <value-lengths-block-size>
//   <value-lengths-block>
//   <values-block-size>
//   <values-block>
//
// record-length := Int
// key-length := Int
// keys-lengths-block-size> := Int
// value-lengths-block-size> := Int
//
// keys-block :: = Byte[keys-block-size]
// values-block :: = Byte[values-block-size]
//
// -- The key-lengths and value-lengths blocks are are a sequence of lengths encoded
// -- in ZeroCompressedInteger (VInt) format.
//
// key-lengths-block :: = Byte[key-lengths-block-size]
// value-lengths-block :: = Byte[value-lengths-block-size]
//
// Byte ::= An eight-bit byte
//
// VInt ::= Variable length integer. The high-order bit of each byte
// indicates whether more bytes remain to be read. The low-order seven
// bits are appended as increasingly more significant bits in the
// resulting integer value.
//
// Int ::= A four-byte integer in big-endian format.
//
// Text ::= VInt, Chars (Length prefixed UTF-8 characters)

#include "exec/base-sequence-scanner.h"

namespace impala {

class DelimitedTextParser;

class HdfsSequenceScanner : public BaseSequenceScanner {
 public:
  // The four byte SeqFile version header present at the beginning of every
  // SeqFile file: {'S', 'E', 'Q', 6}
  static const uint8_t SEQFILE_VERSION_HEADER[4];

  HdfsSequenceScanner(HdfsScanNode* scan_node, RuntimeState* state);

  virtual ~HdfsSequenceScanner();
  
  // Implementation of HdfsScanner interface.
  virtual Status Prepare(ScannerContext* context);

  // Codegen writing tuples and evaluating predicates.
  static llvm::Function* Codegen(HdfsScanNode*,
                                 const std::vector<ExprContext*>& conjunct_ctxs);

 protected:
  // Implementation of sequence container super class methods.
  virtual FileHeader* AllocateFileHeader();
  virtual Status ReadFileHeader();
  virtual Status InitNewRange();
  virtual Status ProcessRange();
  
  virtual THdfsFileFormat::type file_format() const { 
    return THdfsFileFormat::SEQUENCE_FILE; 
  }

 private:
  // Maximum size of a compressed block.  This is used to check for corrupted
  // block size so we do not read the whole file before we detect the error.
  const static int MAX_BLOCK_SIZE = (1024 * 1024 * 1024);

  // The value class name located in the SeqFile Header.
  // This is always "org.apache.hadoop.io.Text"
  static const char* const SEQFILE_VALUE_CLASS_NAME;

  // Read the record header.
  // Sets:
  //   current_block_length_
  Status ReadBlockHeader();

  // Process an entire block compressed scan range.  Block compressed ranges are 
  // more common and can be parsed more efficiently in larger pieces.
  Status ProcessBlockCompressedScanRange();

  // Read a compressed block. Does NOT read sync or -1 marker preceding sync.
  // Decompress to unparsed_data_buffer_ allocated from unparsed_data_buffer_pool_.
  Status ReadCompressedBlock();

  // Utility function for parsing next_record_in_compressed_block_. Called by
  // ProcessBlockCompressedScanRange.
  Status ProcessDecompressedBlock();

  // Read compressed or uncompressed records from the byte stream into memory
  // in unparsed_data_buffer_pool_.  Not used for block compressed files.
  // Output:
  //   record_ptr: ponter to the record.
  //   record_len: length of the record
  Status GetRecord(uint8_t** record_ptr, int64_t *record_len);
  
  // Appends the current file and line to the RuntimeState's error log.
  // row_idx is 0-based (in current batch) where the parse error occurred.
  virtual void LogRowParseError(int row_idx, std::stringstream*);
  
  // Helper class for picking fields and rows from delimited text.
  boost::scoped_ptr<DelimitedTextParser> delimited_text_parser_;
  std::vector<FieldLocation> field_locations_;

  // Data that is fixed across headers.  This struct is shared between scan ranges.
  struct SeqFileHeader : public BaseSequenceScanner::FileHeader {
    // If true, the file uses row compression
    bool is_row_compressed;
  };

  // Struct for record locations and lens in compressed blocks.  
  struct RecordLocation {
    uint8_t* record;
    int64_t len;
  };

  // Records are processed in batches.  This vector stores batches of record locations
  // that are being processed.  
  // TODO: better perf not to use vector?
  std::vector<RecordLocation> record_locations_;

  // Length of the current sequence file block (or record).
  int current_block_length_;

  // Length of the current key.  This is specified as 4 bytes in the format description.
  int current_key_length_;

  // Buffer for data read from HDFS or from decompressing the HDFS data.
  uint8_t* unparsed_data_buffer_;

  // Number of buffered records unparsed_data_buffer_ from block compressed data.
  int64_t num_buffered_records_in_compressed_block_;

  // Next record from block compressed data.
  uint8_t* next_record_in_compressed_block_;
};

} // namespace impala

#endif // IMPALA_EXEC_HDFS_SEQUENCE_SCANNER_H
