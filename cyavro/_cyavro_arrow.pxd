from __future__ import absolute_import
from pyarrow.includes.libarrow cimport *
from pyarrow.lib cimport DataType

from ._cavro cimport avro_file_reader_t
from ._cyavro cimport AvroReaderType

cdef class AvroArrowReader:
    cdef DataType arrow_schema
    cdef avro_file_reader_t _reader
    cdef public int chunk_size
    cdef public list refholder, field_names, field_types, buffer_lst
    cdef public str filename
    cdef int initialized, empty_file, _should_free_buffer
    cdef void *fp_reader_buffer
    cdef int fp_reader_buffer_length
    cdef public bytes filedata
    cdef int filedatalength
    cdef AvroReaderType reader_type
    cdef init_reader_buffer(self)
    cdef init_file_reader(self)
    cdef init_memory_reader(self)


cdef extern from "arrow/builder.h" namespace "arrow":

    cdef cppclass CArrayBuilder" arrow::ArrayBuilder":

    # const std::shared_ptr<DataType>& type, MemoryPool* pool)

        CArrayBuilder* child(int i)
        int num_children()

        int64_t length()
        int64_t null_count()
        int64_t capacity()


        # Append to null bitmap
        CStatus AppendToBitmap(c_bool is_valid)
        # Vector append. Treat each zero byte as a null.   If valid_bytes is null
        # assume all of length bits are valid.
        CStatus AppendToBitmap(const uint8_t* valid_bytes, int64_t length);
        # Set the next length bits to not null (i.e. valid).
        CStatus SetNotNull(int64_t length);

        CStatus Init(int64_t capacity)
        CStatus Resize(int64_t capacity)
        CStatus Reserve(int64_t capacity)

        CStatus Finish(shared_ptr[CArray]* )

        shared_ptr[CDataType] type()

    cdef cppclass CNullBuilder" arrow::NullBuilder"(CArrayBuilder):
        CStatus AppendNull()

    cdef cppclass CBooleanBuilder" arrow::BooleanBuilder"(CArrayBuilder):
        CStatus AppendNull()
        CStatus Append(const c_bool)
        CStatus Append(const uint8_t)

    cdef cppclass CInt32Builder" arrow::Int32Builder"(CArrayBuilder):
        CStatus AppendNull()
        CStatus Append(const int32_t)

    cdef cppclass CInt64Builder" arrow::Int64Builder"(CArrayBuilder):
        CStatus AppendNull()
        CStatus Append(const int64_t)

    cdef cppclass CFloatBuilder" arrow::FloatBuilder"(CArrayBuilder):
        CStatus AppendNull()
        CStatus Append(const float)

    cdef cppclass CDoubleBuilder" arrow::DoubleBuilder"(CArrayBuilder):
        CStatus AppendNull()
        CStatus Append(const double)

    cdef cppclass CInt64Builder" arrow::Int64Builder"(CArrayBuilder):
        CStatus AppendNull()
        CStatus Append(const int64_t)

    cdef cppclass CListBuilder" arrow::ListBuilder"(CArrayBuilder):
        CStatus AppendNull()
        CStatus Append(c_bool isvalid)

    cdef cppclass CBinaryBuilder" arrow::BinaryBuilder"(CArrayBuilder):
        CStatus Append(const uint8_t* value, int32_t length)

        CStatus Append(const char* value, int32_t length)

        CStatus Append(const c_string& value)
        CStatus AppendNull()

    cdef cppclass CStringBuilder" arrow::BinaryBuilder"(CBinaryBuilder):
        pass

    cdef cppclass CFixedSizeBinaryBuilder" arrow::FixedSizeBinaryBuilder"(CArrayBuilder):
        CStatus Append(const uint8_t* value)
        CStatus Append(const uint8_t* data, int64_t length,
                const uint8_t* valid_bytes = nullptr)
        CStatus Append(const char* value, int32_t length)

        CStatus Append(const c_string& value)
        CStatus AppendNull()

    cdef cppclass CStructBuilder" arrow::StructBuilder"(CArrayBuilder):
        CStatus AppendNull()
        CStatus Append(c_bool isvalid)

    cdef CStatus MakeBuilder( CMemoryPool* pool, shared_ptr[CDataType]& type, unique_ptr[CArrayBuilder]* out);


ctypedef CNullBuilder * CPNullBuilder
ctypedef CBinaryBuilder * CPBinaryBuilder
ctypedef CStringBuilder * CPStringBuilder
ctypedef CFixedSizeBinaryBuilder * CPFixedSizeBinaryBuilder
ctypedef CBooleanBuilder * CPBooleanBuilder
ctypedef CInt32Builder * CPInt32Builder
ctypedef CInt64Builder * CPInt64Builder
ctypedef CFloatBuilder * CPFloatBuilder
ctypedef CDoubleBuilder * CPDoubleBuilder

ctypedef CStructBuilder * CPStructBuilder
ctypedef CListBuilder * CPListBuilder