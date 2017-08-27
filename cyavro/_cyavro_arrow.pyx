# Copyright (c) 2015 MaxPoint Interactive, Inc.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#    disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products
#    derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#cython: boundscheck=False
#cython: wraparound=False
#cython: profile=True
#cython: embedsignature=True

"""
This contains the cython based implementation of avro reader and writer classes.

For runtime performance this is compiled without bounds checking or wraparound.

Profile is enabled by default for easier debugging of the code with gdb.
embedsignature allows docstrings to propogate to Python.
"""

from __future__ import absolute_import
from cython.view cimport array as cvarray
from cython.operator cimport dereference as deref
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np

import pyarrow as pa
cimport pyarrow as pa

from libc.stdint cimport int32_t, int64_t
from _cavro cimport *
from _cyavro_arrow cimport *
from six import string_types, binary_type, iteritems
from libc.string cimport memcpy

# from posix.stdio cimport *  # New on cython, not yet released
from posixstdio cimport *

# Globals
cdef int64_t PANDAS_NaT = np.datetime64('nat').astype('int64') # Not a time from pandas
# numpy / pandas do not support nullable int32/int64 dtypes.  These values are treated as avro nulls.
cdef int64_t SENTINEL_INT64_NULL = -9223372036854775808
cdef int32_t SENTINEL_INT32_NULL = -2147483648

###################################################################################################
#
#   Reader Methods and classes
#
###################################################################################################

cdef class AvroArrowReader(object):
    """Avro Reader class for reading chunks from an avro file.

    Parameters
    ----------
    filename : str
        Input filename to be read

    Examples
    --------
    >>> rd = AvroReader('/path/to/file.avro')
    >>> rd.init_reader()
    >>> rd.init_buffers(10000)
    >>> while True:
    ...     df = rd.read_chunk()
    ...     ...
    ...     # Do something with df
    ...     ...
    ...     if len(df) == 0:
    ...         break
    ...
    >>> rd.close()

    """
    cdef pa.DataType arrow_schema

    def __cinit__(self):
        self._reader = NULL
        self.fp_reader_buffer = NULL
        self.fp_reader_buffer_length = 0
        self._should_free_buffer = True

    def __init__(self):
        self.chunk_size = 10000
        self.refholder = []
        self.field_names = []
        self.field_types = []
        self.empty_file = False
        self.reader_type = avro_reader_type_unset
        self.initialized = False

    def init_file(self, str filename):
        self.filename = filename
        self.reader_type = avro_reader_type_file

    def init_bytes(self, bytes data):
        self.filedata = data
        self.filedatalength = len(data) + 1 
        self.reader_type = avro_reader_type_bytes

    def __dealloc__(self):
        # Ensure that we close the reader properly
        cdef avro_file_reader_t filereader
        if self._reader != NULL:
            try:
                filereader = self._reader
                avro_file_reader_close(filereader)
            finally:
                self._reader = NULL
        if self.fp_reader_buffer != NULL:
            free(self.fp_reader_buffer)
            self.fp_reader_buffer = NULL

    def close(self):
        cdef avro_file_reader_t filereader
        if self._reader != NULL:
            filereader = self._reader
            avro_file_reader_close(filereader)
            self._reader = NULL
        if self.fp_reader_buffer != NULL:
            free(self.fp_reader_buffer)
            self.fp_reader_buffer = NULL

    def init_reader(self):
        if self.reader_type == avro_reader_type_file:
            self.init_file_reader()
        elif self.reader_type == avro_reader_type_bytes:
            #raise Exception("DEATH")
            self.init_memory_reader()

    cdef init_memory_reader(self):
        #raise Exception("ASDASD")

        cdef char* cbytes = self.filedata
        cdef int size = len(self.filedata)
        cdef void *dest = malloc(size + 1)
        memcpy(dest, cbytes, size)
        self.fp_reader_buffer = dest
        self.fp_reader_buffer_length = size
        self.init_reader_buffer()

    cdef init_file_reader(self):
        """Initialize the file reader object.  This must be called before calling :meth:`init_buffers`
        """

        cdef avro_file_reader_t filereader
        py_byte_string = self.filename.encode('UTF-8')
        cdef char* c_filename =  py_byte_string

        cdef avro_value_t record
        cdef int rval
        cdef avro_type_t avro_type

        rval = avro_file_reader(c_filename, &filereader)
        if rval != 0:
            avro_error = avro_strerror().decode('UTF-8')
            if 'Cannot read file block count' in avro_error:
                self.empty_file = True
            else:
                raise Exception("Can't read file : {}".format(avro_error))

        self._reader = filereader
        self.initialized = True

    cdef init_reader_buffer(self):
        """Initialize the file reader object based on a File Description using
        avro_file_reader_fp
        """
        cdef FILE* cfile = fmemopen(self.fp_reader_buffer, self.fp_reader_buffer_length, "rb")

        cdef avro_file_reader_t filereader
        cdef rval = avro_file_reader_fp(cfile, "unused", 0, &filereader)

        if rval != 0:
            avro_error = avro_strerror().decode('UTF-8')
            if 'Cannot read file block count' in avro_error:
                self.empty_file = True
            else:
                raise Exception("Can't read file : {}".format(avro_error))
        self._reader = filereader
        self.initialized = True

    def init_buffers(self, size_t chunk_size=0):
        """Initialize the buffers for the reader object.  This must be called before calling :meth:`read_chunk`

        This method also initializes and parses the avro schema for the file in question.

        Parameters
        ----------
        chunk_size : size_t
            Number of records to read in chunk.  This essentially determines how much memory we are allowing the reader
            to use.

        """
        if not self.initialized:
            raise Exception("Reader not initialized")

        # If the file contains nothing: do nothing.
        if self.empty_file:
            return

        if chunk_size != 0:
            self.chunk_size = chunk_size

        cdef:
            avro_file_reader_t filereader = self._reader
            int rval
            avro_schema_t wschema
            avro_value_iface_t *iface
            avro_value_t record
            avro_value_t child
            avro_type_t avro_type
            size_t actual_size
            size_t i
            const char* map_key = NULL
            list thelist
            np.ndarray thearray

        # Retrieve the avro schema stored in the file and get the first record.
        wschema = avro_file_reader_get_writer_schema(filereader)
        if wschema.type != AVRO_RECORD:
            raise Exception('Non-record types are not supported.')

        cdef pa.DataType get_elem_type(avro_schema_t schema):
            # primitives
            cdef:
                avro_schema_t child
                size_t size, idx
                int64_t length
                str name

            avro_type = schema.type
            if avro_type == AVRO_INT32:
                return pa.int32()
            elif avro_type == AVRO_INT64:
                return pa.int64()
            elif avro_type == AVRO_FLOAT:
                return pa.float32()
            elif avro_type == AVRO_DOUBLE:
                return pa.float64()
            elif avro_type in AVRO_BOOLEAN:
                return pa.bool_()
            elif avro_type in AVRO_STRING:
                return pa.string()
            elif avro_type in  AVRO_BYTES:
                return pa.binary()
            elif avro_type in AVRO_FIXED:
                length = avro_schema_fixed_size(avro_type)
                return pa.binary(length)
            elif avro_type in AVRO_ARRAY:
                elem_type = get_elem_type(avro_schema_array_items(schema))
                return pa.list_(elem_type)
            elif avro_type in AVRO_MAP:
                value_type = get_elem_type(avro_schema_map_values(schema))
                raise NotImplementedError("Arrow doesn't do maps yet")
            elif avro_type in AVRO_UNION:
                size = avro_schema_union_size(schema)
                arrow_types = []
                for idx in range(size):
                    child = avro_schema_union_branch(schema, idx)
                    arrow_types.append(get_elem_type(child))
                raise NotImplementedError("Arrow doesn't do unions yet")
            elif avro_type in AVRO_RECORD:
                size = avro_schema_record_size(schema)
                arrow_fields = list()
                for idx in range(size):
                    child = avro_schema_record_field_get_by_index(schema, idx)
                    elem_type = get_elem_type(child)
                    name = avro_schema_record_field_name(schema, idx)
                    pa.field(name, elem_type)
                pa.struct(arrow_fields)
            else:
                raise Exception('Unexpected type ({})'.format(avro_type))

        arrow_schema = get_elem_type(wschema)
        self.arrow_schema = arrow_schema

    def read_chunk(self):
        """Reads a chunk of records from the avro file.

        If the avro file is completed being read, this will return an empty dictionary.  It is the responsibility of the
        caller to then call :meth:`close`

        Returns
        -------
        out : dict
            A dictionary mapping avro record field names to a numpy array / list

        Examples
        --------
        >>> reader = AvroReader(...)
        >>> ...
        >>> import pandas as pd
        >>> chunk = pd.DataFrame(reader.read_chunk())

        """

        if self.empty_file:
            return dict()

        cdef:
            size_t counter = 0
            avro_file_reader_t filereader = self._reader
            avro_schema_t wschema
            avro_value_iface_t *iface
            avro_value_t record

        cdef unique_ptr[CArrayBuilder] builder
        MakeBuilder(memory_pool, self.arrow_schema.sp_type, &builder)

        # Get avro reader schema from the file
        wschema = avro_file_reader_get_writer_schema(filereader)
        iface = avro_generic_class_from_schema(wschema)
        avro_generic_value_new(iface, &record)

        while True:
            rval = avro_file_reader_read_value(filereader, &record)
            if rval != 0:
                break
            # decompose record into Python types
            read_record(record, builder)
            avro_value_reset(&record)
            counter += 1
            if counter == self.chunk_size:
                break

        # builder finish
        cdef shared_ptr[CArray] out;
        deref(builder).Finish(&out)

        pa.StructArray()

        # set sizing for output buffers
        out = dict()
        for name, avro_type, array in zip(self.field_names, self.field_types, self.refholder):
            #print((name, avro_type, type(array)))
            if avro_type == AVRO_BOOLEAN:
                out[name] = array[:counter].astype('bool')
            else:
                out[name] = array[:counter]

        # Reference count cleanup
        avro_value_decref(&record)
        avro_value_iface_decref(iface)
        avro_schema_decref(wschema)

        return out


cdef arrow_reader_from_bytes_c(void *buffer, int length):
    """Returns an AvroReader based on a buffer of bytes.
    Useful for other cython extentions that already have a `void *`
    Caller should call `init_buffers`
    """
    reader = AvroArrowReader()
    reader._should_free_buffer = False
    reader.fp_reader_buffer = buffer
    reader.fp_reader_buffer_length = length
    reader.init_reader_buffer()
    return reader


cdef CStatus read_record(const avro_value_t val, CArrayBuilder builder):
    """Main read function for the root level record type.

    Dispatches to the correct reader function for each of the fields in the record.

    Parameters
    ----------
    val : avro_value_t
        Record to be read into python types
    container : list of list/array
        List of length `n_fields`.  Each element of this list is another container
    row : size_t
        Row number.  This is used to directly index into the list/array
    """
    cdef:
        size_t i
        list subcontainerlist
        np.ndarray subcontainerarray
        avro_type_t avro_type
        avro_value_t child
        CArrayBuilder* child_builder
        CStatus result

    container_length = builder.num_children()
    (<CStructBuilder> builder).Append(True)
    for i in range(container_length):
        avro_value_get_by_index(&val, i, &child, NULL)
        avro_type = avro_value_get_type(&child)
        child_builder = builder.child(i)
        result = generic_read(child, child_builder)
    return result

# Primitive read functions.  These all take the same basic form
#
# * call the appropriate read method from libavro
# * convert to a python type if needed
# * assign that read value to the subcontainer if present
# * return the value.

cdef CStatus read_string(const avro_value_t val, CArrayBuilder builder) nogil:
    cdef:
        size_t strlen
        const char* c_string = NULL
        list l
    avro_value_get_string(&val, &c_string, &strlen)
    return (<CStringBuilder> builder).Append(&c_string, strlen)


cdef CStatus read_bytes(const avro_value_t val, CArrayBuilder builder) nogil:
    cdef:
        size_t strlen
        const char* c_string = NULL
        list l
        bytes py_bytes
    avro_value_get_bytes(&val, <void**> &c_string, &strlen)
    return (<CBinaryBuilder> builder).Append(&c_string, strlen)


cdef CStatus read_fixed(const avro_value_t val, CArrayBuilder builder) nogil:
    cdef:
        size_t strlen
        const char* c_string = NULL
        list l
        bytes py_bytes
    avro_value_get_fixed(&val,  <void**> &c_string, &strlen)
    return (<CFixedSizeBinaryBuilder> builder).Append(&c_string, strlen)


# numpy primitive read functions.  These all write to ndarrays.

cdef CStatus read_int32(const avro_value_t val, CArrayBuilder builder) nogil:
    cdef int32_t out
    avro_value_get_int(&val, &out)
    return (<CInt32Builder> builder).Append(out)


cdef CStatus read_int64(const avro_value_t val, CArrayBuilder builder) nogil:
    cdef int64_t out
    avro_value_get_long(&val, &out)
    return (<CInt64Builder> builder).Append(out)


cdef CStatus read_float64(const avro_value_t val, CArrayBuilder builder) nogil:
    cdef double out
    avro_value_get_double(&val, &out)
    return (<CDoubleBuilder> builder).Append(out)


cdef CStatus read_float32(avro_value_t val, CArrayBuilder builder) nogil:
    cdef float out
    avro_value_get_float(&val, &out)
    return (<CFloatBuilder> builder).Append(out)


cdef CStatus read_bool(const avro_value_t val, CArrayBuilder builder) nogil:
    cdef int32_t out
    cdef int temp
    avro_value_get_boolean(&val, &temp)
    return (<CBooleanBuilder> builder).Append(out)


cdef CStatus read_enum(const avro_value_t val, CArrayBuilder builder) nogil:
    # TODO
    cdef int32_t out
    cdef int temp
    avro_value_get_enum(&val, &temp)
    out = temp
    return (CInt32Builder).Append(out)


cdef CStatus read_null(const avro_value_t val, CArrayBuilder builder) nogil:
    return (<CNullBuilder> builder).AppendNull

# Read methods for union types.

cdef CStatus read_union(const avro_value_t val, CArrayBuilder builder) nogil:
    """Unions are only supported for [null, something types]"""
    cdef avro_value_t child
    avro_value_get_current_branch(&val, &child)
    if avro_value_get_type(&val) == AVRO_NULL:
        # TODO figure out how to cast + append properly
        builder
    else:
        return generic_read(child, builder)

# Read methods for complex types. Method sketch
#
# * initialize container
# * for each value in the complex type
#   * dispatch to the appropriate read method
# * assign that read value to the container
# * return the container.

cdef CStatus read_map(const avro_value_t val, CArrayBuilder builder) nogil:
    """For map types the value of an avro_value_t is similar to a record.

    These are fetched one at a time since maps are stored as an ordered list of values.
    """

    out  = {}
    if subcontainer is not None:
        subcontainer[row] = out
    cdef size_t actual_size
    cdef size_t i
    avro_value_get_size(&val, &actual_size)
    cdef avro_type_t avro_type

    for i in range(actual_size):
        key, value = read_map_elem(val, i)
        out[key] = value

    return out


cdef read_map_elem(const avro_value_t val, size_t i):
    cdef const char* map_key = NULL
    cdef avro_value_t child
    avro_value_get_by_index(&val, i, &child, &map_key)
    key = map_key.decode('utf8')
    retval = generic_read(child)
    return key, retval


cdef CStatus read_array(const avro_value_t val, CArrayBuilder builder) nogil:
    cdef:
        size_t actual_size
        size_t i
        const char* map_key = NULL
        avro_value_t child
        CArrayBuilder* child_builder
        CStatus result

    avro_value_get_size(&val, &actual_size)

    (<CListBuilder>(builder)).Append(True)
    child_builder = builder.child(0)
    for i in range(actual_size):
        avro_value_get_by_index(&val, i,  &child, &map_key)
        result = generic_read(child, &child_builder)

    return result


cdef CStatus generic_read(const avro_value_t val, CArrayBuilder builder):
    """Generic avro type read dispatcher. Dispatches to the various specializations by AVRO_TYPE.

    This is used by the various readers for complex types"""
    cdef avro_type_t avro_type
    avro_type = avro_value_get_type(&val)
    if avro_type == AVRO_STRING:
        return read_string(val, builder)
    elif avro_type == AVRO_BYTES:
        return read_bytes(val, builder)
    elif avro_type == AVRO_INT32:
        return read_int32(val, builder)
    elif avro_type == AVRO_INT64:
        return read_int64(val, builder)
    elif avro_type == AVRO_FLOAT:
        return read_float32(val, builder)
    elif avro_type == AVRO_DOUBLE:
        return read_float64(val, builder)
    elif avro_type == AVRO_BOOLEAN:
        return read_bool(val, builder)
    elif avro_type == AVRO_NULL:
        return read_null(val, builder)
    elif avro_type == AVRO_ENUM:
        return read_enum(val, builder)
    elif avro_type == AVRO_FIXED:
        return read_fixed(val, builder)
    elif avro_type == AVRO_MAP:
        return read_map(val, builder)
    elif avro_type == AVRO_RECORD:
        return read_record(val, builder)
    elif avro_type == AVRO_ARRAY:
        return read_array(val, builder)
    elif avro_type == AVRO_UNION:
        return read_union(val, builder)
    else:
        raise Exception('Unexpected type ({})'.format(avro_type))

