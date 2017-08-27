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
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Cython helper for `from_bytes_c.py`
"""
from __future__ import absolute_import
from libc.stdlib cimport malloc, free
from libc.stdio cimport *
from libc.string cimport memcpy

cimport cyavro._cyavro as cyavro


cpdef get_reader(str fpath):
    py_byte_string = fpath.encode('UTF-8')
    cdef char* c_fpath =  py_byte_string
    cdef FILE* cfile = fopen(c_fpath, "rb")

    fseek(cfile, 0, SEEK_END)
    cdef long fsize = ftell(cfile)
    fseek(cfile, 0, SEEK_SET)

    cdef void *buffer = malloc(fsize + 1)
    fread(buffer, fsize, 1, cfile)

    cdef cyavro.AvroReader reader = cyavro.reader_from_bytes_c(buffer, fsize)
    fclose(cfile)
    return reader
