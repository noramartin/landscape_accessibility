#! /usr/bin/env python

#     sparse_list.py

# Here is a very basic implimentation of a sparse vector type. 
# No support for slicing, negative indices, and much more! 
# It has low RAM usage, and a fast __getitem__.
# Great for a lookup table.

# Written by Kye Hunter
# Last updated 20251201

class SparseList(dict):
    def __init__(self, *args, default = None, **keyword_args):
        if len(args) > 0:
            data = args[0]
            if issubclass(type(data), typing.Generator):
                data = tuple(data)
            if not issubclass(type(data), dict):
                try:
                    data = dict(data)
                except TypeError:
                    data = dict(enumerate(data))
            #super().__init__(None, data, *args[1:], **keyword_args)
            super().__init__(data, *args[1:], **keyword_args)
        else:
            #super().__init__(None, *args, **keyword_args)
            super().__init__(*args, **keyword_args)
        self.default = default

    def __missing__(self, key):
        return self.default

    def __repr__(self):
        return f'SparseList({str(dict(self))}, default = {self.default})'

