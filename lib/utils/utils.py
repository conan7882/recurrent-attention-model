#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: utils.py
# Author: Qian Ge <geqian1001@gmail.com>


def get_shape2D(in_val):
    if isinstance(in_val, int):
        return [in_val, in_val]
    if isinstance(in_val, list):
        assert len(in_val) == 2
        return in_val
    raise RuntimeError('Illegal shape: {}'.format(in_val))