# -*- coding: utf-8 -*-
# Copyright 2024-2024 Lukas Franken (University of Edinburgh, Octopus Energy)
# SPDX-FileCopyrightText: : 2024-2024 Lukas Franken
#
# SPDX-License-Identifier: MIT
"""
Compares dispatch in wholesale market between national and nodal market layout

"""

import logging

logger = logging.getLogger(__name__)

import pypsa
import pandas as pd