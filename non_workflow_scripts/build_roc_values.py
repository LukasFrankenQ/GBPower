# SPDX-FileCopyrightText: : 2024 Lukas Franken
#
# SPDX-License-Identifier: MIT

"""
Script to create the file 'data/roc_values.csv'.
Compiles renewable bidding prices for a larger numbers of days (here arond 600)
and estimates the ROC values for these and other plants of the same technology.
"""

if __name__ == '__main__':

    print(snakemake.input)
    print('=========================')
    print(snakemake.output)


