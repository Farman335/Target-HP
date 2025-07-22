#!/usr/bin/env python
# _*_coding:utf-8_*_

import re
import os
import sys
from collections import Counter

def readFasta(file):
    if not os.path.exists(file):
        print('Error: "' + file + '" does not exist.')
        sys.exit(1)

    with open(file) as f:
        records = f.read()

    if re.search('>', records) is None:
        print('The input file does not appear to be in fasta format.')
        sys.exit(1)

    records = records.split('>')[1:]
    myFasta = []
    for fasta in records:
        array = fasta.split('\n')
        name, sequence = array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '-', ''.join(array[1:]).upper())
        myFasta.append([name, sequence])
    return myFasta

def GAAC(fasta, **kw):
    fastas = readFasta(fasta)
    group = {
        'alphatic': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharge': 'KRH',
        'negativecharge': 'DE',
        'uncharge': 'STCPNQ'
    }

    groupKey = group.keys()

    encodings = []
    header = ['#']
    for key in groupKey:
        header.append(key)
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = [name]
        count = Counter(sequence)
        myDict = {}
        for key in groupKey:
            for aa in group[key]:
                myDict[key] = myDict.get(key, 0) + count[aa]

        total_length = len(sequence)
        if total_length == 0:
            code.extend([0] * len(groupKey))
        else:
            for key in groupKey:
                code.append(myDict[key] / total_length)
        encodings.append(code)

    return encodings

if __name__ == '__main__':
    fastafile = 'CCR_test_583_565.txt'

    vecencoding = GAAC(fastafile)
    with open('GAAC_CCR_test_583_565.csv', 'w') as F:
        for line in vecencoding:
            F.write(','.join(map(str, line)))
            F.write('\n')
    print('Done')
