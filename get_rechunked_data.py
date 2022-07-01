import os
import sys
import numpy as np

def rechunk_file(ogn_file, tgt_file):
    with open(ogn_file, 'r') as fr:
        lines = fr.readlines()

    all_lines = []
    all_lens = []

    for i, line in enumerate(lines):
        if len(line.strip())==0:
            all_lens.append(int(lines[i-1].split()[0]))
        else:
            clean_line = '\t'.join(line.strip().split()[1:])
            all_lines.append(clean_line)

    permutated_lens = np.random.permutation(all_lens)

    start_idx = 0
    with open(tgt_file, 'w') as fw:
        for l in permutated_lens:
            for i in range(l):
                line = all_lines[start_idx+i]
                line_with_idx = str(i+1) + '\t' + line + '\n'
                fw.write(line_with_idx)
            start_idx = start_idx + l
            fw.write('\n')
    print(start_idx)
    print(len(all_lines))



def rechunk_lines(lines):
    all_lines = []
    all_lens = []

    for i, line in enumerate(lines):
        if len(line.strip())==0:
            all_lens.append(int(lines[i-1].split()[0]))
        else:
            clean_line = '\t'.join(line.strip().split()[1:])
            all_lines.append(clean_line)

    permutated_lens = np.random.permutation(all_lens)

    new_lines = []

    start_idx = 0
    for l in permutated_lens:
        for i in range(l):
            line = all_lines[start_idx+i]
            line_with_idx = str(i+1) + '\t' + line + '\n'
            new_lines.append(line_with_idx)
        start_idx = start_idx + l
        new_lines.append('\n')

    return new_lines



if __name__ == '__main__':
    ogn_file = sys.argv[1]
    tgt_file = ogn_file + '.rechunk'
    rechunk_file(ogn_file, tgt_file)