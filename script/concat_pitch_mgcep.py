import argparse
import cPickle as pickle
import numpy as np
import struct
import os

parser = argparse.ArgumentParser(description='Make train data file from pitch and mgcep files')

parser.add_argument('--list_file', '-l', type=str, required=True, help='list file of input files')
parser.add_argument('--output_file', '-o', type=argparse.FileType('wb'), required=True, help='output pickle file')
parser.add_argument('--mgcep_order', '-m', type=int, default=20, help='order of cepstrum')
args         = parser.parse_args()
output_file  = args.output_file
pitch_format = '<f'
pitch_size   = struct.calcsize(pitch_format)
mgcep_format = '<{}f'.format(args.mgcep_order + 1)
mgcep_size   = struct.calcsize(mgcep_format)
matrices     = []

with open(args.list_file) as f:
    file_paths = map(lambda line: line.rstrip(), f.readlines())

for file_path in file_paths:
    pitch_file_name = '{}.pitch'.format(file_path)
    mgcep_file_name = '{}.mgcep'.format(file_path)
    data_length     = os.path.getsize(pitch_file_name) / pitch_size
    matrix          = np.ndarray((data_length, args.mgcep_order + 2), dtype=np.float32)
    with open(pitch_file_name) as pitch_file, open(mgcep_file_name) as mgcep_file:
        for i in range(data_length):
            pitch = struct.unpack(pitch_format, pitch_file.read(pitch_size))
            mgcep = struct.unpack(mgcep_format, mgcep_file.read(mgcep_size))
            matrix[i] = pitch + mgcep
    matrices.append(matrix)

data = np.concatenate(matrices, axis=0)
pickle.dump(data, output_file, pickle.HIGHEST_PROTOCOL)
output_file.close()
