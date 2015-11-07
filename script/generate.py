import argparse
import cPickle as pickle
import glob
import numpy as np
import os
import struct
import chainer
import chainer.functions as F
from chainer import cuda
from trainer import Trainer
from model import Model
from vae_voice_model import VAEVoiceModel
##
from scipy import misc
##

MGCEP_ORDER = 40
#X_WIDTH     = 30
X_WIDTH     = 1
BATCH_SIZE  = 100
MIN_PITCH   = 20.0

parser = argparse.ArgumentParser(description='Train voice')
parser.add_argument('--gpu',     '-g', default=-1,    type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--model',   '-m', required=True, type=str, help='input model file path')
parser.add_argument('--input',   '-i', required=True, type=str, help='input file path')
parser.add_argument('--from_sp', '-f', required=True, type=int, help='speaker index of input')
parser.add_argument('--output',  '-o', required=True, type=str, help='output file path without extension')
parser.add_argument('--to_sp',   '-t', required=True, type=int, help='speaker index of output')
args = parser.parse_args()

model = Model.load(args.model)
if args.gpu >= 0:
    model.to_gpu(args.gpu)
    xp = cuda.cupy
else:
    model.to_cpu()
    xp = np

scale = np.asarray([400.0, 6.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32)
offset = np.asarray([0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
def load_data(file_path):
    with open(file_path, 'rb') as f:
        data   = pickle.load(f)
        shape  = data.shape
        length = shape[0]
        data   /= scale
        data   += offset
        remain = length % X_WIDTH
        data   = np.concatenate([data, np.zeros((X_WIDTH - remain, shape[1]), dtype=np.float32)])
        shape  = data.shape
        return (data.reshape((shape[0] / X_WIDTH, X_WIDTH * shape[1])), length)

def save_data(data, length, file_path):
##
    with open(args.input, 'rb') as f:
        org_data = pickle.load(f)
##

    shape    = data.shape
    reshaped = data.reshape((shape[0] * X_WIDTH, shape[1] / X_WIDTH))[:length]
    reshaped -= offset
    reshaped *= scale
    pitch    = reshaped[:,0]
    pitch[pitch < MIN_PITCH] = 0
    for x, org in zip(reshaped, org_data):
        print '#######'
        print x
        print org
    with open(file_path + '.pitch', 'wb') as f:
        data_format = '<f'
        for x in reshaped:
            f.write(struct.pack(data_format, x[0]))
    with open(file_path + '.mgcep', 'wb') as f:
        x_size = reshaped.shape[1]
        data_format = '<{}f'.format(x_size - 1)
        for x in reshaped:
            f.write(struct.pack(*([data_format] + x[1:].tolist())))

x_data, org_length = load_data(args.input)
shape = x_data.shape
# TODO get category size from model
y_rec = np.zeros((BATCH_SIZE, model.y_size), dtype=np.float32)
y_rec[:, args.from_sp] = 1.0
y_rec = xp.asarray(y_rec)
y_gen = np.zeros((1, model.y_size), dtype=np.float32)
y_gen[:, args.to_sp] = 1.0
y_gen = xp.asarray(y_gen)
# y_rec  = xp.asarray(np.ones((BATCH_SIZE, 1), dtype=np.float32))
# y_gen  = xp.asarray(np.ones((1,1), dtype=np.float32))

ys = []
for i in range(0, shape[0], BATCH_SIZE):
    x = chainer.Variable(xp.asarray(x_data[i: i + BATCH_SIZE]))
    length = x.data.shape[0]
    y = model.generate(x, chainer.Variable(y_rec[:length]), chainer.Variable(y_gen))
    ys.append(cuda.to_cpu(y.data))
save_data(np.concatenate(ys, axis=0), org_length, args.output)
