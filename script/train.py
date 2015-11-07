import argparse
import cPickle as pickle
import glob
import numpy as np
import os
import chainer
import chainer.functions as F
from chainer import cuda
from chainer import optimizers
from trainer import Trainer
from model import Model
from vae_voice_model import VAEVoiceModel

MGCEP_ORDER = 40
X_WIDTH     = 1

parser = argparse.ArgumentParser(description='Train voice')
parser.add_argument('--gpu',       '-g', default=-1,    type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--input',     '-i', default=None,  type=str, help='input model file path')
parser.add_argument('--output',    '-o', required=True, type=str, help='output model file path')
parser.add_argument('--train_dir', '-t', default='.',   type=str, help='training data directory')
parser.add_argument('--iter',            default=100,   type=int, help='number of iteration')
args = parser.parse_args()

gpu_device = None
if args.gpu >= 0:
    cuda.check_cuda_available()
    gpu_device = args.gpu

batch_size  = 100
train_files = glob.glob(os.path.join(args.train_dir, '*.pkl'))
y_size      = len(train_files)

def load_data(file_path):
    y = int(os.path.splitext(os.path.basename(file_path))[0])
    with open(file_path, 'rb') as f:
        os.path.basename(file_path)
        data     = pickle.load(f)
        shape    = data.shape
        data     /= np.asarray([400.0, 6.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32)
        data     += np.asarray([0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        data_len = shape[0] - shape[0] % X_WIDTH
        x_data   = data[:data_len].reshape((data_len / X_WIDTH, X_WIDTH * shape[1]))
        y_data   = np.zeros((data_len / X_WIDTH, y_size), dtype=np.float32)
        y_data[:,y] = 1.0
    return (x_data, y_data)

train_tuples = map(load_data, train_files)
xs, ys       = zip(*train_tuples)
x_train      = np.concatenate(xs, axis=0)
y_train      = np.concatenate(ys, axis=0)
if args.input is not None:
    model = Model.load(args.input)
else:
    model = VAEVoiceModel(x_train.shape[1], y_size)
optimizer = optimizers.Adam(alpha=0.0001)

def loss_func((y, mean, var), target):
    return F.mean_squared_error(y, target) - 0.0001 * 0.5 * F.sum(1 + var - mean ** 2 - F.exp(var)) / float(y.data.size)

def accuracy_func((y, mean, var), target):
    return F.mean_squared_error(y, target)

def progress_func(epoch, loss, accuracy, test_loss, test_accuracy):
    print 'epoch: {} done'.format(epoch)
    print('train mean loss={}, accuracy={}'.format(loss, accuracy))
    if test_loss is not None and test_accuracy is not None:
        print('test mean loss={}, accuracy={}'.format(test_loss, test_accuracy))
    if epoch % 10 == 0:
        model.save(args.output)

Trainer.train(model, (x_train, y_train), x_train, args.iter, batch_size=100, gpu_device=gpu_device, loss_func=loss_func, accuracy_func=accuracy_func, optimizer=optimizer, callback=progress_func)
