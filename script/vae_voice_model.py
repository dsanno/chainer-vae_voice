import numpy as np
import chainer.functions as F
from chainer import cuda
from chainer import Variable
from model import Model

class VAEVoiceModel(Model):
    def __init__(self, x_size, y_size):
        Model.__init__(self,
            rec1_x = F.Linear(x_size, 40),
            rec1_y = F.Linear(y_size, 40),
            rec2 = F.Linear(40, 40),
            rec_mean = F.Linear(40, 10),
            rec_var  = F.Linear(40, 10),
            gen1_z = F.Linear(10, 40),
            gen1_y = F.Linear(y_size, 40),
            gen2 = F.Linear(40, 40),
            gen3 = F.Linear(40, x_size)
#            rec1_x = F.Linear(x_size, 500),
#            rec1_y = F.Linear(y_size, 500),
#            rec2 = F.Linear(500, 500),
#            rec_mean = F.Linear(500, 50),
#            rec_var  = F.Linear(500, 50),
#            gen1_z = F.Linear(50, 500),
#            gen1_y = F.Linear(y_size, 500),
#            gen2 = F.Linear(500, 500),
#            gen3 = F.Linear(500, x_size)
        )
        self.x_size = x_size
        self.y_size = y_size

    def forward(self, (x_var, y_var), train=True):
        xp = cuda.get_array_module(x_var.data)
        h1 = F.relu(self.rec1_x(x_var) + self.rec1_y(y_var))
        h2 = F.relu(self.rec2(h1))
        mean = self.rec_mean(h2)
        var  = 0.5 * self.rec_var(h2)
        rand = xp.random.normal(0, 1, var.data.shape).astype(np.float32)
        z  = mean + F.exp(var) * Variable(rand)
        g1 = F.relu(self.gen1_z(z) + self.gen1_y(y_var))
        g2 = F.relu(self.gen2(g1))
        g3 = F.sigmoid(self.gen3(g2))
        return (g3, mean, var)

    def generate(self, x_rec, y_rec, y_gen):
        assert x_rec.data.shape[0] == y_rec.data.shape[0]
        rec_num = x_rec.data.shape[0]
        gen_num = y_gen.data.shape[0]
        xp = cuda.get_array_module(x_rec.data)
        h1 = F.relu(self.rec1_x(x_rec) + self.rec1_y(y_rec))
        h2 = F.relu(self.rec2(h1))
        mean = self.rec_mean(h2)
        var  = 0.5 * self.rec_var(h2)

        mean_gen = Variable(xp.asarray(np.repeat(cuda.to_cpu(mean.data), gen_num, axis=0)))
        var_gen  = Variable(xp.asarray(np.repeat(cuda.to_cpu(var.data), gen_num, axis=0)))
        y_gen = Variable(xp.asarray(np.repeat(cuda.to_cpu(y_gen.data), rec_num, axis=0)))
        rand = xp.random.normal(0, 1, var_gen.data.shape).astype(np.float32)
        z  = mean_gen + F.exp(var_gen) * Variable(rand)
        g1 = F.relu(self.gen1_z(z) + self.gen1_y(y_gen))
        g2 = F.relu(self.gen2(g1))
        g3 = F.sigmoid(self.gen3(g2))
        return g3
