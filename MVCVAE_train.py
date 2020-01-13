import os

os.environ['KERAS_BACKEND'] = 'theano'
import numpy as np
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
import scipy.io as scio
import sys
import theano
import theano.tensor as T
import math
from sklearn.cluster import KMeans
from keras.models import model_from_json
from sklearn import preprocessing
from sklearn import metrics as mtr
import metrics
import warnings
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse


warnings.filterwarnings("ignore")

theano.config.floatX = 'float32'



class Multiview_VaDE():
    def __init__(self, batch_size, num_views, latent_dim, intermediate_dim, config,weights_path, dataset, ispretrain=True,
                 **kwargs):
        self.batch_size = batch_size
        self.num_views = num_views
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.ispretrain = ispretrain
        self.dataset = dataset
        self.init = 'variancescaling'
        self.weights_path = weights_path
        self.original_dim, self.epoch, self.n_centroid, self.lr_nn, self.lr_gmm, self.decay_n, self.decay_nn, self.decay_gmm, self.alpha, self.datatype = config
        self.sample_output, self.gamma_output, self.vade_model = self.build()

    def encoder(self, original_dim):
        means = []
        inputs = []
        vars = []
        for i in range(0, self.num_views):
            input = Input(batch_shape=(self.batch_size, original_dim[i]), name='input_%d' %i)
            layer1 = Dense(self.intermediate_dim[0], init=self.init, activation='relu', name='encode1_%d' %i)(input)
            layer2 = Dense(self.intermediate_dim[1], init=self.init, activation='relu', name='encode2_%d' %i)(layer1)
            layer3 = Dense(self.intermediate_dim[2], init=self.init, activation='relu', name='encode3_%d' %i)(layer2)
            z_means = Dense(self.latent_dim, init=self.init, activation=None, name='mean_%d' %i)(layer3)
            z_vars = Dense(self.latent_dim, init=self.init, activation=None,name='var_%d' %i)(layer3)
            means.append(z_means)
            vars.append(z_vars)
            inputs.append(input)
        return means, vars, inputs

    def decoder(self, latent, original_dim):
        reconst=[]
        for i in range(0, self.num_views):
            layer4 = Dense(self.intermediate_dim[-1], init=self.init, activation='relu', name='decode1_%d' %i)(latent)
            layer5 = Dense(self.intermediate_dim[-2], init=self.init, activation='relu', name='decode2_%d' %i)(layer4)
            layer6 = Dense(self.intermediate_dim[-3], init=self.init, activation='relu', name='decode3_%d' %i)(layer5)
            decoded= Dense(original_dim[i], init=self.init, activation=self.datatype, name='reconst_%d' %i)(layer6)
            reconst.append(decoded)
        return reconst

    def build(self):
        self.gmmpara_init()
        self.get_zeta()
        means, vars, self.inputs = self.encoder(self.original_dim)
        z_mean = Lambda(self.mixture_u, output_shape=(self.latent_dim,))(means)
        z_log_var = Lambda(self.mixture_z, output_shape=(self.latent_dim,))(vars)
        z = Lambda(self.sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])

        x_decoded = self.decoder(z, self.original_dim)
        output=x_decoded
        output.append(z)
        output.append(z_mean)
        output.append(z_log_var)

        gamma = Lambda(self.get_gamma, output_shape=(self.n_centroid,))(z)

        vade_loss = Lambda(self.vade_loss_function, name='vade_loss', output_shape=([],))(output)
        multiview_output = [vade_loss]

        return Model(self.inputs, z_mean), Model(self.inputs, gamma), Model(self.inputs, multiview_output)

    def gmmpara_init(self):
        theta_init = np.ones(self.n_centroid) / self.n_centroid
        u_init = np.zeros((self.latent_dim, self.n_centroid))
        lambda_init = np.ones((self.latent_dim, self.n_centroid))

        self.theta_p = theano.shared(np.asarray(theta_init, dtype=theano.config.floatX), name="pi")
        self.u_p = theano.shared(np.asarray(u_init, dtype=theano.config.floatX), name="u")
        self.lambda_p = theano.shared(np.asarray(lambda_init, dtype=theano.config.floatX), name="lambda")

    def get_zeta(self):
        zeta_init = np.ones(self.num_views) / self.num_views
        self.zeta = theano.shared(np.asarray(zeta_init, dtype=theano.config.floatX), name="zeta")



    def mixture_u(self, args):
        u =0
        for i in range(0, self.num_views):
            u += self.zeta[i]*args[i]
        return u
    def mixture_z(self, args):
        z =0
        for i in range(0, self.num_views):
            z += self.zeta[i] * K.exp(args[i])
        return K.log(z)

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(self.batch_size, self.latent_dim), mean=0.)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def get_gamma(self, tempz):
        temp_Z = T.transpose(K.repeat(tempz, self.n_centroid), [0, 2, 1])
        temp_u_tensor = T.repeat(self.u_p.dimshuffle('x', 0, 1), self.batch_size, axis=0)

        temp_lambda_tensor = T.repeat(self.lambda_p.dimshuffle('x', 0, 1), self.batch_size, axis=0)
        # version1
        temp_theta_tensor = self.theta_p.dimshuffle('x', 'x', 0) * T.ones(
            (self.batch_size, self.latent_dim, self.n_centroid))
        temp_p_c_z = K.exp(K.sum((K.log(temp_theta_tensor) - 0.5 * K.log(2 * math.pi * temp_lambda_tensor) - \
                                  K.square(temp_Z - temp_u_tensor) / (2 * temp_lambda_tensor)), axis=1)) + 1e-10
        return temp_p_c_z / K.sum(temp_p_c_z, axis=-1, keepdims=True)

        #version2
        # temp_theta_tensor = self.theta_p.dimshuffle('x', 0) * T.ones((self.batch_size, self.n_centroid))
        # temp_p_c_z = K.exp(K.log(temp_theta_tensor) - K.sum((0.5 * K.log(2 * math.pi * temp_lambda_tensor) + \
        #                           K.square(temp_Z - temp_u_tensor) / (2 * temp_lambda_tensor)), axis=1)) + 1e-10
        # return temp_p_c_z / K.sum(temp_p_c_z, axis=1, keepdims=True)

    def vade_loss_function(self, args):
        inputs = self.inputs
        #reconst = self.x_decoded
        reconst, z, z_mean, z_log_var = args[:self.num_views], args[-3], args[-2], args[-1]
        Z = T.transpose(K.repeat(z, self.n_centroid), [0, 2, 1])
        z_mean_t = T.transpose(K.repeat(z_mean, self.n_centroid), [0, 2, 1])
        z_log_var_t = T.transpose(K.repeat(z_log_var, self.n_centroid), [0, 2, 1])
        u_tensor3 = T.repeat(self.u_p.dimshuffle('x', 0, 1), self.batch_size, axis=0)
        lambda_tensor3 = T.repeat(self.lambda_p.dimshuffle('x', 0, 1), self.batch_size, axis=0)

        #version1
        theta_tensor3 = self.theta_p.dimshuffle('x', 'x', 0) * T.ones(
            (self.batch_size, self.latent_dim, self.n_centroid))

        p_c_z = K.exp(K.sum((K.log(theta_tensor3) - 0.5 * K.log(2 * math.pi * lambda_tensor3) - \
                             K.square(Z - u_tensor3) / (2 * lambda_tensor3)), axis=1)) + 1e-10

        gamma = p_c_z / K.sum(p_c_z, axis=-1, keepdims=True)
        gamma_t = K.repeat(gamma, self.latent_dim)

        #version2
        # theta_tensor2 = self.theta_p.dimshuffle('x', 0) * T.ones((self.batch_size, self.n_centroid))
        # p_c_z = K.exp(K.log(theta_tensor2) - K.sum((0.5 * K.log(2 * math.pi * lambda_tensor3) + \
        #                                             K.square(Z - u_tensor3) / (
        #                                                     2 * lambda_tensor3)), axis=1)) + 1e-10
        # gamma = p_c_z / K.sum(p_c_z, axis=1, keepdims=True)
        reconst_loss=0
        for i in range(0, num_views):
            #version 1
            r_loss = self.original_dim[i] * objectives.mean_squared_error(inputs[i], reconst[i])
            #version 2
            #r_loss = self.original_dim[i]*objectives.binary_crossentropy(inputs[i], reconst[i])
            reconst_loss += r_loss
        #version 1
        loss = reconst_loss + self.alpha * (K.sum(0.5 * gamma_t * (
                    self.latent_dim * K.log(math.pi * 2) + K.log(lambda_tensor3) + K.exp(
                z_log_var_t) / lambda_tensor3 + K.square(z_mean_t - u_tensor3) / lambda_tensor3), axis=(1, 2)) \
                               - 0.5 * K.sum(z_log_var + 1, axis=-1) \
                               - K.sum(
                    K.log(K.repeat_elements(self.theta_p.dimshuffle('x', 0), self.batch_size, 0)) * gamma, axis=-1) \
                               + K.sum(K.log(gamma) * gamma, axis=-1))
        #version2
        # loss = reconst_loss + self.alpha * (K.sum(0.5 * gamma * K.sum(
        #         K.log(math.pi * 2) + K.log(lambda_tensor3) + K.exp(
        #     z_log_var_t) / lambda_tensor3 + K.square(z_mean_t - u_tensor3) / lambda_tensor3, axis=1), axis=1) \
        #                                     - 0.5 * K.sum(z_log_var + 1 + K.log(2*math.pi), axis=1) \
        #                                     - K.sum(
        #             K.log(theta_tensor2)* gamma, axis=1) \
        #                                     + K.sum(K.log(gamma) * gamma, axis=1))
        return loss

    def floatX(self, X):
        return np.asarray(X, dtype=theano.config.floatX)

    def load_pretrain_weights(self, vade, weights_path, dataset, inputs,Y):
        vade.load_weights(weights_path + dataset + '.h5')
        sample = self.sample_output.predict(inputs, batch_size=self.batch_size)

        kmeans = KMeans(n_clusters=self.n_centroid, n_init=20)
        kmeans.fit(sample)
        self.u_p.set_value(self.floatX(kmeans.cluster_centers_.T))
        y_pred = kmeans.predict(sample)


        gam = self.gamma_output.predict(inputs, batch_size=batch_size)
        gam_acc = metrics.acc(np.argmax(gam, axis=1), Y)
        print ('pretrain weights loaded!')
        print('Initial_acc:', gam_acc)
        return vade

    def compile(self, inputs, Y):

        if self.ispretrain is True:
            self.vade_model = self.load_pretrain_weights(self.vade_model, self.weights_path, self.dataset, inputs, Y)

        adam_nn = Adam(lr=self.lr_nn, epsilon=1e-4)
        adam_gmm = Adam(lr=self.lr_gmm, epsilon=1e-4)


        self.vade_model.compile(optimizer=adam_nn, loss=lambda y_true, y_pred: y_pred,
                                add_trainable_weights=[self.theta_p, self.u_p, self.lambda_p, self.zeta],
                                add_optimizer=adam_gmm)

        self.epoch_begin = EpochBegin(self.sample_output, self.decay_n, self.gamma_output, self.decay_nn, self.decay_gmm, adam_nn, adam_gmm,
                                      inputs, Y, self.u_p, self.lambda_p, self.theta_p, self.zeta)

    def train(self, inputs):
        none = np.zeros([np.shape(X1)[0]])
        self.vade_model.fit(x=inputs, y=none, shuffle=True, nb_epoch=self.epoch, batch_size=self.batch_size,
                            callbacks=[self.epoch_begin])


def load_dataset(dataset):

    if dataset == 'ORL3':
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        data = scio.loadmat('dataset/ORL_mtv.mat')
        X1 = min_max_scaler.fit_transform(np.transpose(data['X'][0][0]))
        X2 = min_max_scaler.fit_transform(np.transpose(data['X'][0][1]))
        X3 = min_max_scaler.fit_transform(np.transpose(data['X'][0][2]))
        # X3 = np.transpose(data['X'][0][2])
        Y = data['gt'] - 1
        return X1, X2, X3, Y
    if dataset == 'UCI6':
        min_max_scaler = preprocessing.MinMaxScaler()
        path = 'dataset/handwritten.mat'
        data = scio.loadmat(path)
        x1 = min_max_scaler.fit_transform(data['X'][0][0])
        x2 = min_max_scaler.fit_transform(data['X'][0][1])
        x3 = min_max_scaler.fit_transform(data['X'][0][2])
        x4 = min_max_scaler.fit_transform(data['X'][0][3])
        x5 = min_max_scaler.fit_transform(data['X'][0][4])
        x6 = min_max_scaler.fit_transform(data['X'][0][5])
        y = data['Y']
        return x1, x2, x3, x4, x5, x6, y

    if dataset == 'UCI2':
        min_max_scaler = preprocessing.MinMaxScaler()
        path = 'dataset/handwritten.mat'
        data = scio.loadmat(path)
        x1 = min_max_scaler.fit_transform(data['X'][0][0])
        x2 = min_max_scaler.fit_transform(data['X'][0][1])
        x3 = min_max_scaler.fit_transform(data['X'][0][2])
        x4 = min_max_scaler.fit_transform(data['X'][0][3])
        x5 = min_max_scaler.fit_transform(data['X'][0][4])
        x6 = min_max_scaler.fit_transform(data['X'][0][5])
        y = data['Y']
        return x3, x5, y


    if dataset == 'NUS5':
        min_max_scaler = preprocessing.MinMaxScaler()
        data = scio.loadmat('dataset/NUSWIDEOBJ.mat')
        X1 = min_max_scaler.fit_transform(data['X'][0][0])
        X2 = min_max_scaler.fit_transform(data['X'][0][1])
        X3 = min_max_scaler.fit_transform(data['X'][0][2])
        X4 = min_max_scaler.fit_transform(data['X'][0][3])
        X5 = min_max_scaler.fit_transform(data['X'][0][4])
        Y = data['Y']-1
        return X1, X2, X3, X4, X5, Y

    if dataset == 'caltech7':
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        data = scio.loadmat('dataset/Caltech101-7.mat')
        X1 = min_max_scaler.fit_transform(data['X'][0][0])
        X2 = min_max_scaler.fit_transform(data['X'][0][1])
        X3 = min_max_scaler.fit_transform(data['X'][0][2])
        X4 = min_max_scaler.fit_transform(data['X'][0][3])
        X5 = min_max_scaler.fit_transform(data['X'][0][4])
        X6 = min_max_scaler.fit_transform(data['X'][0][5])
        Y = data['Y'] - 1

        return X1, X2, X3, X4, X5, X6, Y

    if dataset == 'Cal2':
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        data = scio.loadmat('dataset/Caltech101-7.mat')
        #X1 = min_max_scaler.fit_transform(data['X'][0][0])
        #X2 = min_max_scaler.fit_transform(data['X'][0][1])
        #X3 = min_max_scaler.fit_transform(data['X'][0][2])
        #X4 = min_max_scaler.fit_transform(data['X'][0][3])
        X5 = min_max_scaler.fit_transform(data['X'][0][4])
        X6 = min_max_scaler.fit_transform(data['X'][0][5])
        Y = data['Y'] - 1

        return X5, X6, Y

    if dataset == 'ORL2':
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        data = scio.loadmat('dataset/ORL_mtv.mat')
        X1 = min_max_scaler.fit_transform(np.transpose(data['X'][0][0]))
        X2 = min_max_scaler.fit_transform(np.transpose(data['X'][0][1]))
        X3 = min_max_scaler.fit_transform(np.transpose(data['X'][0][2]))
        # X3 = np.transpose(data['X'][0][2])
        Y = data['gt'] - 1
        return X2, X3, Y



def config_init(dataset):
    if dataset == 'UCI6':
        return [240, 76, 216, 47, 64, 6], 100, 10, 0.0001, 0.002, 10, 0.9, 0.9, 0.1, 'sigmoid'
    if dataset == 'UCI2':
        return [216, 64], 100, 10, 0.0001, 0.005, 10, 0.9, 0.9, 0.1, 'sigmoid'
    if dataset == 'caltech7':
        return [48, 40, 254, 1984, 512, 928], 100, 7, 0.0001, 0.05, 10, 0.9, 0.9, 1, 'linear'
    if dataset == 'Cal2':
        return [512, 928], 50, 7, 0.0001, 0.05, 5, 0.5, 0.5, 1, 'linear'
    if dataset == 'ORL2':
        return [3304, 6750], 50, 40, 0.0001, 0.01, 5, 0.9, 0.9, 1, 'linear'
    if dataset == 'ORL3':
        return [4096, 3304, 6750], 50, 40, 0.0001, 0.05, 10, 1, 1, 5, 'linear'
    if dataset == 'NUS5':
        return [65, 226, 145, 74, 129], 50, 31, 1e-6, 0.001, 5, 0.9, 0.9, 0.1, 'sigmoid'


class EpochBegin(Callback):
    def __init__(self, sample_output, decay_n, gamma_output, decay_nn, decay_gmm, adam_nn, adam_gmm, inputs, Y, u_p, lambda_p, theta_p,
                 zeta):
        self.sample_output = sample_output
        self.decay_n = decay_n
        self.gamma_output = gamma_output
        self.decay_nn = decay_nn
        self.decay_gmm = decay_gmm
        self.adam_nn = adam_nn
        self.adam_gmm = adam_gmm
        self.inputs = inputs
        self.Y = Y
        self.u_p = u_p
        self.lambda_p = lambda_p
        self.theta_p = theta_p
        self.zeta = zeta

    def on_epoch_begin(self, epoch, logs={}):
        self.epochBegin(epoch)

    def plot_embedding(self, data, label, id):
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)

        fig = plt.figure()
        ax = plt.subplot(111)
        for i in range(data.shape[0]):
            plt.text(data[i, 0], data[i, 1], str(label[i]),
                     color=plt.cm.Set1(label[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 9})

        plt.axis('off')
        plt.savefig('vis_%d' %id)
        return fig

    def epochBegin(self, epoch):
        if epoch % self.decay_n == 0 and epoch != 0:
            self.lr_decay()

        gamma = self.gamma_output.predict(self.inputs, batch_size=batch_size)
        pred = np.argmax(gamma, axis=1)
        acc = self.cluster_acc(pred, self.Y)

        Y = np.reshape(self.Y, [self.Y.shape[0]])
        nmi = metrics.nmi(Y, pred)
        ari = metrics.ari(Y, pred)
        purity = self.purity_score(Y, pred)
        global accuracy
        accuracy = []
        accuracy += [acc[0]]
        if epoch > 0:
            print ('ACC:%0.8f' % acc[0])
            print ('NMI:', nmi)
            print ('ARI:', ari)
            print ('Purity', purity)
        if epoch == 1 and dataset == 'har' and acc[0] < 0.77:
            print ('=========== HAR dataset:bad init!Please run again! ============')
            sys.exit(0)

    def purity_score(self, y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = mtr.cluster.contingency_matrix(y_true, y_pred)
        # return purity
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


    def cluster_acc(self, Y_pred, Y):
        from sklearn.utils.linear_assignment_ import linear_assignment
        assert Y_pred.size == Y.size
        D = max(Y_pred.max(), Y.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(Y_pred.size):
            w[Y_pred[i], Y[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum([w[i, j] for i, j in ind]) * 1.0 / Y_pred.size, w

    def lr_decay(self):
        if dataset == 'mnist' or dataset == 'bbcsport':
            self.adam_nn.lr.set_value(self.floatX(max(self.adam_nn.lr.get_value() * self.decay_nn, 0.0002)))
            self.adam_gmm.lr.set_value(self.floatX(max(self.adam_gmm.lr.get_value() * self.decay_gmm, 0.0002)))
        else:
            self.adam_nn.lr.set_value(self.floatX(self.adam_nn.lr.get_value() * self.decay_nn))
            self.adam_gmm.lr.set_value(self.floatX(self.adam_gmm.lr.get_value() * self.decay_gmm))
        print ('lr_nn:%f' % self.adam_nn.lr.get_value())
        print ('lr_gmm:%f' % self.adam_gmm.lr.get_value())

    def floatX(self, X):
        return np.asarray(X, dtype=theano.config.floatX)



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', dest='dataset', default='UCI2', choices=['UCI2', 'NUS5', 'caltech7', 'UCI6', 'Cal2', 'ORL3', 'ORL2'])
args = parser.parse_args()
dataset = args.dataset
ispretrain = True
batch_size = 200
latent_dim = 10

intermediate_dim = [500, 500, 2000]
# theano.config.floatX='float32'
if dataset in ['UCI2', 'Cal2', 'ORL2']:
    X1, X2, Y = load_dataset(dataset)
    X=[X1, X2]
elif dataset in ['ORL3']:
    X1, X2, X3,Y = load_dataset(dataset)
    X=[X1, X2, X3]
elif dataset in ['UCI6', 'caltech7']:
    X1, X2, X3, X4, X5, X6, Y = load_dataset(dataset)
    X=[X1, X2, X3, X4, X5, X6]
elif dataset in ['NUS5']:
    X1, X2, X3, X4, X5, Y = load_dataset(dataset)
    X=[X1, X2, X3, X4, X5]

num_views = len(X)
weights_path = 'MVCVAE_pretrain/'

vade = Multiview_VaDE(batch_size, num_views, latent_dim, intermediate_dim, config_init(dataset), weights_path, dataset,
                      ispretrain=True)
vade.compile(X,Y)
vade.train(X)
