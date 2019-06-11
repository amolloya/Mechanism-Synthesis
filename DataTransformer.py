import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image
import matplotlib as mpl
#DPI = 96*96

class DataTransformer:
    def __init__(self, name='path', root_dir='./'):
        self.name = name
        self.root_dir = root_dir
        self.datasets = []
        self.dataset_name = []
        self.cp = None
        self.state = None
        self.theta = None
        self.custom = []
        self.label = None

    def load_multiple_datasets(self, env_list, tf_function=None):
        for env_name in env_list:
            self.load_dataset(env_name, tf_function=None)

    def load_dataset(self, env_name, tf_function=None):
        ''' tf_function accepts a dict batch and returns a tensor [N, m, n]
        '''
        with open(self.root_dir + "%s/data.pkl"%env_name, 'rb') as f:
            self.datasets.append(pickle.load(f))
            self.dataset_name.append(env_name)
            f.close()
        self._transform_data(tf_function)

    def isValid(self, seq):
        if len(seq.shape) == 2:
            isVal = np.var(seq[:,0]) <= 5e-3 and np.var(seq[:,1]) <= 5e-3
        else:
            isVal = np.var(seq) <= 5e-3

        if isVal:
            return False
        else:
            return True

    def getValidData(self, cp, state):
        cp_ = []
        state_ = []
        
        for i, cur in enumerate(cp):
            if self.isValid(cur):
                cp_.append(cur)
                state_.append(state[i,:])
            else:
                pass

        return np.array(cp_), np.array(state_)

    def _transform_data(self, tf_function=None):
        if self.name == 'path' or self.name == 'motion' or self.name == 'angle' or self.name == 'state':
            batch = self.datasets[-1]
            cp_ = np.reshape(batch['cp'], [-1, 100, 2])
            state_ = np.reshape(batch['state'], [-1, 100, 6])
            
            cp, state = self.getValidData(cp_, state_)
        
#            theta_ = np.reshape(batch['theta'], [-1, 100])
#            theta = self.getValidData(theta_)
            
            print('Shape of cp: ', cp.shape)
            print('Shape of state: ', state.shape)
            
            if self.cp is None:
                self.cp = cp
#                self.theta = theta
                self.state = state
                self.label = np.zeros((self.cp.shape[0],1))
            else:
                label = np.zeros((cp.shape[0],1)) + len(self.datasets) - 1
                self.cp = np.concatenate((self.cp, cp), axis=0)
                self.state = np.concatenate((self.state, state), axis=0)
#                self.theta = np.concatenate((self.theta, theta), axis=0)
                self.label = np.concatenate((self.label, label), axis=0)
        else:
            batch = self.datasets[-1]
            custom = tf_function(batch)
            self.custom.append(custom)

    def get_dataset(self):
        try:
            if self.name == 'path':
                assert(self.cp is not None)
                assert(self.state is not None)
                return self.cp, self.state, self.label

#            elif self.name == 'motion':
#                assert(self.cp and self.theta)
#                return np.concatenate((self.cp, self.theta), axis=2), self.label
#            
#            elif self.name == 'angle':
#                assert(self.theta)
#                return self.theta, self.label
#            
#            else:
#                assert(self.custom)
#                return self.custom, self.label
        except:
            raise ValueError('Load Datasets First')

    def get_normalized_dataset(self):
        data, mech, label = self.get_dataset()
        data[:,:,0] = data[:,:,0] - np.mean(data[:,:,0], axis=1, keepdims=True)
        data[:,:,1] = data[:,:,1] - np.mean(data[:,:,1], axis=1, keepdims=True)
        denom = np.sqrt(np.var(data[:,:,0], axis=1, keepdims=True) + np.var(data[:,:,1], axis=1, keepdims=True))

        denom = np.expand_dims(denom, axis=2)
        data = data / denom
        
        mech = (mech - np.mean(mech))/np.sqrt(np.var(mech))
        
        return data, mech, label

def make_image_dataset_mlp(seq, prefix='tmp'):
    ''' seq shape is [None, 100, 2]
    '''
    plt.style.use('dark_background')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    fig, ax = plt.subplots(figsize=(1,1))
    mat = np.zeros((seq.shape[0]*3, 200, 200))
    ind = 0
    for suf, s in enumerate(seq):
        for tt in range(3):
            suf_str = '%d'%suf + '%d'%tt
            t = tt*12
            x = s[:,0]*np.cos(t*np.pi/18) - s[:,1]*np.sin(t*np.pi/18)
            y = s[:,0]*np.sin(t*np.pi/18) + s[:,1]*np.cos(t*np.pi/18)
            ax.set_xlim(-3.5, 3.5)
            ax.set_ylim(-3.5, 3.5)
            ax.axis('equal')
            ax.axis('off')
            ax.plot(x, y, color='white')
            fig.savefig('../img/%s'%prefix + suf_str + '.png', bbox_inches='tight', pad_inches=0)
            image = Image.open('../img/%s'%prefix + suf_str + '.png')
            image = image.resize((200,200), Image.ANTIALIAS)
            image.save('../img/%s'%prefix + suf_str + '.png')
            tmp = np.array(image)
            mat[ind] = (tmp[:,:,0] + tmp[:,:,1] + tmp[:,:,2])/3/255
            ind += 1
            ax.clear()

    np.save('../img/path-images.npy', mat)

def make_images_from_paths(seq, save_path=None):
    ''' seq shape is [None, 100, 2]
    '''
    bins = np.linspace(-3.5, 3.5, 63)
    mat = np.zeros((seq.shape[0]*12, bins.shape[0]+1, bins.shape[0]+1))
    i = 0
    for s in seq:
        for tt in range(12):
            t = tt*3
            x = s[:,0]*np.cos(t*np.pi/18) - s[:,1]*np.sin(t*np.pi/18)
            y = s[:,0]*np.sin(t*np.pi/18) + s[:,1]*np.cos(t*np.pi/18)
            x_ind = np.digitize(x, bins)
            y_ind = np.digitize(y, bins)
            mat[i, y_ind, x_ind] = 1
#            plt.imshow(mat[i])
#            plt.pause(0.01)
            i += 1

    if save_path:
        np.save(save_path, mat)


if __name__ == '__main__':
    path_data_handler = DataTransformer('path', root_dir='./data/')

    env_list = ['env-data-sixbar-steph1', 'env-data-sixbar-steph3a', 'env-data-sixbar-watt1']
    path_data_handler.load_multiple_datasets(env_list)
    path_data, mech_data, labels = path_data_handler.get_normalized_dataset()

    print('\nx_max: ', np.max(path_data[:,:,0]), 'x_min: ', np.amin(path_data[:,:,0]))
    print('y_max: ', np.max(path_data[:,:,1]), 'y_min: ', np.amin(path_data[:,:,1]))
    print('x_mean: ', np.mean(path_data[:,:,0]), 'y_mean: ', np.mean(path_data[:,:,1]))
    print('x_std: ', np.std(path_data[:,:,0]), 'y_std: ', np.std(path_data[:,:,1]))
    
#    x_max = [max(a) for a in path_data[:,:,0]]
#    y_max = [max(a) for a in path_data[:,:,1]]
#    x_min = [min(a) for a in path_data[:,:,0]]
#    y_min = [min(a) for a in path_data[:,:,1]]
    
#    fig, ax = plt.subplots(2, 2)
#    ax[0][0].hist(x_max, bins=100)
#    ax[0][1].hist(y_max, bins=100)
#    ax[1][0].hist(x_min, bins=100)
#    ax[1][1].hist(y_min, bins=100)
#    plt.show()
    
#    make_images_from_paths(path_data, './img/images-64.npy')
#    make_images_from_paths(path_data)
#    make_image_dataset_mlp(path_data, 'path')
    
#file = open('data/env-data-fourbar/data.pkl','rb')
#data1 = pickle.load(file)
#file.close()
#
#file = open('data/env-data-sixbar-steph1/data.pkl','rb')
#data2 = pickle.load(file)
#file.close()
#
#file = open('data/env-data-sixbar-steph3a/data.pkl','rb')
#data3 = pickle.load(file)
#file.close()
#
#file = open('data/env-data-sixbar-watt1/data.pkl','rb')
#data4 = pickle.load(file)
#file.close()
