from DataTransformer import DataTransformer
from modelVAETrainer import *
from modelVAE import *

path_data_handler = DataTransformer('path', root_dir='./data/')
env_list = ['env-data-sixbar-steph1', 'env-data-sixbar-steph3a', 'env-data-sixbar-watt1']
path_data_handler.load_multiple_datasets(env_list)
path_data, mech_data, labels = path_data_handler.get_normalized_dataset()

print(path_data.shape, labels.shape)

print('x_max: ', np.max(path_data[:,:,0]), 'x_min', np.amin(path_data[:,:,0]))
print('y_max: ', np.max(path_data[:,:,1]), 'y_min', np.amin(path_data[:,:,1]))
print('x_mean: ', np.mean(path_data[:,:,0]), 'y_mean', np.mean(path_data[:,:,1]))
print('x_std: ', np.std(path_data[:,:,0]), 'y_std', np.std(path_data[:,:,1]))

latent_dim = 8
batch_sz = 64
encoder = FCNN_Encoder(latent_dim)
decoder = FCNN_Decoder()
vae = VAE(latent_dim, encoder, decoder)
trainer = VAETrainer('fcnn-%s'%latent_dim, latent_dim, vae, batch_sz, lr=0.001)
trainer.load_dataset(path_data)
trainer.train(100)
trainer.predict()