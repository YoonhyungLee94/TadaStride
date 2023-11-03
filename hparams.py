################################
# Experiment Parameters        #
################################
seed = 1234
output_directory = './training_log'
iters_per_logging = 100
iters_per_validation = 1000
iters_per_checkpoint = 10000

################################
# Audio Parameters             #
################################
sampling_rate = 16000
filter_length = 1024
hop_length = 256
win_length = 1024
n_mel_channels = 80
mel_fmin = 0
mel_fmax = 8000.0

################################
# Model Parameters             #
################################
hidden_dim = 256
kernel_size = 3
n_classes = 7
sigma_square = 5.0
ver_f = False

################################
# Optimization Hyperparameters #
################################
epochs=100
batch_size = 128
learning_rate = 1e-4