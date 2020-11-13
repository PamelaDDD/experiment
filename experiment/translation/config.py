import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_token = 0
SOS_token = 1
EOS_token = 2
hidden_size = 512
dropout_p = 0.1
teacher_forcing_ratio = 1
BATCH_SIZE = 64
MIN_LENGTH = 3
MAX_LENGTH = 60
source_vocab_size = 4800
target_vocab_size = 50000
n_layers = 4
lr_rate_en = 0.0001
lr_rate_de = 0.0005
lr_decay = True
gamma_encoder = 0.9
gamma_decoder = 0.9
n_epochs = 20
plot_every = 100
print_every = 100
evaluate_every = 100
attn_model = 'dot'
Attention = True
search_method = 'greedy'
beam_size = 10
n_best = 5
sentence_ratio = True


