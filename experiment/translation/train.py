# preparing Trainning Data
import torch
import random
import torch.nn as nn
from torch import optim
import time
import math
from translation.tools import showPlot
from translation.dataset import prepareData
from translation.model import EncoderRNN, AttnDecoderRNN
import os
from translation.config import *
from torch.utils.data import Dataset
import numpy as np
from translation.dataset import *
from translation.evaluate import masked_cross_entropy
from torch.optim.lr_scheduler import ExponentialLR
from sacrebleu import corpus_bleu,TOKENIZERS,DEFAULT_TOKENIZER
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'


# # load data files
# SOS_token = 0
# EOS_token = 1
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MAX_LENGTH = 0
# checkpoint_path = './result/'


def indexesFromStence(lang, sentence):
    idxs = []
    for word in sentence.split(' '):
        try:
            idxs.append(lang.word2index[word])
        except KeyError:
            idxs.append(3)
    idxs.append(EOS_token)
    return idxs


def tensorFromSentence(lang, sentence):
    indexes = indexesFromStence(lang, sentence)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


class VocabDataset(Dataset):
    def __init__(self,input_lang,output_lang,pairs):
        self.source_sen_list = [indexesFromStence(input_lang,pair[0]) for pair in pairs]
        self.target_sen_list = [indexesFromStence(output_lang,pair[1]) for pair in pairs]
    def __len__(self):
        return len(self.source_sen_list)

    def __getitem__(self, key):
        token1_idx = self.source_sen_list[key]
        token2_idx = self.target_sen_list[key]
        return [token1_idx,token2_idx,len(token1_idx),len(token2_idx)]

def Vocab_collate_func(batch):
    source_sen_list = []
    target_sen_list = []
    source_len_list = []
    target_len_list = []
    for datum in batch:  #datum (src,target,src_length,target_length)
        source_len_list.append(datum[2])
        target_len_list.append(datum[3])
    max_len_src = max(source_len_list)
    max_len_trg = max(target_len_list)

    #padding
    for datum in batch:
        #source sentence processing
        padded_source = np.pad(np.array(datum[0]),pad_width=((0,max_len_src - datum[2])),mode='constant',constant_values=PAD_token)
        source_sen_list.append(padded_source)
        #target sentence processing
        padded_target = np.pad(np.array(datum[1]),pad_width=((0,max_len_trg - datum[3])),mode='constant',constant_values=PAD_token)
        target_sen_list.append(padded_target)
    #sort sentences for the batch
    sort_idx = sorted(range(len(source_len_list)),key=source_len_list.__getitem__,reverse=True)
    source_sen_list = np.array(source_sen_list)[sort_idx]
    target_sen_list = np.array(target_sen_list)[sort_idx]
    source_len_list = np.array(source_len_list)[sort_idx]
    target_len_list = np.array(target_len_list)[sort_idx]

    return [torch.tensor(source_sen_list).to(device),
            torch.tensor(target_sen_list).to(device),
            torch.tensor(source_len_list),
            torch.tensor(target_len_list)]





def train(input_tensor, target_tensor, input_lengths,target_lengths,encoder, decoder, encoder_optimizer, decoder_optimizer, clip=10.0):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    batch_size = input_tensor.size(1)
    input_tensor.to(device)
    target_tensor.to(device)

    encoder_hidden = encoder.initHidden(batch_size)
    # 这里encoder_outputs记录的就是编码到每一个单词产生的语义向量，比如10个英语单词的句子就应该有10个语义向量
    encoder_outputs = torch.zeros(input_lengths.max(), batch_size,encoder.hidden_size, device=device)

    encoder_outputs,encoder_hidden,encoder_output_lengths = encoder(input_tensor,input_lengths,encoder_hidden)
    #encoder_outputs:#max_len x batch_size x hidden_size
    #hidden: n_layers * 2 x batch_size x hidden_size
    loss = 0

    decoder_input = torch.tensor([SOS_token] * batch_size).to(device) # decoder_input: torch.Size([1, 32])
    decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder
    all_decoder_outputs = torch.zeros(target_lengths.max(),batch_size,decoder.output_size).to(device)
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False


    if use_teacher_forcing:
        # teacher forcing:feed the target as next input
        # 利用已知的上一步真实的单词去预测下一个单词
        for di in range(target_lengths.max()):
            if Attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
            else:
                decoder_output,decoder_hidden = decoder(
                    decoder_input,decoder_hidden,encoder_outputs
                )

            decoder_input = target_tensor[di]#teacher forcing
            all_decoder_outputs[di] = decoder_output
    else:
        # 利用自己的上一步预测的单词去预测下一个单词
        for di in range(target_lengths.max()):
            if Attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
            else:
                decoder_output,decoder_hidden = decoder(
                    decoder_input,decoder_hidden,encoder_outputs
                )
            # 返回前k个最大值以及其索引
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            all_decoder_outputs[di] = decoder_output
   #loss calculation and backpropagation
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0,1).contiguous(), #->batch x seq
        target_tensor.transpose(0,1).contiguous(), #batch x seq
        target_lengths
    )

    loss.backward()

    #Clip gradient norms
    torch.nn.utils.clip_grad_norm(encoder.parameters(),clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(),clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder,decoder,n_iters,lr_decay=True,gamma_encoder=0.9,gamma_decoder=0.9,print_every=100,plot_every=100,learning_rate_encoder=0.0005,learning_rate_decoder=0.002,evaluate_every=3000):

    start = time.time()
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate_encoder)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate_decoder)

    scheduler_encoder = ExponentialLR(encoder_optimizer, gamma_encoder, last_epoch=-1)
    scheduler_decoder = ExponentialLR(decoder_optimizer, gamma_decoder, last_epoch=-1)

    encoder.to(device)
    decoder.to(device)
    score_max = 0

    plot_losses = []
    test_socres = []



    for epoch in range(1, n_iters + 1):
        print_loss_total = 0
        plot_loss_total = 0

        if lr_decay:
            scheduler_encoder.step()
            scheduler_decoder.step()

        for i, (input_sentences,target_sentences,len1,len2) in enumerate(train_loader):
            encoder.train()
            decoder.train()

            input_tensor = input_sentences.transpose(0,1) # 13*100 to 100*13
            target_tensor = target_sentences.transpose(0,1)

            loss = train(input_tensor,target_tensor,len1,len2,encoder,decoder,encoder_optimizer,decoder_optimizer)
            print_loss_total += loss
            plot_loss_total += loss

            if i > 0 and i % evaluate_every == 0:
                bleu_score,(src_sents,sys_sents,ref_sents) = test_model(encoder,decoder,test_loader)
                print(
                    'Validation Score: {} \n source sentence {} \n predicted sentence {} \n Reference sentence: {}'.format(
                        bleu_score, src_sents, sys_sents, ref_sents))
                test_socres.append(bleu_score)

                if bleu_score > score_max:
                    score_max = bleu_score

                    torch.save({
                                'epoch': epoch,
                                'encoder': encoder.state_dict(),
                                'encoder_optimizer': encoder_optimizer.state_dict(),
                                'decoder': decoder.state_dict(),
                                'decoder_optimizer': decoder_optimizer.state_dict()
                                }, "saved_model/attnIs{}_hiddenSize{}_nLayer{}_batchSize{}_epoch{}_srcVocSize{}_tgtVocSize{}_lrDecay{}_teacherF{}"\
                        .format(Attention,hidden_size,n_layers,BATCH_SIZE,n_iters,source_vocab_size,
                                target_vocab_size,lr_decay,teacher_forcing_ratio))
            if i > 0 and i % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('Time: {}, Epoch: [{}/{}], Step: [{}/{}], Train Loss: {}'.format(
                    timeSince(start, i + 1/len(train_loader)), epoch, n_iters, i,
                    len(train_loader),print_loss_avg))

            if i > 0 and i % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
                torch.save({
                            'plot_losses': plot_losses,
                            'validation_scores': test_socres
                            }, "saved_scores/attnIs{}_hiddenSize{}_nLayer{}_batchSize{}_epoch{}_srcVocSize{}_tgtVocSize{}_lrDecay{}_teacherF{}"\
                    .format(Attention,hidden_size,n_layers,BATCH_SIZE,n_iters,source_vocab_size,
                            target_vocab_size,lr_decay,teacher_forcing_ratio))
            torch.cuda.empty_cache()
        print("plot_losses:",plot_losses)
        print("test_scores:",test_socres)
    showPlot(plot_losses,path='./data/trainloss.png')
    showPlot(test_socres,path='./data/testscore.png')







def evaluate(input_lang, output_lang, encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(pairs, input_lang, output_lang, encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(input_lang, output_lang, encoder, decoder, pair[0], max_length=37)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


#
# def saveModel(epochId):
#     torch.save({'epoch':epochId,
#                 'state':model.state_dict(),
#                 'best_loss':lossMin,
#                 "optimizer":optimizer.state_dict(),
#                 "alpha":loss.alpha
#                 "gamma":loss.gamma
#                 },checkpoint_path + '/m-' + launchTimestamp + '-' + str("%.4f" % lossMin)+'.pth.tar')


def load_checkpoint(encoder, decoder, encoder_optimizer, decoder_optimizer, checkpoint_path):
    if checkpoint_path != None:
        ckpt = torch.load(checkpoint_path)
        encoder.load_state_dict(ckpt['encoder_state'])
        decoder.load_state_dict(ckpt['decoder_state'])
        print("loading checkpoint!")
        encoder_optimizer.load_state_dict(ckpt['encoder_optimizer'])
        decoder_optimizer.load_state_dict(ckpt['decoder_optimizer'])
    return encoder, decoder, encoder_optimizer, decoder_optimizer






input_lang, output_lang, pairs = prepareData('en', 'ch', mode='train', reverse=True)
input_lang = build_topwordVocab(input_lang, vocab_size=source_vocab_size)
output_lang = build_topwordVocab(output_lang, vocab_size=target_vocab_size)

train_dataset = VocabDataset(input_lang,output_lang,pairs)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,collate_fn=Vocab_collate_func,shuffle=True)

_,_,test_pairs = prepareData('en','ch',mode='test',reverse=True)
test_dataset = VocabDataset(input_lang,output_lang,test_pairs)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE,collate_fn=Vocab_collate_func,shuffle=False)



# MAX_LENGTH = input_lang.max_length + 1
# print(MAX_LENGTH)

hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1, max_length=MAX_LENGTH).to(device)
trainIters(input_lang, output_lang, pairs, encoder1, attn_decoder1, n_iters=100000, print_every=5000)

# load checkpoint
encoder_optimizer = optim.SGD(encoder1.parameters(), lr=0.001)
decoder_optimizer = optim.SGD(attn_decoder1.parameters(), lr=0.001)
encoder, decoder, encoder_optimizer, decoder_optimizer = load_checkpoint(encoder1, attn_decoder1, encoder_optimizer,
                                                                         decoder_optimizer,
                                                                         checkpoint_path='./result/loss-1.5297.pth.tar')
test_input_lang, test_output_lang, test_pairs = prepareData('en', 'ch', mode='train', reverse=True)
evaluateRandomly(test_pairs, test_input_lang, test_output_lang, encoder1, attn_decoder1)
