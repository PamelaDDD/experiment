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
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

# load data files
SOS_token = 0
EOS_token = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 0
checkpoint_path = './result/'


def indexesFromStence(lang, sentence):
     return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromStence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # 这里encoder_outputs记录的就是编码到每一个单词产生的语义向量，比如10个英语单词的句子就应该有10个语义向量
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0
    # 一个个单词 feed encoder
    # print('encoder_outputs shape is {}'.format(encoder_outputs.shape))
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        # 记录该单词出的语义向量
        encoder_outputs[ei] = encoder_output[0, 0]




    # SOS 表示句子开始
    decoder_input = torch.tensor([[SOS_token]], device=device)
    # decoder 的 hide state 就是encoder最后一步语义向量
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # teacher forcing:feed the target as next input
        # 利用已知的上一步真实的单词去预测下一个单词
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
    else:
        # 利用自己的上一步预测的单词去预测下一个单词
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # 返回前k个最大值以及其索引
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


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


def trainIters(input_lang, output_lang, pairs, encoder, decoder, n_iters, print_every=1000, plot_every=100,
               learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(input_lang, output_lang, random.choice(pairs)) for i in range(n_iters)]
    criterion = nn.NLLLoss()
    lossMin = 1000
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        # print("input_tensor shape is {}".format(input_tensor.shape))
        # print("output_tensor shape is {}".format(target_tensor.shape))
        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
                     max_length=MAX_LENGTH)

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            # 保存
            if print_loss_avg < lossMin:
                lossMin = print_loss_avg
                launchTimestamp = str(time.time())
                torch.save({'iteration': iter,
                            'encoder_state': encoder.state_dict(),
                            'decoder_state': decoder.state_dict(),
                            'best_loss': lossMin,
                            "encoder_optimizer": encoder_optimizer.state_dict(),
                            "decoder_optimizer": decoder_optimizer.state_dict(),
                            }, checkpoint_path + '/' + 'loss-' + str("%.4f" % lossMin) + '.pth.tar')

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    showPlot(plot_losses)


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


def evaluateRandomly(pairs, input_lang,output_lang,encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(input_lang,output_lang,encoder, decoder, pair[0],max_length=37)
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


def load_checkpoint(encoder,decoder,encoder_optimizer,decoder_optimizer,checkpoint_path):
    if checkpoint_path != None:
        ckpt = torch.load(checkpoint_path)
        encoder.load_state_dict(ckpt['encoder_state'])
        decoder.load_state_dict(ckpt['decoder_state'])
        print("loading checkpoint!")
        encoder_optimizer.load_state_dict(ckpt['encoder_optimizer'])
        decoder_optimizer.load_state_dict(ckpt['decoder_optimizer'])
    return encoder,decoder,encoder_optimizer,decoder_optimizer

input_lang, output_lang, pairs = prepareData('en', 'ch', mode='train',reverse=True)
MAX_LENGTH = input_lang.max_length + 1
print(MAX_LENGTH)
hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1, max_length=MAX_LENGTH).to(device)
trainIters(input_lang, output_lang, pairs, encoder1, attn_decoder1, n_iters=100000, print_every=5000)


#load checkpoint
encoder_optimizer = optim.SGD(encoder1.parameters(), lr=0.001)
decoder_optimizer = optim.SGD(attn_decoder1.parameters(), lr=0.001)
encoder,decoder,encoder_optimizer,decoder_optimizer = load_checkpoint(encoder1,attn_decoder1,encoder_optimizer,decoder_optimizer,checkpoint_path='./result/loss-1.5297.pth.tar')
test_input_lang,test_output_lang,test_pairs = prepareData('en','ch',mode='train',reverse=True)
evaluateRandomly(test_pairs,test_input_lang,test_output_lang,encoder1, attn_decoder1)
