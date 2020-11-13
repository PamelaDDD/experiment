import random
from translation.config import *
from torch import functional
import torch.nn.functional as F
import numpy as np
from translation.config import *
from sacrebleu import corpus_bleu,TOKENIZERS,DEFAULT_TOKENIZER

def evaluateRandomly(pairs, encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def sequence_mask(sequence_length, max_len=None):
    """
    Code paraphrased from
    https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/masked_cross_entropy.py
    """
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len).contiguous()
    seq_range_expand = seq_range_expand.to(device)
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length):
    length = torch.LongTensor(length).to(device)
    """
    Code paraphrased from 
    https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/masked_cross_entropy.py
    """

    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss


class beam_search(object):
    """
    Some code is paraphrased from
    https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/translate/beam.py
    """

    def __init__(self, encoder, decoder, max_length, beam_size, attention=True, sentence_ratio=False):
        super(beam_search, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.max_length = max_length
        self.beam_size = beam_size
        self.sentence_ratio = sentence_ratio

    def search(self, encoder_outputs, decoder_input, decoder_hidden, src_len):

        prob = {k: 0 for k in range(self.beam_size)}
        bestSent = []
        bestScore = []
        decoder_word_choices = {k: [] for k in range(self.beam_size)}
        decoder_hidden_choices = {}
        decoder_input_choices = {}
        decoder_output_choices = {}

        # Initialize beam serach
        if self.attention == True:
            decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input.contiguous(),
                                                                             decoder_hidden.contiguous(),
                                                                             encoder_outputs)
            decoder_output = F.log_softmax(decoder_output, dim=1)
            topv, topi = decoder_output.data.topk(self.beam_size)
        else:
            print("Only available when attention = True")

        # Initialize beam candidates
        for i in range(self.beam_size):
            decoder_word_choices[i].append(topi.squeeze()[i].item())
            decoder_input_choices[i] = topi.squeeze()[i].detach()
            decoder_hidden_choices[i] = decoder_hidden
            prob[i] += topv.squeeze()[i].detach()

        ## running beam search
        cur_len = 0
        max_length = 2 * src_len if self.sentence_ratio else self.max_length
        # delete
        #         print(self.sentence_ratio)
        #         print(src_len)
        #         print(max_length)

        while decoder_hidden_choices and cur_len <= max_length:
            cur_len += 1
            topi = {}
            key_list = list(decoder_hidden_choices.keys())
            scores = []
            for key in key_list:
                decoder_output, decoder_hidden_choices[key], decoder_attn = self.decoder(
                    decoder_input_choices[key].unsqueeze(0), decoder_hidden_choices[key], encoder_outputs)
                decoder_output_choices[key] = F.log_softmax(decoder_output, dim=1)
                topv, topi[key] = decoder_output_choices[key].data.topk(len(decoder_hidden_choices))
                scores.extend((topv + prob[key]).tolist()[0])

            scores = np.array(scores)
            max_candidate_score = scores.argsort()[-len(decoder_hidden_choices):][::-1]
            decoded_sent_score = scores[max_candidate_score]

            choice_sentence = {}
            choiceHidden = {}
            trashOfKeys = []

            for j in range(len(max_candidate_score)):
                prev_choice_idx = key_list[int(np.floor(max_candidate_score[j] / len(decoder_hidden_choices)))]
                if topi[prev_choice_idx].squeeze().dim() == 0:
                    next_idx = topi[prev_choice_idx].squeeze()
                else:
                    next_idx = topi[prev_choice_idx].squeeze()[max_candidate_score[j] % len(decoder_hidden_choices)]

                s_choice = decoder_word_choices[prev_choice_idx].copy()
                s_choice.append(next_idx.item())
                choice_sentence[j] = s_choice
                h_choice = decoder_hidden_choices[prev_choice_idx]
                choiceHidden[j] = h_choice
                decoder_input_choices[j] = next_idx.detach()
                prob[j] = decoded_sent_score[j]

            decoder_word_choices = choice_sentence
            decoder_hidden_choices = choiceHidden

            for key, s in decoder_word_choices.items():
                if EOS_token in s:
                    bestSent.append(s)
                    bestScore.append(prob[key])
                    trashOfKeys.append(key)

            for k in trashOfKeys:
                decoder_hidden_choices.pop(k)
                decoder_word_choices.pop(k)

        if len(bestScore) == 0:
            max_prob = prob[0]
            max_prob_idx = 0
            for k in prob.keys():
                if prob[k] > max_prob:
                    max_prob_idx = k
                    max_prob = prob[k]
            bestScore.append(max_prob)
            bestSent.append(decoder_word_choices[max_prob_idx])

        return bestSent, bestScore


def get_batch_outputs(encoder, decoder, input_sentences, input_lengths, output_lengths,output_lang):
    with torch.no_grad():
        input_tensor = input_sentences.transpose(0, 1).to(device)  # 32*100 to 100*32
        batch_size = input_tensor.size(1)
        encoder_hidden = encoder.initHidden(batch_size)
        encoder_outputs, encoder_hidden, encoder_output_lengths = encoder(input_tensor, input_lengths, encoder_hidden)

        decoder_hidden = encoder_hidden[:decoder.n_layers]

        decoder_input = torch.tensor([SOS_token] * batch_size).to(
            device)  # decoder_input: torch.Size([1, 32])
        decoded_words = np.empty((output_lengths.max(), batch_size), dtype=object)

        for di in range(output_lengths.max()):
            if Attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, batch_size)

            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach().to(device)  # detach from history as input

            decoded_words[di:] = np.array(
                ['<EOS>' if idx == EOS_token else output_lang.index2word[idx] for idx in decoder_input.tolist()])

        return decoded_words.transpose()


def get_beam_batch_outputs(encoder, decoder, input_sentences, input_lengths,output_lang):  #####
    with torch.no_grad():
        input_tensor = input_sentences.transpose(0, 1).to(device)  # 32*100 to 100*32
        batch_size = input_tensor.size(1)
        encoder_hidden = encoder.initHidden(batch_size).to(device)
        encoder_outputs, encoder_hidden, encoder_output_lengths = encoder(input_tensor, input_lengths, encoder_hidden)

        decoder_hidden = encoder_hidden[:decoder.n_layers].to(device)
        my_beam_search = beam_search(encoder, decoder, input_sentences.max().item(), beam_size, True, sentence_ratio)
        beam_search_result = []
        for i in range(batch_size):
            decoder_input = torch.tensor([SOS_token], device=device, requires_grad=False).unsqueeze(
                0)  # .view(1,-1) # take care of different input shape
            sentences, probs = my_beam_search.search(encoder_outputs[:, i, :].unsqueeze(1), decoder_input,
                                                     decoder_hidden[:, i, :].unsqueeze(1), input_lengths[i].item())

            beam_search_result.append(sentences[probs.index(max(probs))])

        padded_beam_search_result = []

        max_length = 0
        for sentence in beam_search_result:
            if len(sentence) > max_length:
                max_length = len(sentence)

        for sentence in beam_search_result:
            while len(sentence) < max_length + 2:
                sentence.append(PAD_token)
            padded_beam_search_result.append(sentence)

        batch_sentences = []

        for sentence in padded_beam_search_result:
            sentence = [output_lang.index2word[k] for k in sentence]
            try:
                end_idx = sentence.index('<EOS>')
                batch_sentences.append(' '.join(sentence[:end_idx]))
            except ValueError:
                batch_sentences.append(' '.join(sentence))

    return batch_sentences


def test_model(encoder, decoder, loader, input_lang,output_lang,search_method='greedy'):
    encoder.eval()
    decoder.eval()

    score = []
    src_sentences = []
    sys_sentences = []
    ref_sentences = []
    encoder.train(False)
    decoder.train(False)
    for i, (input_sentences, target_sentences, len1, len2) in enumerate(loader):
        for sentence in target_sentences:
            trg_list = []
            for idx in sentence:
                if idx.item() == EOS_token:
                    break
                else:
                    trg_list.append(output_lang.index2word[idx.item()])
            ref_sentences.append(' '.join(trg_list))
        for sentence in input_sentences:
            src_list = []
            for idx in sentence:
                if idx.item() == EOS_token:
                    break
                else:
                    src_list.append(input_lang.index2word[idx.item()])
            src_sentences.append(' '.join(src_list))

        # ref_sentences.append(' '.join(sent) for sent in target_sentences)
        # src_sentences.append(' '.join(sent) for sent in input_sentences)
        batch_size = input_sentences.size(0)
        if search_method == 'greedy':
            for sentence in get_batch_outputs(encoder, decoder, input_sentences, len1, len2,output_lang):
                try:
                    end_idx = sentence.tolist().index('<EOS>')
                    sys_sentences.append(' '.join(sentence[:end_idx]))
                except ValueError:
                    sys_sentences.append(' '.join(sentence))

        elif search_method == 'beam':
            translation_output = get_beam_batch_outputs(encoder, decoder, input_sentences,len1,input_lang=input_lang,output_lang=output_lang)
            sys_sentences.extend(translation_output)

    encoder.train(True)
    decoder.train(True)

    score = corpus_bleu(sys_sentences, [ref_sentences], smooth="floor", smooth_floor=0.01, lowercase=False,
                        use_effective_order=True, tokenize=DEFAULT_TOKENIZER).score
    return score, (src_sentences[0:1], sys_sentences[0:1], ref_sentences[0:1])