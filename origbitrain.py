# -*- coding: utf-8 -*-
"""
Translation with a Sequence to Sequence Network and Attention
*************************************************************
**Author**: `Sean Robertson <https://github.com/spro/practical-pytorch>`_

In this project we will be teaching a neural network to translate from
French to English.

::

    [KEY: > input, = target, < output]

    > il est en train de peindre un tableau .
    = he is painting a picture .
    < he is painting a picture .

    > pourquoi ne pas essayer ce vin delicieux ?
    = why not try that delicious wine ?
    < why not try that delicious wine ?

    > elle n est pas poete mais romanciere .
    = she is not a poet but a novelist .
    < she not not a poet but a novelist .

    > vous etes trop maigre .
    = you re too skinny .
    < you re all alone .

... to varying degrees of success.

This is made possible by the simple but powerful idea of the `sequence
to sequence network <http://arxiv.org/abs/1409.3215>`__, in which two
recurrent neural networks work together to transform one sequence to
another. An encoder network condenses an input sequence into a vector,
and a decoder network unfolds that vector into a new sequence.

.. figure:: /_static/img/seq-seq-images/seq2seq.png
   :alt:

To improve upon this model we'll use an `attention
mechanism <https://arxiv.org/abs/1409.0473>`__, which lets the decoder
learn to focus over a specific range of the input sequence.

**Recommended Reading:**

I assume you have at least installed PyTorch, know Python, and
understand Tensors:

-  http://pytorch.org/ For installation instructions
-  :doc:`/beginner/deep_learning_60min_blitz` to get started with PyTorch in general
-  :doc:`/beginner/pytorch_with_examples` for a wide and deep overview
-  :doc:`/beginner/former_torchies_tutorial` if you are former Lua Torch user


It would also be useful to know about Sequence to Sequence networks and
how they work:

-  `Learning Phrase Representations using RNN Encoder-Decoder for
   Statistical Machine Translation <http://arxiv.org/abs/1406.1078>`__
-  `Sequence to Sequence Learning with Neural
   Networks <http://arxiv.org/abs/1409.3215>`__
-  `Neural Machine Translation by Jointly Learning to Align and
   Translate <https://arxiv.org/abs/1409.0473>`__
-  `A Neural Conversational Model <http://arxiv.org/abs/1506.05869>`__

You will also find the previous tutorials on
:doc:`/intermediate/char_rnn_classification_tutorial`
and :doc:`/intermediate/char_rnn_generation_tutorial`
helpful as those concepts are very similar to the Encoder and Decoder
models, respectively.

And for more, read the papers that introduced these topics:

-  `Learning Phrase Representations using RNN Encoder-Decoder for
   Statistical Machine Translation <http://arxiv.org/abs/1406.1078>`__
-  `Sequence to Sequence Learning with Neural
Networks <http://arxiv.org/abs/1409.3215>`__
-  `Neural Machine Translation by Jointly Learning to Align and
   Translate <https://arxiv.org/abs/1409.0473>`__
-  `A Neural Conversational Model <http://arxiv.org/abs/1506.05869>`__


**Requirements**
"""
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################################################
# Loading data files
# ==================
#
# The data for this project is a set of many thousands of English to
# French translation pairs.
#
# `This question on Open Data Stack
# Exchange <http://opendata.stackexchange.com/questions/3888/dataset-of-sentences-translated-into-many-languages>`__
# pointed me to the open translation site http://tatoeba.org/ which has
# downloads available at http://tatoeba.org/eng/downloads - and better
# yet, someone did the extra work of splitting language pairs into
# individual text files here: http://www.manythings.org/anki/
#
# The English to French pairs are too big to include in the repo, so
# download to ``data/eng-fra.txt`` before continuing. The file is a tab
# separated list of translation pairs:
#
# ::
#
#     I am cold.    J'ai froid.
#
# .. Note::
#    Download the data from
#    `here <https://download.pytorch.org/tutorial/data.zip>`_
#    and extract it to the current directory.

######################################################################
# Similar to the character encoding used in the character-level RNN
# tutorials, we will be representing each word in a language as a one-hot
# vector, or giant vector of zeros except for a single one (at the index
# of the word). Compared to the dozens of characters that might exist in a
# language, there are many many more words, so the encoding vector is much
# larger. We will however cheat a bit and trim the data to only use a few
# thousand words per language.
#
# .. figure:: /_static/img/seq-seq-images/word-encoding.png
#    :alt:
#
#


######################################################################
# We'll need a unique index per word to use as the inputs and targets of
# the networks later. To keep track of all this we will use a helper class
# called ``Lang`` which has word → index (``word2index``) and index → word
# (``index2word``) dictionaries, as well as a count of each word
# ``word2count`` to use to later replace rare words.
#

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


######################################################################
# The files are all in Unicode, to simplify we will turn Unicode
# characters to ASCII, make everything lowercase, and trim most
# punctuation.
#

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


######################################################################
# To read the data file we will split the file into lines, and then split
# lines into pairs. The files are all English → Other Language, so if we
# want to translate from Other Language → English I added the ``reverse``
# flag to reverse the pairs.
#

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    #lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
     #   read().strip().split('\n')

    # Split every line into pairs and normalize
    #pairs = [[normalizeString(s) for s in l.split(',')] for l in lines]

    lines1 = open('data/raw_training_contexts.txt', encoding='utf-8').\
        read().strip().split('\n')

    #pairs1 = [normalizeString(l) for l in lines1]
    #pairs1 = []


    '''for l in lines1:
        if len(l.split('__eot__'))>=3:
            pairs1.append(l.split('__eot__')[:3])'''


    lines2 = open('data/raw_training_responses.txt', encoding='utf-8').\
        read().strip().split('\n')

    #airs2 = [[normalizeString(l)] for l in lines2]

    


    pairs =[]

    '''or x in xrange(1,10):
     	pass i in range(len(pairs1)):
    	pairs.append((pairs1[i],pairs2[i]))'''

    for i in range(len(lines1)):
    	l1 = lines1[i]
    	l2 = lines2[i]

    	'''if(len(l1.split('__eot__'))>=3):
    		pairs.append(([normalizeString(s) for s in l1.split('__eot__')[:3]],[normalizeString(l2)]))'''

    	p1 = []
    	p2 = []

    	for s in l1.split('__eot__'):
    		if(len(s)>=1):
    			p1.append(normalizeString(s))

    	p2.append(normalizeString(l2))

    	if(len(p1)>=3):
    		pairs.append((p1[:3],p2))








    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


######################################################################
# Since there are a *lot* of example sentences and we want to train
# something quickly, we'll trim the data set to only relatively short and
# simple sentences. Here the maximum length is 10 words (that includes
# ending punctuation) and we're filtering to sentences that translate to
# the form "I am" or "He is" etc. (accounting for apostrophes replaced
# earlier).
#

MAX_LENGTH = 40

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    if(len(p)>1):

        return len(p[0][0].split(' ')) < 30 and \
        len(p[0][1].split(' ')) < 30 and \
        len(p[0][2].split(' ')) < 30 and \
        len(p[1][0].split(' ')) < 20




'''and \
p[1].startswith(eng_prefixes)'''


def filterPairs(pairs):
    #return [pair for pair in pairs if filterPair(pair)]

    final = []


    for pair in pairs:
    	''''l1 = len(pair[0][0].split(' '))
    	for p in pair[0]:
    		if(len(p.split(' ')))<=l1:
    			l1 = len(p.split(' '))'''

    	'''if(len(pair[1][0].split(' '))<=l1):
    		l1 = len(pair[1][0].split(' '))'''


    	pair2 = []
    	pair3 = []

    	for p in pair[0]:
            if(len(p.split(' '))<=(MAX_LENGTH-1)):
                pair2.append(p)
            else:
                pair2.append(' '.join(p.split(' ')[:MAX_LENGTH-1]))

    	for p in pair[1]:
    		pair3.append(p)



    	final.append((pair2, pair3))


    print("printing final\n")
    print(final[0])


    '''finalpair = []


    for pair in final:
    	if(len(pair[0][0])<=MAX_LENGTH):
    		finalpair.append(pair)'''





    return final;




######################################################################
# The full process for preparing the data is:
#
# -  Read text file and split into lines, split lines into pairs
# -  Normalize text, filter by length and content
# -  Make word lists from sentences in pairs
#

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    print(pairs[0])
    pairs = filterPairs(pairs)
    '''print(pairs)'''
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0][0])
        input_lang.addSentence(pair[0][1])
        input_lang.addSentence(pair[0][2])

        output_lang.addSentence(pair[1][0])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'fra', False)
print("printing a random pair\n")
print(random.choice(pairs))
'''print(random.choice(pairs))
print(random.choice(pairs))
print(random.choice(pairs))
print(random.choice(pairs))
print(random.choice(pairs))
print(random.choice(pairs))
print(random.choice(pairs))
print(random.choice(pairs))
print(random.choice(pairs))
print(random.choice(pairs))
print(random.choice(pairs))
print(random.choice(pairs))'''

pickle.dump(pairs, open('preparepairspickle.sav', 'wb'))
pickle.dump(input_lang, open('prepareinputpickle.sav', 'wb'))
pickle.dump(output_lang, open('prepareoutputpickle.sav', 'wb'))




######################################################################
# The Seq2Seq Model
# =================
#
# A Recurrent Neural Network, or RNN, is a network that operates on a
# sequence and uses its own output as input for subsequent steps.
#
# A `Sequence to Sequence network <http://arxiv.org/abs/1409.3215>`__, or
# seq2seq network, or `Encoder Decoder
# network <https://arxiv.org/pdf/1406.1078v3.pdf>`__, is a model
# consisting of two RNNs called the encoder and decoder. The encoder reads
# an input sequence and outputs a single vector, and the decoder reads
# that vector to produce an output sequence.
#
# .. figure:: /_static/img/seq-seq-images/seq2seq.png
#    :alt:
#
# Unlike sequence prediction with a single RNN, where every input
# corresponds to an output, the seq2seq model frees us from sequence
# length and order, which makes it ideal for translation between two
# languages.
#
# Consider the sentence "Je ne suis pas le chat noir" → "I am not the
# black cat". Most of the words in the input sentence have a direct
# translation in the output sentence, but are in slightly different
# orders, e.g. "chat noir" and "black cat". Because of the "ne/pas"
# construction there is also one more word in the input sentence. It would
# be difficult to produce a correct translation directly from the sequence
# of input words.
#
# With a seq2seq model the encoder creates a single vector which, in the
# ideal case, encodes the "meaning" of the input sequence into a single
# vector — a single point in some N dimensional space of sentences.
#


######################################################################
# The Encoder
# -----------
#
# The encoder of a seq2seq network is a RNN that outputs some value for
# every word from the input sentence. For every input word the encoder
# outputs a vector and a hidden state, and uses the hidden state for the
# next input word.
#
# .. figure:: /_static/img/seq-seq-images/encoder-network.png
#    :alt:
#
#

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, input, hidden):
        #print("enterd internally\n")
        #print(input.size())
        x = input.size()
        #print("printin size\n")
        #print(x[0])

        embedded = self.embedding(input)
        #print('-----', embedded.shape)
        embedded = embedded.view(1, x[0], -1)
        #print(embedded.shape, '------------------')
        output = embedded
        #print("input size finally")
        #print(output.size())
        output, hidden = self.gru(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 2, self.hidden_size, device=device)

######################################################################
# The Decoder
# -----------
#
# The decoder is another RNN that takes the encoder output vector(s) and
# outputs a sequence of words to create the translation.
#


######################################################################
# Simple Decoder
# ^^^^^^^^^^^^^^
#
# In the simplest seq2seq decoder we use only last output of the encoder.
# This last output is sometimes called the *context vector* as it encodes
# context from the entire sequence. This context vector is used as the
# initial hidden state of the decoder.
#
# At every step of decoding, the decoder is given an input token and
# hidden state. The initial input token is the start-of-string ``<SOS>``
# token, and the first hidden state is the context vector (the encoder's
# last hidden state).
#
# .. figure:: /_static/img/seq-seq-images/decoder-network.png
#    :alt:
#
#

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input1, hidden):
        output = self.embedding(input1).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

######################################################################
# I encourage you to train and observe the results of this model, but to
# save space we'll be going straight for the gold and introducing the
# Attention Mechanism.
#


######################################################################
# Attention Decoder
# ^^^^^^^^^^^^^^^^^
#
# If only the context vector is passed betweeen the encoder and decoder,
# that single vector carries the burden of encoding the entire sentence.
#
# Attention allows the decoder network to "focus" on a different part of
# the encoder's outputs for every step of the decoder's own outputs. First
# we calculate a set of *attention weights*. These will be multiplied by
# the encoder output vectors to create a weighted combination. The result
# (called ``attn_applied`` in the code) should contain information about
# that specific part of the input sequence, and thus help the decoder
# choose the right output words.
#
# .. figure:: https://i.imgur.com/1152PYf.png
#    :alt:
#
# Calculating the attention weights is done with another feed-forward
# layer ``attn``, using the decoder's input and hidden state as inputs.
# Because there are sentences of all sizes in the training data, to
# actually create and train this layer we have to choose a maximum
# sentence length (input length, for encoder outputs) that it can apply
# to. Sentences of the maximum length will use all the attention weights,
# while shorter sentences will only use the first few.
#
# .. figure:: /_static/img/seq-seq-images/attention-decoder-network.png
#    :alt:
#
#

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.5, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        '''print("printing the sizes\n")'''
        #print(embedded[0].size())
        #print(hidden[0].size())
        #print("printed \n")
        x = embedded[0].view(1,-1)
        y = hidden[0].view(1,-1)
        '''print(x.size())
        print(y.size())'''


        attn_weights = F.softmax(
            self.attn(torch.cat((x, y), 1)), dim=1)
        '''print("printing attention weights\n")
        print(attn_weights)
        print(attn_weights.size())'''
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        '''print("printing squeezy\n")
        print(attn_weights.unsqueeze(0))
        print(attn_weights.unsqueeze(0).size())
        print(encoder_outputs.unsqueeze(0))
        print(encoder_outputs.unsqueeze(0).size())
        print("printing attn_applied\n")
        print(attn_applied)
        print(attn_applied.size())'''

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


######################################################################
# .. note:: There are other forms of attention that work around the length
#   limitation by using a relative position approach. Read about "local
#   attention" in `Effective Approaches to Attention-based Neural Machine
#   Translation <https://arxiv.org/abs/1508.04025>`__.
#
# Training
# ========
#
# Preparing Training Data
# -----------------------
#
# To train, for each pair we will need an input tensor (indexes of the
# words in the input sentence) and target tensor (indexes of the words in
# the target sentence). While creating these vectors we will append the
# EOS token to both sequences.
#

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor1 = tensorFromSentence(input_lang, pair[0][0])
    input_tensor2 = tensorFromSentence(input_lang, pair[0][1])
    input_tensor3 = tensorFromSentence(input_lang, pair[0][2])

    target_tensor = tensorFromSentence(output_lang, pair[1][0])
    return (input_tensor1, input_tensor2, input_tensor3, target_tensor)


######################################################################
# Training the Model
# ------------------
#
# To train we run the input sentence through the encoder, and keep track
# of every output and the latest hidden state. Then the decoder is given
# the ``<SOS>`` token as its first input, and the last hidden state of the
# encoder as its first hidden state.
#
# "Teacher forcing" is the concept of using the real target outputs as
# each next input, instead of using the decoder's guess as the next input.
# Using teacher forcing causes it to converge faster but `when the trained
# network is exploited, it may exhibit
# instability <http://minds.jacobs-university.de/sites/default/files/uploads/papers/ESNTutorialRev.pdf>`__.
#
# You can observe outputs of teacher-forced networks that read with
# coherent grammar but wander far from the correct translation -
# intuitively it has learned to represent the output grammar and can "pick
# up" the meaning once the teacher tells it the first few words, but it
# has not properly learned how to create the sentence from the translation
# in the first place.
#
# Because of the freedom PyTorch's autograd gives us, we can randomly
# choose to use teacher forcing or not with a simple if statement. Turn
# ``teacher_forcing_ratio`` up to use more of it.
#

teacher_forcing_ratio = 0.5


def train(input_tensor1, input_tensor2, input_tensor3, target_tensor, encoder, gruvish, gruvamc, decoder, encoder_optimizer, encoder_optimizer2, encoder_optimizer3, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden1 = encoder.initHidden()
    encoder_hidden2 = encoder_hidden1
    encoder_hidden3 = encoder_hidden1
    #print("^^^^^^^^^^^^^ ", encoder_hidden1.shape)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    encoder_optimizer2.zero_grad()
    encoder_optimizer3.zero_grad()
    '''print("In training\n")'''

    input_length1 = input_tensor1.size(0)
    input_length2 = input_tensor2.size(0)
    input_length3 = input_tensor3.size(0)

    ''''print("printing input length\n")
    print(input_length)
    print("printing target length\n")'''
    target_length = target_tensor.size(0)
    '''print(target_length)'''

    encoder_outputs1 = torch.zeros(max_length, 2*encoder.hidden_size, device=device)
    encoder_outputs2 = torch.zeros(max_length, 2*encoder.hidden_size, device=device)
    encoder_outputs3 = torch.zeros(max_length, 2*encoder.hidden_size, device=device)
    encoder_outputs123 = torch.zeros(max_length, 2*encoder.hidden_size, device=device)

    '''print("printing inital encoder_outputs\n")
    print(encoder_outputs.size())'''
    loss = 0
    #print("printing input tensors\n")
    #print(input_tensor1)
    #rint(input_tensor1[0].size())

    '''for ei in range(input_length1):
        encoder_output1, encoder_hidden1 = encoder(
            input_tensor1[ei], encoder_hidden1)
       encoder_outputs1[ei] = encoder_output1[0, 0]'''

    encoder_outputs1t, encoder_hidden1 = encoder(input_tensor1, encoder_hidden1)
    #print("encoder encoder_outputs1t----- ", encoder_outputs1t.shape)
    #print("encoder_hidden1 ------ ", encoder_hidden1.shape)
    #print("88888 ", encoder_hidden1.view(1,1,-1).shape)


    '''for ei in range(input_length2):
        encoder_output2, encoder_hidden2 = encoder(
            input_tensor2[ei], encoder_hidden2)
        encoder_outputs2[ei] = encoder_output2[0, 0]'''

    encoder_outputs2t, encoder_hidden2 = encoder(input_tensor2, encoder_hidden2)
   #print("encoder_outputs2 ***** ", encoder_outputs2.size())
   #print("done 555555555 ", encoder_outputs2[0].size())

    ''' ei in range(input_length3):
        encoder_output3, encoder_hidden3 = encoder(
            input_tensor3[ei], encoder_hidden3)
        print("encoder_output -- ", encoder_output3.shape)
        print("tttttttttt fg ", encoder_output3[0,0].shape)

        encoder_outputs3[ei] = encoder_output3[0, 0]'''

    encoder_outputs3t, encoder_hidden3 = encoder(input_tensor3, encoder_hidden3)

    print("input lengh1 *****  ", input_length1)

    for i in range(input_length1):
        encoder_outputs1[i] = encoder_outputs1t[0][i]

    for i in range(input_length2):
        encoder_outputs2[i]=encoder_outputs2t[0][i]

    for i in range(input_length3):
        encoder_outputs3[i]=encoder_outputs3t[0][i]




    decoder_input = torch.tensor([[SOS_token]], device=device)


    for t in range(len(encoder_outputs3)):
    	temp= []
    	temp.append(encoder_outputs1[t])
    	temp.append(encoder_outputs2[t])
    	temp.append(encoder_outputs3[t])
    	#rint("encoder_outputs1[t] ", encoder_outputs1[t].size())
    	hid123 = encoder_outputs1[t]
    	hid123 = hid123.view(1,1,-1)
    	for p in range(0,3):
    		x = temp[p].view(1,1,-1)
    		encoder_output123 , hid123 = gruvish(x, hid123)
    		#print(encoder_output123.size())
    		

    	#encoder_outputs123.append(encoder_output123)

    	encoder_outputs123[t] = encoder_output123[0,0]



    '''print("inital decoder input\n")
    print(encoder_outputs.size())'''

    #decoder_hidden = encoder_hidden

    #encoder_hidden123 comes from the initial 3 encoders!!!

    #encoder_outputs123 is to be calculate from encoder_outputs1,2,3


    e = []
    e1 = encoder_hidden1.view(1,1,-1)
    e.append(e1)
    e2 = encoder_hidden2.view(1,1,-1)
    e.append(e2)
    e3 = encoder_hidden3.view(1,1,-1)
    e.append(e3)


    for t in range(0,3):
    	out123, encoder_hidden123 = gruvamc(e[t])

    print("hidden 123 --- ", encoder_hidden123.shape)




    decoder_hidden = encoder_hidden123 #should come from the other 3 encoders!!!


    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs123)
            '''print("printing temnsors while calc loss\n")
            print(target_tensor[di].size())'''
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs123)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            '''print("printing temnsors while calc loss\n")
            print(target_tensor[di].size())'''

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    encoder_optimizer2.step()
    encoder_optimizer3.step()

    return loss.item() / target_length


######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time
import math


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


######################################################################
# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.
#

def trainIters(encoder, gruvish, gruvamc, decoder, n_iters, print_every=999, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    encoder_optimizer2 = optim.Adam(gruvish.parameters(), lr=learning_rate)
    encoder_optimizer3 = optim.Adam(gruvamc.parameters(), lr=learning_rate)
    '''s = random.choice(pairs)
    print("printing te random pair\n")
    print(s)'''
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    '''print("priniting the tarining pairs\n")
    print(training_pairs[0])
    print("\n")
    print(training_pairs[0][0])
    print("\n")
    print(training_pairs[0][1])
    print("\n")'''
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor1 = training_pair[0]
        input_tensor2 = training_pair[1]
        input_tensor3 = training_pair[2]
        target_tensor = training_pair[3]

        loss = train(input_tensor1, input_tensor2, input_tensor3, target_tensor, encoder, gruvish, gruvamc,
                     decoder, encoder_optimizer, encoder_optimizer2, encoder_optimizer3, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        #print("printing the loss total\n")
        #print(print_loss_total)


        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0




    #	showPlot(plot_losses)


######################################################################
# Plotting results
# ----------------
#
# Plotting is done with matplotlib, using the array of loss values
# ``plot_losses`` saved while training.
#

#import matplotlib.pyplot as plt
#plt.switch_backend('agg')
#import matplotlib.ticker as ticker
import numpy as np


'''def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)'''


######################################################################
# Evaluation
# ==========
#
# Evaluation is mostly the same as training, but there are no targets so
# we simply feed the decoder's predictions back to itself for each step.
# Every time it predicts a word we add it to the output string, and if it
# predicts the EOS token we stop there. We also store the decoder's
# attention outputs for display later.
#

def evaluate(encoder, gruvish, gruvamc, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
    	#print("sentence is ......")
    	#print(sentence)
    	'''l = len(sentence.split(' '))
    	l = l/3
    	l = int(l)'''
    	#print("length is")
    	#print(l)
    	#print("sentence now .....")
    	#print(sentence)
    	#print("\n")

    	'''sent1 = ' '.join(sentence.split(' ')[0:l+1])
    	sent2 = ' '.join(sentence.split(' ')[l+1:2*l+1])
    	sent3 = ' '.join(sentence.split(' ')[2*l+1:])'''

    	sent1 = sentence[0]
    	sent2 = sentence[1]
    	sent3 = sentence[2]

    	'''print(sent1)
    	print(sent2)
    	print(sent3)'''


    	'''print("yeah look above\n")'''


        #input_tensor = tensorFromSentence(input_lang, sentence)
    	input_tensor1 = tensorFromSentence(input_lang, sent1)
    	input_tensor2 = tensorFromSentence(input_lang, sent2)
    	input_tensor3 = tensorFromSentence(input_lang, sent3)


    	input_length1 = input_tensor1.size()[0]
    	input_length2 = input_tensor2.size()[0]
    	input_length3 = input_tensor3.size()[0]

    	encoder_hidden1 = encoder.initHidden()
    	encoder_hidden2 = encoder_hidden1
    	encoder_hidden3 = encoder_hidden1
    	#encoder_hidden123 = encoder_hidden1



        #encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    	encoder_outputs1 = torch.zeros(max_length, 2*encoder.hidden_size, device=device)
    	encoder_outputs2 = torch.zeros(max_length, 2*encoder.hidden_size, device=device)
    	encoder_outputs3 = torch.zeros(max_length, 2*encoder.hidden_size, device=device)
    	encoder_outputs123 = torch.zeros(max_length, 2*encoder.hidden_size, device=device)



    	'''for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]'''





    	'''for ei in range(input_length1):
        	encoder_output1, encoder_hidden1 = encoder(input_tensor1[ei],
                                                     encoder_hidden1)
        	encoder_outputs1[ei] += encoder_output1[0, 0]

    	for ei in range(input_length2):
        	encoder_output2, encoder_hidden2 = encoder(input_tensor2[ei],
                                                     encoder_hidden2)
        	encoder_outputs2[ei] += encoder_output2[0, 0]


    	for ei in range(input_length3):
        	encoder_output3, encoder_hidden3 = encoder(input_tensor3[ei],
                                                     encoder_hidden3)
        	encoder_outputs3[ei] += encoder_output3[0, 0]'''

    	decoder_input = torch.tensor([[SOS_token]], device=device)

    	encoder_outputs1t, encoder_hidden1 = encoder(input_tensor1, encoder_hidden1)
    	encoder_outputs2t, encoder_hidden2 = encoder(input_tensor2, encoder_hidden2)
    	encoder_outputs3t, encoder_hidden3 = encoder(input_tensor3, encoder_hidden3)

    	for i in range(input_length1):
        	encoder_outputs1[i] = encoder_outputs1t[0][i]

    	for i in range(input_length2):
        	encoder_outputs2[i] = encoder_outputs2t[0][i]

    	for i in range(input_length3):
        	encoder_outputs3[i] = encoder_outputs3t[0][i]



        


    	e = []
    	e1 = encoder_hidden1.view(1,1,-1)
    	e.append(e1)
    	e2 = encoder_hidden2.view(1,1,-1)
    	e.append(e2)
    	e3 = encoder_hidden3.view(1,1,-1)
    	e.append(e2)


    	for t in range(0,3):
        	out123, encoder_hidden123 = gruvamc(e[t])

    	decoder_hidden = encoder_hidden123

    	for t in range(len(encoder_outputs3)):
        	temp = []
        	temp.append(encoder_outputs1[t])
        	temp.append(encoder_outputs2[t])
        	temp.append(encoder_outputs3[t])
        	hid123 = encoder_outputs1[t]
        	hid123 = hid123.view(1,1,-1)
        	for p in range(0,3):
        		x = temp[p].view(1,1,-1)
        		encoder_output123 , hid123 = gruvish(x, hid123)

        	#encoder_outputs123.append(encoder_output123)
        	encoder_outputs123[t] = encoder_outputs123[0,0]

    	#encoder_outputs123 = encoder_outputs1
	    	#encoder_outputs123 = encoder_outputs123.view(1,1,-1)

    	decoded_words = []
    	decoder_attentions = torch.zeros(max_length, max_length)
    	'''print("eneterd the evaluating function\n")'''

    	for di in range(max_length):
        	decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs123)
        	decoder_attentions[di] = decoder_attention.data
        	'''print("printing decoder output data\n")
            print(decoder_output.data)
            print(decoder_output.size())'''
        	topv, topi = decoder_output.data.topk(1)
        	'''print("printing topk(1)\n")
            print(decoder_output.data.topk(1))
            print("printing topv\n")
            print(topv)
            print("printing topi\n")
        	print(topi)'''
        	

        	if topi.item() == EOS_token:
            		decoded_words.append('<EOS>')
            		break
        	else:
            		decoded_words.append(output_lang.index2word[topi.item()])
                                

        	decoder_input = topi.squeeze().detach()

    	return decoded_words,decoder_attentions[:di+1]


######################################################################
# We can evaluate random sentences from the training set and print out the
# input, target, and output to make some subjective quality judgements:
#

def evaluateRandomly(encoder, decoder, n=2):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


######################################################################
# Training and Evaluating
# =======================
#
# With all these helper functions in place (it looks like extra work, but
# it makes it easier to run multiple experiments) we can actually
# initialize a network and start training.
#
# Remember that the input sentences were heavily filtered. For this small
# dataset we can use relatively small networks of 256 hidden nodes and a
# single GRU layer. After about 40 minutes on a MacBook CPU we'll get some
# reasonable results.
#
# .. Note::
#    If you run this notebook you can train, interrupt the kernel,
#    evaluate, and continue training later. Comment out the lines where the
#    encoder and decoder are initialized and run ``trainIters`` again.
#

hidden_size = 32
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(2*hidden_size, output_lang.n_words, dropout_p=0.5).to(device)
gruvish = nn.GRU(64,64, batch_first=True).to(device)
gruvamc = nn.GRU(64,64, batch_first=True).to(device)



lrt = 0.001

for i in range(0,9):
	trainIters(encoder1, gruvish, gruvamc, attn_decoder1, 25000, print_every=1000, learning_rate=lrt)
	lrt = lrt - 0.0001

pickle.dump(encoder1, open('encoder1pickle.sav', 'wb'))
pickle.dump(attn_decoder1, open('attn_decoder1pickle.sav', 'wb'))
pickle.dump(gruvish, open('encoder2pickle.sav', 'wb'))
pickle.dump(gruvamc, open('encoder3pickle.sav', 'wb'))

######################################################################
#
'''
evaluateRandomly(encoder1, attn_decoder1)'''


######################################################################
# Visualizing Attention
# ---------------------
#
# A useful property of the attention mechanism is its highly interpretable
# outputs. Because it is used to weight specific encoder outputs of the
# input sequence, we can imagine looking where the network is focused most
# at each time step.
#
# You could simply run ``plt.matshow(attentions)`` to see attention output
# displayed as a matrix, with the columns being input steps and rows being
# output steps:
#

'''output_words, attentions = evaluate(
    encoder1, attn_decoder1, "je suis trop froid .")
plt.matshow(attentions.numpy())'''


######################################################################
# For a better viewing experience we will do the extra work of adding axes
# and labels:
#

'''def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()'''


def evaluateAndShowAttention(input_sentence):
	
    output_words, attentions = evaluate(
        encoder1, gruvish, gruvamc, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    print("\n")

''' showAttention(input_sentence, output_words, attentions)'''


for y in range(0,100):
	evaluateAndShowAttention(random.choice(pairs)[0])

print("outputs of origbitrain.py ***********")


######################################################################
# Exercises
# =========
#
# -  Try with a different dataset
#
#    -  Another language pair
#    -  Human → Machine (e.g. IOT commands)
#    -  Chat → Response
#    -  Question → Answer
#
# -  Replace the embeddings with pre-trained word embeddings such as word2vec or
#    GloVe
# -  Try with more layers, more hidden units, and more sentences. Compare
#    the training time and results.
# -  If you use a translation file where pairs have two of the same phrase
#    (``I am test \t I am test``), you can use this as an autoencoder. Try
#    this:
#
#    -  Train as an autoencoder
#    -  Save only the Encoder network
#    -  Train a new Decoder for translation from there
#
