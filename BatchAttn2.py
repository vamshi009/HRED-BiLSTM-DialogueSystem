
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
import gensim
from gensim.models import Word2Vec


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################################################

'''SOS_token = 0
EOS_token = 1'''
INPUT_LENGTH = 5
class Lang:
	def __init__(self, name):
		self.name = name
		self.word2index = {}
		self.word2count = {}
		self.index2word = {}
		self.n_words = 0  # Count SOS and EOS

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
			s2 = normalizeString(s)
			if(len(s2)>=1):
				p1.append(s2)

		p2.append(normalizeString(l2))

		if((len(p1)>=1) and (len(p1)<=INPUT_LENGTH)):
			pairs.append((p1,p2))

		if(len(p1)>INPUT_LENGTH):
			pairs.append((p1[:INPUT_LENGTH],[p1[INPUT_LENGTH]]))









	# Reverse pairs, make Lang instances
	if reverse:
		pairs = [list(reversed(p)) for p in pairs]
		input_lang = Lang(lang2)
		output_lang = Lang(lang1)
	else:
		input_lang = Lang(lang1)
		output_lang = Lang(lang2)

	return input_lang, output_lang, pairs



MAX_LENGTH = 50

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
			if(len(p.split(' '))<=(MAX_LENGTH-4)):
				temp = "SOS "
				temp = temp + p
				finale = temp + " EOS"
				pair2.append(finale)
			else:
				temp = "SOS "
				temp2 = ' '.join(p.split(' ')[:MAX_LENGTH-4])
				temp = temp + temp2
				finale = temp + " EOS"
				pair2.append(finale)

		for p in pair[1]:
			if(len(p.split(' '))<=(MAX_LENGTH-4)):
				tempi = "SOS "
				tempi = tempi + p
				finale = tempi + " EOS"
				pair3.append(finale)
			else:
				tempi = "SOS "
				tempi2 = ' '.join(p.split(' ')[:MAX_LENGTH-4])
				tempi = tempi + tempi2
				finale = tempi + " EOS"
				pair3.append(finale) 


		final.append((pair2, pair3))


	print("printing final\n")
	print(final[0])


	'''finalpair = []


	for pair in final:
		if(len(pair[0][0])<=MAX_LENGTH):
			finalpair.append(pair)'''





	return final;





sentlist = []
dik = {}
dik2 = {}


def prepareData(lang1, lang2, reverse=False):
	input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
	print("Read %s sentence pairs" % len(pairs))
	print(pairs[0])
	pairs = filterPairs(pairs)
	'''print(pairs)'''
	print("Trimmed to %s sentence pairs" % len(pairs))
	print("Counting words...")
	'''for pair in pairs:
		input_lang.addSentence(pair[0][0])
		input_lang.addSentence(pair[0][1])
		input_lang.addSentence(pair[0][2])

		output_lang.addSentence(pair[1][0])
		input_lang.addSentence(pair[1][0])
		x = pair[0]
		for t in range(len(x)):
			input_lang.addSentence(x[t])
			output_lang.addSentence(x[t])'''
	for p in pairs:
		x = p[0]
		for i in range(len(x)):
			sentlist.append(x[i].split())
		y = p[1][0]
		sentlist.append(y.split())


	modelw2v = Word2Vec(sentences=sentlist, size=64, window=5, min_count=0, workers=1, sg=0)
	modelw2vN2 = Word2Vec(sentences = sentlist, size=128, window=5, min_count=0, workers=1, sg=0)
	pickle.dump(modelw2v, open('picklefiles/BiHREDEpochs1H64wv2AttnBatch5D10LPadd5/w2v64.sav','wb'))
	pickle.dump(modelw2vN2, open('picklefiles/BiHREDEpochs1H64wv2AttnBatch5D10LPadd5/w2v128.sav','wb'))
	

	for word, vocab_obj in modelw2v.wv.vocab.items():
		dik[vocab_obj.index]=word
		input_lang.word2index[word] = vocab_obj.index


	for word, vocab_obj in modelw2vN2.wv.vocab.items():
		dik2[vocab_obj.index] = word
		output_lang.word2index[word] = vocab_obj.index

	for i in range(len(dik)):
		input_lang.index2word[i]=dik[i]

	for i in range(len(dik2)):
		output_lang.index2word[i] = dik2[i]

	input_lang.n_words = len(modelw2v.wv.vocab.items())
	output_lang.n_words = len(modelw2vN2.wv.vocab.items())

	print("Counted words:")
	print(input_lang.name, input_lang.n_words)
	print(output_lang.name, output_lang.n_words)
	return input_lang, output_lang, pairs, modelw2v, modelw2vN2




input_lang, output_lang, pairs, modelw2v, modelw2vN2 = prepareData('eng', 'fra', False)
print("printing a random pair\n")
print(random.choice(pairs))

EOS_token = output_lang.word2index['EOS']


'''input_lang = pickle.load(open('picklefiles/Epochs3/prepareinputpickle.sav','rb'))
output_lang = pickle.load(open('picklefiles/Epochs3/prepareoutputpickle.sav','rb'))
pairs = pickle.load(open('picklefiles/Epochs3/preparepairspickle.sav','rb'))'''


'''pickle.dump(pairs, open('picklefiles/BiHREDEpochs1H64wv2Attn/preparepairspickle.sav', 'wb'))
pickle.dump(input_lang, open('picklefiles/BiHREDEpochs1H64wv2Attn/prepareinputpickle.sav', 'wb'))
pickle.dump(output_lang, open('picklefiles/BiHREDEpochs1H64wv2Attn/prepareoutputpickle.sav', 'wb'))'''




Weights = torch.FloatTensor(modelw2v.wv.syn0, device=device)

print("the word2vec model is +++++++++++++ ", modelw2v)

print(" weights are --- ", Weights.shape)

Weights = torch.cat((Weights, torch.zeros(1,64)))

print("modified weihts are--- ", Weights.shape)

'''print("w2v weights are-- ", Weights)'''

WeightsN2 = torch.FloatTensor(modelw2vN2.wv.syn0, device=device)

print(" second w2v weight srae---", WeightsN2.shape)

WeightsN2 = torch.cat((WeightsN2, torch.zeros(1,128)))

print("second w2v has modified weights ---", WeightsN2.shape)

'''print("2w2v weiht srae+++++ ", WeightsN2)'''


padding_token = input_lang.n_words
'''padding_token=1'''

output_lang.word2index["PAD"] = padding_token
output_lang.index2word[padding_token] = "PAD"



class EncoderRNN(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(EncoderRNN, self).__init__()
		self.hidden_size = hidden_size

		self.embedding = nn.Embedding.from_pretrained(Weights)
		self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)

	def forward(self, input, hidden):


		'''print("input is ******** ", input)'''

		x = input[0]

		embedded = self.embedding(x)
		#print('-----', embedded.shape)
		'''print("embbedded in encoder****** ", embedded.shape)'''
		embedded = embedded.view(1, x.size(0), -1)
		final_embed = embedded

		for i in range(1, len(input)):
			x = input[i]
			embedded = self.embedding(x)
			embedded = embedded.view(1, x.size(0), -1)
			'''print(" the internal embedded shape ####### 99 ",embedded.shape)'''
			final_embed = torch.cat((final_embed, embedded))

		'''print("final embbeding is ---- ", final_embed.shape)'''

		#print(embedded.shape, '------------------')
		output = final_embed
		'''print("device name ==== ", output.device)'''
		output, hidden = self.gru(output)
		'''print("output after gru  is --- ", output.shape)'''
		'''print("hidden after gru is --- ", hidden.shape)'''
		return output, hidden

	def initHidden(self):
		return torch.zeros(1, 2, self.hidden_size, device=device)




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




class AttnDecoderRNN(nn.Module):
	def __init__(self, hidden_size, output_size, dropout_p=0.5, max_length=MAX_LENGTH):
		super(AttnDecoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.dropout_p = dropout_p
		self.max_length = max_length

		self.embedding = nn.Embedding.from_pretrained(WeightsN2)
		self.attn = nn.Linear(self.hidden_size * 2, INPUT_LENGTH*self.max_length)
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
		print(attn_weights.size())'''
		attn_applied = torch.bmm(attn_weights.unsqueeze(0),
								 encoder_outputs.unsqueeze(0))
		'''print("printing squeezy\n")
		print(attn_weights.unsqueeze(0))
		print(attn_weights.unsqueeze(0).size())
		print(encoder_outputs.unsqueeze(0))
		print(encoder_outputs.unsqueeze(0).size())'''
		'''print("printing attn_applied\n")
		print(attn_applied[0].size())'''

		output = torch.cat((embedded[0], attn_applied[0]), 1)
		output = self.attn_combine(output).unsqueeze(0)

		output = F.relu(output)
		output, hidden = self.gru(output, hidden)


		output = F.log_softmax(self.out(output[0]), dim=1)
		return output, hidden, attn_weights

	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=device)


def indexesFromSentence(lang, sentence):
	l = [lang.word2index[word] for word in sentence.split()]
	si = len(l)
	for i in range(si, MAX_LENGTH):
		l.append(padding_token)
	return l


def tensorFromSentence(lang, sentence):
	indexes = indexesFromSentence(lang, sentence)
	'''indexes.append(EOS_token)'''
	return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):

	x = pair[0]
	it = []
	ot = []
	tensorpair = []

	padding_indexes = [(padding_token) for i in range(0,MAX_LENGTH)]
	padding_tensor = torch.tensor(padding_indexes, dtype = torch.long, device=device).view(-1,1)

	for i in range(len(x)):
		it.append(tensorFromSentence(input_lang, pair[0][i]))
	for i in range(len(x), INPUT_LENGTH):
		it.append(padding_tensor)

	'''input_tensor1 = tensorFromSentence(input_lang, pair[0][0])
	input_tensor2 = tensorFromSentence(input_lang, pair[0][1])
	input_tensor3 = tensorFromSentence(input_lang, pair[0][2])

	target_tensor = tensorFromSentence(output_lang, pair[1][0])'''
	ot.append(tensorFromSentence(output_lang, pair[1][0]))
	tensorpair.append((it,ot))
	'''return (input_tensor1, input_tensor2, input_tensor3, target_tensor)'''
	return tensorpair



teacher_forcing_ratio = 0.5


def train(batch_size, training_pair, encoder, gruvish, gruvamc, decoder, encoder_optimizer, encoder_optimizer2, encoder_optimizer3, decoder_optimizer, criterion, max_length=MAX_LENGTH):
	encoder_hidden1 = encoder.initHidden()
	encoder_hidden2 = encoder_hidden1
	encoder_hidden3 = encoder_hidden1
	#print("^^^^^^^^^^^^^ ", encoder_hidden1.shape)

	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()
	encoder_optimizer2.zero_grad()
	encoder_optimizer3.zero_grad()



	x = training_pair[0][0]

	Total_Encoder_outputs = []

	for i in range(batch_size):
		Encoder_outputs=[] 

		for i in range(len(x)):
			encoder_outputs1 = torch.zeros(max_length, 2*encoder.hidden_size, device=device)
			Encoder_outputs.append(encoder_outputs1)

		Total_Encoder_outputs.append(Encoder_outputs)

   
	loss = 0
	

	Encoder_outputsT = []
	Encoder_hidden = []

	Total_list = []
	t = training_pair[0][0]

	'''print("t is ---- ",t)'''

	for i in range(len(t)):
		y = []
		Total_list.append(y)

	for i in range(batch_size):
		temp = training_pair[i][0]

		for d in range(len(temp)):
			Total_list[d].append(temp[d])

	encoder_hidden = encoder.initHidden()




	for i in range(len(t)):
		'''encoder_hidden = encoder.initHidden()'''
		'''print(" sending total list[d] as --- ", len(Total_list[i]))'''
		encoder_outputst, encoder_hidden = encoder(Total_list[i], encoder_hidden)
		'''print(" output t is ######### ", encoder_outputst.shape)'''
		'''print(" hiden is ########## ", encoder_hidden.shape)'''
		Encoder_outputsT.append(encoder_outputst)
		Encoder_hidden.append(encoder_hidden)






	for i in range(batch_size):
		Temp_Encoder_outputs=[]
		for j in range(len(Encoder_outputsT)):
			Temp_Encoder_outputs.append(Encoder_outputsT[j][i])

		'''print(" temp Encodere outputs -- ",Temp_Encoder_outputs[0].shape)'''

		Pre_Encoder_hiddens = []

		for j in range(len(Encoder_hidden)):
			temp = Encoder_hidden[j]
			temp1 = temp[0]
			temp2 = temp[1]
			ans1 = temp1[i]
			ans2 = temp2[i]
			fin = torch.cat((ans1,ans2))
			Pre_Encoder_hiddens.append(fin)

		'''print(" Pre Encoder hiddens arw -- ", Pre_Encoder_hiddens[0].shape)'''
	
		attnoutputs = Temp_Encoder_outputs[0]


		for g in range(1, len(Temp_Encoder_outputs)):
			attnoutputs = torch.cat((attnoutputs, Temp_Encoder_outputs[g]))

		for h in range(len(Temp_Encoder_outputs), INPUT_LENGTH):
			attnoutputs = torch.cat((attnoutputs, torch.zeros(max_length, 2*encoder.hidden_size, device=device)))

		'''print("index is ******** ", output_lang.word2index['SOS'])'''




		decoder_input = torch.tensor([output_lang.word2index['SOS']], device=device)


		'''print(" pre Encoder hidden shape is --", Pre_Encoder_hiddens[0].shape)'''


		batch_encoder_hidden = Pre_Encoder_hiddens[0].view(1,1,-1)
		'''print("before size id ----- ", batch_encoder_hidden.shape)
		print("")'''

		for h in range(1, len(Pre_Encoder_hiddens)):
			batch_encoder_hidden = torch.cat((batch_encoder_hidden, Pre_Encoder_hiddens[h].view(1,1,-1)),1)


		'''print("batch_encoder_hidden size is $$$$ ", batch_encoder_hidden.shape)'''


		'''for t in range(0,3):
			out123, encoder_hidden123 = gruvamc(e[t])'''

		#print("hidden 123 --- ", encoder_hidden123.shape)

		testout, testhidd = gruvamc(batch_encoder_hidden)

		'''print("test output shape--- ", testout.shape)
		print(" test hidden shape---- ", testhidd.shape)'''





		'''decoder_hidden = encoder_hidden123 #should come from the other 3 encoders!!!'''


		decoder_hidden = testhidd
		target_tensor = training_pair[i][1][0]
		target_length = target_tensor.size(0)


		'''use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False'''
		use_teacher_forcing = True

		if use_teacher_forcing:
			# Teacher forcing: Feed the target as the next input
			for di in range(target_length):
				decoder_output, decoder_hidden, decoder_attention = decoder(
					decoder_input, decoder_hidden, attnoutputs)
				'''print("printing temnsors while calc loss\n")
				print(target_tensor[di].size())'''
				loss += criterion(decoder_output, target_tensor[di])
				decoder_input = target_tensor[di]  # Teacher forcing

		else:
			# Without teacher forcing: use its own predictions as the next input
			for di in range(training_pair[0][1][0].size(0)):
				decoder_output, decoder_hidden, decoder_attention = decoder(
					decoder_input, decoder_hidden, attnoutputs)
				topv, topi = decoder_output.topk(1)
				decoder_input = topi.squeeze().detach()  # detach from history as input
				'''print("printing temnsors while calc loss\n")
				print(target_tensor[di].size())'''

				loss += criterion(decoder_output, target_tensor[di])
				if decoder_input.item() == EOS_token:
					break

	print("calculated loss is---- ",loss.item())

	loss.backward()

	encoder_optimizer.step()
	decoder_optimizer.step()
	encoder_optimizer2.step()
	encoder_optimizer3.step()

	return loss.item() / target_length



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


def trainIters(batch_size, encoder, gruvish, gruvamc, decoder, n_iters, encoder_optimizer, decoder_optimizer, encoder_optimizer2, encoder_optimizer3, print_every=999, plot_every=100, learning_rate=0.001):
	start = time.time()
	plot_losses = []
	print_loss_total = 0  # Reset every print_every
	plot_loss_total = 0  # Reset every plot_every

	'''encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
	decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
	encoder_optimizer2 = optim.Adam(gruvish.parameters(), lr=learning_rate)
	encoder_optimizer3 = optim.Adam(gruvamc.parameters(), lr=learning_rate)'''
	'''s = random.choice(pairs)
	print("printing te random pair\n")
	print(s)'''
	'''training_pairs = [tensorsFromPair(random.choice(pairs))
					  for i in range(n_iters)]'''
	'''print("priniting the tarining pairs\n")
	print(training_pairs[0])
	print("\n")
	print(training_pairs[0][0])
	print("\n")
	print(training_pairs[0][1])
	print("\n")'''
	criterion = nn.NLLLoss()

	for iter in range(1, n_iters + 1):
		training_pair = []

		for i in range(batch_size):
			training_pair.extend(tensorsFromPair(random.choice(pairs)))
		'''input_tensor1 = training_pair[0]
		input_tensor2 = training_pair[1]
		input_tensor3 = training_pair[2]
		target_tensor = training_pair[3]'''

		loss = train(batch_size, training_pair, encoder, gruvish, gruvamc,
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


def evaluate(encoder, gruvish, gruvamc, decoder, sentence, max_length=MAX_LENGTH):
	with torch.no_grad():
		

		#input_tensor = tensorFromSentence(input_lang, sentence)
		'''input_tensor1 = tensorFromSentence(input_lang, sent1)
		input_tensor2 = tensorFromSentence(input_lang, sent2)
		input_tensor3 = tensorFromSentence(input_lang, sent3)'''

		intensors = []

		for r in range(len(sentence)):
			intensors.append(tensorFromSentence(input_lang, sentence[r]))



		'''input_length1 = input_tensor1.size()[0]
		input_length2 = input_tensor2.size()[0]
		input_length3 = input_tensor3.size()[0]'''

		encoder_hidden1 = encoder.initHidden()
		encoder_hidden2 = encoder_hidden1
		encoder_hidden3 = encoder_hidden1
		#encoder_hidden123 = encoder_hidden1



		#encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
		'''encoder_outputs1 = torch.zeros(max_length, 2*encoder.hidden_size, device=device)
		encoder_outputs2 = torch.zeros(max_length, 2*encoder.hidden_size, device=device)
		encoder_outputs3 = torch.zeros(max_length, 2*encoder.hidden_size, device=device)
		encoder_outputs123 = torch.zeros(max_length, 2*encoder.hidden_size, device=device)'''

		Encoder_outputs = []


		for r in range(len(sentence)):
			encoder_outputs1 = torch.zeros(max_length, 2*encoder.hidden_size, device=device)
			Encoder_outputs.append(encoder_outputs1)


		encoder_outputsf = torch.zeros(max_length, 2*encoder.hidden_size, device=device)




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

		decoder_input = torch.tensor([output_lang.word2index['SOS']], device=device)

		'''encoder_outputs1t, encoder_hidden1 = encoder(input_tensor1, encoder_hidden1)
		encoder_outputs2t, encoder_hidden2 = encoder(input_tensor2, encoder_hidden2)
		encoder_outputs3t, encoder_hidden3 = encoder(input_tensor3, encoder_hidden3)'''

		Encoder_outputsT = []
		Encoder_hidden = []
		encoder_hidden1 = encoder.initHidden()


		for p in range(len(intensors)):
			encoder_outputst, encoder_hidden1 = encoder([intensors[p]], encoder_hidden1)
			Encoder_outputsT.append(encoder_outputst)
			Encoder_hidden.append(encoder_hidden1)

		'''for i in range(input_length1):
			encoder_outputs1[i] = encoder_outputs1t[0][i]

		for i in range(input_length2):
			encoder_outputs2[i] = encoder_outputs2t[0][i]

		for i in range(input_length3):
			encoder_outputs3[i] = encoder_outputs3t[0][i]'''

		for i in range(len(Encoder_outputsT)):
			d = Encoder_outputsT[i]

			for j in range(len(d)):
				Encoder_outputs[i][j] = d[0][j]
		
		attnoutputs = Encoder_outputs[0]

		for g in range(1, len(Encoder_outputs)):
			attnoutputs = torch.cat((attnoutputs, Encoder_outputs[g]))

		for h in range(len(Encoder_outputs), INPUT_LENGTH):
			attnoutputs = torch.cat((attnoutputs, torch.zeros(max_length, 2*encoder.hidden_size, device=device)))



		


		'''e = []
		e1 = encoder_hidden1.view(1,1,-1)
		e.append(e1)
		e2 = encoder_hidden2.view(1,1,-1)
		e.append(e2)
		e3 = encoder_hidden3.view(1,1,-1)
		e.append(e2)'''

		e = []

		for t in range(len(Encoder_hidden)):
			e1 = Encoder_hidden[t].view(1,1,-1)
			e.append(e1)


		'''for t in range(0,3):
			out123, encoder_hidden123 = gruvamc(e[t])'''

		for t in range(len(e)):
			outf, encoder_hiddenf = gruvamc(e[t])

		'''decoder_hidden = encoder_hidden123'''

		decoder_hidden = encoder_hiddenf

		'''for t in range(len(encoder_outputs3)):
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
		
			encoder_outputs123[t] = encoder_outputs123[0,0]'''



		'''for t in range(len(Encoder_outputs[0])):
			temp = []

			for r in range(len(Encoder_outputs)):
				temp.append(Encoder_outputs[r][t])

			hid123 = Encoder_outputs[0][t]
			hid123 = hid123.view(1,1,-1)
			for p in range(len(temp)):
				x = temp[p].view(1,1,-1)
				encoder_outputf, hid123 = gruvish(x, hid123)

			encoder_outputsf[t] = encoder_outputf[0,0]'''





		decoded_words = []
		decoder_attentions = torch.zeros(max_length, INPUT_LENGTH*max_length)
		'''print("eneterd the evaluating function\n")'''

		for di in range(max_length):
			decoder_output, decoder_hidden, decoder_attention = decoder(
				decoder_input, decoder_hidden, attnoutputs)
			'''print("shape of decoder attentio is -- ", decoder_attention.shape)'''
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


def evaluateRandomly(encoder, decoder, n=2):
	for i in range(n):
		pair = random.choice(pairs)
		print('>', pair[0])
		print('=', pair[1])
		output_words, attentions = evaluate(encoder, decoder, pair[0])
		output_sentence = ' '.join(output_words)
		print('<', output_sentence)
		print('')

lrt = 0.001

hidden_size = 64


encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(2*hidden_size, output_lang.n_words+1, dropout_p=0.5).to(device)
gruvish = nn.GRU(128,128, batch_first=True).to(device)
gruvamc = nn.GRU(128,128, batch_first=True).to(device)

'''encoder1 = pickle.load(open('picklefiles/Epochs3/encoder1pickle.sav','rb'))
attn_decoder1 = pickle.load(open('picklefiles/Epochs3/attn_decoder1pickle.sav','rb'))
gruvish = pickle.load(open('picklefiles/Epochs3/encoder2pickle.sav','rb'))
gruvamc = pickle.load(open('picklefiles/Epochs3/encoder3pickle.sav','rb'))'''



encoder_optimizer = optim.Adam(encoder1.parameters(), lr=lrt)
decoder_optimizer = optim.Adam(attn_decoder1.parameters(), lr=lrt)
encoder_optimizer2 = optim.Adam(gruvish.parameters(), lr=lrt)
encoder_optimizer3 = optim.Adam(gruvamc.parameters(), lr=lrt)




for i in range(0,7):
	lrt = 0.001
	for phi in range(0,10):
		trainIters(5, encoder1, gruvish, gruvamc, attn_decoder1, 5000, encoder_optimizer, decoder_optimizer, encoder_optimizer2, encoder_optimizer3, print_every=1000, learning_rate=lrt)
		lrt = lrt - 0.0001

print("training done!!!")

pickle.dump(encoder1, open('picklefiles/BiHREDEpochs1H64wv2AttnBatch5D10LPadd5/encoder1pickle.sav', 'wb'))
pickle.dump(attn_decoder1, open('picklefiles/BiHREDEpochs1H64wv2AttnBatch5D10LPadd5/attn_decoder1pickle.sav', 'wb'))
pickle.dump(gruvish, open('picklefiles/BiHREDEpochs1H64wv2AttnBatch5D10LPadd5/encoder2pickle.sav', 'wb'))
pickle.dump(gruvamc, open('picklefiles/BiHREDEpochs1H64wv2AttnBatch5D10LPadd5/encoder3pickle.sav', 'wb'))

######################################################################
#
'''
evaluateRandomly(encoder1, attn_decoder1)'''


'''output_words, attentions = evaluate(
	encoder1, attn_decoder1, "je suis trop froid .")
plt.matshow(attentions.numpy())'''



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

life = []
def evaluateAndShowAttention(input_sentence):
	
	output_words, attentions = evaluate(
		encoder1, gruvish, gruvamc, attn_decoder1, input_sentence)
	print('input =', input_sentence)
	print('output =', ' '.join(output_words))
	print("\n")
	w = ' '.join(output_words)
	if(w not in life):
		life.append(w)

''' showAttention(input_sentence, output_words, attentions)'''


for y in range(0,10000):
	x = random.choice(pairs)
	evaluateAndShowAttention(x[0])
	print("expected output----",x[1])
	print("\n")


print("outputs from BiHREDEpochs1H64wv2Attn on 3,00,000 2 epochs with (64,128) and sentence lengthe is 3 ***********")

print("total diffrent ioutputs out of 10,000 are---- ", len(life))

for i in range(len(life)):
	print(life[i])

