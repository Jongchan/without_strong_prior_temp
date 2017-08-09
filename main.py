from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.autograd import Variable
import utils
from data import *

class ResBlock(nn.Module):
	def __init__(self, common_dim):
		super(ResBlock, self).__init__()

		#common_dim+2 --> common_dim
		self.proj = nn.Conv2d(common_dim + 2, common_dim, kernel_size=1, padding=0)
		
		self.conv1 = nn.Conv2d(common_dim, common_dim, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(common_dim, affine=False)
		self.dropout1 = nn.Dropout2d(0.05)	

		self.conv2 = nn.Conv2d(common_dim, common_dim, kernel_size=3, padding=1)
		self.bn2 = nn.BatchNorm2d(common_dim, affine=False)
		self.dropout2 = nn.Dropout2d(0.05)	

	def forward(self, x, gamma1, beta1, gamma2, beta2):
		x = F.relu( self.proj( x ) )
		res = x

		x = self.conv1( x )
		# CBN 1
		x = self.bn1( x )
		gamma1.contiguous()
		beta1.contiguous()
		gamma1 = gamma1.view( gamma1.size(0), gamma1.size(1), 1, 1 )
		gamma1 = gamma1.expand_as( x )
		beta1 = beta1.view( beta1.size(0), beta1.size(1), 1, 1 )
		beta1 = beta1.expand_as( x )
		x = gamma1 * x + beta1
		x = self.dropout1( F.relu( x ) )

		x = self.conv2( x )
		# CBN 2
		x = self.bn2( x )
		gamma2.contiguous()
		beta2.contiguous()
		gamma2 = gamma2.view( gamma2.size(0), gamma2.size(1), 1, 1 )
		gamma2 = gamma2.expand_as( x )
		beta2 = beta2.view( beta2.size(0), beta2.size(1), 1, 1 )
		beta2 = beta2.expand_as( x )
		x = gamma2 * x + beta2 
		x = self.dropout2( x )

		x += res
		return F.relu( x )


class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)

class Seq2SeqCond(nn.Module):
	def __init__(self, q_token_count, embed_size, hidden_size, num_hidden_layer, output_size):
		super(Seq2SeqCond, self).__init__()

		self.num_hidden_layer = num_hidden_layer
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.embed = nn.Embedding(q_token_count, embed_size)
		self.encoder_rnn = nn.GRU(embed_size, hidden_size, num_hidden_layer, batch_first=True)
		self.decoder_linear = nn.Linear(hidden_size, output_size)

	def before_rnn(self, x):
		N, T = x.size()
		idx = torch.LongTensor(N).fill_(T-1)

		x_cpu = x.cpu()
		for i in range(N):
			for t in range(T-1):
				if x_cpu.data[i,t] != 0 and x_cpu.data[i,t+1] == 0:
					idx[i] = t
					break
		idx = idx.type_as(x.data)
		return x, Variable(idx)

	def forward(self, word):
		word, idx = self.before_rnn(word)

		embed = self.embed(word)
		h = Variable(torch.zeros(self.num_hidden_layer, word.size(0), self.hidden_size).type_as(embed.data))
		out, _ = self.encoder_rnn(embed, h)
		idx = idx.view( word.size(0), 1, 1 ).expand( word.size(0), 1, self.hidden_size )
		out = out.gather(1, idx).view( word.size(0), self.hidden_size )

		return self.decoder_linear(out)

def add_x_y_cor(in_tensor):
	xlin = torch.linspace(-1,1,steps=in_tensor.size(2))
	ylin = torch.linspace(-1,1,steps=in_tensor.size(3))
	xcor = Variable(xlin.view((1, 1, in_tensor.size()[2], 1)).expand((1,1,in_tensor.size()[2],in_tensor.size()[3])).expand((in_tensor.size()[0], 1, in_tensor.size()[2], in_tensor.size()[3])), requires_grad=False)
	ycor = Variable(ylin.view((1, 1, 1, in_tensor.size()[3])).expand((1,1, in_tensor.size()[2],in_tensor.size()[3])).expand((in_tensor.size()[0], 1, in_tensor.size()[2], in_tensor.size()[3])), requires_grad=False)
	return torch.cat( [in_tensor, xcor.cuda(), ycor.cuda()], 1)

class ModuleNetCond(nn.Module):
	def __init__(self, cnn_feature_dim, common_dim):
		super(ModuleNetCond, self).__init__()

		self.common_dim = common_dim
		stem_layers = [
			nn.Conv2d(cnn_feature_dim + 2, common_dim, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(common_dim, affine=True),
			nn.ReLU(inplace=True)
		]
		self.stem = nn.Sequential(*stem_layers)
		self.RB1 = ResBlock(common_dim)
		self.RB2 = ResBlock(common_dim)
		self.RB3 = ResBlock(common_dim)
		cls_layers = [
			nn.Conv2d(common_dim + 2, 512, kernel_size=1),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(14, stride=14),
			Flatten(),
			nn.Linear(512,1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(inplace=True),
			nn.Linear(1024,32)
		]
		self.classifier = nn.Sequential(*cls_layers)

	def forward(self, feats, g_b):
		x = add_x_y_cor( feats )

		x = self.stem( x )

		splits = torch.split( g_b, self.common_dim, dim=1 )

		x = add_x_y_cor( x )
		x = self.RB1( x, splits[0] + 1, splits[1], splits[2] + 1, splits[3] )
		x = add_x_y_cor( x )
		x = self.RB2( x, splits[4] + 1, splits[5], splits[6] + 1, splits[7] )
		x = add_x_y_cor( x )
		x = self.RB3( x, splits[8] + 1, splits[9], splits[10] + 1, splits[11] )

		"""
		x = add_x_y_cor( x )
		x = self.RB1( x,
						1 +	g_b[:, :self.common_dim], 
							g_b[:, self.common_dim:self.common_dim*2],
						1 +	g_b[:, self.common_dim*2:self.common_dim*3], 
							g_b[:, self.common_dim*3:self.common_dim*4] )

		x = add_x_y_cor( x )
		x = self.RB2( x,
						1 +	g_b[:, self.common_dim*4:self.common_dim*5], 
							g_b[:, self.common_dim*5:self.common_dim*6],
						1 +	g_b[:, self.common_dim*6:self.common_dim*7], 
							g_b[:, self.common_dim*7:self.common_dim*8] )

		x = add_x_y_cor( x )
		x = self.RB3( x,
						1 +	g_b[:, self.common_dim*8:self.common_dim*9], 
							g_b[:, self.common_dim*9:self.common_dim*10],
						1 +	g_b[:, self.common_dim*10:self.common_dim*11], 
							g_b[:, self.common_dim*11:self.common_dim*12] )
		"""

		x = add_x_y_cor( x )
		return self.classifier( x )

if __name__ == "__main__":
	batch_size = 64
	lr = 3e-4
	wd = 1e-5
	vocab = utils.load_vocab('./data/vocab.json')
	train_loader_kwargs = {
        'question_h5': './data/CLEVR_v1.0/val_questions.h5',
        'feature_h5': './data/CLEVR_v1.0/val_features.h5',
        'vocab': vocab,
        'batch_size': batch_size,
		'shuffle': True,
	#	'num_workers': 5,
	}

	net = ModuleNetCond(1024, 128)
	pg = Seq2SeqCond(100, 200, 4096, 1, 1536)
	net.cuda()
	net.train()
	pg.cuda()
	pg.train()

	print (pg)
	print (net)
	#sys.exit(1)
	with ClevrDataLoader(**train_loader_kwargs) as train_loader:
		loss_fn = nn.CrossEntropyLoss().cuda()
		net_optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
		pg_optimizer = optim.Adam(pg.parameters(), lr=lr, weight_decay=wd)

		train_iter = 0

		for epoch in range( 20 ):
			for batch in train_loader:
				questions, _, feats, answers, programs, _ = batch
				questions_var = Variable(questions.cuda())
				feats_var = Variable(feats.cuda())
				answers_var = Variable(answers.cuda())
    
				net_optimizer.zero_grad()
				pg_optimizer.zero_grad()
    
				g_b = pg( questions_var )
				output = net( feats_var, g_b )
				
				loss = loss_fn( output, answers_var )
				loss.backward()
		
				net_optimizer.step()
				pg_optimizer.step()
			
				preds = output.data.cpu().max(1)[1]
				correct = preds.eq( answers ).sum()
    
				print ("train iter %05d: loss %.6f correct %d" % (train_iter, loss.data[0], correct))
				train_iter += 1
