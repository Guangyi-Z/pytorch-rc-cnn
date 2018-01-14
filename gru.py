import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
cuda = torch.cuda.is_available()

import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from utils import load_data, build_dict, gen_embeddings


class Net(nn.Module):

    def __init__(self, word_dict, entity_dict, embeddings, embedding_dim, hidden_dim):
        super(Net, self).__init__()

        self.IDX_UNK = 0
        
        self.word_dict = word_dict
        self.embeddings = embeddings
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim // 2 * 2

        self.d_gru = nn.GRU(embedding_dim, self.hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        self.q_gru = nn.GRU(embedding_dim, self.hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        
        self.entity_dict = entity_dict
        self.entity_dim = len(entity_dict)
        self.lin = nn.Linear(self.hidden_dim, self.entity_dim)
            
    def init_hidden(self):
        # Variable(num_layers*num_directions, minibatch_size, hidden_dim)
        if cuda:
            return Variable(torch.randn(2, 1, self.hidden_dim // 2).cuda())
        return Variable(torch.randn(2, 1, self.hidden_dim // 2))
    
    def forward(self, d, q):
        d_words = d.split()
        q_words = q.split()
        d_idx = [self.word_dict.get(dw, self.IDX_UNK) for dw in d_words]
        q_idx = [self.word_dict.get(qw, self.IDX_UNK) for qw in q_words]
        d_emb = [self.embeddings[i] for i in d_idx] # !bug: max_words not in word_dict
        q_emb = [self.embeddings[i] for i in q_idx]
        if cuda:
            d_emb = Variable(torch.FloatTensor(d_emb).cuda(), requires_grad=True)
            q_emb = Variable(torch.FloatTensor(q_emb).cuda(), requires_grad=True)
        else:
            d_emb = Variable(torch.FloatTensor(d_emb), requires_grad=True)
            q_emb = Variable(torch.FloatTensor(q_emb), requires_grad=True)
        
        d_hidden = self.init_hidden()
        q_hidden = self.init_hidden()
        d_gru_out, d_hidden = self.d_gru(d_emb.view(len(d_words), 1, -1), # (seq_len, batch, input_size)
                                         d_hidden)
        q_gru_out, q_hidden = self.q_gru(q_emb.view(len(q_words), 1, -1), # (seq_len, batch, input_size)
                                         q_hidden)
        q_gru_out_mean = q_gru_out.view(len(q_words), -1).mean(dim=0)
        
        d_gru_out = d_gru_out.view(len(d_words), self.hidden_dim)
        q_gru_out = q_gru_out.view(len(q_words), self.hidden_dim)
        #sim = torch.mm(d_gru_out, q_hidden.view(self.hidden_dim,1))
        sim = torch.mm(d_gru_out, q_gru_out_mean.view(self.hidden_dim,1))
        o = torch.sum(d_gru_out * sim, dim=0)
        
        ol = self.lin(o)
        if cuda:
            dummy = Variable(torch.FloatTensor([float('-inf')] * self.entity_dim).cuda())
        else:
            dummy = Variable(torch.FloatTensor([float('-inf')] * self.entity_dim).cuda())
        ol2 = torch.cat((ol.view(-1,1), dummy.view(-1,1)), dim=1)
        d_ent_idx = set(list(filter(lambda x: x, 
                                [self.entity_dict.get(dw, None)
                                 for dw in d_words])))
        o_idx = [0 if i in d_ent_idx else 1 for i in range(self.entity_dim)]
        if cuda:
            o = ol2.gather(1, Variable(torch.Tensor(o_idx).cuda().long().view(-1,1)))
        else:
            o = ol2.gather(1, Variable(torch.Tensor(o_idx).long().view(-1,1)))
        
        return F.log_softmax(o, dim=0)


def main():
    fin_train = 'data/cnn/train.txt'
    fin_dev = 'data/cnn/dev.txt'
    sz_train = 50000
    sz_dev = 1000
    sz_test = 5000
    embedding_size = 50
    hidden_dim = 128
    lr = 0.05
    momentum = 0.0

    print('*' * 10 + ' Train Loading')
    train_d, train_q, train_a = load_data(fin_train, sz_train, relabeling=True)
    print('*' * 10 + ' Dev Loading')
    dev_d, dev_q, dev_a = load_data(fin_dev, sz_dev, relabeling=True)

    print('Build dictionary..')
    word_dict = build_dict(train_d + train_q)
    entity_markers = list(set([w for w in word_dict.keys()
                  if w.startswith('@entity')] + train_a))
    entity_markers = ['<unk_entity>'] + entity_markers
    entity_dict = {w: index for (index, w) in enumerate(entity_markers)}
    print('Entity markers: %d' % len(entity_dict))
    num_labels = len(entity_dict)
    
    embeddings = gen_embeddings(word_dict, embedding_size, 'data/glove.6B/glove.6B.{}d.txt'.format(embedding_size))

    torch.manual_seed(42)

    net = Net(word_dict, entity_dict, embeddings, embedding_size, hidden_dim)
    if cuda: 
        net.cuda()
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    #optimizer = optim.Adam(net.parameters(), lr=lr)

    losses = list()
    cnt = 0
    for i,d,q,a in zip(range(len(train_a)), train_d, train_q, train_a):
        net.zero_grad()
        log_probs = net(d, q)
        if cuda:
            target = Variable(torch.LongTensor([entity_dict[a]]).cuda())
        else:
            target = Variable(torch.LongTensor([entity_dict[a]]))
        if cuda: target.cuda()
        loss = loss_function(log_probs.view(1,-1), target)
        losses.append(loss.cpu().data.numpy().tolist()[0])
        loss.backward()
        torch.nn.utils.clip_grad_norm(net.parameters(), 1)
        optimizer.step()
        
        if i % 50 == 0:
            print('COUNT {}, Loss: {}'.format(i, losses[-1]))
    plt.plot(range(len(losses)), losses)
    plt.savefig('tmp-pic-repo/loss-lr{}-momentum{}-trainsz{}-embsz{}-hidsz{}.png'.format(lr, momentum, sz_train, embedding_size, hidden_dim))
    #plt.savefig('tmp-pic-repo/loss-Adamlr{}-trainsz{}-embsz{}-hidsz{}.png'.format(lr, sz_train, embedding_size, hidden_dim))

    acc = 0
    for i,d,q,a in zip(range(len(train_a)), train_d, train_q, train_a):
        if i >= sz_test:
            break
            
        log_probs = net(d, q)
        target = entity_dict[a]
        _, idx = torch.max(log_probs, 0)
        acc += (idx.cpu().data.numpy().tolist()[0]) == target
    print('acc: {}'.format(float(acc)/sz_test))


if __name__ == '__main__':
    print('start')
    print('using gpu: {}'.format(cuda))
    main()
