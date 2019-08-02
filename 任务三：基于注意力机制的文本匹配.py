# Author:Zhangbingbin 
# Time:2019/8/1
from torchtext.data import Iterator, BucketIterator
from torchtext import data
from torchtext.vocab import Vectors
from torchtext.vocab import GloVe
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
torch.manual_seed(1)


### LOAD_DATA
def load_iters(batch_size=1, device="cpu", data_path='data'):
    TEXT = data.Field(batch_first=True, include_lengths=True, lower=True)
    LABEL = data.LabelField(batch_first=True)
    fields = {'sentence1': ('premise', TEXT),
              'sentence2': ('hypothesis', TEXT),
              'gold_label': ('label', LABEL)}

    train_data, dev_data, test_data = data.TabularDataset.splits(
        path=data_path,
        train='snli_1.0_train.jsonl',
        validation='snli_1.0_dev.jsonl',
        test='snli_1.0_test.jsonl',
        format='json',
        fields=fields,
        filter_pred=lambda ex: ex.label != '-'  # filter the example which label is '-'(means unlabeled)
    )

    TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
    LABEL.build_vocab(train_data)
    # TEXT.build_vocab(train_data, vectors=vectors, unk_init=torch.Tensor.normal_)

    train_iter, dev_iter = BucketIterator.splits(
        (train_data, dev_data),
        batch_sizes=(batch_size, batch_size),
        device=device,
        sort_key=lambda x: len(x.premise) + len(x.hypothesis),  #按 合并文本长度 排序，
        sort_within_batch=True,                                 #方便后面pytorch lstm进行pack和pad
        repeat=False,
        shuffle=True
    )

    test_iter = Iterator(test_data,
                         batch_size=batch_size,
                         device=device,
                         sort=False,
                         sort_within_batch=False,
                         repeat=False,
                         shuffle=False)

    return train_iter, dev_iter, test_iter, TEXT, LABEL


### MODELS
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=300, dropout_rate=0.3, layer_num=1):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.bilstm = nn.LSTM(input_size, hidden_size // 2, layer_num, batch_first=True, bidirectional=True)

        self.init_weights()

    def init_weights(self):
        for p in self.bilstm.parameters():
            if p.dim() > 1:
                nn.init.normal_(p)
                p.data.mul_(0.01)
            else:
                p.data.zero_()
                # This is the range of indices for our forget gates for each LSTM cell
                p.data[self.hidden_size // 2: self.hidden_size] = 1

    def forward(self, x, lens):
        '''
        :param x: (batch, seq_len, input_size)
        :param lens: (batch, )
        :return: (batch, seq_len, hidden_size)
        '''
        ordered_lens, index = lens.sort(descending=True)
        ordered_x = x[index]

        packed_x = nn.utils.rnn.pack_padded_sequence(ordered_x, ordered_lens, batch_first=True)
        packed_output, _ = self.bilstm(packed_x)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        recover_index = index.argsort()
        recover_output = output[recover_index]
        return recover_output


class ESIM(nn.Module):
    def __init__(self, vocab_size, num_labels, embed_size, hidden_size, dropout_rate=0.3, layer_num=1,
                 pretrained_embed=None, freeze=False):
        super(ESIM, self).__init__()
        self.pretrained_embed = pretrained_embed
        if pretrained_embed is not None:
            self.embed = nn.Embedding.from_pretrained(pretrained_embed, freeze)
        else:
            self.embed = nn.Embedding(vocab_size, embed_size)
        self.bilstm1 = BiLSTM(embed_size, hidden_size, dropout_rate, layer_num)
        self.bilstm2 = BiLSTM(hidden_size, hidden_size, dropout_rate, layer_num)
        self.fc1 = nn.Linear(4 * hidden_size, hidden_size)
        self.fc2 = nn.Linear(4 * hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout_rate)

        self.init_weight()

    def init_weight(self):
        if self.pretrained_embed is None:
            nn.init.normal_(self.embed.weight)
            self.embed.weight.data.mul_(0.01)
        nn.init.normal_(self.fc1.weight)
        self.fc1.weight.data.mul_(0.01)
        nn.init.normal_(self.fc2.weight)
        self.fc2.weight.data.mul_(0.01)
        nn.init.normal_(self.fc3.weight)
        self.fc3.weight.data.mul_(0.01)


    def soft_align_attention(self, x1, x1_lens, x2, x2_lens):
        '''
        local inference modeling
        :param x1: (batch, seq1_len, hidden_size)
        :param x1_lens: (batch, )
        :param x2: (batch, seq2_len, hidden_size)
        :param x2_lens: (batch, )
        :return: x1_align (batch, seq1_len, hidden_size)
                 x2_align (batch, seq2_len, hidden_size)
        '''
        seq1_len = x1.size(1)
        seq2_len = x2.size(1)
        batch_size = x1.size(0)

        attention = torch.matmul(x1, x2.transpose(1, 2))  # (batch, seq1_len, seq2_len)
        mask1 = torch.arange(seq1_len).expand(batch_size, seq1_len).to(x1.device) >= x1_lens.unsqueeze(
            1)  # (batch, seq1_len), 1 means <pad>
        mask2 = torch.arange(seq2_len).expand(batch_size, seq2_len).to(x1.device) >= x2_lens.unsqueeze(
            1)  # (batch, seq2_len)
        mask1 = mask1.float().masked_fill_(mask1, float('-inf'))
        mask2 = mask2.float().masked_fill_(mask2, float('-inf'))
        weight2 = F.softmax(attention + mask2.unsqueeze(1), dim=-1)  # (batch, seq1_len, seq2_len)
        x1_align = torch.matmul(weight2, x2)  # (batch, seq1_len, hidden_size)
        weight1 = F.softmax(attention.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)  # (batch, seq2_len, seq1_len)
        x2_align = torch.matmul(weight1, x1)  # (batch, seq2_len, hidden_size)
        return x1_align, x2_align

    def composition(self, x, lens):
        x = F.relu(self.fc1(x))
        x_compose = self.bilstm2(self.dropout(x), lens)  # (batch, seq_len, hidden_size)
        p1 = F.avg_pool1d(x_compose.transpose(1, 2), x.size(1)).squeeze(-1)  # (batch, hidden_size)
        p2 = F.max_pool1d(x_compose.transpose(1, 2), x.size(1)).squeeze(-1)  # (batch, hidden_size)
        return torch.cat([p1, p2], 1)  # (batch, hidden_size*2)

    def forward(self, x1, x1_lens, x2, x2_lens):
        '''
        :param x1: (batch, seq1_len)
        :param x1_lens: (batch,)
        :param x2: (batch, seq2_len)
        :param x2_lens: (batch,)
        :return: (batch, num_class)
        '''
        # Input encoding
        embed1 = self.embed(x1)  # (batch, seq1_len, embed_size)
        embed2 = self.embed(x2)  # (batch, seq2_len, embed_size)
        new_embed1 = self.bilstm1(self.dropout(embed1), x1_lens)  # (batch, seq1_len, hidden_size)
        new_embed2 = self.bilstm1(self.dropout(embed2), x2_lens)  # (batch, seq2_len, hidden_size)

        # Local inference collected over sequence
        x1_align, x2_align = self.soft_align_attention(new_embed1, x1_lens, new_embed2, x2_lens)

        # Enhancement of local inference information
        x1_combined = torch.cat([new_embed1, x1_align, new_embed1 - x1_align, new_embed1 * x1_align],
                                dim=-1)  # (batch, seq1_len, 4*hidden_size)
        x2_combined = torch.cat([new_embed2, x2_align, new_embed2 - x2_align, new_embed2 * x2_align],
                                dim=-1)  # (batch, seq2_len, 4*hidden_size)

        # Inference composition
        x1_composed = self.composition(x1_combined, x1_lens)  # (batch, 2*hidden_size), v=[v_avg; v_max]
        x2_composed = self.composition(x2_combined, x2_lens)  # (batch, 2*hidden_size)
        composed = torch.cat([x1_composed, x2_composed], -1)  # (batch, 4*hidden_size)

        # MLP classifier
        out = self.fc3(self.dropout(torch.tanh(self.fc2(self.dropout(composed)))))
        return out

### RUN

BATCH_SIZE = 200
HIDDEN_SIZE = 600  # every LSTM's(forward and backward) hidden size is half of HIDDEN_SIZE
EPOCHS = 5
DROPOUT_RATE = 0.3
LAYER_NUM = 1
LEARNING_RATE = 0.003
PATIENCE = 5
CLIP = 10
EMBEDDING_SIZE = 300
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# vectors = Vectors('glove.840B.300d.txt', 'D:\Glove')
freeze = False
data_path = 'data'


def show_example(premise, hypothesis, label, TEXT, LABEL):
    tqdm.write('Label: ' + LABEL.vocab.itos[label])
    tqdm.write('premise: ' + ' '.join([TEXT.vocab.itos[i] for i in premise]))
    tqdm.write('hypothesis: ' + ' '.join([TEXT.vocab.itos[i] for i in hypothesis]))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def eval(data_iter, name, epoch=None, use_cache=False):
    if use_cache:
        model.load_state_dict(torch.load('best_model.ckpt'))
    model.eval()
    correct_num = 0
    err_num = 0
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(data_iter):
            premise, premise_lens = batch.premise
            hypothesis, hypothesis_lens = batch.hypothesis
            labels = batch.label

            output = model(premise, premise_lens, hypothesis, hypothesis_lens)
            predicts = output.argmax(-1).reshape(-1)
            loss = loss_func(output, labels)
            total_loss += loss.item()
            correct_num += (predicts == labels).sum().item()
            err_num += (predicts != batch.label).sum().item()

    acc = correct_num / (correct_num + err_num)
    if epoch is not None:
        tqdm.write(
            "Epoch: %d, %s Acc: %.3f, Loss %.3f" % (epoch + 1, name, acc, total_loss))
    else:
        tqdm.write(
            "%s Acc: %.3f, Loss %.3f" % (name, acc, total_loss))
    return acc

def train(train_iter, dev_iter, loss_func, optimizer, epochs, patience=5, clip=5):
    best_acc = -1
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_iter):
            premise, premise_lens = batch.premise
            hypothesis, hypothesis_lens = batch.hypothesis
            labels = batch.label
            # show_example(premise[0],hypothesis[0], labels[0], TEXT, LABEL)

            model.zero_grad()
            output = model(premise, premise_lens, hypothesis, hypothesis_lens)
            loss = loss_func(output, labels)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        tqdm.write("Epoch: %d, Train Loss: %d" % (epoch + 1, total_loss))

        acc = eval(dev_iter, "Dev", epoch)
        if acc<best_acc:
            patience_counter +=1
        else:
            best_acc = acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.ckpt')
        if patience_counter >= patience:
            tqdm.write("Early stopping: patience limit reached, stopping...")
            break

if __name__ == "__main__":
    train_iter, dev_iter, test_iter, TEXT, LABEL = load_iters(BATCH_SIZE, device, data_path )

    model = ESIM(len(TEXT.vocab), len(LABEL.vocab.stoi),
                 EMBEDDING_SIZE, HIDDEN_SIZE, DROPOUT_RATE, LAYER_NUM,
                 TEXT.vocab.vectors, freeze).to(device)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_func = nn.CrossEntropyLoss()

    train(train_iter, dev_iter, loss_func, optimizer, EPOCHS, PATIENCE, CLIP)
    eval(test_iter, "Test", use_cache=True)
