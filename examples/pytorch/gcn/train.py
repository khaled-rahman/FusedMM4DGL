import argparse
import time,sys
#import numpy as np
import torch
import dgl.function as fn
import torch.nn.functional as F
import dgl
import pdb
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import CoauthorPhysicsDataset, AmazonCoBuyComputerDataset, RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset
import traceback
from sklearn.preprocessing import OneHotEncoder
from gcn import GCN, GCN2
#from gcn_mp import GCN
#from gcn_spmv import GCN
import traceback
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def citations_network(args):
    if args.dataset == 'cora':
        data = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
    return data

def create_masks(g, num_train = 1000):
    train_mask = torch.zeros([g.num_nodes()], dtype=torch.bool)
    val_mask = torch.zeros([g.num_nodes()], dtype=torch.bool)
    test_mask = torch.zeros([g.num_nodes()], dtype=torch.bool)
    # shuffle indices of masks
    # indices = random.sample(range(g.num_nodes()), g.num_nodes())
    indices = list(range(g.num_nodes()))
    train_length = num_train # int(len(indices) * 0.3)
    val_length = train_length + int(len(indices) * 0.1)
    train_mask[indices[:train_length]] = True
    val_mask[indices[train_length:val_length]] = True
    test_mask[indices[val_length:]] = True
    return train_mask, val_mask, test_mask

def main(args):
    # load and preprocess dataset
    if args.dataset in ('cora', 'citeseer', 'pubmed'):
        data = citations_network(args)
        g = data[0]
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
    elif args.dataset == 'PUBMED':
        g = PubmedGraphDataset()[0]
        train_mask, val_mask, test_mask = create_masks(g, args.tsamples)
    elif args.dataset == 'coauthorp':
        g = CoauthorPhysicsDataset()[0]
        train_mask, val_mask, test_mask = create_masks(g, args.tsamples)
    elif args.dataset == 'amazon':
        g = AmazonCoBuyComputerDataset()[0]
        train_mask, val_mask, test_mask = create_masks(g, args.tsamples)
    elif args.dataset == 'ogbn-protein':
        dataset = DglNodePropPredDataset(name='ogbn-proteins')[0]
        g = dataset[0]
        g.update_all(fn.copy_e("feat", "feat_copy"), fn.sum("feat_copy", "feat"))
        g.ndata['label'] = dataset[1].sum(dim=1) # view(-1)
        train_mask, val_mask, test_mask = create_masks(g, args.tsamples)
    elif args.dataset == 'reddit':
        g = RedditDataset()[0]
        train_mask, val_mask, test_mask = create_masks(g, args.tsamples)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    features = g.ndata['feat'] # [train_mask] would make incompatible with matrix multiplication
    labels = g.ndata['label'] #
    in_feats = features.shape[1]
    n_classes = len(set(g.ndata['label'].tolist()))
    n_edges = g.num_edges()
    cuda = False
    
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              train_mask.int().sum().item(),
              val_mask.int().sum().item(),
              test_mask.int().sum().item()))

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    if args.gcn2:
        model = GCN2(g,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                F.relu,
                args.dropout)
    else:
        model = GCN(g,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                F.relu,
                args.dropout)

    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        # pdb.set_trace()
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, features, labels, val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             acc, n_edges / np.mean(dur) / 1000))

    print("Total Time:", sum(dur))
    test_start = time.time()
    acc = evaluate(model, features, labels, test_mask)
    test_end = time.time()
    print("Test time: {:.2} sec., Test accuracy: {:.2%}".format((test_end-test_start), acc))
    outp_file = open("out_time.txt", "a")
    if args.gcn2:
        outp_file.write(str(args.dataset) + " " + str(args.n_hidden) + " FusedMM " + str(sum(dur)) + " " + str(test_end-test_start) + "\n")
    else:
        outp_file.write(str(args.dataset) + " " + str(args.n_hidden) + " DGL " + str(sum(dur)) + " " + str(test_end-test_start) + "\n")
    outp_file.close()


if __name__ == '__main__':

    if True:
        torch.manual_seed(123)
        np.random.seed(123)
        random.seed(123)

    # this version contains both DGL and FusedMM implementation
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default="cora",
                        help="Dataset name ('cora', 'citeseer', 'pubmed').")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--tsamples", type=int, default=1000,
                        help="number of training samples")
    parser.add_argument("--n-hidden", type=int, default=64,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--gcn2", action='store_true',
                        help="FusedMM-GCN (default=False)")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    # print(args)
    main(args)
    
