import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import random
import math
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CitationFull
import pickle
import numpy as np
import scipy.sparse as sp


def rand_int(lower, upper):
    return random.randint(lower, upper)


def shuffle_list(target):
    return random.shuffle(target)


dataset = "dblp"


class LPLoader:
    @staticmethod
    def load():
        if dataset == "dblp":
            return LPLoader.load_dblp()

        return None

    @staticmethod
    def load_dblp():
        # dataset = CitationFull('data/graphs/', 'DBLP', transform=T.NormalizeFeatures())
        # data = dataset[0]
        # x = data.x.cpu().detach().numpy()
        # edge_index = data.edge_index.cpu().detach().numpy()
        # y = data.y.cpu().detach().numpy()

        # with open("dblp.pkl", 'wb') as file:
        #     pickle.dump([x, edge_index, y], file)

        with open("./../datasets/dblp/dblp.pkl", 'rb') as file:
            x, edge_index, y = pickle.load(file)

        graph = dict()
        for i in range(x.shape[0]):
            graph[i] = dict()
            graph[i]['edges'] = dict()
            graph[i]['feature'] = None
        for i in range(edge_index.shape[1]):
            from_node = edge_index[0][i]
            to_node = edge_index[1][i]
            graph[from_node]['edges'][to_node] = 1
        
        features = x
        labels = y
        edge_index = edge_index
        n_node = len(features)
        adj_orig = np.zeros([n_node, n_node])
        for i in range(len(edge_index[0])):
            from_node = edge_index[0][i]
            to_node = edge_index[1][i]
            adj_orig[from_node][to_node] = 1
        adj_orig = sp.csr_matrix(adj_orig)
        a = list(range(n_node))
        train_ratio = 0.4
        val_ratio = 0.1
        tvt_nids = [
            a[:int(n_node * train_ratio)], 
            a[int(n_node * train_ratio):int(n_node * (1 - val_ratio - train_ratio))], 
            a[int(n_node * (1 - val_ratio - train_ratio)):]
        ]

        return tvt_nids, adj_orig, features, labels, graph


class LPEval():
    @staticmethod
    def eval(graph, emb, multi_class=False, use_shuffle=False, neg_pos_ratio=1, split_ratio=0.5):
        # feeder
        positive_sample = []
        negative_sample = []

        for node_id in graph:
            for edge in graph[node_id]["edges"]:
                if edge in graph:
                    positive_sample.append((node_id, edge, graph[node_id]["edges"][edge]))

        if not multi_class:
            node_num = len(graph)
            target_neg_num = math.ceil(len(positive_sample) * neg_pos_ratio)
            if target_neg_num < 1:
                target_neg_num = 1
            while len(negative_sample) < target_neg_num:
                rand_from_node = rand_int(0, node_num - 1)
                rand_to_node = rand_int(0, node_num - 1)
                from_node = list(graph.keys())[rand_from_node]
                to_node = list(graph.keys())[rand_to_node]
                if (from_node, to_node) not in positive_sample:
                    negative_sample.append((from_node, to_node, 0))

        # shuffle train and test
        if use_shuffle:
            shuffle_list(positive_sample)
            shuffle_list(negative_sample)

        train_pos = positive_sample[:int(len(positive_sample) * split_ratio)]
        test_pos = positive_sample[int(len(positive_sample) * split_ratio):]

        train_neg = negative_sample[:int(len(negative_sample) * split_ratio)]
        test_neg = negative_sample[int(len(negative_sample) * split_ratio):]

        train_edges, train_edges_false, test_edges, test_edges_false = train_pos, train_neg, test_pos, test_neg
        
        # compute embeddings
        emb_mappings = dict()
        for i in range(len(emb)):
            emb_mappings[i] = emb[i]

        # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
        def get_edge_embeddings(edge_list):
            embs = []
            for edge in edge_list:
                node1 = edge[0]
                node2 = edge[1]
                emb1 = emb_mappings[node1]
                emb2 = emb_mappings[node2]
                edge_emb = np.multiply(emb1, emb2)
                embs.append(edge_emb)
            embs = np.array(embs)
            return embs

        # Train-set edge embeddings
        pos_train_edge_embs = get_edge_embeddings(train_edges)
        neg_train_edge_embs = get_edge_embeddings(train_edges_false)
        if multi_class:
            train_edge_embs = pos_train_edge_embs
            train_edge_labels = np.array([e[2] for e in train_edges])
        else:
            train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])
            train_edge_labels = np.array([e[2] for e in (train_edges + train_edges_false)])

        # Test-set edge embeddings, labels
        pos_test_edge_embs = get_edge_embeddings(test_edges)
        neg_test_edge_embs = get_edge_embeddings(test_edges_false)
        if multi_class:
            test_edge_embs = pos_test_edge_embs
            test_edge_labels = np.array([e[2] for e in test_edges])
        else:
            test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])
            test_edge_labels = np.array([e[2] for e in (test_edges + test_edges_false)])

        # Train logistic regression classifier on train-set edge embeddings
        if multi_class:
            edge_classifier = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
        else:
            edge_classifier = LogisticRegression(random_state=0)
        edge_classifier.fit(np.array(train_edge_embs), np.array(train_edge_labels))

        # Predicted edge scores: probability of being of class "1" (real edge)
        test_preds = edge_classifier.predict(test_edge_embs)

        # record result
        predicted = list()
        ground_truth = list()
        for i in range(len(test_edge_labels)):
            # print("--- {} - {} ---".format(test_preds[i], test_edge_labels[i]))
            predicted.append(test_preds[i])
            ground_truth.append(test_edge_labels[i])

        if len(predicted) == 0:
            print("predicted value is empty.")
            return

        if multi_class:
            # accuracy
            accuracy = accuracy_score(ground_truth, predicted)

            labels = set()
            for e in ground_truth:
                labels.add(e)

            # Micro-F1
            micro_f1 = f1_score(ground_truth, predicted, labels=list(labels), average="micro")

            # Macro-F1
            macro_f1 = f1_score(ground_truth, predicted, labels=list(labels), average="macro")

            print("Acc: {:.4f} Micro-F1: {:.4f} Macro-F1: {:.4f}".format(accuracy, micro_f1, macro_f1))
        else:
            # auc
            auc = roc_auc_score(ground_truth, predicted)

            # accuracy
            accuracy = accuracy_score(ground_truth, predicted)

            # recall
            recall = recall_score(ground_truth, predicted)

            # precision
            precision = precision_score(ground_truth, predicted)

            # F1
            f1 = f1_score(ground_truth, predicted)

            print("Acc: {:.4f} AUC: {:.4f} Pr: {:.4f} Re: {:.4f} F1: {:.4f}".format(accuracy, auc, precision, recall, f1))
