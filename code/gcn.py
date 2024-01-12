import time
import torch
import numpy as np
from tqdm import trange
import torch.nn.init as init
from torch.nn import Parameter
import torch.nn.functional as F
from utils import calculate_auc, setup_features
from convolution import GraphConvolutionalNetwork, AttentionNetwork
from dataset import get_snapshot_index


class TrustGuard(torch.nn.Module):
    def __init__(self, device, args, X, num_labels):
        super(TrustGuard, self).__init__()
        self.args = args
        torch.manual_seed(self.args.seed)  # fixed seed == 42
        self.device = device
        self.X = X
        self.dropout = self.args.dropout
        self.num_labels = num_labels
        self.build_model()
        self.regression_weights = Parameter(torch.Tensor(self.args.layers[-1]*2, self.num_labels))
        init.xavier_normal_(self.regression_weights)  # initialize regression_weights

    def build_model(self):
        """
        Constructing spatial and temporal layers.
        """
        self.structural_layer = GraphConvolutionalNetwork(self.device, self.args, self.X, self.num_labels)
        self.temporl_layer = AttentionNetwork(input_dim=self.args.layers[-1],n_heads=self.args.attention_head,num_time_slots=self.args.train_time_slots,attn_drop=0.5,residual=True)

    def calculate_loss_function(self, z, train_edges, target):
        """
        Calculating loss.
        :param z: Node embedding.
        :param train_edges: [2, #edges]
        :param target: Label vector storing 0 and 1.
        :return loss: Value of loss.
        """
        start_node, end_node = z[train_edges[0], :], z[train_edges[1], :]

        features = torch.cat((start_node, end_node), 1)
        predictions = torch.mm(features, self.regression_weights)

        # deal with imbalance data
        class_weight = torch.FloatTensor(1 / np.bincount(target.cpu()) * features.size(0))
        criterion = torch.nn.CrossEntropyLoss(weight=class_weight).to(self.device)
        loss_term = criterion(predictions, target)  # target与sorted_train_edges相对应

        return loss_term

    def forward(self, train_edges, y, y_train, index_list):
        structural_out = []
        index0 = 0
        for i in range(self.args.train_time_slots):
            structural_out.append(self.structural_layer(train_edges[:, index0:index_list[i]], y_train[index0:index_list[i], :]))
            index0 = index_list[i]

        structural_out = torch.stack(structural_out)
        structural_out = structural_out.permute(1,0,2)  # [N,T,F] [5881,7,32]
        temporal_all = self.temporl_layer(structural_out)  # [N,T,F]
        temporal_out = temporal_all[:, self.args.train_time_slots-1, :].squeeze()  # [N,F]

        loss = self.calculate_loss_function(temporal_out, train_edges, y)

        return loss, temporal_out


class GCNTrainer(object):
    """
    Object to train and score the TrustGuard, log the model behaviour and save the output.
    """
    def __init__(self, args, edges):
        """
        Constructing the trainer instance and setting up logs.
        :param args: Arguments object.
        :param edges: Edge data structure.
        """
        self.args = args
        self.edges = edges
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_start_time = time.time()
        self.setup_logs()

    def setup_logs(self):
        """
        Creating a log dictionary for recording performance.
        """
        self.logs = {}
        self.logs["parameters"] = vars(self.args)
        self.logs["performance"] = [["Epoch", "MCC", "AUC", "ACC_Balanced", "AP", "F1_Micro", "F1_Macro"]]
        self.logs["training_time"] = [["Epoch", "Seconds"]]

    def setup_dataset(self):
        """
        Creating training snapshots and testing snapshots.
        """
        self.index_list = get_snapshot_index(self.args.time_slots, data_path=self.args.data_path)
        index_t = self.index_list[self.args.train_time_slots-1]

        print("--------------- Getting training and testing snapshots starts ---------------")
        print('Snapshot index',self.index_list,index_t)  # index_t denotes the index of snapshot t
        self.train_edges = self.edges['edges'][:index_t]
        self.y_train = self.edges['labels'][:index_t]
        train_set = set(list(self.train_edges.flatten()))

        # single-timeslot prediction
        if self.args.single_prediction:
            # index_t_1 = self.index_list[self.args.train_time_slots-2]  # index of snapshot t-1, used for single-timeslot prediction on unobserved nodes, i.e., task 3
            # train_pre = set(list(self.train_edges[:index_t_1].flatten()))  # for task 3
            # train_t = set(list(self.train_edges[index_t_1:].flatten()))  # for task 3

            index_t1 = self.index_list[self.args.train_time_slots]  # index of snapshot t+1
            self.test_edges = self.edges['edges'][index_t:index_t1]
            self.y_test = self.edges['labels'][index_t:index_t1]
            print('{} edges at snapshot t+1'.format(len(self.test_edges)))

            self.obs = []  # observed nodes' edges
            self.y_test_obs = []
            # self.unobs = []  # for task 3
            # self.y_test_unobs = []  # for task 3
            for i in range(len(self.test_edges)):
                tr = self.test_edges[i][0]
                te = self.test_edges[i][1]
                if tr in train_set and te in train_set:
                    self.obs.append(self.test_edges[i])
                    self.y_test_obs.append(self.y_test[i])
                # for task 3
                # if tr in train_t and tr not in train_pre and te not in train_t and te in train_pre:
                #     self.unobs.append(self.test_edges[i])
                #     self.y_test_unobs.append(self.y_test[i])
                # elif te in train_t and te not in train_pre and tr not in train_t and tr in train_pre:
                #     self.unobs.append(self.test_edges[i])
                #     self.y_test_unobs.append(self.y_test[i])
                # elif tr in train_t and te in train_t and tr not in train_pre and te not in train_pre:
                #     self.unobs.append(self.test_edges[i])
                #     self.y_test_unobs.append(self.y_test[i])

            self.pos_count = 0
            self.neg_count = 0
            for i in range(len(self.y_test_obs)):
                if self.y_test_obs[i][0] == 1:
                    self.pos_count += 1
                else:
                    self.neg_count += 1
            print('Trust and distrust distribution:',self.pos_count,self.neg_count)
            # print('Observed single-timeslot test edges'.format(len(self.unobs)))  # for task 3
            print('Observed single-timeslot test edges:',len(self.obs))

            self.obs = np.array(self.obs)
            self.y_test_obs = np.array(self.y_test_obs)
            # self.unobs = np.array(self.unobs)  # for task 3
            # self.y_test_unobs = np.array(self.y_test_unobs)  # for task 3
        else:   # Multi-timeslot prediction, predict latter three snapshots, i.e., task 2
            index_pre = self.index_list[self.args.train_time_slots - 1]
            index_lat = self.index_list[self.args.train_time_slots + 2]
            self.test_edges = self.edges['edges'][index_pre:index_lat]
            self.y_test = self.edges['labels'][index_pre:index_lat]
            print('{} edges from snapshot t+1 to snapshot t+3'.format(len(self.test_edges)))

            self.obs = []  # observed nodes' edges
            self.y_test_obs = []

            for i in range(len(self.test_edges)):
                if self.test_edges[i][0] in train_set and self.test_edges[i][1] in train_set:
                    self.obs.append(self.test_edges[i])
                    self.y_test_obs.append(self.y_test[i])

            self.pos_count = 0
            self.neg_count = 0
            for i in range(len(self.y_test_obs)):
                if self.y_test_obs[i][0] == 1:
                    self.pos_count += 1
                else:
                    self.neg_count += 1
            print('Trust and distrust distribution:',self.pos_count,self.neg_count)
            print('Observed multi-timeslot test edges:',len(self.obs))

            self.obs = np.array(self.obs)
            self.y_test_obs = np.array(self.y_test_obs)
        print("--------------- Getting training and testing snapshots ends ---------------")

        self.X = setup_features(self.args)  # Setting up the node features as a numpy array.
        self.num_labels = np.shape(self.y_train)[1]

        self.y = torch.from_numpy(self.y_train[:,1]).type(torch.long).to(self.device)
        # convert vector to number 0/1, 0 represents trust and 1 represents distrust

        self.train_edges = torch.from_numpy(np.array(self.train_edges, dtype=np.int64).T).type(torch.long).to(self.device)  # (2, #edges)
        self.y_train = torch.from_numpy(np.array(self.y_train, dtype=np.float32)).type(torch.float).to(self.device)
        self.num_labels = torch.from_numpy(np.array(self.num_labels, dtype=np.int64)).type(torch.long).to(self.device)
        self.X = torch.from_numpy(self.X).to(self.device)

    def create_and_train_model(self):
        """
        Model training and scoring.
        """
        print("\nTraining started.\n")
        self.model = TrustGuard(self.device, self.args, self.X, self.num_labels).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate,weight_decay=self.args.weight_decay)
        self.model.train()
        epochs = trange(self.args.epochs, desc="Loss")
        for epoch in epochs:
            start_time = time.time()
            self.optimizer.zero_grad()
            loss, final_embedding = self.model(self.train_edges, self.y, self.y_train, self.index_list)
            loss.backward()
            epochs.set_description("TrustGuard (Loss=%g)" % round(loss.item(), 4))
            self.optimizer.step()
            self.score_model(epoch)
            self.logs["training_time"].append([epoch + 1, time.time() - start_time])
            self.score_model(epoch)

        self.logs["training_time"].append(["Total", time.time() - self.global_start_time])

    def score_model(self, epoch):
        """
        Score the model on the test set edges in each epoch.
        :param epoch: Epoch number.
        """
        self.model.eval()
        loss, self.train_z = self.model(self.train_edges, self.y, self.y_train, self.index_list)
        score_edges = torch.from_numpy(np.array(self.obs, dtype=np.int64).T).type(torch.long).to(self.device)
        test_z = torch.cat((self.train_z[score_edges[0, :], :], self.train_z[score_edges[1, :], :]), 1)
        # score_edges[0, :] is the index of trustors, while score_edges[1, :] is the index of trustees
        scores = torch.mm(test_z, self.model.regression_weights.to(self.device))

        predictions = F.softmax(scores, dim=1)

        mcc, auc, acc_balanced, precision, f1_micro, f1_macro = calculate_auc(predictions, self.y_test_obs)
        print('mcc, auc, acc_balanced, precision, f1_micro, f1_macro \n')
        print('%.4f' % mcc, '%.4f' % auc, '%.4f' % acc_balanced, '%.4f' % precision, '%.4f' % f1_micro, '%.4f' % f1_macro)

        self.logs["performance"].append([epoch + 1, mcc, auc, acc_balanced, precision, f1_micro, f1_macro])
