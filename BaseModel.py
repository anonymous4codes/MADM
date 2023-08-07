import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random 
import scipy.sparse as sp


class GraphRec(nn.Module):
    def __init__(self, n_users, n_items, hidden, hop, drop):
        super().__init__()
        self.hop = 1
        self.hidden = hidden
        self.social_weight_dict = self.init_weight()
        self.item_weight_dict = self.init_weight()
        self.dropout = drop
        self.tanh = nn.Tanh()
        self.n_users = n_users
        self.n_items = n_items
        
    def init_weight(self):
        initializer = nn.init.xavier_uniform_
        weight_dict = nn.ParameterDict()
        for k in range(self.hop):
            weight_dict.update({'W_%d'%k: nn.Parameter(initializer(torch.empty(self.hidden,
                                                                      self.hidden)))})
        return weight_dict
    
    def GCN_a(self, users_emb, items_emb, adj):
        num_users = users_emb.size()[0]
        num_items = items_emb.size()[0]
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        for k in range(self.hop):
            all_emb = torch.sparse.mm(adj, all_emb) # sparse x sparse -> sparse sparse x dense -> dense
            all_emb = self.tanh(torch.matmul(all_emb, self.item_weight_dict['W_%d' %k]))
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [num_users, num_items])
        return users, items
    
    def GCN_s(self, user_embs, adj):
        adj = F.dropout(adj, p=self.dropout, training=self.training) #
        for k in range(self.hop):
            new_user_embs = torch.matmul(adj, user_embs)
            user_embs = self.tanh(torch.matmul(new_user_embs, self.social_weight_dict['W_%d' %k])) + user_embs
        return user_embs
    
    def forward(self, users, pos, neg, user_embs, item_embs, S, A):
        all_user_embs_S = self.GCN_s(user_embs, S) 
        all_user_embs_A, all_item_embs = self.GCN_a(user_embs, item_embs, A) 
        all_user_embs = 0.5*all_user_embs_S + 0.5*all_user_embs_A # twinGCN
        
        users_emb = all_user_embs[users]
        pos_emb = all_item_embs[pos]
        neg_emb = all_item_embs[neg] # 
        return users_emb, pos_emb, neg_emb

    def rec_loss(self, user_emb, pos_emb, neg_emb):
        pos_scores = torch.sum(user_emb*pos_emb, dim=1)  # [B]
        neg_scores = torch.sum(user_emb*neg_emb, dim=1)  # [B]
        sup_logits = pos_scores-neg_scores
        bpr_loss = -torch.mean(F.logsigmoid(sup_logits))
        return bpr_loss

    def l2_loss(self, *weights):
        loss = 0.0
        for w in weights:
            loss += torch.mean(torch.pow(w, 2))

        return 0.5*loss
    
    def reg_loss(self, users, pos, neg, user_embs, item_embs):
        reg_loss = self.l2_loss(
            user_embs[users],
            item_embs[pos],
            item_embs[neg],
        )
        return reg_loss

class DiffNet_plus(nn.Module):
    def __init__(self, n_users, n_items, hidden, hop, drop):
        super().__init__()
        self.hop = hop
        self.hidden = hidden
        self.weight_dict = self.init_weight()
        self.dropout = drop
        self.tanh = nn.Tanh()
        self.n_users = n_users
        self.n_items = n_items
        
    def init_weight(self):
        initializer = nn.init.xavier_uniform_
        weight_dict = nn.ParameterDict()
        for k in range(self.hop):
            weight_dict.update({'W_%d'%k: nn.Parameter(initializer(torch.empty(self.hidden*2,
                                                                      self.hidden)))})
        return weight_dict
    
    def GCN_a(self, users_emb, items_emb, adj):
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        for _ in range(1):
            all_emb = torch.sparse.mm(adj, all_emb) # sparse x sparse -> sparse sparse x dense -> dense
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_items])
        return users, items
    
    def GCN_s(self, user_embs, adj):
        for k in range(self.hop):
            new_user_embs = torch.matmul(adj, user_embs)
            user_embs = torch.matmul(torch.cat([new_user_embs, user_embs], dim=1), self.weight_dict['W_%d' %k])
            user_embs = self.tanh(user_embs)
            if k < self.hop-1:
                user_embs = F.dropout(user_embs, p=self.dropout, training=self.training)
        return user_embs
    
    def forward(self, users, pos, neg, user_embs, item_embs, S, A):
        all_user_embs_S = self.GCN_s(user_embs, S) 
        all_user_embs_A, all_item_embs = self.GCN_a(user_embs, item_embs, A) 
        all_user_embs = 0.5*all_user_embs_S + 0.5*all_user_embs_A # 
        
        users_emb = all_user_embs[users]
        pos_emb = all_item_embs[pos]
        neg_emb = all_item_embs[neg] 
        return users_emb, pos_emb, neg_emb

    def rec_loss(self, user_emb, pos_emb, neg_emb):
        pos_scores = torch.sum(user_emb*pos_emb, dim=1)  # [B]
        neg_scores = torch.sum(user_emb*neg_emb, dim=1)  # [B]
        sup_logits = pos_scores-neg_scores
        bpr_loss = -torch.mean(F.logsigmoid(sup_logits))
        return bpr_loss

    def l2_loss(self, *weights):
        loss = 0.0
        for w in weights:
            loss += torch.mean(torch.pow(w, 2))

        return 0.5*loss
    
    def reg_loss(self, users, pos, neg, user_embs, item_embs):
        reg_loss = self.l2_loss(
            user_embs[users],
            item_embs[pos],
            item_embs[neg],
        )
        return reg_loss


class SEPT(nn.Module):
    
    def __init__(self, config, args, device):
        super(SEPT, self).__init__()
        self.device = device
        self.n_users = config['n_users']
        self.n_items = config['n_items']
        
        # Graph
        S = config['S'] # normalized S np
        RRT = config['RRT_tr'] # normalized RRT np
        social_mat, sharing_mat = self.get_social_related_views(S, RRT)
        self.social_mat, self.sharing_mat = self.convert_numpy_to_tensor(social_mat), self.convert_numpy_to_tensor(sharing_mat)
        A = config['A_tr']
        self.A = self.sparse_mx_to_torch_sparse_tensor(A)
        self.R = config['R_tr'] # un-normalized R sp
        
        # training hyper-parameter
        self.hidden = args.hidden
        self.neg = args.neg
        self.std = args.std 
        self.hop = args.hop
        self.drop = args.dropout
        self.decay = args.decay
        self.instance_cnt = args.instance_cnt
        self.ssl_reg = args.ssl_reg
        self.ssl_drop = args.ssl_drop

        # layer
        self.user_embs = nn.Embedding(self.n_users, self.hidden) 
        self.item_embs = nn.Embedding(self.n_items, self.hidden)
        nn.init.xavier_uniform_(self.user_embs.weight)
        nn.init.xavier_uniform_(self.item_embs.weight)
        # xavier_uniform nn.init.xavier_uniform_
        # self.GCN_S = LightGCN(hop=self.hop, drop=self.drop)
        # self.GCN_A = LightGCN_sp(hop=self.hop, drop=self.drop)
    
    def convert_numpy_to_tensor(self, adj):
        adj = torch.FloatTensor(adj).to(self.device)
        return adj
    
    def GCN_a(self, users_emb, items_emb, adj):
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        for _ in range(self.hop):
            all_emb = torch.sparse.mm(adj, all_emb) # sparse x sparse -> sparse sparse x dense -> dense
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_items])
        return users, items
    
    def GCN_s(self, users_emb, adj):
        embs = [users_emb]
        for _ in range(self.hop):
            if adj.is_sparse:
                users_emb = torch.sparse.mm(adj, users_emb) # sparse x sparse -> sparse sparse x dense -> dense
            else:
                users_emb = torch.matmul(adj, users_emb)
            embs.append(users_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        return light_out

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        Graph = torch.sparse.FloatTensor(indices, values, shape)
        Graph = Graph.coalesce().to(self.device)
        return Graph
    
    def label_prediction(self, emb, users):
        emb = emb[users]
        emb = F.normalize(emb, p=2, dim=-1)
        aug_emb = self.aug_user_embs[users]
        aug_emb = F.normalize(aug_emb, p=2, dim=-1)
        prob = torch.matmul(emb, aug_emb.t())
        prob = F.softmax(prob, dim=-1)
        return prob
        
    
    def generate_pesudo_labels(self, prob1, prob2):
        positive = (prob1+prob2)/2
        pos_example = torch.topk(positive, self.instance_cnt)[1] # [B,K]
        return pos_example
    
    def neighbor_discrimination(self, positive, emb, users):
        emb = emb[users]
        emb = F.normalize(emb, p=2, dim=-1)
        aug_emb = self.aug_user_embs[users]
        aug_emb = F.normalize(aug_emb, p=2, dim=-1) # [B,H]
        pos_emb = aug_emb[positive] # [B,K,H]
        
        emb2 = emb.unsqueeze(1) # [B,1,H]
        # emb2 = torch.repeat(emb2, [1, self.instance_cnt, 1]) # [B, K, H]
        pos_score = torch.sum(pos_emb*emb2, dim=2) # [B,K]
        ttl_score = torch.matmul(emb, aug_emb.t()) # [B,B]
        pos_score = torch.sum(torch.exp(pos_score / 0.1), dim=1) # [B]
        ttl_score = torch.sum(torch.exp(ttl_score / 0.1), dim=1) # [B]
        ssl_score = -torch.sum(torch.log(pos_score/ttl_score))
        
        return ssl_score
    
    def get_social_related_views(self, S, RRT):
        social_mat = np.dot(S, S)
        social_mat = social_mat*S + np.eye(self.n_users, dtype=np.float32)
        # social_mat = normalize_dense(social_mat)
        
        sharing_mat = RRT*S + np.eye(self.n_users, dtype=np.float32)
        # sharing_mat = normalize_dense(sharing_mat)
        
        return social_mat, sharing_mat 
        
    
    def forward(self, users, pos, neg):
        """
        users:[B] pos:[B] neg:[B, neg]
        """     
        self.rec_user_embs, self.rec_item_embs = self.GCN_a(self.user_embs.weight, self.item_embs.weight, self.A) 
        self.sharing_view_embs = self.GCN_s(self.user_embs.weight, self.sharing_mat)
        self.friend_view_embs = self.GCN_s(self.user_embs.weight, self.social_mat)
        
        users_emb = self.rec_user_embs[users]
        pos_emb = self.rec_item_embs[pos]
        neg_emb = self.rec_item_embs[neg] #
        
        users_emb_ego = self.user_embs(users)
        pos_emb_ego = self.item_embs(pos)
        neg_emb_ego = self.item_embs(neg)
        
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def calculate_bpr_loss(self, users, pos, neg):
        """
        Only this function appears in train()
        """
        (users_emb, pos_emb, neg_emb,
        userEmb0,  posEmb0, negEmb0) = self.forward(users.long(), pos.long(), neg.long())
        # users_emb:[B, d]  pos_emb:[B,d] neg_emb:[B, neg, d]
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb) # [B,d]*[B,d]=[B,d]
        pos_scores = torch.sum(pos_scores, dim=1) # [B]
        users_emb = users_emb.unsqueeze(1) # [B,1,d]
        neg_scores = torch.mul(users_emb, neg_emb) # [B,1,d]*[B,neg,d]=[B, neg, d]
        neg_scores = torch.sum(torch.sum(neg_scores, dim=-1), dim=-1) # [B]
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        loss = loss + self.decay*reg_loss
        
        return loss
    
    def edge_dropout(self, sp_adj):
        """Input: a sparse user-item adjacency matrix and a dropout rate."""
        adj_shape = sp_adj.get_shape()
        edge_count = sp_adj.count_nonzero()
        row_idx, col_idx = sp_adj.nonzero()
        keep_idx = random.sample(range(edge_count), int(edge_count * (1 - self.ssl_drop)))
        user_np = np.array(row_idx)[keep_idx]
        item_np = np.array(col_idx)[keep_idx]
        edges = np.ones_like(user_np, dtype=np.float32)
        dropped_adj = sp.csr_matrix((edges, (user_np, item_np)), shape=adj_shape)
        return dropped_adj
    
    def convert_to_normalized_lap(self, adj_mat):
        adj_shape = adj_mat.get_shape()
        n_nodes = adj_shape[0]+adj_shape[1]
        (user_np_keep, item_np_keep) = adj_mat.nonzero()
        ratings_keep = adj_mat.data
        tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + adj_shape[0])),shape=(n_nodes, n_nodes),dtype=np.float32)
        tmp_adj = tmp_adj + tmp_adj.T
        # tmp_adj = normalize_sp(tmp_adj)
        
        return tmp_adj.tocsr()
   
    def calculate_contrastive_loss(self, users):
        """
        Only this function appears in train()
        """
        R_aug = self.edge_dropout(self.R)
        A_aug = self.convert_to_normalized_lap(R_aug)
        self.A_aug = self.sparse_mx_to_torch_sparse_tensor(A_aug)
        self.aug_user_embs, self.aug_item_embs = self.GCN_A(self.user_embs.weight, self.item_embs.weight, self.A_aug)
        
        social_prediction = self.label_prediction(self.friend_view_embs, users) # [B,B]
        sharing_prediction = self.label_prediction(self.sharing_view_embs, users)
        rec_prediction = self.label_prediction(self.rec_user_embs, users)
        
        self.f_pos = self.generate_pesudo_labels(sharing_prediction, rec_prediction) # [B,K]
        self.sh_pos = self.generate_pesudo_labels(social_prediction, rec_prediction)
        self.r_pos = self.generate_pesudo_labels(social_prediction, sharing_prediction)
        
        neighbor_dis_loss = self.neighbor_discrimination(self.f_pos, self.friend_view_embs, users) + \
            self.neighbor_discrimination(self.sh_pos, self.sharing_view_embs, users) + \
                self.neighbor_discrimination(self.r_pos, self.rec_user_embs, users)
                
        neighbor_dis_loss = self.ssl_reg*neighbor_dis_loss
        
        return neighbor_dis_loss
    

class DESIGN(nn.Module):
    def __init__(self, n_users, n_items, hidden, hop, drop):
        super().__init__()
        self.hop = hop
        self.hidden = hidden
        self.n_users = n_users
        self.n_items = n_items
        self.user1_embs = nn.Embedding(self.n_users, self.hidden)
        self.item1_embs = nn.Embedding(self.n_items, self.hidden)
        self.user2_embs = nn.Embedding(self.n_users, self.hidden)
        self.item2_embs = nn.Embedding(self.n_items, self.hidden)
        nn.init.xavier_uniform_(self.user1_embs.weight)
        nn.init.xavier_uniform_(self.item1_embs.weight)
        nn.init.xavier_uniform_(self.user2_embs.weight)
        nn.init.xavier_uniform_(self.item2_embs.weight)
        self.sigmoid = nn.Sigmoid()
        
    def GCN_a(self, users_emb, items_emb, adj):
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        for _ in range(self.hop):
            all_emb = torch.sparse.mm(adj, all_emb) # sparse x sparse -> sparse sparse x dense -> dense
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_items])
        return users, items
    
    def GCN_s(self, users_emb, adj):
        embs = [users_emb]
        for _ in range(self.hop):
            if adj.is_sparse:
                users_emb = torch.sparse.mm(adj, users_emb) # sparse x sparse -> sparse sparse x dense -> dense
            else:
                users_emb = torch.matmul(adj, users_emb)
            embs.append(users_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        return light_out
    
    def forward(self, users, pos, neg, user_embs, item_embs, S, A):
        all_user_embs_S = self.GCN_s(user_embs, S) 
        all_user_embs_A, all_item_embs = self.GCN_a(user_embs, item_embs, A) 
        all_user_embs = 0.5*all_user_embs_S + 0.5*all_user_embs_A # twinGCN
        
        all_user_embs_social = self.GCN_s(self.user1_embs.weight, S) # socialGCN
        all_item_embs_social = self.item1_embs.weight
        
        all_user_embs_rating, all_item_embs_rating = self.GCN_a(self.user2_embs.weight, self.item2_embs.weight, A) # ratingGCN
        
        users_emb = all_user_embs[users]
        pos_emb = all_item_embs[pos]
        neg_emb = all_item_embs[neg] 
        
        users_emb_social = all_user_embs_social[users]
        pos_emb_social = all_item_embs_social[pos]
        neg_emb_social = all_item_embs_social[neg]

        users_emb_rating = all_user_embs_rating[users]
        pos_emb_rating = all_item_embs_rating[pos]
        neg_emb_rating = all_item_embs_rating[neg]
        
        return users_emb, pos_emb, neg_emb, users_emb_social, pos_emb_social, neg_emb_social, users_emb_rating, pos_emb_rating, neg_emb_rating

    def rec_loss(self, user_emb, pos_emb, neg_emb):
        pos_scores = torch.sum(user_emb*pos_emb, dim=1)  # [B]
        neg_scores = torch.sum(user_emb*neg_emb, dim=1)  # [B]
        sup_logits = pos_scores-neg_scores
        bpr_loss = -torch.mean(F.logsigmoid(sup_logits))
        return bpr_loss

    def l2_loss(self, *weights):
        loss = 0.0
        for w in weights:
            loss += torch.mean(torch.pow(w, 2))

        return 0.5*loss
    
    def reg_loss(self, users, pos, neg, user_embs, item_embs):
        reg_loss = self.l2_loss(
            user_embs[users],
            item_embs[pos],
            item_embs[neg],
        )
        reg_loss_social = self.l2_loss(
            self.user1_embs(users),
            self.item1_embs(pos),
            self.item1_embs(neg),
        )
        reg_loss_rating = self.l2_loss(
            self.user2_embs(users),
            self.item2_embs(pos),
            self.item2_embs(neg),
        )
        return reg_loss+reg_loss_social+reg_loss_rating

    def compute_distill_loss(self, pre_a, pre_b):
        pre_a = self.sigmoid(pre_a)
        pre_b = self.sigmoid(pre_b)
        distill_loss = - torch.mean(pre_b * torch.log(pre_a) + (1 - pre_b) * torch.log(1 - pre_a))
        return distill_loss   
    
    def KL_loss(self, users_emb, pos_emb, neg_emb, users_emb_social, pos_emb_social, neg_emb_social, users_emb_rating, pos_emb_rating, neg_emb_rating):

        pos_emb = pos_emb.unsqueeze(1) # [B,1,H]
        neg_emb = neg_emb.unsqueeze(1) # [B,1,H]
        all_item_embs = torch.cat([pos_emb, neg_emb], dim=1) # [B,2,H] 
        users_emb = users_emb.unsqueeze(1) # [B,1,H]
        pre = torch.mul(users_emb, all_item_embs) # [B,1,H]*[B,2,H]=[B,2,H]
        pre = torch.mean(pre, dim=-1) # [B,2]
        
        pos_emb_social = pos_emb_social.unsqueeze(1)
        neg_emb_social = neg_emb_social.unsqueeze(1)
        all_item_embs_social = torch.cat([pos_emb_social, neg_emb_social], dim=1) 
        users_emb_social = users_emb_social.unsqueeze(1)
        pre_social = torch.mul(users_emb_social, all_item_embs_social) # [B,1,H]*[B,2,H]=[B,2,H]
        pre_social = torch.mean(pre_social, dim=-1) # [B,2]
        
        pos_emb_rating = pos_emb_rating.unsqueeze(1)
        neg_emb_rating = neg_emb_rating.unsqueeze(1)
        all_item_embs_rating = torch.cat([pos_emb_rating, neg_emb_rating], dim=1) 
        users_emb_rating = users_emb_rating.unsqueeze(1)
        pre_rating = torch.mul(users_emb_rating, all_item_embs_rating) # [B,1,H]*[B,2,H]=[B,2,H]
        pre_rating = torch.mean(pre_rating, dim=-1) # [B,2]
        kl_loss = 0
        kl_loss += self.compute_distill_loss(pre, pre_social)
        kl_loss += self.compute_distill_loss(pre, pre_rating)
        kl_loss += self.compute_distill_loss(pre_social, pre)
        kl_loss += self.compute_distill_loss(pre_social, pre_rating)
        kl_loss += self.compute_distill_loss(pre_rating, pre)
        kl_loss += self.compute_distill_loss(pre_rating, pre_social)
        return kl_loss