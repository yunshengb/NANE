import tensorflow as tf
from random import shuffle
import random, collections

flags = tf.app.flags
FLAGS = flags.FLAGS


def neg_sampling(input, batch, pos_labels, neg_labels, num_neg=5,
                 embed_dim=100):
    def generate_batch():
        return tf.reshape(tf.nn.embedding_lookup(input, batch),
                          shape=(-1, embed_dim, 1))

    def generate_samples():
        return tf.concat([tf.nn.embedding_lookup(input, pos_labels),
                          tf.nn.embedding_lookup(input, neg_labels)], 1)

    sims_col = num_neg + 1
    if FLAGS.need_higher:
        sims_col = 8

    sims = tf.reshape(tf.matmul(generate_samples(), generate_batch()), shape=(
        -1, sims_col))
    return sims


class NegSampler(object):
    def __init__(self, num_neg):
        self.ptr = 0
        self.num_neg = num_neg
        self.round = 0

    def init(self, N):
        self.li = list(range(N))
        if N % self.num_neg != 0:
            need = self.num_neg - N % self.num_neg
            self.li += list(range(need))
        assert (len(self.li) % self.num_neg == 0)

    def get_neg(self, pos_labels):
        cand_list = self.li[self.ptr:self.ptr + self.num_neg]
        s = (self.ptr + self.num_neg) % len(self.li)
        for i, cand in enumerate(cand_list):
            if cand in pos_labels:
                cand_list[i], s = self._find_next(pos_labels, s)
        return cand_list

    def increment(self):
        self.ptr = self.ptr + self.num_neg
        if self.ptr == len(self.li):
            self.ptr = 0
            shuffle(self.li)
            self.round += 1
        return self.round

    def _find_next(self, pos_labels, s):
        while True:
            cand = self.li[s]
            if cand not in pos_labels:
                return cand, (s + 1) % len(self.li)
            s = (s + 1) % len(self.li)


class DynamicSampler(object):
    class State(object):
        def __init__(self):
            self.which_list = 1 # start with using 1st order neighbors
            self.li_idx = 0
            self.take_idx = 0


    def __init__(self):
        self.higher_neigh_ids = collections.defaultdict(list)
        self.higher_neigh_orders = collections.defaultdict(list)
        self.K = 2 # number of samples to return
        import numpy as np
        self.GR = 20 # growth rate
        self.order_limit = 3

    def init(self, neighbor_map):
        self.neighbor_map = neighbor_map
        self.states = {}
        print('DynamicSampler initializing states')
        for i, ns in neighbor_map.items():
            self.states[i] = DynamicSampler.State()
        print('DynamicSampler initialized states')

    def grow(self, i):
        for _ in range(self.GR): # try to add self.GR neighbors
            try:
                (take, order) = self._explore(i)
                #if take != i and take not in self.neighbor_map[i] and take
                # not \
                #        in self.higher_neigh_ids[i]:
                self.higher_neigh_ids[i].append(take)
                self.higher_neigh_orders[i].append(order)
            except RuntimeError as e:
                #print(e)
                pass

    def _explore(self, i):
        s = self.states[i]
        base_li, higher_list, order = self._get_item_lists(s, i)
        if order > self.order_limit:
            raise RuntimeError('Only 2')
        take = higher_list[s.take_idx] # actual id of the higher order neighbor
        s.take_idx += 1
        if s.take_idx == len(higher_list):
            # Move on to the next entry in base_li
            s.li_idx += 1
            s.take_idx = 0
            if s.li_idx == len(base_li):
                # Move on to the higher order neighbor map
                if s.which_list == 1:
                    s.which_list = 2
                    s.li_idx = 0
                else:
                    s.which_list = -1 # run out of higher order neighbors
        return (take, order)


    def _get_item_lists(self, s, i):
        if s.which_list == -1:
            raise RuntimeError('Node {} runs out of higher-order neighbors'.format(i))
        elif s.which_list == 1:
            base_li = self.neighbor_map[i] # looking for a 2nd order neighbor
            order = 2
        else:
            base_li = self.higher_neigh_ids[i]
            if not base_li:
                raise RuntimeError(
                    'Node {} has empty higher_neigh_ids'.format(i))
            order = self.higher_neigh_orders[i][s.li_idx] + 1
        return base_li, self.neighbor_map[base_li[s.li_idx]], order


    def get_higher_order_neighbors(self, i):
        if FLAGS.fs:
            ns = self.neighbor_map[i]
            return [random.choice(self.neighbor_map[random.choice(ns)]),
                    random.choice(self.neighbor_map[random.choice(ns)])], \
                   [0.5, 0.25, 0.25]
        if not self.higher_neigh_ids[i]: # no higher order neighbors
            return [int(len(self.neighbor_map)*random.uniform(0, 1)) \
                    for _ in range(self.K)], [1]+[0 for _ in range(self.K)]
        idxs = [random.randint(0, len(self.higher_neigh_ids[i])-1) for _ in
                range(self.K)]
        ids = []
        tot_len = 1
        for idx in idxs:
            ids.append(self.higher_neigh_ids[i][idx])
            tot_len += 1/self.higher_neigh_orders[i][idx] # the higher the
            # order, the less the weight
        weights = [1/(tot_len)]
        for idx in idxs:
            weights.append(1/self.higher_neigh_orders[i][idx]/tot_len)
        return ids, weights


if __name__ == '__main__':
    # test()
    # n = NegSampler(5)
    # n.init(7)
    # for i in range(4):
    #     for j in range(7):
    #         print(i, n.get_neg([0, 1, 2, 3]))
    #     n.increment()

    d = DynamicSampler()
    neighbor_map = {0: [1, 2, 4], 1: [0], 2: [0], 3: [4], 4: [3, 0]}
    d.init(neighbor_map)
    for i, ns in neighbor_map.items():
        d.grow(i)
        d.grow(i)
        d.grow(i)
    for i in neighbor_map.keys():
        ids, weights = d.get_higher_order_neighbors(i)
        print('i:{} ids:{} weights:{}'.format(i, ids, weights))

