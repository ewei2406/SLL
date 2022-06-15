import Utils.Utils as Utils
import torch

class SamplingMatrix:
    def __init__(self, g0, gX, adj, sample_size=3):
        self.g0_ratio = torch.tensor(1)
        self.gX_ratio = torch.tensor(1)
        self.g0gX_ratio = torch.tensor(1)
        self.sample_size = sample_size

        self.g0 = g0
        self.g0_idx = Utils.bool_to_idx(g0).t()
        self.gX = gX
        self.gX_idx = Utils.bool_to_idx(gX).t()

        self.g0_sampling = torch.zeros_like(adj)
        self.g0_sampling.index_fill_(0, Utils.bool_to_idx(g0).squeeze(), 1)
        self.g0_sampling.index_fill_(1, Utils.bool_to_idx(gX).squeeze(), 0)
        self.g0_sampling.fill_diagonal_(0)

        self.gX_sampling = torch.zeros_like(adj)
        self.gX_sampling.index_fill_(0, Utils.bool_to_idx(gX).squeeze(), 1)
        self.gX_sampling.index_fill_(1, Utils.bool_to_idx(g0).squeeze(), 0)
        self.gX_sampling.fill_diagonal_(0)

        self.g0gX_sampling = torch.ones_like(adj)
        self.g0gX_sampling -= self.g0_sampling + self.gX_sampling
        self.g0gX_sampling.fill_diagonal_(0)

        self.updateSamplingMatrix()

    def getRatio(self):
        total = self.g0_ratio + self.gX_ratio + self.g0gX_ratio
        # print(f"G0:  {self.g0_ratio / total:.2f}")
        # print(f"GX:  {self.gX_ratio / total:.2f}")
        # print(f"G0GX:{self.g0gX_ratio / total:.2f}")
    
    def updateSamplingMatrix(self):
        total = self.g0_ratio + self.gX_ratio + self.g0gX_ratio
        constant_g0 =       ((self.g0_ratio * self.sample_size * 2) / (total * self.g0_sampling.sum()))
        constant_gX =       ((self.gX_ratio * self.sample_size * 2) / (total * self.gX_sampling.sum()))
        constant_g0gX =    ((self.g0gX_ratio * self.sample_size * 2) / (total * self.g0gX_sampling.sum()))

        self.sampling_matrix = \
            self.g0gX_sampling * constant_g0gX + \
            self.g0_sampling * constant_g0 + \
            self.gX_sampling * constant_gX

        self.sampling_matrix = torch.clamp(self.sampling_matrix, min=0, max=1)

        self.sampling_matrix.triu_(diagonal=1)
        # self.sampling_matrix.fill_diagonal_(0)

    def get_sample(self):
        return Utils.to_edges(torch.bernoulli(self.sampling_matrix))
    
    def updateRatio(self, g0_ratio, gX_ratio, g0gX_ratio):
        self.g0_ratio = g0_ratio
        self.gX_ratio = gX_ratio
        self.g0gX_ratio = g0gX_ratio
        total = self.g0_ratio + self.gX_ratio + self.g0gX_ratio
        self.g0_ratio /= total
        self.gX_ratio /= total
        self.g0gX_ratio /= total

        self.updateSamplingMatrix()

    def updateByGrad(self, adj_grad, count):
        min_sample = self.sample_size / 10

        g0r_count = (count * self.g0_sampling).sum() + min_sample
        gXr_count = (count * self.gX_sampling).sum() + min_sample
        g0gXr_count = count.sum() - (g0r_count + gXr_count) + min_sample

        abs_grad = adj_grad.abs()
        g0r = (abs_grad * self.g0_sampling).sum() / g0r_count
        gXr = (abs_grad * self.gX_sampling).sum() / gXr_count
        g0gXr = (abs_grad.sum() - (g0r + gXr)) / g0gXr_count

        # abs_grad = adj_grad.abs()
        # g0r = (abs_grad * self.g0_sampling)
        # g0r = torch.median(g0r[g0r.nonzero()])

        # gXr = (abs_grad * self.gX_sampling)
        # gXr = torch.median(gXr[gXr.nonzero()])

        # g0gXr = (abs_grad * self.g0gX_sampling)
        # g0gXr = torch.median(g0gXr[g0gXr.nonzero()])

        # print(g0r, gXr, g0gXr)
        # print(g0r_count, gXr_count, g0gXr_count)

        total = g0r + gXr + g0gXr
        g0r /= total
        gXr /= total
        g0gXr /= total

        self.updateRatio(
            g0_ratio=(self.g0_ratio + g0r) / 2, 
            gX_ratio=(self.gX_ratio + gXr) / 2, 
            g0gX_ratio=(self.g0gX_ratio + g0gXr) / 2
        )

if __name__ == "__main__":
    # mymatrix = SamplingMatrix(torch.tensor([True, True, False, False]), torch.tensor([False, False, True, True]), torch.tensor([
    #     [1, 1, 1, 1],
    #     [1, 1, 1, 1],
    #     [1, 1, 1, 1],
    #     [1, 1, 1, 1],
    # ]))

    # print(mymatrix.sampling_matrix)

    # a = mymatrix.get_sample()
    # print(a)


    import utils.GraphData as GraphData
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'==== Dataset: ====')

    graph = GraphData.getGraph("./datasets", "cora", "gcn", 123, device)
    graph.summarize()

    g0 = torch.rand(graph.features.shape[0]) <= 0.14
    # g0 = graph.labels == 5 

    gX = ~g0

    print(f"Number of protected nodes: {g0.sum():.0f}")
    print(f"Protected Size: {g0.sum() / graph.features.shape[0]:.2%}")

    test2 = SamplingMatrix(g0, gX, graph.adj, 500)
    print(test2.g0_idx)

    s = test2.get_sample()

    grad = torch.ones_like(graph.adj)
    count = torch.zeros_like(graph.adj)

    test2.updateByGrad(grad, count)
    print(g0.sum(), gX.sum(), graph.features.shape[0] - (g0.sum() + gX.sum()))