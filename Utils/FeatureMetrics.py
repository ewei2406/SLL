import torch
import numpy as np

def categorical_accuracy(truth: torch.int, guess: torch.int) -> float:
    """
    Returns the accuracy of guesses as the percent correct
    """
    correct = (guess == truth).sum().item()
    acc = correct / guess.size(dim=0)
    return acc


def discretize(tensor_a: torch.tensor, n_bins=50, force_bins=False) -> torch.tensor:
    """
    Discretizes a tensor by the number of bins
    """
    if ((not tensor_a.is_floating_point()) and (not tensor_a.is_complex())) and not force_bins:
        return (tensor_a - tensor_a.min()).int()
    
    tensor_a_max = tensor_a.max().item()
    tensor_a_min = tensor_a.min().item()
    d = tensor_a_max - tensor_a_min

    if d == 0:
        return torch.zeros_like(tensor_a, device=tensor_a.device).int()

    boundaries = torch.arange(start=tensor_a_min, end=tensor_a_max, step = d / n_bins, device=tensor_a.device)
    bucketized = torch.bucketize(tensor_a, boundaries, right=True).to(tensor_a.device)
    result = bucketized - bucketized.min()
    assert result.shape[0] == tensor_a.shape[0]
    return result.int().to(tensor_a.device)


def dist(tensor_a: torch.tensor, n_bins=50, force_bins=False) -> torch.tensor:
    """
    Returns the distribution of frequencies of values in a discretized tensor
    """
    
    if tensor_a.nelement() == 0: 
        return torch.tensor([0], device=tensor_a.device)
    if ((not tensor_a.is_floating_point()) and (not tensor_a.is_complex())) and not force_bins:
        offset = tensor_a.min().item()
        dist = torch.zeros([tensor_a.max().item() - offset + 1], device=tensor_a.device)
        for f in tensor_a:
            dist[f - offset] += 1
    else:
        if force_bins:
            tensor_a = tensor_a.float()
        dist = torch.histc(tensor_a, bins=n_bins).to(tensor_a.device)
    
    return dist


def p_dist(tensor_a: torch.tensor, n_bins=50, force_bins=False) -> torch.tensor:
    """
    Returns the distribution of values in a tensor as a vector of probabilities
    If the tensor is not discrete, bin using n_bins
    """
    distribution = dist(tensor_a, n_bins=n_bins, force_bins=force_bins)
    return distribution / distribution.sum()


def joint_pdf(tensor_a: torch.tensor, tensor_b: torch.tensor, n_bins=50, force_bins=False) -> torch.tensor:
    """
    Returns a m*n tensor of joint probabilities for a and b
    """

    assert tensor_a.shape[0] == tensor_b.shape[0]
    assert tensor_a.device == tensor_b.device

    a_binned = discretize(tensor_a, n_bins=n_bins, force_bins=force_bins)
    b_binned = discretize(tensor_b, n_bins=n_bins, force_bins=force_bins)

    if a_binned.max().item() == 0:
        if b_binned.max().item() == 0:
            return torch.tensor([[0]], device=tensor_a.device)
        return p_dist(b_binned).unsqueeze(0)
    if b_binned.max().item() == 0:
        if a_binned.max().item() == 0:
            return torch.tensor([[0]], device=tensor_a.device)
        return p_dist(a_binned).unsqueeze(1)

    cumulative = torch.zeros((a_binned.max().item() + 1, b_binned.max().item() + 1), device=tensor_a.device)
    for i in range(a_binned.shape[0]):
        cumulative[a_binned[i]][b_binned[i]] += 1

    return cumulative / cumulative.sum()


def shannon_entropy(tensor_a: torch.tensor, n_bins=50) -> float:
    """
    Returns the Shannon (information) entropy of a tensor in bits (b=2)
    if tensor is an int tensor, n_bins is ignored
    """
    dist = p_dist(tensor_a)
    return ((dist * torch.log2(dist)).nan_to_num().sum() * -1).item()


def pearson_r(tensor_a: torch.tensor, tensor_b: torch.tensor) -> float:
    """
    Returns the pearson r correlation coefficient of two tensors
    """
    assert tensor_a.shape[0] == tensor_b.shape[0]
    assert tensor_a.device == tensor_b.device

    cat = torch.cat((tensor_a.unsqueeze(0), tensor_b.unsqueeze(0)), device=tensor_a.device)
    return torch.corrcoef(cat)[0][1].item()


def information_gain(tensor_a: torch.tensor, tensor_b: torch.tensor, discrete=True, n_bins=10) -> float:
    """
    Returns the information gain of a with respect to b
    IG = Entropy(parent) - M_Entropy(children)
    The children are split by integer value (if tensor is an int tensor) or by n_bins
    """

    assert tensor_a.shape[0] == tensor_b.shape[0]
    assert tensor_a.device == tensor_b.device

    parent_entropy = shannon_entropy(tensor_a)
    children_entropy = 0

    tensor_b_max = tensor_b.max().item()
    tensor_b_min = tensor_b.min().item()
    d = tensor_b_max - tensor_b_min

    if (not tensor_b.is_floating_point()) and (not tensor_b.is_complex()):
        for i in range(tensor_b_min, tensor_b_max + 1):
            split = tensor_b == i
            weight = split.sum().item() / tensor_b.shape[0]
            children_entropy += shannon_entropy(tensor_a[split]) * weight
    else:
        step_size = d / n_bins
        for i in np.arange(tensor_b_min, tensor_b_max + 1, step_size):
            split = (tensor_b - i).abs() < step_size / 2
            weight = split.sum().item() / n_bins
            children_entropy += shannon_entropy(tensor_a[split]) * weight
    
    return parent_entropy - children_entropy


def mutual_information(tensor_a: torch.tensor, tensor_b: torch.tensor) -> float:
    """
    Returns the mutual information between a and b
    """

    assert tensor_a.shape[0] == tensor_b.shape[0]
    assert tensor_a.device == tensor_b.device

    j_pdf = joint_pdf(tensor_a, tensor_b)
    cumulative = 0
    sum_X = torch.sum(j_pdf, 1)
    sum_Y = torch.sum(j_pdf, 0)
    log_pY = torch.log2(sum_Y)

    for idx_X in range(j_pdf.shape[0]):
        for idx_Y in range(j_pdf.shape[1]):
            p_xy = j_pdf[idx_X][idx_Y]
            cumulative += (p_xy * torch.log2(p_xy / (sum_X[idx_X] * sum_Y[idx_Y]))).nan_to_num().item()

    return cumulative


def chi_squared(tensor_a: torch.tensor, tensor_b: torch.tensor) -> float:
    """
    Returns the chi-sqaured statistic (WITHOUT CONTINUITY CORRECTION) of two variables
    """
    assert tensor_a.shape[0] == tensor_b.shape[0]
    assert tensor_a.device == tensor_b.device

    j_pdf = joint_pdf(tensor_a, tensor_b)
    # print(j_pdf)
    cumulative = 0
    sum_X = torch.sum(j_pdf, 0).to(tensor_a.device)
    sum_Y = torch.sum(j_pdf, 1).to(tensor_a.device)
    
    for i in range(j_pdf.shape[0]):
        E_X = sum_X * sum_Y[i]
        # print((((j_pdf[i] - E_X) ** 2) / E_X))
        cumulative += (((j_pdf[i] - E_X) ** 2) / E_X).nan_to_num().sum().item()
    
    return cumulative


def sample_by_quantiles(tensor_a: torch.tensor, tensor_b: torch.tensor, n_bins=4, n_samples=50) -> torch.tensor:
    """
    Returns a boolean index tensor of n samples distributed amongst tensor a and b in n_bins
    """
    assert tensor_a.shape[0] == tensor_b.shape[0]
    assert tensor_a.device == tensor_b.device

    dist = torch.zeros([2, tensor_a.shape[0]], device=tensor_a.device)

    a_range = torch.arange(tensor_a.min(), tensor_a.max() - 0.000001, (tensor_a.max() - tensor_a.min()) / (n_bins - 1), device=tensor_a.device)
    dist[0] = torch.bucketize(tensor_a, a_range)

    b_range = torch.arange(tensor_b.min(), tensor_b.max() - 0.000001, (tensor_b.max() - tensor_b.min()) / (n_bins - 1), device=tensor_a.device)
    dist[1] = torch.bucketize(tensor_b, b_range)

    prob = torch.zeros([dist.shape[1]], device=tensor_a.device)
    for i in range(n_bins):
        for j in range(n_bins):
            selected = (dist[0] == i) * (dist[0] == j)
            if selected.sum() == 0:
                prob[selected] = 0
            else:
                prob[selected] = 1 / selected.sum()

    prob *= n_samples / prob.sum()
    idx = torch.bernoulli(prob.clamp(0, 1)) == 1
    return idx.to(tensor_a.device)


if __name__ == "__main__":
    a = torch.tensor([1, 1, 2, 1, 2, 1, 2.])
    b = torch.tensor([1, 2, 1, 2, 2, 2, 2])

    z = chi_squared(a, b)
    print(z)



    # z = shannon_entropy(torch.tensor([1, 1, 2]))
    # print(z)