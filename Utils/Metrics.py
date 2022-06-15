import torch
from . import Utils
import numpy as np
import random

def acc(predictions, labels):
    correct = (predictions.argmax(1) == labels).sum()
    acc = correct / predictions.size(dim=0)
    return acc.item()

def calc_acc(model, features, adj, labels, idx=False):
    if not idx:
        idx = torch.ones_like(labels) > 0

    pred = model(features, adj)

    correct = (pred.argmax(1)[idx] == labels[idx]).sum()
    acc = correct / idx.sum()
    return acc.item()

def partial_acc(predictions, labels, g0, g_g0, verbose=True):
    g0_acc = acc(predictions[g0], labels[g0])
    gX_acc = acc(predictions[g_g0], labels[g_g0])

    if verbose:
        print(f"G0: {g0_acc:.2%}")
        print(f"GX: {gX_acc:.2%}")

    return {
        "g0": g0_acc,
        "gX": gX_acc
    }

def mask_adj(adj, bool_list, device):
    idx = Utils.bool_to_idx(bool_list).squeeze().to(device)

    temp_adj = adj.clone().to(device)
    temp_adj.index_fill_(dim=0, index=idx, value=0)
    diff = adj - temp_adj

    temp_adj = diff.clone().to(device)
    temp_adj.index_fill_(dim=1, index=idx, value=0)
    diff = diff - temp_adj

    # add = int(diff.clamp(0,1).sum() / 2)
    # remove = int(diff.clamp(-1,0).abs().sum() / 2)

    return diff

def show_metrics(changes, labels, g0, device, verbose=True):
    """
    Prints the changes in edges with respect to g0 and labels of a diff
    
    Parameters
    ---
    par_name : par_type
        par_description
    
    Returns
    ---
    out : ret_type
        ret_description
    
    Examples
    ---
    >>>example
    
    """
    
    def print_same_diff(type, adj):
        edges = Utils.to_edges(adj)
        same = 0
        for edge in edges.t():
            same += int(labels[edge[0]].item() == labels[edge[1]].item())
        
        diff = edges.shape[1] - same

        if verbose: print(f"     {type}   {int(same)}  \t{int(diff)}  \t{int(same+diff)}")
        return { "same": int(same), "diff": int(diff), "total": int(same + diff)}

    def print_add_remove(adj):
        add = adj.clamp(0,1)
        remove = adj.clamp(-1,0).abs()
        if verbose: print("                A-A\tA-B\tTOTAL")
        numAdd = print_same_diff("     (+)", add)
        numRemove = print_same_diff("     (-)", remove)

        return {"add": numAdd, "remove": numRemove, "total": numAdd["total"] + numRemove["total"]}
    # print_add_remove(changes)

    r = {}

    if verbose: print("     Within G0 ====")
    g0_adj = mask_adj(changes, g0, device)
    r["g0"] = print_add_remove(g0_adj)

    if verbose: print("     Within GX ====")
    gX_adj = mask_adj(changes, ~g0, device)
    r["gX"] = print_add_remove(gX_adj)

    if verbose: print("     Between G0-GX ====")
    g0gX_adj = (changes - g0_adj - gX_adj)
    r["g0gX"] = print_add_remove(g0gX_adj)

    if verbose: print()
    print_same_diff("   TOTAL", changes)

    return r

def calc_entropy(data: torch.tensor, numbins=50):
    bins = torch.histc(data, bins=numbins)
    bins /= bins.sum()
    return ((bins * torch.log2(bins)).nan_to_num().sum() * -1).item()

def calc_correlation(tensor1: torch.tensor, tensor2: torch.tensor):
    cat = torch.cat((tensor1.unsqueeze(0).cpu(), tensor2.unsqueeze(0).cpu())).numpy()
    return np.corrcoef(cat)[0][1]


def get_ent_cor(features: torch.tensor, labels: torch.tensor, num: int=10, rand=False, offset=0):
    """
    Return the features with most entropy and/or correlation
    
    Parameters
    ---
    par_name : par_type
        par_description
    
    Returns
    ---
    entropy, correlation, index
    
    Examples
    ---
    >>>example
    
    """
    
    ent_cor = torch.zeros(3, features.shape[1])

    for r in range(features.shape[1]):
        feat = features.t()[r]
        entropy = calc_entropy(feat)
        correlation = abs(calc_correlation(feat, labels))
        ent_cor[0][r] = entropy
        ent_cor[1][r] = correlation
        ent_cor[2][r] = entropy + correlation

    ent_cor.nan_to_num_()

    if rand:
        idx = torch.tensor(random.sample(range(features.shape[1]), num))
    else:
        idx = torch.topk(ent_cor[2], num + offset, sorted=True).indices[offset:]
    
    data = ent_cor[:,idx]

    return data[0], data[1], idx