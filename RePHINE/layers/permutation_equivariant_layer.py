import torch


def permutation_equivariant_layer(inp, dimension, perm_op, lbda, b, gamma):
    """DeepSet PersLay"""
    dimension_before, num_pts = inp.shape[2], inp.shape[1]
    b = b.unsqueeze(0).unsqueeze(0).to(inp.device)
    lbda = lbda.to(inp.device)
    A = torch.matmul(inp, lbda).reshape(-1, num_pts, dimension)

    if perm_op is not None:
        if perm_op == "max":
            beta = torch.unsqueeze(torch.max(inp, dim=1).values, 1).expand(
                -1, num_pts, -1
            )
        elif perm_op == "min":
            beta = torch.unsqueeze(torch.min(inp, dim=1).values, 1).expand(
                -1, num_pts, -1
            )
        elif perm_op == "sum":
            beta = torch.unsqueeze(torch.sum(inp, dim=1), 1).expand(-1, num_pts, -1)
        else:
            raise Exception("perm_op should be min, max or sum")
        gamma = gamma.to(inp.device)
        B = torch.matmul(beta, gamma).reshape(-1, num_pts, dimension)
        return A - B + b
    else:
        return A + b


def image_layer(inp, image_size, image_bnds, sg):
    """Persistence Image PersLay"""
    aux = torch.tensor([[1.0, -1.0], [0.0, 1.0]]).float().to(inp.device)
    bp_inp = torch.einsum("ijk,kl->ijl", inp, aux)
    dimension_before, num_pts = inp.shape[2], inp.shape[1]
    coords = [
        torch.linspace(image_bnds[i][0], image_bnds[i][1], steps=image_size[i])
        for i in range(dimension_before)
    ]
    M = torch.meshgrid(*coords)
    mu = torch.cat([tens.unsqueeze(0) for tens in M], dim=0).to(inp.device)
    bc_inp = bp_inp.view(-1, num_pts, dimension_before, 1, 1)
    aux = -torch.square(bc_inp - mu)
    aux2 = torch.sum(aux / (2 * torch.square(sg)), axis=2)
    result = torch.exp(aux2) / (2 * 3.14 * torch.square(sg))
    return result.unsqueeze(-1)
