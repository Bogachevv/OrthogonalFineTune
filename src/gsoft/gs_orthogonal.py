import math
import numpy as np

import torch
from torch.nn import functional as F
import torch.nn as nn

from blockdiag_butterfly_multiply import BlockdiagButterflyMultiply


class GSOrthogonal(nn.Module):
    def __init__(self, 
                 n: int, 
                 nblocks: int, 
                 orthogonal=True, 
                 method="cayley", 
                 block_size=None,
                 base_tensor: torch.Tensor = None,
                 ):

        if block_size is not None:
            assert n % block_size == 0
            nblocks = n // block_size

        assert n % nblocks == 0
        if not orthogonal:
            raise ValueError("orthogonal == False deprecated")

        super().__init__()

        if base_tensor is None:
            self.gsoft_A = nn.Parameter(torch.empty(nblocks, n // nblocks, n // nblocks, dtype=torch.float32))
            # self.gsoft_R = nn.Parameter(torch.empty(nblocks, n // nblocks, n // nblocks))
            # self.gsoft_L = nn.Parameter(torch.empty(nblocks, n // nblocks, n // nblocks))
        else:
            self.gsoft_A = nn.Parameter(base_tensor.new_empty(nblocks, n // nblocks, n // nblocks, dtype=torch.float32))
            # self.gsoft_R = nn.Parameter(base_tensor.new_empty(nblocks, n // nblocks, n // nblocks, dtype=torch.float32))
            # self.gsoft_L = nn.Parameter(base_tensor.new_empty(nblocks, n // nblocks, n // nblocks, dtype=torch.float32))

        self.orthogonal = orthogonal
        self.n = n
        self.nblocks = nblocks
        self.block_size = n // nblocks
        self.method = method

        self.blockdiag_butterfly_multiply = BlockdiagButterflyMultiply.apply

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        # initialize whole layer as identity matrix

        if self.orthogonal:
            torch.nn.init.zeros_(self.gsoft_A)
            # torch.nn.init.zeros_(self.gsoft_L)
            # torch.nn.init.zeros_(self.gsoft_R)
        else:
            raise ValueError("orthogonal == False deprecated")
    
    def exp_full(self, data):
        # skew = 0.5 * (data - data.transpose(1, 2))
        skew = data - data.transpose(1, 2)
        return torch.matrix_exp(skew)

    def cayley_batch(self, data):
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        # skew = 0.5 * (data - data.transpose(1, 2))
        skew = data - data.transpose(1, 2)
        I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)

        # Perform the Cayley parametrization
        Q = torch.linalg.solve(I - skew, I + skew, left=False)
        return Q
    
    def forward(self, x):
        gsoft_L = torch.triu(self.gsoft_A)
        gsoft_R = torch.tril(self.gsoft_A)

        if self.orthogonal:
            if self.method == "cayley":
                L = self.cayley_batch(gsoft_L)
                R = self.cayley_batch(gsoft_R)
            elif self.method == "exp":
                L = self.exp_full(gsoft_L)
                R = self.exp_full(gsoft_R)
            else:
                raise NotImplementedError(f"Method {self.method} is not supported. Use 'cayley' or 'exp'.")
        else:
            raise ValueError("orthogonal == False deprecated")

        return self.blockdiag_butterfly_multiply(x, R, L)

    def __repr__(self):
        return f"GSOrthogonal(n={self.n}, nblocks={self.nblocks}, orthogonal={self.orthogonal}, method={self.method}, block_size={self.block_size})"
