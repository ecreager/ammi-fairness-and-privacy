import torch
import torch.nn.functional as F

def get_mmd_inds(n0, n1):
  '''Helper function for MMD torch.
  
  Suppose A, B are tensors of length n0 and n1, respectively.
  Let AB represent the length-(n0 + n1) concatenation of A and B.
  Let dists = torch.nn.functional.pdist(AB) --- this is the calculation of all
    pairwaise distances of rows in AB.
  
  Then this function returns indices identifying which elements of dists
    represent distances between two elements of A, two elements of B, or
    an element of A and an element of B.
    
  Args:
    n0 (int): number of rows in A.
    n1 (int): number of rows in B.
  Returns:
    v0 (Torch tensor): An ((n0 + n1) * (n0 + n1 - 1) / 2,)-shaped tensor, with
      a 1 where the output of dists represents a distance between two elements
      of A, and 0 elsewhere.
    v1 (Torch tensor): An ((n0 + n1) * (n0 + n1 - 1) / 2,)-shaped tensor, with
      a 1 where the output of dists represents a distance between two elements
      of B, and 0 elsewhere.    
    v01 (Torch tensor): An ((n0 + n1) * (n0 + n1 - 1) / 2,)-shaped tensor, with
      a 1 where the output of dists represents a distance between one element
      of A and one element of B, and 0 elsewhere.      
  '''
  n = n0 + n1
        
  x_mmd_0_1 = torch.mul(torch.triu(torch.ones(n, n)) - torch.eye(n), 
                           torch.mul(
                               torch.reshape(
                                   torch.arange(n) < n0, [-1, 1]).float(), 
                               torch.reshape(
                                   torch.arange(n) >= n0, [1, -1]).float()))
  x_mmd_0 = torch.mul(torch.triu(torch.ones(n, n)) - torch.eye(n), 
                           torch.mul(
                               torch.reshape(
                                   torch.arange(n) < n0, [-1, 1]).float(), 
                               torch.reshape(
                                   torch.arange(n) < n0, [1, -1]).float()))
  x_mmd_1 = torch.mul(torch.triu(torch.ones(n, n)) - torch.eye(n), 
                           torch.mul(
                               torch.reshape(
                                   torch.arange(n) >= n0, [-1, 1]).float(), 
                               torch.reshape(
                                   torch.arange(n) >= n0, [1, -1]).float()))


  v01 = x_mmd_0_1[torch.triu(torch.ones(n, n)) - torch.eye(n) == 1].byte()
  v0 = x_mmd_0[torch.triu(torch.ones(n, n)) - torch.eye(n) == 1].byte()
  v1 = x_mmd_1[torch.triu(torch.ones(n, n)) - torch.eye(n) == 1].byte()
  return v0, v1, v01




def MMD_torch(Z, A, bandwidth=1.):
  '''Calculates the MMD between the two groups (as defined by A) in Z, using
     a Gaussian kernel.

  MMD stands for Maximum Mean Discrepancy.
  As calculated in https://arxiv.org/pdf/1511.00830.pdf

  Args:
    Z (Torch tensor):  an (n x d)-shaped tensor, representing some data about 
                       each individual.
    A (Torch tensor):  an (n x 1) or (n,)-shaped tensor, representing a binary 
                       sensitive attribute for each individual.
    bandwidth (float, optional): bandwidth paramater for the kernel.
  Returns:
    mmd: an estimate of the MMD between the distributions P(Z | A = 0) and
                       P(Z | A = 1).
  '''
  
  A = A.flatten().byte()
  
  # Separate the A = 0 and A = 1 groups in Z.
  Z0 = Z[1 - A, :]
  Z1 = Z[A, :]
  Z01 = torch.cat([Z0, Z1], dim=0)
  n0 = torch.sum(1 - A)
  n1 = torch.sum(A)

  # Calculate the pairwise distances between rows of Z.
  dists = F.pdist(Z01)
  
  # Determine which elements of dists represent distances between rows from
  # which groups.
  v0, v1, v01 = get_mmd_inds(n0, n1)
  Z0_dist = dists[v0]
  Z1_dist = dists[v1]
  Z_dist = dists[v01]

  # Calculate MMD using these distances.
  kernel_sum_0 = 2 * torch.sum(torch.exp(-bandwidth * Z0_dist)) + n0
  kernel_sum_1 = 2 * torch.sum(torch.exp(-bandwidth * Z1_dist)) + n1
  kernel_sum_01 = torch.sum(torch.exp(-bandwidth * Z_dist))
  mmd = (kernel_sum_0 / (n0 ** 2)
         + kernel_sum_1 / (n1 ** 2)
         - 2 * kernel_sum_01 / (n0 * n1)
        )
  return mmd
