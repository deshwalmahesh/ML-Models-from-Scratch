import torch

class LoRA(torch.nn.Module):
    def __init__(self, input_feat_dim:int, out_feat_dim:int, rank:int, controlling_alpha:float = 1.0):
      """
      It's just a simple matrix multiplication where matrix "A" gets initialized with random scaled values and "B" gets intialized with zero values
      args:
        input_feat_dim: Number of input features coming from the previous layer
        out_feat_dim: Number of output features that will go out as input to next layer
        rank: determines the "compression" size of LoRA layer and is usually proportional to knowledge gained by LoRA. Increasing it will make the bigger in size (thus more capacity)
        alpha: This is another param I found on the blog:https://lightning.ai/lightning-ai/studios/code-lora-from-scratch#coding-lora-from-scratch. Not sure if it was there in the original
                paper. It basically scales (controls) the results by LoRA. 0.5 means half the effect, 2.0 means double the numbers.
      """
      super().__init__()
      scaling_factor = 1 / torch.sqrt(torch.tensor(rank).float()) # in the orig papaer, it scales the random values for Matrix A
      self.A = torch.nn.Parameter(torch.randn(input_feat_dim, rank) * scaling_factor) # random weights scaled by the factor. This is input facing matrix (not sure why in the blog was shown as output facing)
      self.B = torch.nn.Parameter(torch.zeros(rank, out_feat_dim))  # output facing matrix. In the beginning it is zero because we want ful pretrained weights contribution as random weights can destabilist the entire system
      self.controlling_alpha = controlling_alpha # I'm not sure if it is needed because it'll scale the "whole" output in "every" direction. Instead we can test it with a learnable param so it adjusts

      def forward(self, input_tensor):
        return self.controlling_alpha * (input_tensor @ self.A @ self.B) # This output will get "ADDED" to the output of the parent layer of this LoRA layer