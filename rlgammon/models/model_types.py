"""File storing types associated with models."""
import torch as th

LayerList = list[th.nn.Module]
ActivationList = list[th.nn.ReLU | th.nn.Sigmoid | th.nn.Tanh | th.nn.Softmax]

BaseOutput = th.Tensor
MCTSOutput = tuple[th.Tensor, th.Tensor]
