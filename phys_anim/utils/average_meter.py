# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import torch
import torch.nn as nn


class AverageMeter(nn.Module):
    def __init__(self, in_shape, max_size):
        super(AverageMeter, self).__init__()
        self.max_size = max_size
        self.register_buffer("mean", torch.zeros(in_shape, dtype=torch.float32))
        self.register_buffer("current_size", torch.tensor(0, dtype=torch.int64))  # Ensure it is saved

    def update(self, values):
        size = values.size()[0]
        if size == 0:
            return
        new_mean = torch.mean(values.float(), dim=0)
        size = np.clip(size, 0, self.max_size)
        old_size = min(self.max_size - size, self.current_size.item())  # Convert tensor to int
        size_sum = old_size + size
        self.current_size.fill_(size_sum)  # Update as tensor
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self):
        self.current_size.fill_(0)
        self.mean.fill_(0)

    def __len__(self):
        return self.current_size.item()

    def get_mean(self):
        return self.mean.squeeze(0).cpu().numpy()


class TensorAverageMeter:
    def __init__(self):
        self.tensors = []

    def add(self, x):
        if len(x.shape) == 0:
            x = x.unsqueeze(0)
        self.tensors.append(x)

    def mean(self):
        if len(self.tensors) == 0:
            return torch.tensor(0.0)
        cat = torch.cat(self.tensors, dim=0)
        if cat.numel() == 0:
            return torch.tensor(0.0)
        else:
            return cat.mean()

    def std(self):
        if len(self.tensors) == 0:
            return torch.tensor(0.0)
        cat = torch.cat(self.tensors, dim=0)
        if cat.numel() == 0:
            return torch.tensor(0.0)
        else:
            return cat.std()

    def clear(self):
        self.tensors = []

    def mean_and_clear(self):
        mean = self.mean()
        self.clear()
        return mean

    def state_dict(self):
        """Save the list of tensors as a single stacked tensor."""
        return {"tensors": torch.cat(self.tensors, dim=0) if self.tensors else torch.empty(0)}

    def load_state_dict(self, state_dict):
        """Load state with backward compatibility (support empty tensors)."""
        self.tensors = list(state_dict.get("tensors", torch.empty(0)).unbind())


class TensorAverageMeterDict:
    def __init__(self):
        self.data = {}

    def add(self, data_dict):
        for k, v in data_dict.items():
            # Originally used a defaultdict, this had lambda
            # pickling issues with DDP.
            if k not in self.data:
                self.data[k] = TensorAverageMeter()
            self.data[k].add(v)

    def compute(self):
        mean_dict = {k + '_mean': v.mean() for k, v in self.data.items()}
        std_dict = {k + '_std': v.std() for k, v in self.data.items()}
        return {**mean_dict, **std_dict}

    def clear(self):
        self.data = {}

    def compute_and_clear(self):
        output = self.compute()
        self.clear()
        return output

    def state_dict(self):
        """Save all TensorAverageMeter objects as a dictionary."""
        return {k: v.state_dict() for k, v in self.data.items()}

    def load_state_dict(self, state_dict):
        """Load state with backward compatibility."""
        for k, v in state_dict.items():
            if k not in self.data:
                self.data[k] = TensorAverageMeter()
            self.data[k].load_state_dict(v)
