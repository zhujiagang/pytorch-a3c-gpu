import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from utils import normalized_columns_initializer, weights_init

class ActorCriticLSTM(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(ActorCriticLSTM, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.lstm = nn.LSTMCell(1024, 512)

        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_outputs)

        self.apply(weights_init)
        # self.actor_linear.weight.data = normalized_columns_initializer(
        #     self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.reset()

    def reset(self):
        self.require_init = True

    def forward(self, inputs, keep_same_state=False):
        batch = inputs.size(0)

        if self.require_init:
            self.require_init = False
            self.hx = Variable(inputs.data.new().resize_((batch, 512)).zero_())
            self.cx = Variable(inputs.data.new().resize_((batch, 512)).zero_())

        x = F.relu(F.max_pool2d(self.conv1(inputs), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv3(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv4(x), kernel_size=2, stride=2))

        x = x.view(batch, -1)
        hx, cx = self.lstm(x, (self.hx, self.cx))
        x = hx

        if not keep_same_state:
            self.hx, self.cx = hx, cx

        return self.critic_linear(x), self.actor_linear(x)

    def detach(self):
        if self.require_init:
            return

        self.hx.detach_()
        self.cx.detach_()