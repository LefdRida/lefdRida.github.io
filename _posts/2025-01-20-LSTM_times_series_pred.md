---
layout: post
title: Energy Forecasting with LSTM
date: 2025-01-20 11:12:00-0400
description: 
tags: 
categories: Technical-post
related_posts: false
---


In this post we will cover LSTM through time series forecasting. The aim is to explain LSTM and its advantage regarding RNNs. We will cover an application of Energy consumption and cover also some statistical algorithm for comparison. 

As we explained previously RNN takes as input a sequence of elements and outputs a sequece of hidden states. Each unit in RNN take as input an element from sequence and the hidden state of the previous element from the sequence. 
Dealing with sequences makes RNN suitable for making prediction using time series as data. Time series is a sequence of elements, that could be real value, images or texts, that is ordered in a chronological order. 

A first problem of RNN is its inability to perform well on tasks that require the use of information distant from the current point of procssing. The hidden states tends to be local and relevant only the most recent parts of the input sequence. The second problem is vanishing gradients, as during training the error needs to backpropagate through time and then we have repeated multiplications which results in that gradients are eventualy driven to zero.

To address this limitations, more complex network was designed. And LSTM is the most commonly used as an extension to RNN.

### Long Short Term Memory : LSTM

Long Short term memory solves the limitation of neglicting information distant from the current time step, encountered in RNN. It achieves this by adding a context layer to the architecture that output a context vector $$c_t$$. And, adding, also, gates that allow either to remove information that are no longer needed or to add information to be needed for later decision making. 

The first gate to consider is the _forget gate_ $$f_t$$. Its purpose is to delete information that are no longer needed and computed as follows:

$$f_t = \sigma (U_f h_{t-1} + W_fx_t)$$

$$k_t = c_t \odot f_t$$

The second gate to consider is the _add gate_ which aims to select information to add to the current context and computed as follows:

$$i_t = \sigma (U_i h_{t-1} + W_i x_t)$$

$$j_t = g_t \odot i_t$$

where $$g_t$$ is the actual information we need to extract from the previous hidden state and current input :

$$g_t = tanh (U_g h_{t-1} + W_g x_t)$$

the output of the add gate and the forget gate are summed then to get the current context vector:

$$c_t = j_t + k_t$$

finally we have the output gate that decides what information is required for the current hidden state. Is is computed as follows:

$$o_t = \sigma (U_o h_{t-1} + W_o x_t)$$

$$h_t = o_t \odot tanh(c_t)$$

We can remark that the gates have a similar design. They contain a feedforward layer, followed by a sigmoid function, and finally followed by an element-wise multiplication with the layer being gated.\
The choice of _sigmoid_ function arises from its tendency to push its output to either 0 or 1. With the use of the element-wise multiplication, we have the effect of a binary mask. This allows then the values in the layer being gated that align with values near to 1 to pass and the values that align vaues near 0 to be erased. This makes the intuition behind how the add gate or the forget gate select or delete information. 

The LSTM can be implemented using Pytorch simply as follows:

```python 
import torch
lstm = nn.LSTM(
            input_size=d, # The size of the input
            hidden_size=n_units, # The size of the hidden state 
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=True,
        )
```


to be continued ...