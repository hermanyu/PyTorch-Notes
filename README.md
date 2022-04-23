# PyTorch Notes

These notebooks are notes I wrote down while learning the basics of PyTorch.
- Notebooks 00 - 08 follow <a href='https://www.youtube.com/watch?v=c36lUUr864M&t=12863s'> Python Engineer's wonderful youtube crashcourse </a> and starts from ```torch.tensor()``` objects all the way to Convolutional Neural Networks.
- (In Progress) Notebook 09 will go over Recurrent Neural Networks, GRU, and LSTM layers.
- (Future) Notebooks 10+ will follow a Coursera course on GANs.

---

# PyTorch: A Beautiful Framework

I first learned how to build neural networks using Keras. The design of Keras is elegant and clean, with syntax that blended in nicely with Sci-kit Learn. This made it easy for me to pickup and I was able to build some good neural networks without getting hung-up on the code.

The reason I started learning PyTorch was to really force myself to dig into the mechanics of deep learning. Even though I don't have to code a dense layer or write my own autograd system from scratch, the fact that I actually have to manually code a training loop by calling the feed-forward process, the backpropogation process, and the gradient descent step really gave me a much more down-to-Earth experience. I enjoyed getting my hands relatively dirty and running into bugs because it would often make me think about subtle details I always glossed over when using Keras. The following is some example code for an LSTM network I built for <a href='https://github.com/hermanyu/the-onion-classifier'> my Onion Classifier </a> using PyTorch.

```python
import torch
import torch.nn as nn

# we "build" our model by creating an actual class. This makes it really feel like we are
# in the driver's seat right from the get-go.
class OnionNet(nn.Module): 
    
    def __init__(self):
        super(OnionNet, self).__init__()
        # all the layers we want to use go here. 
        self.lstm = nn.LSTM(input_size=100, hidden_size=64, batch_first=True, device='cuda')
        self.hidden = nn.Linear(in_features=64, out_features=32, device='cuda')
        self.output = nn.Linear(in_features=64, out_features=1, device='cuda')
    
    # in PyTorch, we have to define our own forward-pass. This is where we connect the layers.
    def forward(self, x):
        # because we have to define our own forward-pass, we really have to understand
        # what exactly comes out of the nn.LSTM layer 
        lstm_out, (h_n,c_n) = self.lstm(x)
        z = lstm_out[-1]
        # feed LSTM output into next layer; Tanh activation
        z = nn.Tanh(self.hidden(z))
        z = self.output(z)
        z = nn.Sigmoid(z)
        return z
```

<br>

All-in-All I would say PyTorch and Keras are both very nice and comfortable to use. Keras has the benefit of feeling like an extension of Sklearn, while PyTorch really makes you feel like you are in the driver's seat.