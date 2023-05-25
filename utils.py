import matplotlib.pyplot as plt
import numpy as np


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# caculate model parameters
def network_parameters(nets):
    num_params = sum(param.numel() for param in nets.parameters())
    return num_params
