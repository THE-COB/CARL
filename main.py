import torch
import torchvision
import matplotlib.pyplot as plt

texture = torchvision.io.read_image('textures/tomatoes.png')
plt.imshow( texture.permute(1, 2, 0)  )
plt.show()
