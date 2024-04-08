import torch 
import torchvision
import matplotlib.pyplot as plt

texture = torchvision.io.read_image('textures/brickwall2.png')
plt.imshow( texture.permute(1, 2, 0)  )
plt.show()

num_samples = texture.shape[0] * texture.shape[1] * texture.shape[2]
p = torch.ones(num_samples)/num_samples
index = torch.multinomial(input=p, num_samples=num_samples, replacement=True)
init = texture.flatten()[index].reshape(texture.shape)
plt.imshow( init.permute(1, 2, 0)  )
plt.show()