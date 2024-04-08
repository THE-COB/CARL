import torch 
import torchvision
import matplotlib.pyplot as plt

texture = torchvision.io.read_image('textures/tomatoes.png').permute(1, 2, 0)
plt.imshow( texture  )
plt.show()

num_samples = texture.shape[0] * texture.shape[1] 
p = torch.ones(num_samples)/num_samples
index = torch.multinomial(input=p, num_samples=num_samples, replacement=True)
init = texture.view((num_samples, -1))[index].reshape(texture.shape)
plt.imshow( init )
plt.show()