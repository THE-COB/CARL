import torch 
import torchvision
import matplotlib.pyplot as plt

texture = torchvision.io.read_image('textures/tomatoes.png')
plt.imshow( texture.permute(1, 2, 0)  )
plt.show()

num_samples = texture.shape[1] * texture.shape[2] 
p = torch.ones(num_samples)/num_samples
index = torch.multinomial(input=p, num_samples=num_samples, replacement=True)
init = texture.permute((1,2,0)).view((num_samples, -1))[index].reshape(texture.shape[1], texture.shape[2], texture.shape[0])
plt.imshow( init )
plt.show()