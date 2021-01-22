## import packages
import os

import torch
import random
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset

import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'



## Checks for the availability of GPU 
if torch.cuda.is_available():
    print("working on gpu!")
    device = 'cuda'
else:
    print("No gpu! only cpu ;)")
    device = 'cpu'
    
## The following random seeds are just for deterministic behaviour of the code and evaluation

if device == 'cpu':    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
elif device == 'cuda':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = '0'


############################################################################### 
import torchvision
import torchvision.transforms as transforms
import os

if not os.path.isdir('./data'):
    os.mkdir('./data')
root = './data/'

train_bs = 128

transform = transforms.Compose([transforms.ToTensor(),
         transforms.Normalize(mean=[0.5],
                                std=[0.5])
        ])

training_data = np.zeroes([831,1,36,36])
index = 0
for font in glob.glob(root):
    fontImage = Image.open(font)
    # convert image to numpy array
    data = np.asarray(image)
    training_data[index] = data
    index+=1
# training_data = torchvision.datasets.MNIST(root, train=True, transform=transform,download=True)
train_loader=torch.utils.data.DataLoader(dataset=training_data, batch_size=train_bs, shuffle=True, drop_last=True)


###############################################################################
def noise(bs, dim):
    """Generate random Gaussian noise.
    
    Inputs:
    - bs: integer giving the batch size of noise to generate.
    - dim: integer giving the dimension of the the noise to generate.
    
    Returns:
    A PyTorch Tensor containing Gaussian noise with shape [bs, dim]
    """
    
    out = (torch.randn((bs, dim))).to(device)
    return out


###############################################################################

class Generator(nn.Module):
    def __init__(self, noise_dim=100, out_size=1296):
        super(Generator, self).__init__()
        '''
        REST OF THE MODEL HERE
        # define a fully connected layer (self.layer1) from noise_dim -> 256 neurons
        # define a leaky relu layer(self.leaky_relu) with negative slope=0.2.
        # define a fully connected layer (self.layer2) from 256 -> 512 neurons
        # define a fully connected layer (self.layer3) from 512 -> 1024 neurons
        # define a fully connected layer (self.layer4) from 1024 -> out_size neurons
        # define a tanh activation function (self.tanh)
        '''
        self.layer1 = nn.Linear(in_features=noise_dim, out_features=256)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.layer2 = nn.Linear(in_features=256, out_features=512)
        self.layer3 = nn.Linear(in_features=512, out_features=1024)
        self.layer4 = nn.Linear(in_features=1024, out_features=out_size)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        '''
        Make a forward pass of the input through the generator. Leaky relu is used as the activation 
        function in all the intermediate layers. Tanh activation function is only used at the end (which
        means only after self.layer4)
        
        Note that, generator takes an random noise as input and gives out fake "images". Hence, the output 
        after tanh activation function is reshaped into the same size as the real images. i.e., 
        [batch_size, n_channels, H, W] == (batch_size, 1,36,36)
        '''
        x = self.layer1(x)
        x = self.leaky_relu(x)
        x = self.layer2(x)
        x = self.leaky_relu(x)
        x = self.layer3(x)
        x = self.leaky_relu(x)
        x = self.layer4(x)
        x = self.tanh(x)
        x = x.reshape(train_bs, 1, 36, 36)
        
        return x
             

###############################################################################
generator = Generator().to(device)
###############################################################################
## Similar to the Generator, we now define a Discriminator which takes in a vector and output a single scalar 
## value. 

class Discriminator(nn.Module):
    def __init__(self, input_size=1296):
        super(Discriminator, self).__init__()
        '''
        REST OF THE MODEL HERE
        # define a fully connected layer (self.layer1) from input_size -> 512 neurons
        # define a leaky relu layer(self.leaky_relu) with negative slope=0.2.
        # define a fully connected layer (self.layer2) from 512 -> 256 neurons
        # define a fully connected layer (self.layer3) from 256 -> 1 neurons
        '''
        self.layer1 = nn.Linear(in_features=input_size, out_features=512)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.layer2 = nn.Linear(in_features=512, out_features=256)
        self.layer3 = nn.Linear(in_features=256, out_features=1)
    
    def forward(self, x):
        '''
        The Discriminator takes a vectorized input of the real and generated fake images. Reshape the input 
        to match the Discriminator architecture. 
        
        Make a forward pass of the input through the Discriminator and return the scalar output of the 
        Discriminator.
        '''
        x = x.view(x.size(0), 36*36)
        y = self.layer1(x)
        y = self.leaky_relu(y)
        y = self.layer2(y)
        y = self.leaky_relu(y)
        y = self.layer3(y)
        
        return y
        

###############################################################################
discriminator = Discriminator()
discriminator = discriminator.to(device)
############################################################################### 
bce_loss = nn.BCEWithLogitsLoss()
############################################################################### 
def DLoss(logits_real, logits_fake, targets_real, targets_fake):
    '''
    d1 - binary cross entropy loss between outputs of the Discriminator with real images 
         (logits_real) and targets_real.
    d2 - binary cross entropy loss between outputs of the Discriminator with the generated fake images 
         (logits_fake) and targets_fake.
    '''
    d1 = bce_loss(logits_real, targets_real)
    d2 = bce_loss(logits_fake, targets_fake)
    
    total_loss = d1 + d2
    return total_loss
    

############################################################################### 
def GLoss(logits_fake, targets_real):
    '''
    The aim of the Generator is to fool the Discriminator into "thinking" the generated images are real.
    g_loss - binary cross entropy loss between the outputs of the Discriminator with the generated fake images 
         (logits_fake) and targets_real.
         
    Thus, the gradients estimated with the above loss corresponds to generator producing fake images that 
    fool the discriminator.
    '''
    g_loss = bce_loss(logits_fake, targets_real)
    return g_loss


############################################################################### 
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

epochs = 50
noise_dim = 100
############################################################################### 
## Training loop

for epoch in range(epochs):
    for i, (images, _) in enumerate(train_loader):
        
        # We set targets_real and targets_fake to non-binary values(soft and noisy labels).
        # This is a hack for stable training of GAN's.  
        # GAN hacks: https://github.com/soumith/ganhacks#6-use-soft-and-noisy-labels
        
        targets_real = (torch.FloatTensor(images.size(0), 1).uniform_(0.8, 1.0)).to(device)
        targets_fake = (torch.FloatTensor(images.size(0), 1).uniform_(0.0, 0.2)).to(device)
        
        images = images.to(device)

        ## D-STEP:
        optimizer_D.zero_grad()
        logits_real = discriminator.forward(images)
        fake_images = generator(noise(train_bs, noise_dim)).detach()
        logits_fake = discriminator.forward(fake_images)
        discriminator_loss = DLoss(logits_real, logits_fake, targets_real, targets_fake)
        discriminator_loss.backward()
        optimizer_D.step()
        
        ## G-STEP:
        optimizer_G.zero_grad()
        fake_images = generator(noise(train_bs, noise_dim))
        logits_fake = discriminator.forward(fake_images)
        generator_loss = GLoss(logits_fake, targets_real)
        generator_loss.backward()
        optimizer_G.step()
        
    
    print("D Loss: ", discriminator_loss.item())
    print("G Loss: ", generator_loss.item())
          
    if epoch % 2 == 0:
        viz_batch = fake_images.data.cpu().numpy()
        fig = plt.figure(figsize=(8,10))
        for i in np.arange(1, 10):
            ax = fig.add_subplot(3, 3, i)
            img = viz_batch[i].squeeze()
            plt.imshow(img)
        plt.show()
