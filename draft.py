import torch
from torchsummary import summary
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
inp =  torch.randn((1,3,64,64))
torch.save(model, "resnet18.pth")
summary(model, inp)