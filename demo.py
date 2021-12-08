import torch
from torch.autograd import Variable
from torchvision import transforms
from models.nn import sexnet

model = sexnet()
model.load_state_dict(torch.load('output/params_100.pth'))
print(type(torch.load('output/params_100.pth')))
# model = torch.load('output/model.pth')
model.eval()

height = float(input('Height:'))/2.0
weight = float(input('Weight:'))/80.0

# pred = torch.max(prediction, 1)[1]