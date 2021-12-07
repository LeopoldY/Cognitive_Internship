import torch
from torch.autograd import Variable
from models.nn import sexnet

sexDic = {
    1:'Male',
    0:'Female'
}

model = sexnet()
model.eval()
model.load_state_dict(torch.load('output/params_100.pth'))
# model = torch.load('output/params_100.pth')

height = float(input('Height:'))/2.0
weight = float(input('Weight:'))/80.0

tensor = torch.tensor([height, weight])
out = model(Variable(tensor))
print(out.size())
pred = torch.max(out, 0)[1]
print(sexDic[pred.item()])
