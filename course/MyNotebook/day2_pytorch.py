import torch
import torchsummary
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt

repo_path = '//Users/sumiya/git/LastBlackBox'
box_path = repo_path + '/boxes/learning'
image_path = box_path + '/transfer/backbone/_data/dog.jpg'
labels_path = box_path + '/transfer/backbone/_data/imagenet_labels.txt'

class_labels = np.array(open(labels_path).read().splitlines())

image = cv2.imread(image_path)

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
resized = cv2.resize(rgb, (224,224))

preprocess = transforms.Compose([
                                transforms.ToTensor(),
                                transforms .Normalize (mean= [0.485, 0.456, 0.486], std=[0.229, 0.224, 6.225])
                                ])
input_tensor = preprocess(resized)
input_batch = input_tensor.unsqueeze(0)

model = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=True)
model.eval()

# Summary of model
torchsummary.summary(model, (3,224,244))

# Get cpu or gpu device
# if torch.cuda.is_available():
#     input_batch = input_batch.to('cuda')
#     model.to('cuda')

# Run model
with torch.no_grad():
    output = model(input_batch)

# Find top 5 classes
top5 = torch.topk(output, 5)
top5 = top5.indices.detach().numpy()
print(class_labels[top5])

# Plot class probabilities
probabilities = torch.nn.functional.softmax(output[0], dim=0)
probabilities =probabilities.detach().numpy()
plt.plot(probabilities)
plt.show()

# FIN