import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


image_path = r'C:\Users\Admin\Documents\code\image_captioning_project\cat.jpg'  
image = Image.open(image_path)


print("Original Image Size:", image.size)  
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')  
plt.show()

#Preprocess the Image
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),          
    transforms.Normalize(           
        mean=[0.485, 0.456, 0.406],  
        std=[0.229, 0.224, 0.225]     
    )
])


image_tensor = preprocess(image)

print("Preprocessed Image Tensor Shape:", image_tensor.shape)  
print("Tensor Values (Min, Max):", image_tensor.min().item(), image_tensor.max().item())


image_processed = image_tensor.permute(1, 2, 0).numpy()  

image_processed = (image_processed - image_processed.min()) / (image_processed.max() - image_processed.min())


print("Processed Image Shape (for display):", image_processed.shape)  
print("Processed Image Values (Min, Max):", image_processed.min(), image_processed.max())

plt.imshow(image_processed)
plt.title("Preprocessed Image")
plt.axis('off')
plt.show()