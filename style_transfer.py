import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import os
#test
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def forward(self, x):
        G = self.gram_matrix(x)
        self.loss = F.mse_loss(G, self.target)
        return x
    
    def gram_matrix(self, input):
        batch_size, n_channels, height, width = input.size()
        features = input.view(batch_size * n_channels, height * width)
        G = torch.mm(features, features.t())
        return G.div(batch_size * n_channels * height * width)

class StyleTransfer:
    def __init__(self):
        # Load pre-trained VGG19 model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn = models.vgg19(pretrained=True).features.to(self.device).eval()
        
        # Define normalization mean and std
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        
        # Content and style layers
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        
        # Directory for style images
        self.style_dir = "static/styles"
        os.makedirs(self.style_dir, exist_ok=True)
        
        # Load some default styles
        self.default_styles = {
            "starry_night": "static/styles/starry_night.jpg",
            "kandinsky": "static/styles/kandinsky.jpg",
            "picasso": "static/styles/picasso.jpg",
            "monet": "static/styles/monet.jpg"
        }
    
    def load_image(self, path, size=512):
        """Load an image and convert it to a tensor"""
        image = Image.open(path)
        loader = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cnn_normalization_mean, std=self.cnn_normalization_std)
        ])
        image = loader(image).unsqueeze(0).to(self.device)
        return image
    
    def get_model_and_losses(self, content_img, style_img):
        """Set up the model, losses, and feature extraction"""
        # Create a sequential model with content and style losses
        model = nn.Sequential()
        content_losses = []
        style_losses = []
        
        # Current position in the model
        i = 0
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i}'
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i}'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{i}'
            else:
                raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')
            
            model.add_module(name, layer)
            
            # Add content loss
            if name in self.content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module(f"content_loss_{i}", content_loss)
                content_losses.append(content_loss)
            
            # Add style loss
            if name in self.style_layers:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module(f"style_loss_{i}", style_loss)
                style_losses.append(style_loss)
        
        # Trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
        
        model = model[:(i + 1)]
        
        return model, style_losses, content_losses
    
    def apply_style(self, content_img_path, style_name, output_path, num_steps=300, 
                   style_weight=1000000, content_weight=1):
        """Apply a style to a content image"""
        # Load content image
        content_img = self.load_image(content_img_path)
        
        # Load style image
        if style_name in self.default_styles:
            style_img_path = self.default_styles[style_name]
        else:
            style_img_path = os.path.join(self.style_dir, f"{style_name}.jpg")
        
        style_img = self.load_image(style_img_path)
        
        # Create input image (content image clone)
        input_img = content_img.clone()
        
        # Set up the optimizer
        optimizer = torch.optim.LBFGS([input_img.requires_grad_()])
        
        # Get model and losses
        model, style_losses, content_losses = self.get_model_and_losses(content_img, style_img)
        
        # Run the optimization
        run = [0]
        while run[0] <= num_steps:
            def closure():
                input_img.data.clamp_(0, 1)
                
                optimizer.zero_grad()
                model(input_img)
                
                style_score = 0
                content_score = 0
                
                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss
                
                style_score *= style_weight
                content_score *= content_weight
                
                loss = style_score + content_score
                loss.backward()
                
                run[0] += 1
                if run[0] % 50 == 0:
                    print(f"Run {run[0]}: Style Loss: {style_score.item():4f} Content Loss: {content_score.item():4f}")
                
                return style_score + content_score
            
            optimizer.step(closure)
        
        # Denormalize the output image
        input_img.data.clamp_(0, 1)
        
        # Convert tensor to PIL image
        unloader = transforms.ToPILImage()
        output_img = input_img[0].cpu().clone()
        output_img = unloader(output_img)
        
        # Save the output image
        output_img.save(output_path)
        
        return output_path
