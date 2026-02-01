import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import copy
import os

# --- Configuration ---
# Set up the device (use GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Desired image size for processing. Larger sizes require more VRAM and time.
imsize = 512 if torch.cuda.is_available() else 128

# File paths
CONTENT_IMG_PATH = "images/content.jpg"
STYLE_IMG_PATH = "images/style.jpg"
OUTPUT_DIR = "outputs"
OUTPUT_IMG_NAME = "output.jpg"

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Image Loading and Pre-processing ---
# Pre-processing transformations to resize and convert images to tensors
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()])

def image_loader(image_name):
    """Loads an image, applies transformations, and adds a batch dimension."""
    try:
        image = Image.open(image_name)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_name}. Please check the path.")
        exit()
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# Function to convert a tensor back to a PIL image for display/saving
unloader = transforms.ToPILImage()

def imshow(tensor, title=None):
    """Displays a tensor as an image."""
    image = tensor.cpu().clone()  # Clone the tensor to not do changes on it
    image = image.squeeze(0)      # Remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # Pause a bit so that plots are updated

# --- Loss Function Definitions ---
class ContentLoss(nn.Module):
    """
    Computes the content loss (Mean Squared Error) between the feature maps
    of the target image and the content image.
    """
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # We 'detach' the target content from the computation graph.
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    """Computes the Gram matrix for a given batch of feature maps."""
    b, c, h, w = input.size()  # b=batch size, c=channels, (h,w)=dimensions
    features = input.view(b * c, h * w)  # Reshape to (channels, height*width)
    G = torch.mm(features, features.t())  # Compute the gram product
    # Normalize by dividing by the number of elements in each feature map.
    return G.div(b * c * h * w)

class StyleLoss(nn.Module):
    """
    Computes the style loss (Mean Squared Error) between the Gram matrices
    of the target image and the style image.
    """
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

# --- Model Building ---
# Load the pre-trained VGG19 model's features section
cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

# Normalization module using ImageNet mean and std
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std

# Define which VGG layers to use for style and content extraction
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    """
    Builds a new model by adding our custom loss layers to a pre-trained VGG19.
    """
    cnn = copy.deepcopy(cnn)
    
    normalization_mean = torch.tensor([0.485, 0.456, 0.406])
    normalization_std = torch.tensor([0.229, 0.224, 0.225])
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv layer
    for layer in cnn.children():
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

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # Trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], (ContentLoss, StyleLoss)):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

# --- Optimization Loop ---
def get_input_optimizer(input_img):
    """Creates the L-BFGS optimizer for the input image."""
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def run_style_transfer(cnn, content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Main optimization loop for Neural Style Transfer."""
    print('Building the style transfer model...')
    model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing...')
    run = [0]
    while run[0] <= num_steps:
        def closure():
            # Correct the values of the updated input image to be between 0 and 1
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = sum([sl.loss for sl in style_losses])
            content_score = sum([cl.loss for cl in content_losses])

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"run {run[0]}: Style Loss: {style_score.item():.4f} Content Loss: {content_score.item():.4f}")

            return style_score + content_score

        optimizer.step(closure)

    # Final correction
    input_img.data.clamp_(0, 1)

    return input_img

# --- Main Execution Block ---
if __name__ == '__main__':
    style_img = image_loader(STYLE_IMG_PATH)
    content_img = image_loader(CONTENT_IMG_PATH)

    # Ensure both images are the same size
    assert style_img.size() == content_img.size(), \
        "Style and content images must be the same size"

    # We start with the content image as the base for optimization
    input_img = content_img.clone()
    
    print("Starting Style Transfer...")
    output = run_style_transfer(cnn, content_img, style_img, input_img, num_steps=300)

    # Save the result
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_IMG_NAME)
    output_image = unloader(output.squeeze(0))
    output_image.save(output_path)
    print(f"\nOptimization complete. Image saved to: {output_path}")

    # Display the result
    plt.figure()
    imshow(output, title='Generated Image')
    plt.ioff()
    plt.show()