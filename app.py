# app.py

# ==============================================================================
# PART 1: IMPORTS AND SETUP
# ==============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import copy
import os
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import uuid 

# Flask App Configuration
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Ensure the upload and result directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Device configuration - Force CPU for free hosting to avoid CUDA errors
device = torch.device("cpu")

# --- DEPLOYMENT OPTIMIZATION ---
# Using 128px keeps memory usage within the 512MB limit of free hosting
imsize = 128 

# ==============================================================================
# PART 2: NEURAL STYLE TRANSFER MODEL CODE
# ==============================================================================

# Image loading and transforming
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()])

def image_loader(image_name):
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

unloader = transforms.ToPILImage()

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

# Load pre-trained VGG19 (features only)
# Using weights instead of deprecated 'pretrained' argument
cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
    def forward(self, img):
        return (img - self.mean.to(device)) / self.std.to(device)

def get_style_model_and_losses(cnn, style_img, content_img,
                               content_layers=['conv_4'],
                               style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    cnn = copy.deepcopy(cnn)
    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)
    
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        
        model.add_module(name, layer)
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], (ContentLoss, StyleLoss)):
            break
    model = model[:(i + 1)]
    return model, style_losses, content_losses

def run_style_transfer(cnn, content_img, style_img, input_img, num_steps=150,
                       style_weight=1000000, content_weight=1):
    model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img)
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    
    run = [0]
    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            
            style_score *= style_weight
            content_score *= content_weight
            loss = style_score + content_score
            loss.backward()
            run[0] += 1
            return loss
        optimizer.step(closure)
    
    input_img.data.clamp_(0, 1)
    return input_img

# ==============================================================================
# PART 3: FLASK ROUTING
# ==============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stylize', methods=['POST'])
def stylize():
    if 'content_image' not in request.files or 'style_image' not in request.files:
        return "Error: Missing content or style image", 400

    content_file = request.files['content_image']
    style_file = request.files['style_image']

    if content_file.filename == '' or style_file.filename == '':
        return "Error: No files selected", 400

    # Generate unique filenames
    unique_id = uuid.uuid4()
    content_filename = secure_filename(f"c-{unique_id}.png")
    style_filename = secure_filename(f"s-{unique_id}.png")
    output_filename = f"r-{unique_id}.png"
    
    content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
    style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
    output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
    
    content_file.save(content_path)
    style_file.save(style_path)

    try:
        # Lower default steps for hosted stability
        num_steps = int(request.form.get('steps', 100)) 
        style_weight = int(request.form.get('style_weight', 1000000))
    except (ValueError, TypeError):
        return "Error: Invalid settings", 400
    
    # Run Style Transfer
    content_img = image_loader(content_path)
    style_img = image_loader(style_path)
    input_img = content_img.clone()

    output = run_style_transfer(cnn, content_img, style_img, input_img, 
                               num_steps=num_steps, style_weight=style_weight)

    # Save output
    output_image = unloader(output.squeeze(0))
    output_image.save(output_path)
    
    # Use url_for to generate correct paths for the template
    return render_template('result.html', 
                           content_img=f"static/uploads/{content_filename}", 
                           style_img=f"static/uploads/{style_filename}", 
                           output_img=f"static/results/{output_filename}")

if __name__ == '__main__':
    # Use environment variable PORT if available (required for some hosts)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
