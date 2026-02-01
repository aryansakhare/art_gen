🎨Neural Style Transfer Web App

An interactive, AI-powered web application that uses Deep Learning to merge the content of one image with the artistic style of another. Transform your everyday photos into masterpieces inspired by Van Gogh, Picasso, and other iconic artists.

🚀 Live Demo: 
[click here ](https://art-gen-vd6x.onrender.com)

🧐 What is Neural Style Transfer?

Neural Style Transfer (NST) is an optimization technique that takes two images—a Content Image and a Style Reference Image—and blends them together. The output image retains the identifiable objects of the content image but adopts the textures, color schemes, and brushstrokes of the style image.

This is achieved by utilizing a pre-trained VGG19 Convolutional Neural Network (CNN). The network allows us to separate:

Content: Represented by the feature maps in the deeper layers of the network.

Style: Represented by the correlations (Gram Matrices) between feature maps across multiple layers.

🛠️ Tech Stack

Deep Learning Framework: PyTorch

Model Architecture: VGG19 (Pre-trained on ImageNet)

Web Backend: Flask (Python)

Frontend: HTML5, CSS3 (Modern Responsive UI)

Image Processing: Pillow (PIL), Torchvision

Deployment: Render (with Gunicorn)

⚙️ How It Works

Pre-processing: Images are resized and normalized to be compatible with the VGG19 model.

Feature Extraction: The Content and Style images are passed through the network to extract target feature maps.

Loss Functions:

Content Loss: Measures how much the objects in the generated image differ from the original.

Style Loss: Measures the difference in artistic texture using Gram Matrices.

Optimization: We use the L-BFGS optimizer to iteratively update the pixels of the input image until the total loss is minimized.

💻 Local Installation

To run art_gen on your local machine, follow these steps:

Clone the Repo:

git clone [https://github.com/your-username/art_gen.git](https://github.com/your-username/art_gen.git)
cd art_gen


Create a Virtual Environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:

pip install -r requirements.txt


Launch the App:

python app.py


Navigate to http://127.0.0.1:5000 in your web browser.

☁️ Deployment Note

This application is optimized for deployment on free-tier hosting (like Render or Railway).

Memory Optimization: The imsize variable in app.py is set to 128 to prevent "Out of Memory" errors on CPU-only servers.

Processing Time: On a free CPU server, stylization may take 1–3 minutes depending on the optimization steps selected. For high-resolution results, running locally on a GPU is recommended (imsize = 512).

📂 Project Structure

├── app.py              # Main Flask server & NST logic
├── templates/          # HTML files (index.html, result.html)
├── static/
│   ├── uploads/        # Temporary storage for user uploads
│   └── results/        # Generated artistic images
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation


📜 License

Distributed under the MIT License.
