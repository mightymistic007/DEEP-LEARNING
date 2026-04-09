# NeuroStyle: Architectural Neural Style Transfer (NST)

NeuroStyle is a deep learning synthesis engine that decouples and recombines artistic style from photographic content. By utilizing a custom-built **Convolutional Neural Network (CNN)** with **Residual Blocks**, the system can "re-paint" any photograph using the textures and color profiles of 39+ distinct art movements.

## 🚀 Key Features

* **Residual-Block Architecture**: Employs a sophisticated CNN featuring Downsampling, Residual Skip-Connections for gradient stability, and Upsampling (Conv2DTranspose) for high-fidelity output generation.
* **Dual-Feature Extraction**:
    * **Content Extraction**: Captures semantic scenery and object arrangement from deep network layers ($x3, x6$).
    * **Style Extraction**: Captures texture correlations via **Gram Matrices** from multi-scale layers ($x1, x2, x5$).
* **Vast Style Library**: Includes specialized support for 39+ unique styles, including **Street Art**, **Ukiyo-e**, **Suprematism**, **Bauhaus**, and **Surrealism**.
* **Latent Space Optimization**: Utilizes the **Adam Optimizer** to iteratively evolve a noise-initialized tensor into a final masterpiece.

## 🛠️ Technologies Used

* **TensorFlow 2.x**: Core framework for model construction and `tf.GradientTape` optimization.
* **Numpy & PIL**: Used for image-to-tensor transformations and high-fidelity LANCZOS resampling.
* **Matplotlib**: For real-time training visualization and final rendering.

## 🧪 Methodology

Grounding its logic in the seminal research by **Gatys et al.**, NeuroStyle operates on the principle that style and content are separable in the feature spaces of performance-optimized neural networks.

1.  **Gram Matrix Calculation**: Computes the inner product between vectorized feature maps to capture texture while discarding global arrangement.
2.  **Loss Joint-Minimization**:
    $$\mathcal{L}_{total} = \alpha\mathcal{L}_{content} + \beta\mathcal{L}_{style}$$
3.  **Backpropagation**: The gradient of the total loss is used to iteratively update the pixel values of the generated image $\vec{x}$.

## 📂 Project Structure

```text
NeuroStyle
├── ART_STYLES/      # 39+ distinct style images (Renaissance, Street Art, etc.)
├── reference_paper/ # Scientific foundations (Gatys_Image_Style_Transfer.pdf)
├── DL_PROJECT.ipynb # Main implementation and training pipeline
└── README.md        # Project documentation
```

## ⚡ Getting Started

* Requirements
  **Ensure you have a GPU environment and the following libraries installed:
  **pip install tensorflow pillow matplotlib numpy

* Configuration
  **Update the content_image_path and style_image_path in DL_PROJECT.ipynb to point to your desired files in /ART_STYLES/.

* Run
  **Execute the notebook cells. The system will iteratively print the loss and render your stylized output.


