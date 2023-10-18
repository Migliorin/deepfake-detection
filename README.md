# Deepfake Detection

## Project Overview

This repository is the result of a Project of Support for Scientific Initiation (PAIC) that aimed to test image preprocessing techniques to enhance the performance of the network proposed in the research paper "Combining EfficientNet and Vision Transformers for Video Deepfake Detection," which can be accessed via this [Coccomini et al.](https://doi.org/10.48550/arXiv.2107.02612). The project focused on detecting deepfakes in videos.

However, if you would like see the original work please acess the follow [GitHub](https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection)

## Preprocessing Techniques

To improve the model's performance, the following preprocessing techniques were applied to the input images:

- Histogram Equalization
- Sobel Filter
- Mean Subtract Channel

These techniques were chosen to enhance the quality of the input data for deepfake detection.

## Code Refactoring

The code in this repository has been refactored according to the preferences of the author. It is designed to improve the overall readability and maintainability of the codebase. However, please note that the refactored code may still have some issues or bugs.

If you encounter any problems or have questions about the code, please do not hesitate to contact Lucas Migliorin (Me) at lucasmigliorin@hotmail.com or lmdr.eng19@uea.edu.br.

## Research Publication

The exploration of preprocessing techniques conducted in this project has led to a publication in the ENIAC conference, which is promoted by the Brazilian Computer Society (SBC) and the Federal University of Minas Gerais (UFMG). The details of this publication will be made available soon, and it will serve as a reference for this work. All authors of original work have been properly referenced. With you find some problem or referenced authors missing please contact me.

## System Configuration

The deepfake detection model was trained on a computer with the following hardware specifications:

- GPU: NVIDIA RTX 2080 with 8GB of VRAM
- RAM: 64GB
- Storage: 1.5 TB SSD
- CPU: 10th generation Intel Core i7

## Experiment Details

In the paper, two networks, EfficientViT and Cross Efficient ViT, were proposed for video deepfake detection. However, due to time and computational constraints, the experiments in this repository are focused on the EfficientViT network.

## Usage

Please refer to the documentation and code within this repository for more details on how to use the deepfake detection model.

## Acknowledgments

- This research project was made possible by the support provided through the Project of Support for Scientific Initiation (PAIC).
- The deepfake detection model is based on the techniques described in the research paper "Combining EfficientNet and Vision Transformers for Video Deepfake Detection."

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute the code, subject to the terms and conditions of the license.

## Contact

For any questions, issues, or inquiries, please contact Lucas Migliorin (Me) at lucasmigliorin@hotmail.com or lmdr.eng19@uea.edu.br.
