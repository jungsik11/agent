# AI-Powered Application and Model Training Repository

This repository contains a collection of code for building and training AI models, as well as a full-stack application that leverages these models. The project includes a Flutter-based frontend, Python backends, and extensive scripts for machine learning, with a focus on natural language processing (NLP) for the Korean language.

## Directory Structure

Here's an overview of the main directories in this repository:

-   `front/app`: A mobile application built with Flutter, serving as the user interface for interacting with the AI services.
-   `back/server` & `webserver/`: Python-based backend servers that provide APIs for the frontend application. These servers likely handle requests and interact with the trained AI models.
-   `lora_finetune/`: Contains scripts for fine-tuning language models using LoRA (Low-Rank Adaptation), a parameter-efficient fine-tuning technique. Includes a detailed README on how to use the scripts.
-   `models/`: Scripts and notebooks for creating, training, and saving machine learning models.
-   `train_example/`: A collection of example scripts for training various models, including `Qwen3-MoE` models for the Korean language.
-   `data/`: Intended for storing datasets used in model training, such as the `korean_textbooks_qa` dataset.
-   `document/`: Contains documentation and manuals related to the project, such as model training guides.
-   `docker/`: Docker-related files, such as `docker-compose.yml`, for containerizing and deploying the applications.

## Key Technologies and Concepts

-   **Machine Learning Frameworks:** The project appears to use MLX for accelerated machine learning on Apple silicon, alongside other common frameworks.
-   **Language Models:** There is a focus on fine-tuning and deploying various language models, including Google's Gemma and MoE (Mixture-of-Experts) models like Qwen.
-   **Fine-Tuning Techniques:** The repository heavily utilizes LoRA for efficient fine-tuning of language models, particularly for Korean NLP tasks.
-   **Application Stack:**
    -   **Frontend:** Flutter
    -   **Backend:** Python (likely with frameworks such as FastAPI or Flask)
-   **Language:** A significant portion of the work is dedicated to Korean language processing, as seen in the datasets and model configurations.

## Getting Started

To get started with a specific part of the project, please refer to the `README.md` file within the corresponding directory (e.g., `lora_finetune/README.md`).
