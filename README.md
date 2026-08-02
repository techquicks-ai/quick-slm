# Quick SLM

Quick SLM is an experimental LLaMA-style decoder-only language model (~103M parameters) trained from scratch to test grounded tool-calling. This training run ended with negative results, serving as an educational post-mortem showing that while structure is cheap, reliable grounding at this scale is not.

## Overview
This repository contains the complete training pipeline and framework for Quick SLM. It is primarily designed to be run in Jupyter Notebooks and Google Colab.

### Repository Layout
- **`notebooks/`**: The core pipeline. Contains all the Colab-ready notebooks for data preparation, pretraining, supervised fine-tuning (SFT), evaluation, and diagnostics.
- **`framework/`**: The core Python package (`quick_slm_trainer`), testing suite, and DPO labelling backend. All complex training logic lives here.
- **`docs/`**: Source files for the static documentation site.
- **`app/`**: Front-end applications, including the DPO labelling studio UI.
- **`papers/`**: Research-related materials and Docker configurations.
- **`scripts/`**: Utility scripts for managing the project configuration.
- **`notes.md`**: Comprehensive, consolidated documentation covering training details, SFT plans, audit logs, and future recipes.

## Getting Started (Google Colab)

The training pipeline is structured to be executed sequentially in Colab.

1. Mount your Google Drive to save datasets and checkpoints.
2. Clone this repository into your Colab environment.
3. Navigate to the `notebooks/` directory and begin with `01_data_preparation.ipynb`.

## Hardware Requirements
The primary training run was executed on an **RTX 6000 PRO Blackwell Server Edition (Google G4 in Colab)**. However, with correct parameters, even a **T4** GPU can run the training of the model.

## Documentation
For in-depth details on the model architecture, token budgets, data mixture, and the SFT strategy, please read the [notes.md](file:///Users/aidev/Documents/GitHub/quick-slme/notes.md) file located in the root of this repository, or visit the documentation site at [techquicks-ai.github.io/quick-slm](https://techquicks-ai.github.io/quick-slm).
