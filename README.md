# DIB-X: Formulating Explainability Principles for a Self-Explainable Model Through Information Theoretic Learning

## Overview

DIB-X introduces a novel self-explainable deep learning approach designed for image data, grounded in the principles of minimal, sufficient, and interactive explanations. This methodology seeks to integrate these well-defined explainability principles directly into the learning process, aiming for their achievement through optimization. 

Accepted for publication at ICASSP 2024, DIB-X exemplifies the use of information theoretic learning to create models that not only perform with high accuracy but also provide insights into their decision-making processes.

## Abstract

Recent advancements in self-explainable deep learning have focused on embedding explainability principles within the learning process to achieve more transparent models. DIB-X represents a pioneering approach in this direction for image data, adhering to minimal, sufficient, and interactive explanation principles. The minimality and sufficiency principles derive from the information bottleneck framework's trade-off relationship, with DIB-X quantifying minimality through matrix-based R\'enyi's $\alpha$-order entropy functional, thus avoiding variational approximation and distributional assumptions. Interactivity is incorporated by leveraging domain knowledge as prior explanations, promoting alignment with established domain insights. Empirical evaluations on MNIST and two marine environment monitoring datasets demonstrate DIB-X's ability to enhance explainability without sacrificing, and in some cases improving, classification performance.

## Datasets

In our research, we employed three datasets:
- MNIST: A well-known dataset for handwritten digit recognition.
- Multi-frequency echosounder data: Utilized for marine environment monitoring.
- Aerial imagery of seal pups: Another dataset for marine environment monitoring.

Due to privacy and accessibility constraints, the multi-frequency echosounder data and the aerial imagery of seal pups datasets are not publicly available. However, we have made the code for experiments using the MNIST dataset open for research and educational purposes.

## Code Usage

The repository contains the code for the MNIST dataset experiments. 
