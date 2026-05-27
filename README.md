# Hybrid Fingerprint Biometric Vault using Convex Hull Normalization and Collatz Transformation

## Project Overview

This project explores a revocable biometric template framework for fingerprint security.

The system:

- Extracts fingerprint minutiae
- Applies Convex Hull normalization
- Finds centroid and shifts coordinates relative to it
- Generates PIN-dependent perturbation using Collatz steps
- Creates transformed templates for secure storage

## Features

- Translation invariant fingerprint representation
- PIN-dependent transformed templates
- Revocability support
- Visualization of transformation process

## Pipeline

Fingerprint  
→ Minutiae Extraction  
→ Convex Hull  
→ Centroid Normalization  
→ Collatz Transformation  
→ Secure Vault  

## Future Work

- FAR/FRR/EER evaluation
- Correlation analysis
- Enrollment and authentication workflow
- Numerical metrics