# Spherical Diffusion PINN (Singularity Experiment)

**Author:** Thaer Abushawer

## Motivation
In the Single Particle Model (SPM) for batteries, the diffusion equation is defined in spherical coordinates. A known numerical issue arises at the center (r=0) due to the term 2/r, which often requires mesh refinement in standard FEM solvers.

I wrote this code to test if a **Physics-Informed Neural Network (PINN)** could solve this PDE continuously from r=0 by incorporating the mathematical limit directly into the loss function.

## Mathematical Formulation
The governing PDE is:
$\frac{\partial C}{\partial t} = D \left( \frac{\partial^2 C}{\partial r^2} + \frac{2}{r} \frac{\partial C}{\partial r} \right)$

At the center (r=0), I applied **L'Hôpital's Rule** to resolve the undefined term:
$\lim_{r \to 0} \frac{2}{r} \frac{\partial C}{\partial r} = 2 \frac{\partial^2 C}{\partial r^2}$

This simplifies the equation at the center to:
$\frac{\partial C}{\partial t} = 3D \frac{\partial^2 C}{\partial r^2}$

## Implementation Details
* **Framework:** DeepXDE / TensorFlow
* **Geometry:** 1D Interval [0, 1] (starting exactly at 0).
* **Logic:** Used `tf.where` to switch between the standard equation (for r > 0) and the L'Hôpital form (for r near 0).

## Results
The model was trained for 10,000 iterations. The plot below compares the PINN prediction against the expected analytical parabolic profile.

![Comparison Plot](result_plot.png)
*(The Green line represents the PINN solution, showing smooth behavior at the center without numerical artifacts.)*

## Usage
To run the code:

```bash
pip install deepxde
python main.py
