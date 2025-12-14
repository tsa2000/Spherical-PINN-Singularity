# Spherical Diffusion PINN (Singularity Experiment)

**Author:** Thaer Abushawer

## Motivation
In the Single Particle Model (SPM) for batteries, the diffusion equation is defined in spherical coordinates. A known numerical issue arises at the center (r=0) due to the geometrical term, which often requires mesh refinement in standard solvers.

I wrote this code to test if a **Physics-Informed Neural Network (PINN)** could solve this PDE continuously from r=0 by incorporating the mathematical limit directly into the loss function.

## Mathematical Formulation
The governing PDE for spherical diffusion is:

    ∂C/∂t = D · [ ∂²C/∂r² + (2/r) · ∂C/∂r ]

At the center (r=0), the term (2/r) becomes undefined. I used **L'Hôpital's Rule** to find the limit:

    Limit (r→0):  (2/r) · ∂C/∂r  =  2 · ∂²C/∂r²

Substituting this back, the equation at the center simplifies to:

    ∂C/∂t = 3D · ∂²C/∂r²

## Implementation Details
* **Framework:** DeepXDE / TensorFlow
* **Geometry:** 1D Interval [0, 1] (starting exactly at 0).
* **Logic:** Used `tf.where` to switch between the standard equation (for r > 0) and the L'Hôpital form (for r ≈ 0).

## Results
The model was trained for 10,000 iterations. The plot below compares the PINN prediction against the expected analytical parabolic profile.

![Comparison Plot](result_plot.png)
*(The Green line represents the PINN solution, showing smooth behavior at the center without numerical artifacts.)*

## Usage
To run the code:

```bash
pip install deepxde
python main.py
