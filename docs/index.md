# Google Summer of Code 2023 Report

## The Project: Differential Programming in Material Point Method

Brief summary of  [the
project](https://summerofcode.withgoogle.com/programs/2023/projects/RZJ09HkH) is as follows:


In this project we are building a new library implementing the Material Point Method. The Material Point Method is a computational technique used in solid mechanics and fluid dynamics to simulate the behaviour of materials undergoing large deformation.  

The primary objectives of the library is to provide the support for backward differentiation which will help in solving inverse problems. The library is based on the [C++ Implementation](https://github.com/cb-geo/mpm) of the method. The C++ Implementation does not have support for inverse problem solving. The library is being developed in JAX to efficiently utilize modern GPUs to accelerate MPM simulations.  

## What is Differential Programming?  

Differentiable programming is a paradigm that allows us to seamlessly integrate traditional programming and differentiable calculus, opening up exciting opportunities for optimization, sensitivity analysis, and more. Unlike conventional programming, where code execution follows explicit instructions, differentiable programming enables the calculation of gradients of functions with respect to their inputs. This characteristic is particularly powerful in scenarios where optimization is key, as gradients provide crucial information about how small changes in inputs affect the output.  
Differential Programming has many applications in scientific computing, it is used to solve inverse problems, probabilistic programming. Differentiable programming creats a computational graph of the program and when the backward pass is called it calculates gradient at each node of the computational graph by chain rule. A sample computational graph:

![Computational Graph](https://blog.paperspace.com/content/images/2019/03/full_graph.png)
*Image Source: [Paperspace](https://blog.paperspace.com/pytorch-101-understanding-graphs-and-automatic-differentiation/)*

The gradient of the input is thus calculated with respect to the loss L and thus we can apply gradient based optimisation methods to minimise the loss and find the appropriate parameters.  

## What is Material Point Method?

Material Point Method (MPM) is a numerical method which is used in simulation of interaction of bodies under various conditions. This mathod is a particle based method that represents the material as a collection of material points. This method is highly effective in the context of large body deformations. There are various drawbacks of the Finite Element Method (FEM) which are overcome by MPM. The main drawback of FEM is that it is not much effective in large deformations due to mesh distortion, which is addressed by MPM by having a fixed mesh and moving the material points.  

The MPM can be divided into 4 major steps mapping the particles to the node, finding the solution at the nodal points, mapping the nodal solution back to the particles, updating the particles.

![Material Point Method](https://www.cb-geo.com/images/cb-geo/research/mpm/mpm-algorithm.png)
*Image Source: [CB-Geo](https://www.cb-geo.com/research/mpm/)*

## Inverse Problems and Applications

Inverse problems play a pivotal role in scientific computing by unraveling hidden information from observed data. These problems involve inferring the causes or parameters of a system based on its observable outcomes. In scientific computing, they find applications across diverse fields, such as medical imaging, geophysics, and engineering simulations.  

Using this library we can find the input material properties by iteratively updating the input material properties and minimizing the loss function which is the norm of the expected output and the actual output. Diff-MPM is a novel tool which can provide gradient information and can be used along with existing ML algorithms to generate optimization in robotics.

Following is an example of inverse problem solving using Diff-MPM: