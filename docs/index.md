# Google Summer of Code 2023 Report

## The Project: Differential Programming in Material Point Method

Brief summary of  [the
project](https://summerofcode.withgoogle.com/programs/2023/projects/RZJ09HkH) is as follows:


In this project we are building a new library implementing the Material Point Method. The Material Point Method is a computational technique used in solid mechanics and fluid dynamics to simulate the behaviour of materials undergoing large deformation.  

The primary objectives of the library is to provide the support for backward differentiation which will help in solving inverse problems. The library is based on the [C++ Implementation](https://github.com/cb-geo/mpm) of the method. The C++ Implementation does not have support for inverse problem solving. The library is being developed in JAX to efficiently utilize modern GPUs to accelerate MPM simulations.  

## What is Differential Programming?  

Differentiable programming is a paradigm that allows us to seamlessly integrate traditional programming and differentiable calculus, opening up exciting opportunities for optimization, sensitivity analysis, and more. Unlike conventional programming, where code execution follows explicit instructions, differentiable programming enables the calculation of gradients of functions with respect to their inputs. This characteristic is particularly powerful in scenarios where optimization is key, as gradients provide crucial information about how small changes in inputs affect the output.
<!-- The Stingray project is a sub-organization of
[OpenAstronomy](https://openastronomy.org/), which, as given on its
landing page, is a collaboration between open-source astronomy and
astrophysics projects to share resources, ideas, and to improve code.

While the proposed milestones only included proof-of-concept and final
implementations of the Multitaper algorithm, a proof-of-concept
implementation of a recently showcased technique ([A. Springford et al.
(2020)](https://iopscience.iop.org/article/10.3847/1538-3881/ab7fa1)) which uses the Multitapering concept to improve the Lomb-Scargle
periodogram, a popular technique to obtain spectral estimates of time
series with uneven temporal sampling, was also proposed as an optional
milestone.

For exhaustive details of the project, such as motivation, improvements over other methods, references and more,
please refer the [project proposal](https://gist.github.com/dhruv9vats/fdbf193600cefdaf8c8e2f54bfd83083).

Given below is a brief summary of the Multitaper spectrum estimate:

<script src="https://gist.github.com/dhruv9vats/49b9d27ca313882723a4492471aa499c.js">
</script>

## The Work

Not only the was Multitaper algorithm successfully implemented and
merged, the Multitaper based Lomb-Scargle periodogram, of which only a
proof-of-concept was proposed, was also fully implemented, and merged,
extending on the prior implementation.

### Repositories

<https://github.com/StingraySoftware/stingray>

<https://github.com/StingraySoftware/notebooks>

### Milestones

- [X]  Proof-of-concept implementation of the Multitaper algorithm.

- [X]  Full implementation, integration of the algorithm into the
Stingray framework, including tests and relevant documentation.

- [X]  Proof-of-concept implementation of the Multitaper Lomb-Scargle
algorithm. **(Optional)**

- [x]  Full implementation, integration of the Multitaper Lomb-Scargle
algorithm into the Stingray framework, including tests and relevant
documentation. **(Unproposed)**

- [X]  Tutorial Notebook.

- [ ]  Support for overlapping (tapered) data segments for use in
averaged periodograms, like in the Welch method. **(Optional)** (The
project took a different direction, and work was being done to add
support for uneven time series)

### Pull Requests

| Pull Request                                                                                                                                          | Commit                                                                                                            | Status     | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ----------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Multitaper Periodogram using SciPy](https://github.com/StingraySoftware/stingray/pull/578)                                                           | [b053ebd](https://github.com/StingraySoftware/stingray/pull/578/commits/b053ebdbbca52229c04cb63463b83d0d21df499d) | ![badge](https://shields.io/badge/PR-Merged-blueviolet?style=for-the-badge&logo=appveyor)   | The core of the project, encapsulating almost all the work associated with the proposed milestones, including test and documentation.                                                                                                                                                                                                                                                                                                                                                                                                |
| [Extending the Multitapering concept to unevenly sampled time-series: Multitaper Lomb-Scargle](https://github.com/StingraySoftware/stingray/pull/584) | [9826db4](https://github.com/StingraySoftware/stingray/pull/584/commits/9826db4b4e18cd34ef2d519b33e10f77e4afd2b0) | <img width=200/>![badge](https://shields.io/badge/PR-Merged-blueviolet?style=for-the-badge&logo=appveyor)   | Contains the implementation of the Multitaper Lomb-Scargle, as derived from the [research paper](https://iopscience.iop.org/article/10.3847/1538-3881/ab7fa1), with tests and documentation. Is implemented using Astropy’s Lomb-Scargle routine, for fast computation.                                                                                                                                                                                                                                                                                                                             |
| [Extending the Multitapering concept to unevenly sampled time-series: Multitaper Lomb-Scargle](https://github.com/StingraySoftware/stingray/pull/583) | [40104c5](https://github.com/StingraySoftware/stingray/pull/583/commits/40104c5e42b86a9fe82819db186bfd5a187166d4) | ![badge](https://shields.io/badge/PR-Closed-red?style=for-the-badge&logo=appveyor)  | This, too, contains the implementation of the Multitaper Lomb-Scargle. The default brute-force way to calculate the Lomb-Scargle Periodogram runs in O(n^2) time complexity and is thus impractical for research purposes. Thus, this PR contained an O(n log(n)) implementation of the same, implemented in Stingray itself, which is presented in [Press & Rybicki (2012)](https://www.researchgate.net/publication/258561369_Fast_calculation_of_the_Lomb-Scargle_periodogram_using_nonequispaced_fast_Fourier_transforms). It was closed in favor of the PR above to minimize potential break points and use Astropy’s implementation instead, as Astropy was already a dependency. |
| [Proof-of-Concept Multitaper Periodogram Implementation](https://github.com/StingraySoftware/stingray/pull/548)                                       | [aef0f39](https://github.com/StingraySoftware/stingray/pull/548/commits/aef0f39c76f471698b53ba565150f6c0f3af130a) | ![badge](https://shields.io/badge/PR-Closed-red?style=for-the-badge&logo=appveyor)  | This contains the initial proof-of-concept implementation, which was a wrapper around another package called ‘nitime’. This PR was created before submitting the proposal for prototyping purposes. Final implementation uses SciPy for greater granularity and flexibility, and also to keep dependencies, thus, break points, at a minimum.               |
| [Multitaper example notebook](https://github.com/StingraySoftware/notebooks/pull/56)                                                                  | [ 4ea1b5d](https://github.com/dhruv9vats/notebooks/commit/4ea1b5d0df23483ad122465fc50888a203c9283b)               | ![badge](https://shields.io/badge/PR-In_Review-lightgreen?style=for-the-badge&logo=appveyor) | A jupyter notebook showcasing the new Multitaper method, while also giving an intuitive insight at the workings of the technique.                                                                                                                                                                                                                                                                                                                                                                                                    |


## The Result

This project, completed under the Google Summer of Code 2021 program,
successfully implemented, and integrated a spectral estimation technique
which hopefully proves to be a valuable tool in a time series analyst's
arsenal, accounting to is desirable noise properties.

Below is a jupyter notebook, showcasing the use of this technique:

<script src="https://gist.github.com/dhruv9vats/767ed9c110e8597150f6497e5f341f88.js"></script>

## Beyong GSoC, logical next steps

At the time of writing, support for time series with uneven temporal sampling (Lomb-Scargle)
for the `Powerspectrum` class is being worked upon, with possible extension
to the `Crossspectrum` class.

- [ ] Add support for Lomb-Scargle periodograms/uneven time-series to `Powerspectrum`
- [ ] Add support for Lomb-Scargle periodograms/uneven time-series to `Crossspectrum`

## Acknowledgements

I would like to thank the mentors, [Daniela Huppenkothen](https://github.com/dhuppenkothen)
and [Matteo Bachetti](https://github.com/matteobachetti), for their 
*incredible* support from day one, insightful in-depth reviews and 
so much more. I seriously could not have asked for anything more.

## Blogs

These are the blogs written during the GSoC '21 program

1.  [Google Summer of Code: My Introduction to
    OpenSource](https://dhruv9vats.medium.com/google-summer-of-code-8c89d5535bd6)

2.  [GSoC Progress Report? Almost
    Done!](https://dhruv9vats.medium.com/gsoc-progress-report-almost-done-6239f301b23)

3.  [A Month into
    GSoC](https://dhruv9vats.medium.com/a-month-into-gsoc-805d42b1b5ce)

4.  [Halfway into
    GSoC](https://dhruv9vats.medium.com/halfway-into-gsoc-b6f9ec014333)

5.  [A Glimpse into my GSoC
    project](https://dhruv9vats.medium.com/a-glimpse-into-my-gsoc-project-25c0fe3296dd)

## Profiles

GitHub: <https://github.com/dhruv9vats>

LinkedIn: <https://www.linkedin.com/in/dhruv9vats/>

Medium: <https://dhruv9vats.medium.com/> -->
