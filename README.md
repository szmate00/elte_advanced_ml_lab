<img src="https://ttkhok.elte.hu/sites/default/files/mindentudas-egyeteme/elte_cimer_ff.jpg" height="180" />

# Advanced Machine Learning Lab - Spring 2021
Repository for my coursework in the "[Advanced Machine Learning Lab](https://csabaibio.github.io/advanced-ml-lab/)" course at ELTE<br>
<br>
**Subject code:** dsadvmachlf20lx<br>
**Credits:** 4<br>

## Project description
### [Hamiltonian neural networks](https://arxiv.org/abs/1906.01563)

*Abstract:* Even though neural networks enjoy widespread use, they still struggle to learn the basic laws of physics. How might we endow them with better inductive biases? In this paper, we draw inspiration from Hamiltonian mechanics to train models that learn and respect exact conservation laws in an unsupervised manner. We evaluate our models on problems where conservation of energy is important, including the two-body problem and pixel observations of a pendulum. Our model trains faster and generalizes better than a regular neural network. An interesting side effect is that our model is perfectly reversible in time.

* Read the above article and understand the basics of it.
* Take a simple Hamiltonian and run a number of simulations with it in order to generate training data [code required]
* Set up the training pipeline for the Hamiltonian neural net manually and run it on your data.
* Evaluate the predictions compared to the simple Euler method.
