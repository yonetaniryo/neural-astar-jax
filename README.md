# Faster Neural A\* Implemented in Jax

This is a third-party implementation of [Neural A\* search [Yonetani+, ICML 2021]](https://github.com/omron-sinicx/neural-astar/).
I have reimplemented the Neural A* model and training scripts in [JAX](https://github.com/google/jax) and [Flax](https://github.com/google/flax), and expect an overall speedup thanks to JAX's jit compile and vmapping.

**The implementation is under active development;** the quantitative performance is still limited due to potential bugs and implementation differences between PyTorch and JAX.
