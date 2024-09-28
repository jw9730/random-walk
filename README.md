# Random Walk Neural Network (PyTorch)

[![arXiv](https://img.shields.io/badge/arXiv-2407.01214-b31b1b.svg)](https://arxiv.org/abs/2407.01214) \
**Revisiting Random Walks for Learning on Graphs** \
[Jinwoo Kim](https://jw9730.github.io), [Olga Zaghen*](http://bit.ly/olga-zaghen), [Ayhan Suleymanzade*](https://github.com/MisakiTaro0414), [Youngmin Ryou](https://www.linkedin.com/in/miinyou/), [Seunghoon Hong](https://maga33.github.io) (* equal contribution) \
ICML 2024 Workshop on Geometry-grounded Representation Learning and Generative Modeling

![image-random-walk](./docs/random-walk.png)

## Updates

Sep 28, 2024

- Released the code for random walks and their records, and DeBERTa experiments.

## Setup

Using ```Dockerfile``` (recommended)

```bash
git clone https://github.com/jw9730/random-walk.git /random-walk
cd random-walk
docker build --no-cache --tag rw:latest .
docker run -it --gpus all --ipc host --name rw -v /home:/home rw:latest bash
# upon completion, you should be at /rw inside the container
```

Using ```pip```

```bash
git clone https://github.com/jw9730/random-walk.git /random-walk
cd random-walk
bash install.sh
```

## Running Experiments

To try out random walks and their records, see the examples in the following files:

```bash
python3 test_walk_statistics.py
python3 test_walk_records.py
```

We will update the instructions for DeBERTa and llama 3 experiments soon.

## Trained Models

We will release the trained model checkpoints for DeBERTa experiments soon.

## References

Our implementation is based on code from the following repositories:

- [kerighan/graph-walker](https://github.com/kerighan/graph-walker) for random walks
- [ELENE](https://github.com/nur-ag/ELENE) for graph separation experiments
- [Homomorphism Expressivity](https://github.com/subgraph23/homomorphism-expressivity) for substructure counting experiments

## Citation

If you find our work useful, please consider citing it:

```bib
@article{kim2024revisiting,
  author    = {Jinwoo Kim and Olga Zaghen and Ayhan Suleymanzade and Youngmin Ryou and Seunghoon Hong},
  title     = {Revisiting Random Walks for Learning on Graphs},
  journal   = {arXiv},
  volume    = {abs/2407.01214},
  year      = {2024},
  url       = {https://arxiv.org/abs/2407.01214}
}
```
