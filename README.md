# `gpt2-mlx`

A re-implementation of GPT-2 in Apple's new machine learning framework, [MLX](https://github.com/ml-explore/mlx)

Run OpenAI's 1.5 billion parameter model or train custom GPT-style models from scratch, all on your Mac GPU!

<p align="center">
  <img src="gpt2-mlx.gif" alt="GIF of GPT2-XL decoding">
  <br>
  <em>GPT-2 XL 1.5B real-time text generation on M1 Pro 16GB</em>
</p>

## Install

Use a device with Apple silicon

```shell
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

See the full GPT-2 neural network architecture, implemented in MLX, in [transformer.py](transformer.py)

## Run

Download the pre-trained GPT-2 model weights from [Hugging Face](https://huggingface.co/gpt2-xl)

Convert the PyTorch model weights to the MLX format
```shell
$ python convert_weights.py --weights_path="path/to/pytorch_model.bin" --model_name="gpt2-xl"
```

Generate text
```shell
$ python generate.py --model_name="gpt2-xl" --prompt "In a shocking finding, scientists discovered a herd of unicorns"
```

## Train

With `gpt2-mlx`, it is possible to train a custom GPT-style model on your own data

First, gather your training data and save it as a text file, i.e. `train.txt`

Run the following script to pre-process and tokenize the text data into a format compatible with the model

```shell
$ python prepare_data.py --data_path="path/to/train.txt"
```

Train the model, natively on your device

```shell
$ python train.py --data_path="path/to/train.npy" --checkpoint_dir="path/to/save/checkpoints"
```

The [training script](train.py) loads data in batches at each training step to avoid loading the entire dataset into memory

It also has an implementation of gradient accumulation, allowing you to train larger models with larger batch sizes than would otherwise fit in memory

To generate text with your custom trained model, use the same `generate.py` file, but with the checkpoint directory containing your custom model definition and weights

```shell
$ python generate.py --checkpoint_dir="path/to/checkpoints" --prompt="Hello world,"
```

