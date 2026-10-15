# The Well Baselines

Tuned Fourier Neural Operator (FNO) baselines for [The Well](https://github.com/PolymathicAI/the_well). Each of 17 datasets gets its own FNO configuration, with Tucker-factorized spectral weights and a 12-hour training budget, to give stronger FNO baselines than those originally reported with The Well. Trained checkpoints for every dataset are released on Hugging Face.

## Install

Python 3.10 to 3.12

```bash
conda create -n well python=3.10 && conda activate well
pip install -r requirements.txt
```

The FNO implementation comes from [neuraloperator](https://github.com/neuraloperator/neuraloperator). Clone it into the repository root, where the scripts add it to `sys.path` at runtime. The released checkpoints were trained with commit `53c1817`:

```bash
git clone https://github.com/neuraloperator/neuraloperator.git
git -C neuraloperator checkout 53c1817bbe28d48996a0aa4976f9dd9bd0d3fcc2
```

## Data

Datasets are read from `WELL_DATASETS_PATH`, which defaults to `./datasets`. From the repository root, this downloads every split of a dataset, plus its normalization statistics, into that default location:

```bash
python dataloaders/download.py --well_dataset_name shear_flow --root_dir .
```
For example,
```
datasets/shear_flow/
  stats.yaml
  data/{train,valid,test}/*.hdf5
```

To keep the data elsewhere, pass `--root_dir /path/to/well` and `export WELL_DATASETS_PATH=/path/to/well/datasets`. Evaluation only reads the test split which The Well's own downloader can fetch by itself:

```bash
the-well-download --base-path . --dataset shear_flow --split test
```
## Evaluating a released checkpoint

Checkpoints for all 17 datasets are at
[huggingface.co/yashpincha/TheWellBaselines](https://huggingface.co/yashpincha/TheWellBaselines/tree/main/ckpts),
one `ckpts/<dataset>.pt` per dataset. To fetch one and compute its single-step VRMSE on the test split:

```bash
mkdir -p ckpts
curl -L -o ckpts/shear_flow.pt \
  https://huggingface.co/yashpincha/TheWellBaselines/resolve/main/ckpts/shear_flow.pt
python evaluate.py --ckpt ckpts/shear_flow.pt
```

Every checkpoint records the config it was trained with under `args`.

```bash
python evaluate.py --ckpt ckpts/planetswe.pt
```

VRMSE is a mean of per-batch means, so `--batch-size` shifts the number slightly; it defaults to the batch size each model was trained with.

## Training
`train.py` trains one FNO on one dataset. This command uses the configuration of the released `shear_flow` checkpoint:

```bash
python train.py \
  --dataset shear_flow \
  --n_layers 6 --modes 64 --width 32 \
  --factorization Tucker --rank 0.1 --domain_padding 0.1 \
  --normalize 1 --batch_size 64 --lr 0.0005 --patience 8 \
  --max_train_hours 12 --eval_at_end_only 1
```
Each run writes to `runs/<run_name>/` (we change it with `--log_dir`): `<run_name>__last.pt` after every epoch, and `<run_name>__best.pt` whenever an evaluation improves the single-step test VRMSE. Either file can be passed to `evaluate.py`. Metrics go to Weights & Biases, so run `wandb login` or set `WANDB_API_KEY` first, or pass `--debug_mode 1` to train without logging.


## Layout

```
train.py          training entry point
evaluate.py       single-step VRMSE of one checkpoint on a test split
dataloaders/
  metadata.py     per-dataset paths, grid size, channels (DATASET_DICT, CHANNELS)
  dataset.py      WellHDF5Dataset: HDF5 reading and normalization
  download.py     fetch a dataset from The Well
neuraloperator/   cloned FNO implementation (see Install)
```

## Authors

Originally created by [Andy Dimnaku](https://github.com/adimnaku) and
[Andrew Chiang](https://github.com/achiang17).

## Citation

If you use these baselines, please cite:

```bibtex
@misc{pincha2026performance,
  title={Performance of Neural Operator Baselines on The Well Datasets},
  author={Pincha, Yash and Dimnaku, Andy and Chiang, Andrew and Ganeshram, Adarsh and Duruisseaux, Valentin and Anandkumar, Anima},
  year={2026}
}
```

and The Well:

```bibtex
@article{ohana2024well,
  title={The well: a large-scale collection of diverse physics simulations for machine learning},
  author={Ohana, Ruben and McCabe, Michael and Meyer, Lucas and Morel, Rudy and Agocs, Fruzsina and Beneitez, Miguel and Berger, Marsha and Burkhart, Blakesly and Dalziel, Stuart and Fielding, Drummond and others},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={44989--45037},
  year={2024}
}
```

## License

[neuraloperator](https://github.com/neuraloperator/neuraloperator) is MIT licensed. It is cloned at install time, not redistributed here.
