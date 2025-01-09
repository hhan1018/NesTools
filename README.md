# NesTools: A Dataset for Evaluating Nested Tool Learning Abilities of Large Language Models

Code and data for our paper: [NesTools: A Dataset for Evaluating Nested Tool Learning Abilities of Large Language Models](https://arxiv.org/abs/2410.11805)

## News

- **[2025.01.09]** Release the code for inference and evaluation.
- **[2025.01.08]** Release the data and code for data construction.
- **[2025.01.07]** Paper available on [ArXiv](https://arxiv.org/abs/2410.11805).


## 🔨 Preparations

```bash
$ git clone https://github.com/hhan1018/NesTools.git
$ cd NesTools
$ pip install -r requirements.txt
```

## 🍰 Get started

Our test data can be found in `data/test_data.jsonl`.

### Data construction

If you want to experience our data construction method, please follow the steps:
1. Set your api key and url in `data_construction/settings.py`. 
Meanwhile, you can change the ICL examples to satisfy your taste in `data_construction/settings.py`.
2. Start the data construction:

```bash
python data_construction/main.py --refine
```

### Build evaluation settings
Our test prompt can be found in `inference/test_prompt.jsonl`, which can be used for evaluation directly or as a reference.

Downloading [gte-large-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5) or other embedding models.

```bash
bash scripts/build.sh
```

### Inference
1. Set your api key and url in `scripts/inference.sh`.
2. Start the Inference process:

```bash
bash scripts/inference.sh
```

### Evaluation

```bash
bash scripts/eval.sh
```



## 📝 Citation

If you find our work useful in your research, please cite our work:
```
@article{han2024nestools,
  title={NesTools: A Dataset for Evaluating Nested Tool Learning Abilities of Large Language Models},
  author={Han, Han and Zhu, Tong and Zhang, Xiang and Wu, Mengsong and Xiong, Hao and Chen, Wenliang},
  journal={arXiv preprint arXiv:2410.11805},
  year={2024}
}
```
