# NesTools: A Dataset for Evaluating Nested Tool Learning Abilities of Large Language Models

Code and data for our paper: [NesTools: A Dataset for Evaluating Nested Tool Learning Abilities of Large Language Models](https://arxiv.org/abs/2410.11805)

## 🔨 Preparations

```bash
$ git clone https://github.com/hhan1018/NesTools.git
$ cd NesTools
$ pip install -r requirements.txt
```

##   Get started

### data construction
Our test data can be found in data/test_data.jsonl.

If you want to experience our data construction method, please follow the steps:
Set your api key and url in data_construction/settings.py. You can change the ICL example to satisfy your taste.
```bash
python data_construction/main.py --refine
```

### build evaluation settings
Our test prompt can be found in inference/test_prompt.jsonl, which can be used for evaluation directly or as a reference.

Downloading [gte-large-en-v1.5](https://huggingface.co/Alibaba-NLP/
gte-large-en-v1.5) or any other retrievers.

```bash
bash build_test_prompt.sh
```

### Evaluate

```bash
bash build_test_prompt.sh
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
