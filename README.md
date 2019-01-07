
## 中文ULMFiT
[Universal Language Model Fine-tuning for Text Classification
](https://arxiv.org/abs/1801.06146)

[下载预训练的模型](https://drive.google.com/open?id=1Z9b1gVqfFjPaEEuU0Y-XfgsnmHr9yB_m)


创建虚拟环境（可以[配置清华conda源](https://mirror.tuna.tsinghua.edu.cn/help/anaconda/)）  
```bash
conda env create -f env.yml
```

解压中文维基百科语料
```bash
python -m gensim.scripts.segment_wiki -i -f /data/zhwiki-latest-pages-articles.xml.bz2 -o tmp/wiki2018-11-14.json.gz
```

分词维基百科语料
```bash
python preprocessing.py segment-wiki --input_file=tmp/wiki2018-11-14.json.gz --output_file=tmp/wiki2018-11-14.words.pkl
```

分词领域语料
```bash
python preprocessing.py segment-csv --input_file=data/ch_auto.csv --output_file=tmp/ch_auto.words.pkl --label_file=tmp/ch_auto.labels.npy
```

tokenize维基百科语料
```bash
python preprocessing.py tokenize --input_file=tmp/wiki2018-11-14.words.pkl --output_file=tmp/wiki2018-11-14.ids.npy --mapping_file=tmp/wiki2018-11-14.mapping.pkl
```

tokenize领域语料
```bash
python preprocessing.py tokenize --input_file=tmp/ch_auto.words.pkl --output_file=tmp/ch_auto.ids.npy --mapping_file=tmp/ch_auto.mapping.pkl
```

预训练
```bash
python pretraining.py --input_file=tmp/wiki2018-11-14.ids.npy --mapping_file=tmp/wiki2018-11-14.mapping.pkl --dir_path=tmp
```

微调
```bash
python finetuning.py --input_file=tmp/ch_auto.ids.npy --mapping_file=tmp/ch_auto.mapping.pkl --pretrain_model_file=tmp/models/wiki2018-11-14.h5 --pretrain_mapping_file=tmp/wiki2018-11-14.mapping.pkl --dir_path=tmp --model_id=ch_auto
```

训练分类器
```bash
python3 train_classifier.py  --id_file=tmp/ch_auto.ids.npy --label_file=tmp/ch_auto.labels.npy --mapping_file=tmp/ch_auto.mapping.pkl  --encoder_file=ch_auto_enc
```

测试
```bash
python3 predicting.py --mapping_file=tmp/ch_auto.mapping.pkl --classifier_filename=tmp/models/classifier_1.h5 --num_class=2
```
