***Thanks to the [CompeteSMoE](https://github.com/giangdip2410/CompeteSMoE) team for providing the basis of this repo*** 

## Prerequisites

- pytorch
- fastmoe: https://github.com/laekov/fastmoe

##### Prepare Enwik8, Text8 and Wikitext-103 Datasets: 

- Download the enwik8, text8, wikitext-103 dataset from [here](https://github.com/laekov/fastmoe/blob/master/examples/transformer-xl/scripts/getdata.sh), then change bash scripts based on your local data paths.
```bash
data_folder/
└── pretraining
    └── enwik8
        ├── test.txt
        ├── train.txt
        └── valid.txt
    └── text8
        ├── test.txt
        ├── train.txt
        └── valid.txt
    └── wikitext-103
        ├── test.txt
        ├── train.txt
        └── valid.txt
```

- Select the SwitchTransformers or GLaM  architecture, its scale, and the type of SMoE layer. We support:

|                     | SMoE | SMoE-Dropout | XMoE | StableMoE | CompeteSMoE |
|---------------------|------|--------------|------|-----------|-------------|
| Transformer (S/M)   |  ✅  |     ✅       |  ✅  |     ✅    |      ✅     |
| GLaM (S/M)          |  ✅  |     ✅       |  ✅  |     ✅    |      ✅     |


##### Pretraining SwitchTransformers on Enwik8, Text8 and Wikitext-103: 

``` # Enwik8 dataset:
bash scripts/pretraining/enwik8/transformers/small/smoe-s.sh
bash scripts/pretraining/enwik8/transformers/small/smoe-dropout-s.sh 
bash scripts/pretraining/enwik8/transformers/small/xmoe-s.sh
bash scripts/pretraining/enwik8/transformers/small/stablemoe-s.sh    
bash scripts/pretraining/enwik8/transformers/small/competesmoe-s.sh 
```

``` # Text8 dataset: 
bash scripts/pretraining/text8/transformers/small/smoe-s.sh
bash scripts/pretraining/text8/transformers/small/smoe-dropout-s.sh 
bash scripts/pretraining/text8/transformers/small/xmoe-s.sh
bash scripts/pretraining/text8/transformers/small/stablemoe-s.sh    
bash scripts/pretraining/text8/transformers/small/competesmoe-s.sh 
```


``` # Wikitext103 dataset: 
bash scripts/pretraining/wikitext-103/transformers/small/smoe-s.sh
bash scripts/pretraining/wikitext-103/transformers/small/smoe-dropout-s.sh 
bash scripts/pretraining/wikitext-103/transformers/small/xmoe-s.sh
bash scripts/pretraining/wikitext-103/transformers/small/stablemoe-s.sh    
bash scripts/pretraining/wikitext-103/transformers/small/competesmoe-s.sh 
```

##### Pretraining GLaM on Enwik8, Text8 and Wikitext-103: 

``` # Enwik8 dataset:
bash scripts/pretraining/enwik8/glam/small/smoe-s.sh
bash scripts/pretraining/enwik8/glam/small/smoe-dropout-s.sh 
bash scripts/pretraining/enwik8/glam/small/xmoe-s.sh
bash scripts/pretraining/enwik8/glam/small/stablemoe-s.sh    
bash scripts/pretraining/enwik8/glam/small/competesmoe-s.sh 
```

``` # Text8 dataset: 
bash scripts/pretraining/text8/glam/small/smoe-s.sh
bash scripts/pretraining/text8/glam/small/smoe-dropout-s.sh 
bash scripts/pretraining/text8/glam/small/xmoe-s.sh
bash scripts/pretraining/text8/glam/small/stablemoe-s.sh    
bash scripts/pretraining/text8/glam/small/competesmoe-s.sh 
```


``` # Wikitext103 dataset: 
bash scripts/pretraining/wikitext-103/glam/small/smoe-s.sh
bash scripts/pretraining/wikitext-103/glam/small/smoe-dropout-s.sh 
bash scripts/pretraining/wikitext-103/glam/small/xmoe-s.sh
bash scripts/pretraining/wikitext-103/glam/small/stablemoe-s.sh    
bash scripts/pretraining/wikitext-103/glam/small/competesmoe-s.sh 
```



##### Fine-tuning on BANKING77, IMDb, SST-2, and SST-5:

```
bash scripts/finetuning/banking77/transformers/small/competesmoe-s.sh 
bash scripts/finetuning/imdb/transformers/small/competesmoe-s.sh 
bash scripts/finetuning/sst2/transformers/small/competesmoe-s.sh 
bash scripts/finetuning/sst5/transformers/small/competesmoe-s.sh 
```


