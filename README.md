Audio Deepfake Detection with XLS-R and SLS classfier
===============
This repository contains our implementation of the paper ["Audio Deepfake Detection with XLS-R and SLS classfier  Qishan Zhang, Shuangbing Wen, Tao Hu ACM MM 2024"]（https://openreview.net/pdf?id=acJMIXJg2u)



## Installation
First, clone the repository locally, create and activate a conda environment, and install the requirements :
```
$ git clone https://github.com/QiShanZhang/SLSforASVspoof-2021-DF.git
$ cd SLSforASVspoof-2021-DF
$ unzip fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1.zip
$ conda create -n SLS python=3.7
$ conda activate SLS
$ pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
$ cd fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
$ pip install --editable ./
$ cd ..
$ pip install -r requirements.txt
```

### Local setup used in this workspace
This workspace has no `conda` command available, so the current reproducible environment was prepared with the existing `venv`:
```
unzip -n fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1.zip
venv/bin/python -m pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
venv/bin/python -m pip install -r requirements.txt
venv/bin/python -m pip install --editable ./fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
wget -O xlsr2_300m.pt https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt
```

Verified locally:
```
venv/bin/python main.py --help
venv/bin/python -c "import fairseq; model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(['xlsr2_300m.pt']); print(type(model[0]).__name__)"
venv/bin/python -c "import argparse, torch; from model import Model; args=argparse.Namespace(xlsr_checkpoint='xlsr2_300m.pt'); model=Model(args,'cpu').eval(); print(tuple(model(torch.zeros(1,64600)).shape))"
```

Current machine note: `torch.cuda.is_available()` is `False` and `nvidia-smi` cannot communicate with the NVIDIA driver, so full training should be run after GPU access is fixed or on another CUDA-ready machine.


## Experiments

### Dataset
Our experiments are performed on the public dataset logical access (LA) and deepfake (DF) partition of the ASVspoof 2021 dataset and In-the-Wild dataset(train on 2019 LA training and evaluate on 2021 LA and DF, In-the-Wild evaluation database).

The ASVspoof 2019 dataset, which can can be downloaded from [here](https://datashare.is.ed.ac.uk/handle/10283/3336).

The ASVspoof 2021 database is released on the zenodo site.

LA [here](https://zenodo.org/record/4837263#.YnDIinYzZhE)

DF [here](https://zenodo.org/record/4835108#.YnDIb3YzZhE)

The In-the-Wild dataset can be downloaded from [here](https://deepfake-total.com/in_the_wild)

For ASVspoof 2021 dataset keys (labels) and metadata are available [here](https://www.asvspoof.org/index2021.html)

## Pre-trained wav2vec 2.0 XLS-R (300M)
Download the XLS-R models from [here](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec/xlsr)

## Training model
To train the model run:
```
CUDA_VISIBLE_DEVICES=0 python main.py --track=DF --lr=0.000001 --batch_size=5 --loss=WCE  --num_epochs=50
```
## Testing

To evaluate your own model on the DF, LA, and In-the-Wild evaluation datasets: The code below will generate three 'score.txt' files, one for each evaluation dataset, and these files will be used to compute the EER(%).
```
CUDA_VISIBLE_DEVICES=0 python main.py   --track=DF --is_eval --eval 
                                        --model_path=/path/to/your/best_model.pth
                                        --protocols_path=database/ASVspoof_DF_cm_protocols/ASVspoof2021.DF.cm.eval.trl.txt 
                                        --database_path=/path/to/your/ASVspoof2021_DF_eval/ 
                                        --eval_output=/path/to/your/scores_DF.txt

CUDA_VISIBLE_DEVICES=0 python main.py   --track=LA --is_eval --eval 
                                        --model_path=/path/to/your/best_model.pth
                                        --protocols_path=database/ASVspoof_DF_cm_protocols/ASVspoof2021.LA.cm.eval.trl.txt 
                                        --database_path=/path/to/your/ASVspoof2021_LA_eval/ 
                                        --eval_output=/path/to/your/scores_LA.txt

CUDA_VISIBLE_DEVICES=0 python main.py   --track=In-the-Wild --is_eval --eval 
                                        --model_path=/path/to/your/best_model.pth
                                        --protocols_path=database/ASVspoof_DF_cm_protocols/in_the_wild.eval.txt 
                                        --database_path=/path/to/your/release_in_the_wild/ 
                                        --eval_output=/path/to/your/scores_In-the-Wild.txt
```
We also provide a pre-trained model. To use it, you can download from [here](https://drive.google.com/drive/folders/13vw_AX1jHdYndRu1edlgpdNJpCX8OnrH?usp=sharing) and change the --model_path to our pre-trained model.

[Here](https://pan.baidu.com/s/1dj-hjvf3fFPIYdtHWqtCmg?pwd=shan) is the baidu download link.

### Local DF subset test
The local workspace is configured with:
```
xlsr2_300m.pt
MMpaper_model.pth
data/ASVspoof2021_DF_eval/flac/
database/ASVspoof_DF_cm_protocols/ASVspoof2021.DF.cm.eval.tiny10.trl.txt
database/ASVspoof_DF_cm_protocols/ASVspoof2021.DF.cm.eval.subset.trl.txt
```

Run a 10-file smoke test first:
```
source venv/bin/activate
bash scripts/eval_df_tiny10.sh
```

Then run the available DF subset:
```
bash scripts/eval_df_subset.sh
```

Run a 5000-file subset:
```
bash scripts/eval_df_5000.sh
bash scripts/eer_df_5000.sh
```

Run a 20000-file subset:
```
bash scripts/eval_df_20000.sh
bash scripts/eer_df_20000.sh
```

For a 4GB GPU, keep `EVAL_BATCH_SIZE=1`. If memory allows, try:
```
EVAL_BATCH_SIZE=2 bash scripts/eval_df_subset.sh
```

### Local LLM web assistant

The project includes a lightweight DeepSeek-backed web assistant for reproduction summaries and experiment Q&A.

Configure `.env` first:
```
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```

Then run:
```
source venv/bin/activate
bash scripts/run_llm_web.sh
```

Open:
```
http://127.0.0.1:7860
```

If port 7860 is already in use, the server will automatically try the next available port and print the final URL. You can also choose a port manually:
```
PORT=7861 bash scripts/run_llm_web.sh
```

Compute the EER(%) use three 'scores.txt' file
```
python evaluate_2021_DF.py scores/scores_DF.txt ./keys eval

python evaluate_2021_LA.py scores/scores_LA.txt ./keys eval

python evaluate_in_the_wild.py scores/scores_Wild.txt ./keys eval
``` 

## Results using pre-trained model:
EER: 1.92 % on ASVspoof 2021 DF dataset.
EER: 2.87 % on ASVspoof 2021 LA dataset.
EER: 7.46 % on In-the-Wild dataset.


