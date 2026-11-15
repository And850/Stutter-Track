# Stutter Track

Per-speaker stuttering rate from audio. Entry for the Veritas AI International High School Hackathon 2026, AI + Health track (own idea). Solo entry by Andrii Ryzhanskyi.

**This project is a prototype and is not medical advice.**

## What it does

Stutter Track takes a recording of one speaker, cuts it into 3-second clips and gives each clip a probability of containing a stutter (block, prolongation, sound repetition or word repetition). The speaker's stuttering rate is the mean of those probabilities. The demo app also shows a timeline of the recording.

The intended user is a speech-language therapist who measures how much a client stutters, or a person who stutters and wants to follow their own progress between sessions.

## Links

- Notebook on Colab: https://colab.research.google.com/drive/1ZDl9j-8Yf4I9Fqh65iGtjiJljdkcfF-1
- Demo video: https://youtu.be/xixwQc0l-Tg

## Repository contents

| Path | What it is |
|---|---|
| `VeritasHackathon.ipynb` | Full pipeline with saved outputs: data and speaker split, MFCC baseline, frozen wav2vec2, calibration correction, fine-tuned wav2vec2, threshold choice, failure analysis |
| `demo/app.py` | Gradio demo app |
| `demo/requirements.txt`, `demo/packages.txt` | Dependencies of the app |
| `demo/test_clips.csv`, `demo/test_speakers.csv` | Saved test-set predictions for the 155 held-out speakers with at least 10 clips |

## Results

All results are on the test set: 6,011 clips from 242 speakers that were never used for training or for any choice. Speaker results cover the 155 test speakers with 10 or more clips.

| Model | AUROC | F1 | FN | FP | Speaker MAE | Spearman |
|---|---|---|---|---|---|---|
| MFCC + LR (baseline) | 0.533 | 0.413 | 1,227 | 1,443 | 0.213 | 0.199 |
| Frozen wav2vec2 + LR | 0.714 | 0.560 | 924 | 1,033 | 0.102 | 0.646 |
| Fine-tuned wav2vec2, threshold 0.30 | 0.802 | 0.655 | 749 | 744 | 0.083 | 0.749 |

For the fine-tuned row, speaker MAE and Spearman use the probability rate. For the other two rows they use the label rate.

The estimated rate of the fine-tuned model is off by about 8 points per speaker, against 21 points for the baseline.

## Limitations

- The output is the share of stuttered 3-second clips. It is not %SS and not an SSI-4 score.
- Silent blocks are missed most often (recall 0.520 on block-only clips, against 0.814 for sound repetitions).
- Speakers with a true stuttering rate of 0.4 to 0.6 are underestimated by 0.110 on average.
- Speaker IDs come from automatic clustering, so one ID may mix several people.
- 270 of 381 episodes have clips in more than one split. Speakers never overlap, but recording conditions can.
- The data is adult English podcast speech. The live demo on new recordings has been checked on one speaker only.

The full failure analysis is in the last section of the notebook.

## Reproducing the notebook

Open the notebook in Colab with a T4 GPU runtime and run all cells. The data is downloaded by the notebook.

The fine-tuned model is saved as `ft_best.pt`. This file is not in the repository because of its size. With `RETRAIN = False` the notebook loads it from the author's Google Drive. Without that file, training runs automatically (about 10 minutes on a T4 GPU) and saves a new `ft_best.pt`. Seeds are fixed at 42, but GPU training is not exactly repeatable, so a retrained model can differ slightly from the reported numbers.

## Running the demo

The app needs `ft_best.pt` in the same folder as `app.py`. Produce it with the fine-tuning section of the notebook, then copy it into `demo/`.

`requirements.txt` does not list Gradio, because the app was written for Hugging Face Spaces, which installs Gradio separately. Install it yourself: the app was built and tested with `gradio==6.32.0`.

In Colab, with the `demo` folder on Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

import sys
demo_dir = '/content/drive/MyDrive/stutter_demo'   # folder with app.py, the CSV files and ft_best.pt

!pip -q install gradio==6.32.0
sys.path.insert(0, demo_dir)

import app
print("Model loaded:", app.MODEL is not None, app.MODEL_ERROR)
app.launch(share=True)
```

On a local machine: `pip install -r requirements.txt gradio==6.32.0`, install ffmpeg, then `python app.py`.

The "Held-out test speakers" tab works without the model file, because it uses the saved predictions.

## Sources

- Lea et al. (2021). SEP-28k. [arXiv:2102.12394](https://arxiv.org/abs/2102.12394), [Apple repository](https://github.com/apple/ml-stuttering-events-dataset)
- Bayerl et al. (2022). SEP-28k-E speaker labels and split. [Repository](https://github.com/th-nuernberg/ml-stuttering-events-dataset-extended)
- Audio mirror: [saeedzou/sep28k-fluencybank-stutter-dataset](https://huggingface.co/datasets/saeedzou/sep28k-fluencybank-stutter-dataset) on Hugging Face (no licence stated)
- Baevski et al. (2020). wav2vec 2.0. [arXiv:2006.11477](https://arxiv.org/abs/2006.11477). Checkpoint [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base)
- librosa, scikit-learn, PyTorch, Hugging Face transformers and datasets, pandas, NumPy, SciPy, Gradio

## AI tools

Gemini in Google Colab (code editing). Claude (prompts, checking, human-assisted failure analysis, demo app code, first draft and outline of the write-up, this README).

## Acknowledgement

Platon Shchegolev, a former team member, researched how therapists measure stuttering today (%SS and the SSI-4).

## Licence

The code is under the MIT License (see `LICENSE`). The files `demo/test_clips.csv` and `demo/test_speakers.csv` contain labels derived from SEP-28k and SEP-28k-E and remain under the licences of those datasets (SEP-28k-E labels: CC BY-NC 4.0). Personal names of podcast speakers are replaced by neutral labels.
