"""Stuttering rate estimator: demo app (Gradio, Hugging Face Spaces).

Tab 1 runs the fine-tuned wav2vec2 model live on an uploaded or recorded file.
Tab 2 shows the saved test-set predictions of the same model next to the
human annotations, for the 155 held-out speakers with at least 10 clips.

This project is a prototype and is not medical advice.
"""
import base64
import io
import os

import gradio as gr
import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from matplotlib.patches import Patch
from transformers import Wav2Vec2Config, Wav2Vec2Model

# ---------------------------------------------------------------- settings
SR = 16000                 # sample rate used in training
CLIP_SEC = 3               # the model was trained on 3-second clips
CLIP_LEN = SR * CLIP_SEC
THRESHOLD = 0.30           # chosen on the calibration split (best F1), not on test
MIN_LAST_SEC = 1.0         # a shorter leftover at the end is dropped
MAX_MINUTES = 10           # longer recordings are cut, to keep the demo fast
MIN_CLIPS = 10             # evaluation only covered speakers with >= 10 clips
SPEAKER_MAE = 0.083        # test result: mean absolute error of the speaker rate
CHECKPOINT = os.environ.get("CHECKPOINT", "ft_best.pt")
HERE = os.path.dirname(os.path.abspath(__file__))

BLUE, ORANGE, GREY, INK = "#2a78d6", "#eb6834", "#c9c8c2", "#0b0b0b"
DISCLAIMER = "**This project is a prototype and is not medical advice.**"


# ---------------------------------------------------------------- model
class Wav2Vec2FineTuned(nn.Module):
    """Same architecture as in the training notebook, so the weights fit."""

    def __init__(self):
        super().__init__()
        # Only the small config file is downloaded. The weights come from ft_best.pt.
        config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base")
        self.backbone = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(768, 1)

    def forward(self, input_values, true_lengths):
        last_hidden = self.backbone(input_values).last_hidden_state  # [B, T, 768]
        # Number of model frames that hold real audio (the rest is zero padding)
        frame_lengths = self.backbone._get_feat_extract_output_lengths(true_lengths)
        pooled = [last_hidden[i, :n].mean(dim=0) for i, n in enumerate(frame_lengths)]
        return self.head(self.dropout(torch.stack(pooled))).squeeze(-1)


MODEL = None
MODEL_ERROR = ""


def load_model():
    global MODEL, MODEL_ERROR
    path = os.path.join(HERE, CHECKPOINT)
    if not os.path.exists(path):
        MODEL_ERROR = f"Model file {CHECKPOINT} was not found in the Space."
        return
    try:
        model = Wav2Vec2FineTuned()
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)  # strict: fails loudly if the weights do not fit
        model.eval()
        MODEL = model
    except Exception as e:  # shown in the app instead of crashing it
        MODEL_ERROR = f"Model could not be loaded: {e}"


# ---------------------------------------------------------------- tab 1: live analysis
STEP_OVERLAP = 1.5  # seconds between window starts in overlapping mode


def make_windows(y, step_sec):
    """Cut audio into 3-second windows that start every step_sec seconds.
    With step 3 the windows are consecutive clips, as in the evaluation.
    With step 1.5 each part of the audio is seen by two windows, so a stutter
    that falls on a cut is also seen whole by the window centred on it."""
    step = int(step_sec * SR)
    clips, lengths, starts = [], [], []
    for start in range(0, len(y), step):
        part = y[start:start + CLIP_LEN]
        if len(part) < MIN_LAST_SEC * SR:
            break
        clips.append(np.pad(part, (0, CLIP_LEN - len(part))))
        lengths.append(len(part))
        starts.append(start / SR)
        if start + CLIP_LEN >= len(y):
            break
    return np.array(clips, dtype=np.float32), np.array(lengths), np.array(starts)


def find_speech(clips, lengths):
    """Simple loudness check: a clip much quieter than the loud parts counts as no speech.
    This is a heuristic added for the demo. It was not part of the evaluation."""
    rms = np.array([np.sqrt(np.mean(c[:n] ** 2)) for c, n in zip(clips, lengths)])
    return rms >= max(0.1 * np.percentile(rms, 90), 1e-4)


def predict(clips, lengths, batch_size=8):
    """Probability of a stutter for each clip. Raw 16 kHz audio goes in, as in training."""
    probs = []
    with torch.no_grad():
        for i in range(0, len(clips), batch_size):
            x = torch.from_numpy(clips[i:i + batch_size])
            n = torch.from_numpy(lengths[i:i + batch_size])
            probs.extend(torch.sigmoid(MODEL(x, n)).numpy())
    return np.array(probs)


def segments_for_display(probs, speech, starts, lengths, step_sec, total_sec):
    """Bars for the timeline. In consecutive mode one bar = one window.
    In overlapping mode one bar = one 1.5-second segment, showing the highest
    probability of the windows that cover it."""
    if step_sec >= CLIP_SEC:
        ends = starts + lengths / SR
        return starts, ends, probs, speech
    n_seg = int(np.ceil(total_sec / step_sec))
    seg_start = np.arange(n_seg) * step_sec
    seg_end = np.minimum(seg_start + step_sec, total_sec)
    value = np.zeros(n_seg)
    has_speech = np.zeros(n_seg, dtype=bool)
    for p, s, st, n in zip(probs, speech, starts, lengths):
        if not s:
            continue
        covered = (seg_start >= st - 1e-6) & (seg_end <= st + n / SR + 1e-6)
        value[covered] = np.maximum(value[covered], p)
        has_speech |= covered
    return seg_start, seg_end, value, has_speech


def fmt_time(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


def timeline_html(y, seg_start, seg_end, value, speech, mode_note):
    """Audio player plus an SVG timeline. The script in PLAYER_JS moves a cursor
    across the timeline and highlights the bar being played. Clicking a bar jumps there."""
    buf = io.BytesIO()
    sf.write(buf, y, SR, format="MP3")
    src = "data:audio/mpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    total = float(seg_end[-1])
    W, H, L, R, T, B = 1000, 250, 52, 12, 14, 42
    pw, ph = W - L - R, H - T - B
    x = lambda t: L + pw * t / total
    yv = lambda v: T + ph * (1 - v)

    parts = []
    for v in (0, 0.25, 0.5, 0.75, 1.0):  # grid and y labels
        parts.append(f'<line x1="{L}" x2="{W - R}" y1="{yv(v):.1f}" y2="{yv(v):.1f}" stroke="#e6e5e0"/>'
                     f'<text x="{L - 8}" y="{yv(v) + 4:.1f}" text-anchor="end" class="stl-ax">{v:.2f}</text>')
    tick = next(s for s in (5, 10, 15, 30, 60, 120) if total / s <= 12)
    for t in np.arange(0, total + 1e-6, tick):  # x labels
        parts.append(f'<text x="{x(t):.1f}" y="{H - B + 18}" text-anchor="middle" class="stl-ax">{fmt_time(t)[:-2]}</text>')
    for i, (a, b, v, s) in enumerate(zip(seg_start, seg_end, value, speech)):
        color = GREY if not s else (ORANGE if v >= THRESHOLD else BLUE)
        h = v if s else 0.03
        x0, x1 = x(a) + 1, x(b) - 1  # 2 px gap between bars
        label = "no speech" if not s else f"p = {v:.2f}"
        parts.append(f'<rect class="stl-bar" data-i="{i}" data-start="{a:.3f}" data-end="{b:.3f}" '
                     f'data-label="{label}" x="{x0:.1f}" y="{yv(h):.1f}" width="{max(x1 - x0, 1):.1f}" '
                     f'height="{yv(0) - yv(h):.1f}" fill="{color}" rx="2"><title>{fmt_time(a)} to {fmt_time(b)}, {label}</title></rect>')
    parts.append(f'<line x1="{L}" x2="{W - R}" y1="{yv(THRESHOLD):.1f}" y2="{yv(THRESHOLD):.1f}" '
                 f'stroke="{INK}" stroke-dasharray="6 4"/>')
    parts.append(f'<line class="stl-cursor" x1="{L}" x2="{L}" y1="{T}" y2="{yv(0)}" stroke="{INK}" stroke-width="2"/>')
    parts.append(f'<text x="{L + pw / 2}" y="{H - 4}" text-anchor="middle" class="stl-ax">'
                 f'Time in the recording (minutes:seconds). Click a bar to play from there.</text>')

    legend = "".join(
        f'<span><i style="background:{c}"></i>{t}</span>'
        for c, t in [(ORANGE, "Flagged as stutter"), (BLUE, "Not flagged"), (GREY, "No speech (skipped)")])
    legend += f'<span><i class="dash"></i>Threshold {THRESHOLD:.2f}</span>'

    return f"""
<div class="stl" data-total="{total:.3f}" data-left="{L}" data-width="{pw}">
  <div class="stl-title">Stutter timeline: {mode_note}</div>
  <audio class="stl-audio" controls preload="auto" src="{src}"></audio>
  <div class="stl-now">Press play: the current bar is outlined and the black line follows the audio.</div>
  <svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Stutter probability over time">{''.join(parts)}</svg>
  <div class="stl-legend">{legend}</div>
</div>"""


def analyse(audio_path, overlap):
    if MODEL is None:
        raise gr.Error(MODEL_ERROR or "Model is not loaded.")
    if not audio_path:
        raise gr.Error("Please upload or record audio first.")
    y, _ = librosa.load(audio_path, sr=SR, mono=True)  # resample to 16 kHz mono
    note = ""
    if len(y) > MAX_MINUTES * 60 * SR:
        y = y[:MAX_MINUTES * 60 * SR]
        note = f"\n\nOnly the first {MAX_MINUTES} minutes were analysed."
    step = STEP_OVERLAP if overlap else CLIP_SEC
    clips, lengths, starts = make_windows(y, step)
    if len(clips) == 0:
        raise gr.Error("The recording is shorter than 1 second.")
    speech = find_speech(clips, lengths)
    if speech.sum() == 0:
        raise gr.Error("No speech was found in the recording.")
    probs = predict(clips, lengths)

    n = int(speech.sum())
    rate = float(probs[speech].mean())                    # headline: mean probability
    n_flagged = int((probs[speech] >= THRESHOLD).sum())   # windows above the threshold
    low, high = max(rate - SPEAKER_MAE, 0), min(rate + SPEAKER_MAE, 1)
    unit = "overlapping 3-second windows" if overlap else "3-second clips"
    n_clip_equiv = n * step / CLIP_SEC                    # speech time in 3-second clips

    text = (
        f"## Estimated stuttering rate: {rate:.1%}\n"
        f"Share of 3-second clips that contain a stutter, estimated as the mean probability "
        f"over {n} {unit} with speech.\n\n"
        f"- Windows flagged at threshold {THRESHOLD:.2f}: **{n_flagged} of {n}** ({n_flagged / n:.1%})\n"
        f"- Typical error on unseen speakers in testing: about {SPEAKER_MAE * 100:.0f} percentage "
        f"points, so a plausible range is {low:.0%} to {high:.0%}\n"
        f"- Windows skipped as no speech: {int((~speech).sum())}"
    )
    if overlap:
        text += ("\n\nOverlapping mode is not how the model was evaluated (consecutive clips). "
                 "The reported rate may differ a little from consecutive mode.")
    if n_clip_equiv < MIN_CLIPS:
        text += (f"\n\n**Caution:** less than {MIN_CLIPS * CLIP_SEC} seconds of speech. "
                 f"The estimate was only tested on speakers with at least {MIN_CLIPS} clips "
                 f"and is unstable below that.")
    text += note + "\n\n" + DISCLAIMER

    seg = segments_for_display(probs, speech, starts, lengths, step, len(y) / SR)
    mode_note = ("one bar per 1.5 seconds, showing the highest probability of the two windows covering it"
                 if overlap else "one bar per 3-second clip")
    html = timeline_html(y, *seg, mode_note)

    table = pd.DataFrame({
        "Start (s)": np.round(starts, 1),
        "End (s)": np.round(starts + lengths / SR, 1),
        "Probability": np.round(probs, 3),
        "Result": np.where(~speech, "no speech (skipped)",
                           np.where(probs >= THRESHOLD, "flagged", "not flagged")),
    })
    return text, html, table


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


# Styles and script for the synced timeline. The listener sits on the document,
# so it keeps working each time Gradio replaces the timeline HTML.
PLAYER_HEAD = """
<style>
.stl {font-family: inherit;}
.stl-title {font-weight: 600; margin-bottom: 6px;}
.stl audio {width: 100%; margin-bottom: 4px;}
.stl-now {font-size: 14px; min-height: 20px; margin-bottom: 4px;}
.stl-ax {font-size: 12px; fill: #52514e;}
.stl-bar {cursor: pointer;}
.stl-bar.stl-cur {stroke: #0b0b0b; stroke-width: 2.5;}
.stl-legend {display: flex; gap: 18px; flex-wrap: wrap; font-size: 13px; margin-top: 2px;}
.stl-legend i {display: inline-block; width: 14px; height: 10px; margin-right: 6px; border-radius: 2px;}
.stl-legend i.dash {border-top: 2px dashed #0b0b0b; height: 0; border-radius: 0; vertical-align: middle;}
</style>
<script>
(function () {
  if (window.__stlReady) return;
  window.__stlReady = true;
  function fmt(t) { var m = Math.floor(t / 60), s = t - 60 * m; return m + ":" + (s < 10 ? "0" : "") + s.toFixed(1); }
  function update(audio) {
    var box = audio.closest(".stl"); if (!box) return;
    var t = audio.currentTime, total = parseFloat(box.dataset.total);
    var left = parseFloat(box.dataset.left), width = parseFloat(box.dataset.width);
    var cx = left + width * Math.min(t, total) / total;
    var cur = box.querySelector(".stl-cursor");
    cur.setAttribute("x1", cx); cur.setAttribute("x2", cx);
    var found = null;
    box.querySelectorAll(".stl-bar").forEach(function (b) {
      var on = t >= parseFloat(b.dataset.start) && t < parseFloat(b.dataset.end);
      b.classList.toggle("stl-cur", on); if (on) found = b;
    });
    var bars = box.querySelectorAll(".stl-bar").length;
    box.querySelector(".stl-now").textContent = found
      ? "Now " + fmt(t) + ": bar " + (+found.dataset.i + 1) + " of " + bars + ", " +
        fmt(+found.dataset.start) + " to " + fmt(+found.dataset.end) + ", " + found.dataset.label
      : "Now " + fmt(t);
  }
  ["timeupdate", "seeked", "play"].forEach(function (ev) {
    document.addEventListener(ev, function (e) {
      if (e.target && e.target.classList && e.target.classList.contains("stl-audio")) update(e.target);
    }, true);
  });
  // timeupdate fires only about 4 times a second; this keeps the cursor smooth while playing
  function tick() {
    document.querySelectorAll(".stl-audio").forEach(function (a) { if (!a.paused) update(a); });
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
  document.addEventListener("click", function (e) {
    var bar = e.target.closest && e.target.closest(".stl-bar"); if (!bar) return;
    var audio = bar.closest(".stl").querySelector(".stl-audio");
    audio.currentTime = parseFloat(bar.dataset.start) + 0.01;
    audio.play();
  });
})();
</script>
"""

# ---------------------------------------------------------------- tab 2: held-out test speakers
CLIPS = pd.read_csv(os.path.join(HERE, "test_clips.csv"))
SPEAKERS = pd.read_csv(os.path.join(HERE, "test_speakers.csv"))
DEFAULT_SPEAKER = "StrongVoices_21_1"


def speaker_view(speaker):
    row = SPEAKERS[SPEAKERS.speaker == speaker].iloc[0]
    c = CLIPS[CLIPS.speaker == speaker].reset_index(drop=True)
    pred = c.ft_prob >= THRESHOLD
    true = c.stutter == 1
    fn, fp = int((true & ~pred).sum()), int((~true & pred).sum())

    text = (
        f"## {speaker}: estimated {row.ft_prob_rate:.1%}, annotated {row.true_rate:.1%}\n"
        f"- Clips: {int(row.n_clips)} (this speaker was never used for training or for any choice)\n"
        f"- Error of the estimate: {(row.ft_prob_rate - row.true_rate) * 100:+.1f} percentage points\n"
        f"- At threshold {THRESHOLD:.2f}: {fn} missed stutters (false negatives), "
        f"{fp} false alarms (false positives)\n\n"
        f"These are saved predictions from the notebook, not a live run. " + DISCLAIMER
    )

    # Chart 1: this speaker's clips in recording order
    fig1, ax = plt.subplots(figsize=(10, 3.2))
    x = np.arange(len(c))
    ax.bar(x, c.ft_prob, width=0.85, color=np.where(pred, ORANGE, BLUE))
    ax.scatter(x[true], np.full(true.sum(), 1.04), marker="v", s=28, color=INK, clip_on=False)
    ax.axhline(THRESHOLD, color=INK, linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.6, len(c) - 0.4)
    ax.set_xlabel("Clips of this speaker, in recording order")
    ax.set_ylabel("Probability of a stutter")
    ax.set_title("Model output per clip, with the human annotation on top", loc="left", pad=16)
    ax.legend(handles=[Patch(color=ORANGE, label="Flagged by the model"),
                       Patch(color=BLUE, label="Not flagged"),
                       plt.Line2D([], [], marker="v", color=INK, linestyle="", label="Annotated as stutter"),
                       plt.Line2D([], [], color=INK, linestyle="--", label=f"Threshold {THRESHOLD:.2f}")],
              loc="upper center", bbox_to_anchor=(0.5, -0.24), ncols=4, frameon=False, fontsize=9)
    style_axes(ax)
    fig1.tight_layout()

    # Chart 2: all 155 test speakers, this one highlighted
    fig2, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="#8a897f", linestyle="--", linewidth=1)
    ax.scatter(SPEAKERS.true_rate, SPEAKERS.ft_prob_rate, s=22, color=BLUE, alpha=0.55,
               edgecolor="white", linewidth=0.5)
    ax.scatter([row.true_rate], [row.ft_prob_rate], s=110, color=ORANGE, edgecolor=INK,
               linewidth=1.2, zorder=3)
    ax.annotate("selected speaker", (row.true_rate, row.ft_prob_rate), xytext=(8, 8),
                textcoords="offset points", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.text(0.97, 0.93, "perfect estimate", ha="right", fontsize=9, color="#52514e", rotation=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Annotated rate (human labels)")
    ax.set_ylabel("Estimated rate (model)")
    ax.set_title("155 held-out speakers\nMAE 0.083, Spearman 0.749", loc="left", fontsize=11)
    style_axes(ax)
    fig2.tight_layout()

    table = pd.DataFrame({
        "Episode": c.EpId, "Clip": c.ClipId, "Probability": c.ft_prob.round(3),
        "Model": np.where(pred, "flagged", "not flagged"),
        "Annotation": np.where(true, "stutter", "no stutter"),
        "Outcome": np.select([true & pred, true & ~pred, ~true & pred],
                             ["correct", "missed stutter", "false alarm"], "correct"),
    })
    return text, fig1, fig2, table


# ---------------------------------------------------------------- interface
ABOUT = f"""
**What it does.** The recording is cut into 3-second clips. A fine-tuned wav2vec2 speech model
gives each clip a probability of containing a stutter (block, prolongation, sound repetition or
word repetition). The stuttering rate is the mean of these probabilities.

**Results on unseen speakers** (6,011 test clips, 242 speakers, used once):

| Model | Recall | Precision | AUROC | Missed stutters (FN) | False alarms (FP) |
|---|---|---|---|---|---|
| MFCC + logistic regression (baseline) | 0.434 | 0.395 | 0.533 | 1,227 | 1,443 |
| Fine-tuned wav2vec2, threshold 0.30 | 0.655 | 0.656 | 0.802 | 749 | 744 |

Speaker rate (155 speakers with at least 10 clips): mean absolute error 0.083, Spearman 0.749.
The baseline reaches 0.213 and 0.199.

**Known limits.**
- The number is the share of stuttered 3-second clips. It is not %SS and not an SSI-4 score.
- Silent blocks are missed most often (recall 0.520 on block-only clips).
- High rates are underestimated: speakers with a true rate of 0.4 to 0.6 by about 11 points.
- Trained on English podcast speech of adults (SEP-28k). Not tested on children, clinical
  recordings or other languages.
- Live recordings are cut every 3 seconds, while the training clips were selected by the
  dataset authors. The live rate has not been checked against human labels.
- The recording should contain one speaker. The no-speech filter is a simple loudness check.

Data: SEP-28k (Lea et al., 2021) with speaker labels from SEP-28k-E (Bayerl et al., 2022).
Model: facebook/wav2vec2-base (Baevski et al., 2020), fine-tuned.

{DISCLAIMER}
"""

with gr.Blocks(title="Stuttering rate estimator") as demo:
    gr.Markdown("# Stuttering rate estimator\n"
                "Estimates how much of a person's speech contains stuttering, for speech-language "
                "therapists and people who stutter who want to follow progress between sessions.\n\n"
                + DISCLAIMER)
    with gr.Tab("Analyse a recording"):
        with gr.Row():
            with gr.Column(scale=1):
                audio = gr.Audio(sources=["upload", "microphone"], type="filepath",
                                 label="Recording of one speaker (30 seconds or more)")
                run = gr.Button("Estimate stuttering rate", variant="primary")
                overlap = gr.Checkbox(value=False, label="Overlapping windows (a new 3-second window every 1.5 s)",
                                      info="Use for natural recordings, so a stutter on a cut is also seen whole. "
                                           "Leave off for files built from 3-second dataset clips.")
            with gr.Column(scale=2):
                out_text = gr.Markdown("Upload or record audio, then press the button.")
        out_html = gr.HTML(head=PLAYER_HEAD)
        with gr.Accordion("Table of clips", open=False):
            out_table = gr.Dataframe()
        run.click(analyse, inputs=[audio, overlap], outputs=[out_text, out_html, out_table])

    with gr.Tab("Held-out test speakers"):
        gr.Markdown("Speakers from the test set, which the model never heard in training. "
                    "The estimate is compared with the rate from human annotators.")
        choices = [(f"{s} ({n} clips)", s) for s, n in zip(SPEAKERS.speaker, SPEAKERS.n_clips)]
        pick = gr.Dropdown(choices=choices, value=DEFAULT_SPEAKER, label="Test speaker")
        with gr.Row():
            with gr.Column(scale=2):
                spk_text = gr.Markdown()
                spk_plot = gr.Plot(label="Clips")
            with gr.Column(scale=1):
                spk_scatter = gr.Plot(label="All test speakers")
        with gr.Accordion("Table of clips", open=False):
            spk_table = gr.Dataframe()
        outs = [spk_text, spk_plot, spk_scatter, spk_table]
        pick.change(speaker_view, inputs=pick, outputs=outs)
        demo.load(speaker_view, inputs=pick, outputs=outs)

    with gr.Tab("About and limits"):
        gr.Markdown(ABOUT)

load_model()



def launch(**kwargs):
    """Start the app. In Colab use app.launch(share=True). PLAYER_HEAD is passed here
    as well as to the HTML component, so the timeline script loads in every Gradio 6 setup."""
    return demo.launch(head=PLAYER_HEAD, **kwargs)


if __name__ == "__main__":
    launch()
