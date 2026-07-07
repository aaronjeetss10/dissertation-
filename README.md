# SARTriage

**A post-mission triage pipeline for UAV search-and-rescue footage.**

SARTriage ingests raw drone video, detects and tracks people on the ground, classifies their posture, and produces a **priority-ranked event timeline** — ordered *lying > standing > running > walking* — so that operators review the highest-priority moments first instead of scrubbing through hours of recording.

It is delivered as a Dockerised Flask web application, evaluated against seven pre-registered acceptance criteria, and backed by a 63-test harness.
---

## The problem

When a UAV flies a search-and-rescue mission at operational altitude, the people it searches for are tiny. Across the evaluation footage, **96% of tracked persons occupy fewer than 50 pixels** on the shorter bounding-box dimension. At that scale, pixel-based action recognition fails *structurally*, not incidentally.

We confirm this empirically on Okutama-Action under perfect detection: five fine-tuned state-of-the-art video-action classifiers spanning every dominant paradigm — MViTv2-S, R3D-18, I3D, SlowFast, and VideoMAE — collapse uniformly to chance (21.6–27.3% accuracy, Cohen's κ ≤ 0.04). That uniform collapse is the signature of a regime limit, not a model-specific artefact.

**The response is a representational shift from pixels to bounding-box geometry.** The aspect ratio *w/h* is scale-invariant by construction and carries the posture signal that the pixels cannot.

---

## Pipeline overview

SARTriage is an 11-component pipeline (~21,600 lines) built around six novel contributions:

| # | Component | What it does | Headline result |
|---|-----------|--------------|-----------------|
| **C1** | **TMS-16** | 16-feature trajectory-motion-statistics posture classifier for sub-50 px targets | 51.5% four-way posture accuracy, casualty AUC 0.873 — **+24.2 pp** over the best pixel baseline |
| **C2** | **TrajMAE** | Masked trajectory autoencoder for representation learning | **+11.4 pp** |
| **C3** | **SCTE** | Scale-contrastive temporal embedding | **+8.2 pp** cross-scale |
| **C4** | **TCE** | Temporal Criticality Engine with dwell-duration escalation | 76.3% casualty Recall@3 |
| **C5** | **EMI** | Ego-Motion Intelligence — reinterprets drone motion as an attention signal rather than noise | — |
| **C6** | **AAI-v2** | Scale-adaptive fusion across trajectory and pixel streams | Correctly weights trajectories at 90% below 25 px |

Supporting these is a **centroid-distance tracker**, motivated by a closed-form IoU-collapse derivation (IoU ≈ 1 − 2d/w) that reframes a long-standing tracker-tuning problem as a geometric property of the matching metric. The project also reports the **first documented failure of ByteTrack at sub-25 px scales**.

---

## Results

- Surfaces casualties at **median rank 21 of 215** on Okutama-Action (**NDCG@3 = 0.612**).
- Clears operational recall on a DJI Neo demonstration benchmark captured below the scale wall.
- Runs at **~47.5 ms/frame on commodity CPU** — no GPU required for inference.
- Seven pre-registered acceptance criteria: **five pass**; two self-supervised alternatives are reported diagnostically as honest negatives.

---

## Datasets

| Dataset | Role | Scale |
|---------|------|-------|
| [Okutama-Action](http://okutama-action.org/) | Primary evaluation & training | 2,834 tracks |
| [VisDrone-MOT](https://github.com/VisDrone/VisDrone-Dataset) | Cross-dataset generalisation | 1,021 tracks |
| DJI Neo SAR-proxy set | Operational demonstration benchmark | Custom capture, two altitudes |

> **Note:** The DJI Neo evaluation clips and trained model weights are hosted externally (see [Setup](#setup)) rather than committed to the repository, owing to file-size limits.

---

## Setup

**Requirements:** Python 3.11, Docker (recommended). CUDA is *not* required — inference runs on CPU.

### Option A — Docker (recommended)

```bash
git clone https://github.com/<your-username>/sartriage.git
cd sartriage

docker build -t sartriage .
docker run -p 5000:5000 sartriage
```

Then open <http://localhost:5000> and upload a video to review.

### Option B — Local install

```bash
git clone https://github.com/<your-username>/sartriage.git
cd sartriage

python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Model weights & evaluation clips

The trained weights and the DJI Neo evaluation clip set are hosted externally:

- **Model weights:** `<add download link>` → place in `weights/`
- **DJI Neo clips:** `<add download link>` → place in `data/dji_neo/`

---

## Usage

### Run the web app

```bash
python app.py
# → serves the triage UI at http://localhost:5000
```

Upload a drone video; SARTriage returns a ranked timeline of events with each detection's posture, dwell duration, and criticality score.

### Reproduce the evaluation

```bash
# Run the full acceptance-criteria evaluation (H1–H8)
python -m evaluation.run_all

# Or a single hypothesis
python -m evaluation.run --hypothesis H1
```

### Run the test suite

```bash
pytest                 # all 63 tests
pytest -q tests/       # quiet mode
```

> The exact script names above (`app.py`, `evaluation.run_all`, etc.) are placeholders — swap them for your real entry points if they differ.

---

## Repository structure

```
sartriage/
├── app.py                  # Flask web application entry point
├── pipeline/               # The 11-component triage pipeline
│   ├── tms16/              # C1 — trajectory-motion-statistics classifier
│   ├── trajmae/            # C2 — masked trajectory autoencoder
│   ├── scte/               # C3 — scale-contrastive embedding
│   ├── tce/                # C4 — temporal criticality engine
│   ├── emi/                # C5 — ego-motion intelligence
│   ├── fusion/             # C6 — AAI-v2 scale-adaptive fusion
│   └── tracking/           # centroid-distance tracker
├── evaluation/             # Pre-registered hypothesis experiments
├── weights/                # Trained model weights (downloaded separately)
├── data/                   # Datasets (downloaded separately)
├── tests/                  # 63-test harness
├── Dockerfile
├── requirements.txt
└── README.md
```

*(Adjust to match your actual layout.)*

---

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{sandhu2026sartriage,
  author = {Aaron Sandhu},
  title  = {SARTriage: Trajectory-Centric Multi-Stream Triage of UAV Search-and-Rescue Footage},
  school = {University of Bath},
  year   = {2026}
}
```

---

## Acknowledgements

Supervised by Dr. Chen at the University of Bath. Thanks to the 20 SAR drone operators and emergency-response practitioners who completed the stakeholder questionnaire that shaped the system's design. The Okutama-Action dataset was provided by the NII, Tokyo; the VisDrone dataset by AISKYEYE, Tianjin University. Ethics reference: 14297-16768.

---

## License

Released under the MIT License — see [`LICENSE`](LICENSE). *(Pick whichever licence you prefer; MIT is a sensible default for a portfolio project.)*
