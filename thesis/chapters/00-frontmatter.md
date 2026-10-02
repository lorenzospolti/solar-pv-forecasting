---
title: "Day-Ahead Photovoltaic Power Forecasting on Consumer Hardware: Time-Series Foundation Models versus Purpose-Trained Models"
author: "Lorenzo Spolti"
# TODO(Lorenzo): confirm the official wording of the degree/institution line for the cover.
institution: "Bachelor of Science in Artificial Intelligence — Università degli Studi di Milano, Università degli Studi di Milano-Bicocca, Università degli Studi di Pavia"
supervisor: "Prof. Claudio Cusano (Università degli Studi di Pavia)"
date: "December 2026"
bibliography: references.bib
link-citations: true
# Libertinus (in tectonic's bundle, loaded by file name): covers the symbols Latin Modern lacks (≈ ≥ ⁻²)
mainfont: LibertinusSerif-Regular.otf
mainfontoptions:
  - BoldFont=LibertinusSerif-Bold.otf
  - ItalicFont=LibertinusSerif-Italic.otf
  - BoldItalicFont=LibertinusSerif-BoldItalic.otf
sansfont: LibertinusSans-Regular.otf
mathfont: LibertinusMath-Regular.otf
# wide results tables (12 columns) need a smaller font than the 11 pt body
header-includes:
  - \usepackage{etoolbox}
  - \AtBeginEnvironment{longtable}{\small}
# pandoc-crossref (make manuscript): chapter-relative numbering, "Figure 3.1"
chapters: true
chaptersDepth: 1
figPrefix: ["Figure", "Figures"]
tblPrefix: ["Table", "Tables"]
secPrefix: ["Section", "Sections"]
eqnPrefix: ["Equation", "Equations"]
linkReferences: true
---

<!--
One file per chapter, concatenated in file-name order by `make manuscript`
(pandoc + pandoc-crossref + citeproc + tectonic -> thesis/build/thesis.pdf).
Chapters are level-1 headings; cross-reference with @sec:, @fig:, @tbl:, @eq:
labels; cite with [@key] from references.bib. Image paths are relative to thesis/.
Numbers, tables and figures come from thesis/tables/ and thesis/figures/
(regenerate with `make report`).
-->

# Abstract {.unnumbered}

*Scope note: 150–250 words. State the question, the dataset (HKUST rooftop PV,
2021–2023), the harness (one shared clear-sky-index pipeline), the model families
compared, the headline finding (do foundation models — zero-shot and fine-tuned —
match/beat purpose-trained models on day-ahead hourly PV power on an M4/16 GB), and
the compute cost. End with one sentence of significance.*

This thesis asks whether time-series **foundation models** (Chronos-2, TimesFM-2.5
and TiRex) — used **zero-shot** and, for Chronos-2, **fine-tuned** — can match or beat
purpose-trained models (SARIMA, LightGBM, LSTM) and reference forecasts (smart
persistence, CLIPER) for **day-ahead hourly photovoltaic (PV) power** on **consumer
hardware**, and at what compute cost. […]
