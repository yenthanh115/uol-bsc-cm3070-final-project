#import "@preview/cetz:0.3.4"

// Chronological Model Development and Evaluation Strategy.
// Shows the two-level temporal partitioning (80/20 chronological split, with
// expanding-window CV inside the training partition), how the three models are
// tuned and compared on the training data only, and how the single winner is
// retrained then evaluated exactly once on the strictly-later held-out test set.
// The whole figure is anchored on one guarantee (goal G2): every test record
// occurs after every training record, so no future information leaks back.

#set page(width: auto, height: auto, margin: 14pt)
#set text(font: "Arial", size: 10pt)

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // ---- palette ----
  let train-fill = rgb("#e1f5fe"); let train-stroke = rgb("#0288d1") // training / past
  let val-fill   = rgb("#c8e6c9"); let val-stroke   = rgb("#2e7d32") // validation
  let test-fill  = rgb("#ffcdd2"); let test-stroke  = rgb("#c62828") // held-out future test
  let cv-fill    = rgb("#fff9c4"); let cv-stroke    = rgb("#f9a825") // CV / tuning
  let axis-col   = rgb("#455a64")
  let now-col    = rgb("#e65100")

  set-style(mark: (end: "stealth", fill: axis-col, scale: 0.55))
  let arrow(a, b, ..s) = line(a, b, stroke: axis-col + 1.1pt, ..s)

  // ======================================================================
  // TITLE-LEVEL LAYOUT: a chronological time axis runs left -> right.
  // x maps a 0..100 "share of data by timestamp" onto the canvas.
  // ======================================================================
  let x(p) = p * 0.14         // 100 units -> 14 cm
  let splitP = 80             // 80/20 chronological cut

  // ---------- LEVEL 1: the 80/20 chronological split band ----------
  let bandTop = 0
  let bandBot = -1.5

  // training partition (earliest 80%)
  rect((x(0), bandBot), (x(splitP), bandTop),
       fill: train-fill, stroke: train-stroke + 1.4pt, radius: 3pt, name: "train")
  content((x(splitP / 2), (bandTop + bandBot) / 2), align(center)[
    #text(weight: "bold", size: 10pt, fill: train-stroke)[TRAINING PARTITION] \
    #text(size: 8pt)[earliest 80% by timestamp]
  ])

  // held-out test partition (final 20%)
  rect((x(splitP), bandBot), (x(100), bandTop),
       fill: test-fill, stroke: test-stroke + 1.4pt, radius: 3pt, name: "test")
  content((x((splitP + 100) / 2), (bandTop + bandBot) / 2), align(center)[
    #text(weight: "bold", size: 9.5pt, fill: test-stroke)[HELD-OUT TEST] \
    #text(size: 7.5pt)[final 20% - scored once]
  ])

  // chronological split marker
  line((x(splitP), bandTop + 0.85), (x(splitP), bandBot - 3.6),
       stroke: (paint: now-col, thickness: 1.5pt, dash: "dashed"))
  content((x(splitP), bandTop + 0.9), anchor: "south",
          text(size: 8pt, weight: "bold", fill: now-col)[80 / 20 split (by time)])

  // time axis under the band
  let ay = bandBot - 0.45
  line((x(0) - 0.3, ay), (x(100) + 0.6, ay), stroke: axis-col + 1.1pt)
  content((x(100) + 0.7, ay), anchor: "west",
          text(size: 8pt, style: "italic", fill: axis-col)[time →])
  content((x(0), ay - 0.28), anchor: "north", text(size: 7pt, fill: axis-col)[earliest])
  content((x(100), ay - 0.28), anchor: "north", text(size: 7pt, fill: axis-col)[latest])

  // "every test record is strictly later" guarantee
  content((x(splitP) - 4.5, bandBot - 4.15), anchor: "north", box(
    width: 8.5cm, inset: 5pt, radius: 3pt, fill: rgb("#c8e6c9"), stroke: val-stroke + 1pt,
    text(size: 7.5pt, fill: val-stroke)[
      #text(weight: "bold")[Guarantee (G2):] every test record occurs strictly after every training record, so no future information leaks into training or tuning.
    ]))

  // ---------- LEVEL 2: expanding-window CV inside the training band ----------
  // three expanding splits shown as stacked mini-bars beneath the training box.
  let cvTop = bandBot - 1.0
  let rowH  = 0.55
  let rowGap = 0.18
  let cvLabelX = x(0) - 0.35

  // vertical CV label to the left of the three split rows
  let cvRowsMid = cvTop - (1.5 * (rowH + rowGap)) + rowH / 2
  content((cvLabelX - 1.4, cvRowsMid),
          std.rotate(-90deg, reflow: true,
            align(center, text(size: 7.5pt, weight: "bold", fill: cv-stroke)[Expanding-window CV \ (k = 4 → 3 splits)])))

  // each split: train blocks (cv-fill) then one validation block (val-fill).
  // training partition spans 0..80; divide into 4 chronological blocks of 20.
  let blocks = (0, 20, 40, 60, 80)   // block boundaries within 0..80
  let splits = (
    (trainEnd: 1, val: 1),   // train B1, val B2
    (trainEnd: 2, val: 2),   // train B1-2, val B3
    (trainEnd: 3, val: 3),   // train B1-3, val B4
  )
  for (i, s) in splits.enumerate() {
    let ty = cvTop - i * (rowH + rowGap)
    // training portion
    rect((x(blocks.at(0)), ty - rowH), (x(blocks.at(s.trainEnd)), ty),
         fill: cv-fill, stroke: cv-stroke + 1pt, radius: 2pt)
    content((x(blocks.at(s.trainEnd) / 2), ty - rowH / 2),
            text(size: 7pt, fill: rgb("#8a6d00"))[train])
    // validation block
    rect((x(blocks.at(s.val)), ty - rowH), (x(blocks.at(s.val + 1)), ty),
         fill: val-fill, stroke: val-stroke + 1pt, radius: 2pt)
    content((x((blocks.at(s.val) + blocks.at(s.val + 1)) / 2), ty - rowH / 2),
            text(size: 7pt, weight: "bold", fill: val-stroke)[val])
    content((cvLabelX + 0.05, ty - rowH / 2), anchor: "east",
            text(size: 7pt, fill: axis-col)[Split #(i + 1)])
  }

  // ======================================================================
  // WORKFLOW STRIP (below): develop -> select -> retrain -> test -> analyse
  // ======================================================================
  let wY = bandBot - 5.6
  let boxW = 4.2
  let boxH = 2.0
  let gap  = 0.8
  let startX = x(0) - 4.5

  let steps = (
    (title: "1 · Develop",
     body: "Grid-search LR, RF, XGB;\nper-fold StandardScaler;\ncost-sensitive weighting",
     fill: train-fill, stroke: train-stroke),
    (title: "2 · Select + tune",
     body: "Pick best mean-val AUC;\nlock F1-optimised threshold\non last validation fold",
     fill: cv-fill, stroke: cv-stroke),
    (title: "3 · Retrain winner",
     body: "Refit chosen config on\nthe full 80% training\npartition (fresh scaler)",
     fill: train-fill, stroke: train-stroke),
    (title: "4 · Test once",
     body: "Score the strictly-later\n20% holdout with the\nlocked threshold",
     fill: test-fill, stroke: test-stroke),
    (title: "5 · Analyse",
     body: "AUC-ROC + P/R/F1;\n1000-boot CIs; McNemar;\nsingle-feature & transfer",
     fill: val-fill, stroke: val-stroke),
  )

  let sx(i) = startX + i * (boxW + gap)
  for (i, st) in steps.enumerate() {
    let lx = sx(i)
    rect((lx, wY - boxH), (lx + boxW, wY),
         fill: st.fill, stroke: st.stroke + 1.3pt, radius: 4pt, name: "st" + str(i))
    content((lx + boxW / 2, wY - 0.4), text(weight: "bold", size: 9pt, fill: st.stroke)[#st.title])
    content((lx + boxW / 2, wY - boxH / 2 - 0.25),
            align(center)[#text(size: 7.5pt)[#st.body]])
    if i > 0 {
      arrow((sx(i - 1) + boxW, wY - boxH / 2), (lx, wY - boxH / 2))
    }
  }

  // connect the data bands to the workflow: training band -> develop; test band -> test-once
  arrow((x(splitP / 2), bandBot), (sx(0) + boxW / 2, wY),
        stroke: (paint: train-stroke, thickness: 1pt, dash: "dotted"),
        mark: (end: "stealth", fill: train-stroke, scale: 0.5))
  arrow((x((splitP + 100) / 2), bandBot), (sx(3) + boxW / 2, wY),
        stroke: (paint: test-stroke, thickness: 1pt, dash: "dotted"),
        mark: (end: "stealth", fill: test-stroke, scale: 0.5))
})
