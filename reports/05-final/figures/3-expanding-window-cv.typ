#import "@preview/cetz:0.3.4"

// Expanding-Window Temporal Cross-Validation.
// Training window grows each fold while the validation block moves forward
// in time, so no fold ever validates on data earlier than it trained on.

#set page(width: auto, height: auto, margin: 12pt)
#set text(font: "Arial", size: 10pt)

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // ---- palette (mirrors the Mermaid theme) ----
  let train-fill   = rgb("#e1f5fe"); let train-stroke = rgb("#0288d1") // done/train
  let val-fill     = rgb("#c8e6c9"); let val-stroke   = rgb("#2e7d32") // active/val
  let axis-col     = rgb("#9e9e9e")

  // ---- folds: (train_start, train_end, val_start, val_end) on a 0..100 time axis ----
  let folds = (
    (train: (0, 25), val: (25, 50)),
    (train: (0, 50), val: (50, 75)),
    (train: (0, 75), val: (75, 100)),
  )

  // ---- geometry ----
  let scale = 0.10          // 100 time units -> 10 cm
  let bh    = 0.9           // bar height
  let gap   = 0.6           // gap between fold rows
  let row   = bh + gap
  let x(t)  = t * scale     // map time -> x
  let n     = folds.len()

  let top-y(i) = -i * row              // top edge of fold i's row
  let ctr-y(i) = top-y(i) - bh / 2

  // ---- title ----
  content((x(50), row), anchor: "south",
          text(weight: "bold", size: 12pt)[Expanding-Window Temporal CV (k = 3)])

  // ---- light vertical grid at each 25-unit boundary ----
  for g in (0, 25, 50, 75, 100) {
    line((x(g), row - 0.35), (x(g), top-y(n - 1) - bh - 0.15),
         stroke: (paint: axis-col, thickness: 0.5pt, dash: "dotted"))
  }

  // ---- bars per fold ----
  for (i, f) in folds.enumerate() {
    let ty = top-y(i)
    // row label
    content((x(0) - 0.4, ty - bh / 2), anchor: "east",
            text(weight: "bold", size: 9.5pt)[Fold #(i + 1)])

    // train bar
    rect((x(f.train.at(0)), ty - bh), (x(f.train.at(1)), ty),
         fill: train-fill, stroke: train-stroke + 1.1pt, radius: 2pt)
    content(((x(f.train.at(0)) + x(f.train.at(1))) / 2, ty - bh / 2),
            text(size: 8.5pt, weight: "bold", fill: train-stroke)[Train])

    // val bar
    rect((x(f.val.at(0)), ty - bh), (x(f.val.at(1)), ty),
         fill: val-fill, stroke: val-stroke + 1.1pt, radius: 2pt)
    content(((x(f.val.at(0)) + x(f.val.at(1))) / 2, ty - bh / 2),
            text(size: 8.5pt, weight: "bold", fill: val-stroke)[Val])
  }

  // ---- time axis ----
  let ay = top-y(n - 1) - bh - 0.55
  set-style(mark: (end: "stealth", fill: axis-col, scale: 0.5))
  line((x(0), ay), (x(100) + 0.4, ay), stroke: axis-col + 1pt)
  for g in (0, 25, 50, 75, 100) {
    line((x(g), ay + 0.1), (x(g), ay - 0.1), stroke: axis-col + 1pt)
    content((x(g), ay - 0.28), anchor: "north", text(size: 7.5pt, fill: axis-col)[#g])
  }
  content((x(100) + 0.5, ay), anchor: "west",
          text(size: 8pt, style: "italic", fill: axis-col)[time →])

  // ---- legend ----
  let ly = ay - 1.0
  let lx = x(0)
  rect((lx, ly - 0.3), (lx + 0.5, ly + 0.1), fill: train-fill, stroke: train-stroke + 1pt, radius: 2pt)
  content((lx + 0.65, ly - 0.1), anchor: "west", text(size: 8pt)[Training (expands)])
  let lx2 = x(62)
  rect((lx2, ly - 0.3), (lx2 + 0.5, ly + 0.1), fill: val-fill, stroke: val-stroke + 1pt, radius: 2pt)
  content((lx2 + 0.65, ly - 0.1), anchor: "west", text(size: 8pt)[Validation (rolls forward)])
})
