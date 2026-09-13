#import "@preview/cetz:0.3.4"

// Data Pipeline Architecture (six-stage linear pipeline).
// Each stage consumes the previous stage's output and writes an intermediate
// artefact to disk. Shading marks the three critical design points:
//   Stage 4 Target Labelling  -> leakage prevention
//   Stage 6 Model Training     -> temporal validation
//   Stage 6 Evaluation         -> statistical rigour

#set page(width: auto, height: auto, margin: 12pt)
#set text(font: "Arial", size: 10pt)

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // ---- palette ----
  let neutral-fill   = rgb("#e1f5fe")
  let neutral-stroke = rgb("#0288d1")
  // critical design-point colours
  let leak-fill   = rgb("#ffcdd2"); let leak-stroke   = rgb("#c62828") // leakage prevention
  let temp-fill   = rgb("#c8e6c9"); let temp-stroke   = rgb("#2e7d32") // temporal validation
  let stat-fill   = rgb("#fff3e0"); let stat-stroke   = rgb("#e65100") // statistical rigour

  // ---- six stages (top -> bottom) ----
  let stages = (
    (n: "1", title: "Data Loading & Preprocessing",
     resp: "Ingest raw CSV, clean text, extract tickers by\nregex, explode into record–ticker pairs",
     fill: neutral-fill, stroke: neutral-stroke, crit: none),
    (n: "2", title: "Temporal Windowing",
     resp: "Per-ticker forward/backward 24 h posting counts\nvia vectorised binary search",
     fill: neutral-fill, stroke: neutral-stroke, crit: none),
    (n: "3", title: "Sentiment Computation",
     resp: "VADER compound score per record,\ntitle-fallback when selftext absent",
     fill: neutral-fill, stroke: neutral-stroke, crit: none),
    (n: "4", title: "Target Labelling",
     resp: "Temporal 80/20 split, train-only z-scores,\nbinary targets via composite thresholding",
     fill: leak-fill, stroke: leak-stroke, crit: "Leakage prevention"),
    (n: "5", title: "Feature Engineering",
     resp: "Extract eleven backward-looking features",
     fill: neutral-fill, stroke: neutral-stroke, crit: none),
    (n: "6", title: "Model Training & Evaluation",
     resp: "Expanding-window CV, hyperparameter tuning,\nholdout evaluation, statistical testing",
     fill: temp-fill, stroke: temp-stroke, crit: "Temporal validation + statistical rigour"),
  )

  // ---- geometry ----
  let bw   = 8.4    // box width
  let bh   = 1.5    // box height
  let gap  = 1.05   // vertical gap between boxes (arrow space)
  let step = bh + gap
  let cx   = 0      // horizontal centre of the column

  let top-y(i) = -i * step               // top edge y of box i
  let ctr-y(i) = top-y(i) - bh / 2        // centre y of box i

  // ---- draw stage boxes ----
  for (i, s) in stages.enumerate() {
    let ty = top-y(i)
    rect(
      (cx - bw / 2, ty - bh), (cx + bw / 2, ty),
      fill: s.fill, stroke: s.stroke + 1.3pt, radius: 4pt,
      name: "s" + s.n,
    )
    // stage number badge (left)
    circle((cx - bw / 2 + 0.7, ty - bh / 2), radius: 0.42,
           fill: s.stroke, stroke: none)
    content((cx - bw / 2 + 0.7, ty - bh / 2),
            text(fill: white, weight: "bold", size: 11pt)[#s.n])
    // title + responsibility (right of badge)
    content(
      (cx - bw / 2 + 1.55, ty - bh / 2),
      anchor: "west",
      align(left)[
        #text(weight: "bold", size: 10.5pt)[#s.title] \
        #text(size: 8pt, fill: rgb("#37474f"))[#s.resp]
      ],
    )
  }

  // ---- arrows between stages (each writes an artefact to disk) ----
  set-style(mark: (end: "stealth", fill: black, scale: 0.55))
  for i in range(stages.len() - 1) {
    line((cx, top-y(i) - bh), (cx, top-y(i + 1)), stroke: 1.3pt)
    // "artefact -> disk" tag on the connector
    content((cx + 0.25, top-y(i) - bh - gap / 2), anchor: "west",
            text(size: 7pt, style: "italic", fill: rgb("#607d8b"))[writes artefact → disk])
  }

  // ---- critical design-point callouts (right side) ----
  let call-x = cx + bw / 2 + 0.6
  let callout(i, body, stroke, fill) = {
    let y = ctr-y(i)
    // connector
    line((cx + bw / 2, y), (call-x, y),
         stroke: (paint: stroke, thickness: 1pt, dash: "dotted"))
    content((call-x, y), anchor: "west", box(
      inset: 5pt, radius: 3pt, fill: fill, stroke: stroke + 1pt,
      text(size: 8pt, weight: "bold", fill: stroke)[#body]))
  }
  callout(3, "Critical: leakage\nprevention", leak-stroke, leak-fill)
  callout(5, "Critical: temporal\nvalidation + statistical rigour", temp-stroke, temp-fill)

  // ---- endpoints: raw input (top) and results (bottom) ----
  let ry = top-y(0) + gap
  // ---- title (centred, above the top endpoint) ----
  content((cx, ry + 0.9), anchor: "south",
          text(weight: "bold", size: 12pt)[Data Pipeline Architecture])
  content((cx, ry), anchor: "south",
          text(size: 8.5pt, weight: "bold", fill: rgb("#455a64"))[Raw Reddit submissions (CSV)])
  line((cx, ry - 0.28), (cx, top-y(0)), stroke: 1.3pt)

  let by = top-y(stages.len() - 1) - bh - gap
  line((cx, top-y(stages.len() - 1) - bh), (cx, by + 0.28), stroke: 1.3pt)
  content((cx, by), anchor: "north",
          text(size: 8.5pt, weight: "bold", fill: rgb("#455a64"))[Evaluation results & statistical tests])
})
