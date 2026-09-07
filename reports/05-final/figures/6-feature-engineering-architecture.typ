#import "@preview/cetz:0.3.4"

// Feature Engineering Architecture.
// Shows how a raw Reddit record-ticker pair (observed at time t) becomes the
// eleven backward-looking predictive variables, grouped by the four feature
// categories (Content, Temporal, Activity, Interaction) defined in the report,
// then fed to the classifiers. The backward-only constraint is made explicit:
// every feature is computed from information at or before t.

#set page(width: auto, height: auto, margin: 14pt)
#set text(font: "Arial", size: 10pt)

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // ---- palette ----
  let src-fill  = rgb("#fff9c4"); let src-stroke  = rgb("#f9a825") // raw source
  let cat-fill  = rgb("#e1f5fe"); let cat-stroke  = rgb("#0288d1") // category header
  let feat-fill = white;          let feat-stroke  = rgb("#607d8b") // individual feature
  let inter-fill= rgb("#fff3e0"); let inter-stroke = rgb("#e65100") // interaction (derived)
  let model-fill= rgb("#c8e6c9"); let model-stroke = rgb("#2e7d32") // model / output
  let excl-col  = rgb("#c62828")                                     // excluded signal

  set-style(mark: (end: "stealth", fill: rgb("#455a64"), scale: 0.55))
  let arrow(a, b, ..s) = line(a, b, stroke: rgb("#455a64") + 1.1pt, ..s)

  // =================================================================
  // ROW 2 geometry — four category columns (defined first so the
  // source box can span their full width)
  // =================================================================
  let colW = 3.9
  let gapX = 0.35
  let nCols = 4
  let totalW = nCols * colW + (nCols - 1) * gapX
  let x0 = -totalW / 2
  let colX(i) = x0 + i * (colW + gapX)   // left edge of column i

  // =================================================================
  // ROW 1 — raw source (top), spanning the full four-column width
  // =================================================================
  let srcY = 0
  rect((x0, srcY), (x0 + totalW, srcY + 1.35),
       fill: src-fill, stroke: src-stroke + 1.4pt, radius: 4pt, name: "src")
  content(((x0 + x0 + totalW) / 2, srcY + 0.68), align(center)[
    #text(weight: "bold", size: 11pt)[Raw Reddit record–ticker pair  @ time t] \
    #text(size: 8.5pt)[text • title • timestamp • ticker • prior same-ticker posts]
  ])

  // excluded-signals note (above, right-aligned over the source)
  content((x0 + totalW, srcY + 1.35 + 0.3), anchor: "south-east", box(
    inset: 5pt, radius: 3pt, fill: rgb("#ffcdd2"), stroke: excl-col + 1pt,
    text(size: 7.5pt, fill: excl-col)[
      #text(weight: "bold")[Excluded (post-hoc):] upvotes, num_comments  →  would leak the future
    ]))

  let headY = -2.6
  let headH = 0.85

  let cats = (
    (name: "CONTENT", note: "what the post says",
     feats: ("sentiment_score", "word_count", "title_length", "num_tickers_mentioned")),
    (name: "TEMPORAL", note: "when it was posted",
     feats: ("hour_of_day", "day_of_week")),
    (name: "ACTIVITY", note: "recent posting dynamics",
     feats: ("time_since_previous", "ticker_post_rate_24h", "ticker_post_acceleration")),
    (name: "INTERACTION", note: "cross-feature products",
     feats: ("word_count_x_hour", "accel_x_time_since_prev")),
  )

  // feature-box geometry
  let fH = 0.6
  let fGap = 0.22

  for (i, c) in cats.enumerate() {
    let lx = colX(i)
    let cx = lx + colW / 2
    let isInter = c.name == "INTERACTION"
    let hFill = if isInter { inter-fill } else { cat-fill }
    let hStroke = if isInter { inter-stroke } else { cat-stroke }

    // category header
    rect((lx, headY - headH), (lx + colW, headY),
         fill: hFill, stroke: hStroke + 1.3pt, radius: 3pt, name: "cat" + str(i))
    content((cx, headY - headH / 2), align(center)[
      #text(weight: "bold", size: 9.5pt, fill: hStroke)[#c.name] \
      #text(size: 7pt, style: "italic", fill: rgb("#546e7a"))[#c.note]
    ])

    // arrow from source into this header
    arrow((cx, srcY), (cx, headY))

    // feature boxes
    for (j, f) in c.feats.enumerate() {
      let ty = headY - headH - 0.45 - j * (fH + fGap)
      let ffill = if isInter { inter-fill.lighten(40%) } else { feat-fill }
      let fstroke = if isInter { inter-stroke } else { feat-stroke }
      rect((lx + 0.2, ty - fH), (lx + colW - 0.2, ty),
           fill: ffill, stroke: fstroke + 0.9pt, radius: 2pt)
      content((cx, ty - fH / 2),
              text(size: 7.5pt, font: "Consolas")[#raw(f)])
    }
  }

  // note for interaction column: it derives from the other columns
  let interCx = colX(3) + colW / 2
  content((interCx, headY - headH - 0.15), anchor: "north",
          text(size: 6.5pt, style: "italic", fill: inter-stroke)[built from Content × Temporal / Activity])

  // =================================================================
  // ROW 3 — feature vector X(t)
  // =================================================================
  // find lowest feature bottom to place the vector below everything
  let maxFeats = 4
  let lowestY = headY - headH - 0.45 - maxFeats * (fH + fGap)
  let vecY = lowestY - 0.7
  let vecH = 0.95
  rect((x0, vecY - vecH), (x0 + totalW, vecY),
       fill: cat-fill.darken(3%), stroke: cat-stroke + 1.4pt, radius: 4pt, name: "vec")
  content(((x0 + x0 + totalW) / 2, vecY - vecH / 2), align(center)[
    #text(weight: "bold", size: 10pt, fill: cat-stroke)[Feature vector  X(t)  —  11 backward-looking features] \
    #text(size: 7.5pt, style: "italic")[all computed from information at or before t (no lookahead)]
  ])

  // arrows from each category's last feature down into the vector
  for i in range(nCols) {
    let cx = colX(i) + colW / 2
    let nf = cats.at(i).feats.len()
    let bottomY = headY - headH - 0.45 - nf * (fH + fGap) - (nf - 1) * 0 // bottom of last box row start
    let by = headY - headH - 0.45 - (nf - 1) * (fH + fGap) - fH
    arrow((cx, by), (cx, vecY))
  }

  // =================================================================
  // ROW 4 — model + output
  // =================================================================
  let modY = vecY - vecH - 1.15
  let vecCx = (x0 + x0 + totalW) / 2   // horizontal centre of the feature vector
  // MODEL node — centred under the feature vector
  content((vecCx - 2.0, modY), box(inset: (x: 10pt, y: 6pt), radius: 4pt,
    fill: model-fill, stroke: model-stroke + 1.4pt,
    text(size: 10pt, weight: "bold", fill: model-stroke)[Classifier \ (LR / RF / XGB)]), name: "model")
  // output node
  content((vecCx + 2.0, modY), box(inset: (x: 9pt, y: 6pt), radius: 4pt,
    fill: rgb("#ffcdd2"), stroke: rgb("#c62828") + 1.3pt,
    text(size: 9.5pt, weight: "bold", fill: rgb("#c62828"))[P(surge)]), name: "out")

  arrow((vecCx - 2.0, vecY - vecH), "model.north")
  arrow("model.east", "out.west")
})
