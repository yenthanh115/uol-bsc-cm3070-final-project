#import "@preview/cetz:0.3.4"

// Online Attention Lifecycle Model
// Emergence -> Growth -> Peak -> Decline, with two annotation callouts.

#set page(width: auto, height: auto, margin: 10pt)
#set text(font: "Arial", size: 10pt)

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // ---- palette (fill, stroke) mirroring the Mermaid styles ----
  let stages = (
    (id: "A", x: 0,  title: "Emergence", sub: "Few posts, low signal",   fill: rgb("#e1f5fe"), stroke: rgb("#0288d1")),
    (id: "B", x: 4,  title: "Growth",    sub: "Accelerating activity",   fill: rgb("#fff9c4"), stroke: rgb("#f9a825")),
    (id: "C", x: 8,  title: "Peak",      sub: "Maximum attention",       fill: rgb("#ffcdd2"), stroke: rgb("#c62828")),
    (id: "D", x: 12, title: "Decline",   sub: "Activity fading",         fill: rgb("#f5f5f5"), stroke: rgb("#9e9e9e")),
  )

  let bw = 3.2   // box width
  let bh = 1.6   // box height
  let y  = 0     // stage row centre

  // ---- draw stage boxes ----
  for s in stages {
    let cx = s.x + bw / 2
    rect(
      (s.x, y - bh / 2), (s.x + bw, y + bh / 2),
      fill: s.fill, stroke: s.stroke + 1.2pt, radius: 3pt,
      name: s.id,
    )
    content(
      (cx, y),
      align(center)[
        #text(weight: "bold")[#s.title] \
        #text(style: "italic", size: 8pt)[#s.sub]
      ],
    )
  }

  // ---- solid arrows between stages ----
  set-style(mark: (end: "stealth", fill: black, scale: 0.6))
  for i in range(stages.len() - 1) {
    line(
      (stages.at(i).x + bw, y),
      (stages.at(i + 1).x, y),
      stroke: 1.2pt,
    )
  }

  // ---- annotation callouts (parallelogram / flag shape) ----
  // helper: draw a slanted "note" box centred at (cx, cy)
  let note(cx, cy, body, fill, stroke, w: 3.6, h: 1.5, skew: 0.35) = {
    let x0 = cx - w / 2
    let x1 = cx + w / 2
    let y0 = cy - h / 2
    let y1 = cy + h / 2
    line(
      (x0 + skew, y1), (x1 + skew, y1),
      (x1 - skew, y0), (x0 - skew, y0),
      close: true, fill: fill, stroke: stroke + 1.2pt,
    )
    content((cx, cy), align(center)[#body])
  }

  let ny = -3.5  // annotation row centre

  // E: prediction point, linked to Emergence (A)
  note(
    stages.at(0).x + bw / 2, ny,
    [#text(size: 8.5pt)[🎯 This project's \ prediction point]],
    rgb("#c8e6c9"), rgb("#2e7d32"),
  )
  // F: traditional models note, linked to Growth (B)
  note(
    stages.at(1).x + bw / 2, ny,
    [#text(size: 8.5pt)[Traditional models \ require data here]],
    rgb("#fff3e0"), rgb("#e65100"),
  )

  // ---- dotted connectors from stage to its annotation ----
  line((stages.at(0).x + bw / 2, y - bh / 2), (stages.at(0).x + bw / 2, ny + 0.75),
       stroke: (paint: rgb("#2e7d32"), thickness: 1pt, dash: "dotted"))
  line((stages.at(1).x + bw / 2, y - bh / 2), (stages.at(1).x + bw / 2, ny + 0.75),
       stroke: (paint: rgb("#e65100"), thickness: 1pt, dash: "dotted"))
})
