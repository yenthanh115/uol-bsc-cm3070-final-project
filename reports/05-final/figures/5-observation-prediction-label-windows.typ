#import "@preview/cetz:0.3.4"

// Temporal Observation, Prediction and Surge-Label Windows.
// A prediction-time divider splits the timeline: the left OBSERVATION window
// supplies every feature (computed only from [t-k, t]), the right FUTURE/LABEL
// window is used solely to determine whether a surge occurs. The X(t) -> MODEL
// -> P(surge) flow underneath makes the leakage-free contract explicit.

#set page(width: auto, height: auto, margin: 14pt)
#set text(font: "Arial", size: 10pt)

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // ---- palette ----
  let obs-fill   = rgb("#e1f5fe"); let obs-stroke = rgb("#0288d1") // observation (known)
  let lab-fill   = rgb("#ffcdd2"); let lab-stroke = rgb("#c62828") // future / label
  let now-col    = rgb("#e65100")                                   // prediction instant
  let ok-col     = rgb("#2e7d32")                                   // model / leakage-safe
  let axis-col   = rgb("#455a64")

  // ---- geometry ----
  let xL = -8       // left edge
  let xR = 8        // right edge
  let xM = 0        // prediction-time divider (centre)
  let axisY = 0     // timeline y

  // ================= prediction-time divider =================
  let divTop = 3.6
  let divBot = -6.8
  content((xM, divTop + 0.15), anchor: "south",
          text(weight: "bold", size: 11pt, fill: now-col)[Prediction time])
  // down arrow into the divider
  set-style(mark: (end: "stealth", fill: now-col, scale: 0.6))
  line((xM, divTop), (xM, axisY + 0.35), stroke: now-col + 1.6pt)
  // dashed divider continues below the axis
  line((xM, axisY - 0.35), (xM, divBot),
       stroke: (paint: now-col, thickness: 1.4pt, dash: "dashed"))

  // ================= timeline =================
  set-style(mark: (end: "stealth", fill: axis-col, scale: 0.55))
  line((xL - 0.5, axisY), (xR + 0.7, axisY), stroke: axis-col + 1.3pt)
  content((xR + 0.8, axisY), anchor: "west",
          text(size: 8.5pt, style: "italic", fill: axis-col)[time])
  // marker dot at prediction time
  circle((xM, axisY), radius: 0.12, fill: now-col, stroke: none)

  // ================= window bands =================
  let bandBot = axisY + 0.55
  let bandTop = axisY + 1.75

  // observation window (left)
  rect((xL, bandBot), (xM - 0.18, bandTop),
       fill: obs-fill, stroke: obs-stroke + 1.3pt, radius: 3pt)
  content(((xL + xM) / 2, (bandBot + bandTop) / 2), align(center)[
    #text(weight: "bold", size: 10pt, fill: obs-stroke)[OBSERVATION WINDOW] \
    #text(size: 8.5pt)[\[ t − k  ............  t \]]
  ])

  // future / label window (right)
  rect((xM + 0.18, bandBot), (xR, bandTop),
       fill: lab-fill, stroke: lab-stroke + 1.3pt, radius: 3pt)
  content(((xM + xR) / 2, (bandBot + bandTop) / 2), align(center)[
    #text(weight: "bold", size: 10pt, fill: lab-stroke)[FUTURE / LABEL WINDOW] \
    #text(size: 8.5pt)[\[ t + 1  ..........  t + h \]]
  ])

  // ================= descriptions under each window =================
  let descY = axisY - 0.7
  content(((xL + xM) / 2, descY), anchor: "north", align(center)[
    #text(size: 8.5pt, weight: "bold", fill: obs-stroke)[Features calculated] \
    #text(size: 8.5pt, weight: "bold", fill: obs-stroke)[ONLY from this period]
  ])
  content(((xM + xR) / 2, descY), anchor: "north", align(center)[
    #text(size: 8.5pt, weight: "bold", fill: lab-stroke)[Determine whether] \
    #text(size: 8.5pt, weight: "bold", fill: lab-stroke)[a surge occurs]
  ])

  // ================= feature list (left column) =================
  let featY = descY - 1.15
  content(((xL + xM) / 2, featY), anchor: "north", align(center)[
    #set text(size: 8.5pt)
    #text(fill: obs-stroke)[volume] \
    #text(fill: obs-stroke)[growth rate] \
    #text(fill: obs-stroke)[activity] \
    #text(fill: obs-stroke)[sentiment] \
    #text(fill: obs-stroke, style: "italic")[etc.]
  ])

  // label side: y target note
  content(((xM + xR) / 2, featY), anchor: "north", align(center)[
    #set text(size: 8.5pt)
    #text(fill: lab-stroke, style: "italic")[not visible to the model] \
    #text(fill: lab-stroke)[surge target  y ∈ {0, 1}]
  ])

  // ================= X(t) -> MODEL -> P(surge) flow =================
  let flowY = -5.9
  set-style(mark: (end: "stealth", fill: ok-col, scale: 0.6))

  // X(t) node (fed by observation window)
  content((xL + 1.6, flowY), box(inset: 5pt, radius: 3pt,
    fill: obs-fill, stroke: obs-stroke + 1.2pt,
    text(size: 9pt, weight: "bold", fill: obs-stroke)[X(t)]), name: "xt")
  // MODEL node
  content((xM, flowY), box(inset: (x: 9pt, y: 5pt), radius: 3pt,
    fill: rgb("#c8e6c9"), stroke: ok-col + 1.3pt,
    text(size: 9.5pt, weight: "bold", fill: ok-col)[MODEL]), name: "model")
  // P(surge) node
  content((xR - 1.3, flowY), box(inset: 5pt, radius: 3pt,
    fill: lab-fill, stroke: lab-stroke + 1.2pt,
    text(size: 9pt, weight: "bold", fill: lab-stroke)[P(surge)]), name: "psurge")

  line("xt.east", "model.west", stroke: ok-col + 1.3pt)
  line("model.east", "psurge.west", stroke: ok-col + 1.3pt)

  // dotted feed from observation window down to X(t)
  line(((xL + xM) / 2, featY - 1.3), (xL + 1.6, flowY + 0.4),
       stroke: (paint: obs-stroke, thickness: 1pt, dash: "dotted"),
       mark: (end: "stealth", fill: obs-stroke, scale: 0.5))

  // ================= leakage-free caption =================
  content((xM, flowY - 1.1), anchor: "north", box(
    inset: 6pt, radius: 3pt, fill: rgb("#c8e6c9"), stroke: ok-col + 1pt,
    text(size: 8pt, fill: ok-col)[
      #text(weight: "bold")[Leakage-free:] the model sees only X(t) from the
      observation window; the future window defines the label, never a feature.
    ]))
})
