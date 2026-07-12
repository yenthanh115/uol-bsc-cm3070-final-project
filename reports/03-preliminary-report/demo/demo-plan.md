## Surge Detection Prototype — Demo Script (4 min)

---

### Stage 1: Introduction & Input Data (45 sec)

**Action:** Show the raw CSV briefly in the editor or file explorer.

**Dialogue:**

> "So this project is based on the CM3005 Data Science Project Idea, Predictive Modelling of Social Media Trend Emergence. The question I'm exploring is: can we look at stock discussions on Reddit and spot a surge before it actually takes off?
>
> Here's the raw data. It's a public Kaggle dataset, roughly 55,000 posts from r/pennystocks during 2021, right in the middle of the meme stock craze. Each row is a single post: title, body, timestamp, that sort of thing.
>
> What the pipeline does first is pull out ticker symbols, things like $BNGO or $CTRM, using regex, and filters out common false positives like DD or YOLO. Then it explodes each post into one row per ticker mention, which gets us to about 80,000 records. Basically a time-stamped activity log per ticker.>
> From there, it looks at 24-hour windows around each record. How many posts came before, how many came after. It measures volume growth and sentiment shift. Posts without enough future data get excluded, which drops about 89%, leaving around 8,500 records where we actually know what happened next.
>
> Those metrics get normalised and combined into a composite score. If it crosses our threshold tau, we call it a surge. The threshold is configurable so you can make it more or less sensitive depending on the use case. The key thing is: surges are defined entirely by what happened *after* the post. So let me run the pipeline and show you what that looks like."

---

### Stage 2: Setup & Installation (30 sec)

**Action:** Show the terminal, create virtual environment, install dependencies.

**Dialogue:**

> "Before running anything, let me set up the environment. We create a virtual environment, activate it, and install the dependencies from requirements.txt. That pulls in pandas, numpy, scikit-learn, textblob, matplotlib, and a few others. Nothing unusual, all standard data science packages."

---

### Stage 3: Running the Pipeline (1 min)

**Action:** Run the full pipeline.

```bash
python run_labeling.py --file-path ../data/raw/r_pennystocks_submissions_reddit.csv --output-dir ../output/demo --verbose
```

**Dialogue:**

> "This runs the full labelling pipeline end to end: dataset loading , stock ticker extraction, activity window calculation, sentiment scoring, and surge classification.
>
> [points to summary] Right, so it loaded the 80,000 exploded records, filtered out the ones without enough future data, and produced the final labelled dataset with about a 3% surge rate. That's our supervised learning problem. Let's see if we can predict it."

---

### Stage 4: Training the Model (1 min 15 sec)

**Action:** Run the training script.

```bash
python run_training.py --data-path ../data/processed/labelled_dataset.csv --output-dir ../output/demo/evaluation --seed 42 --verbose
```

**Dialogue:**

> "So here's the real question: can we predict these surges using only what we know at the moment someone hits 'post'?
>
> The model uses 9 features: things like sentiment, time of day, how long since the last post about that ticker, the posting rate, word count, and so on. What we deliberately leave out is engagement, upvotes and comments, because those come in after the fact and would be cheating.
>
> For the model itself, I went with Logistic Regression. It's simple, it's interpretable, and if even a basic linear model picks up signal here, that's a strong indicator. Training uses temporal cross-validation, always training on older data, testing on newer, so it matches real-world conditions.
>
> [waits for output] Alright, let's see how it did."

---

### Stage 5: Evaluation & Conclusion (1 min)

**Action:** Point to the printed metrics and open the generated figures in the output folder.

**Dialogue:**

> "OK so the ROC-AUC comes in at 0.733. For context, random guessing is 0.5, so 0.733 means the model ranks a true surge above a non-surge about 73% of the time, and it's doing that with only backward-looking features.>
> [opens confusion matrix] Here's the confusion matrix. You can see the trade-off between catching surges and false positives. [opens ROC curve] And the ROC curve gives us the full picture across all possible classification thresholds.>
> So what does this tell us? Discussion patterns that are visible at posting time *do* carry predictive signal. A simple baseline already gets meaningful performance, which validates the approach. More complex models like random forests or gradient boosting could push this further, but the prototype's goal was to establish that feasibility.>
> That's the demo. Thanks."

---
