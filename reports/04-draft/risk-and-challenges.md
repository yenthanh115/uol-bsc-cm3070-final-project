For the **Draft Report milestone**, the project is entering its highest-risk phase. Unlike earlier phases, most remaining risks are no longer about planning—they concern **research quality, implementation quality, and producing convincing evidence**. These are the areas heavily assessed in the draft and final report rubrics.

| Risk                                                      | Likelihood | Impact | Mitigation                                                                                                                                                           |
| --------------------------------------------------------- | ---------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Model performance is below expectations                   | Medium     | High   | Compare multiple models, improve feature engineering, tune hyperparameters, analyse failure cases rather than only chasing higher accuracy.                          |
| Surge definition proves unreliable                        | Medium     | High   | Validate the surge definition with exploratory analysis and justify it mathematically. Document assumptions and limitations.                                         |
| Feature leakage discovered late                           | Medium     | High   | Verify every feature is available at prediction time. Audit the entire pipeline before experiments.                                                                  |
| Evaluation lacks sufficient evidence                      | Medium     | High   | Include baseline comparison, cross-validation, multiple metrics, confusion matrix, ROC/PR curves, feature importance, and error analysis.                            |
| Literature review remains descriptive instead of critical | Medium     | High   | Compare previous work, identify research gaps, explain how your approach differs, and justify design decisions using the literature.                                 |
| Implementation is incomplete or unstable                  | Low–Medium | High   | Freeze the project scope. Prioritise completing and stabilising the existing pipeline rather than adding new functionality.                                          |
| Poor reproducibility                                      | Medium     | Medium | Organise the project, document dependencies, fix random seeds, and automate experiment execution where possible.                                                     |
| Weak discussion and critical analysis                     | Medium     | High   | Explain why models perform as they do, analyse limitations, discuss practical implications, and relate findings back to the research question.                       |
| Report writing falls behind implementation                | High       | High   | Write each chapter immediately after completing the corresponding implementation or experiment instead of leaving all writing until the end.                         |
| Time pressure before submission                           | High       | High   | Prioritise essential deliverables: complete implementation, experiments, evaluation, and report. Defer optional enhancements (e.g., additional models) if necessary. |

### Technical challenges

Your project has several inherently challenging aspects:

* Defining a robust and defensible mathematical definition of "surge".
* Engineering predictive features without introducing future information leakage.
* Handling class imbalance if surge events are relatively rare.
* Selecting appropriate temporal train/test splits to reflect real prediction scenarios.
* Demonstrating that sentiment features provide measurable value beyond engagement features.
* Producing interpretable models and explaining feature importance.

### Research challenges

The report must go beyond implementation by demonstrating:

* A critical synthesis of existing research rather than a simple literature summary.
* Clear justification for methodological choices.
* Evidence that the evaluation directly addresses the research question.
* Honest discussion of limitations, threats to validity, and future work.

### Project management challenges

* Balancing implementation work with report writing.
* Managing multiple concurrent responsibilities (CM3070, CM3055, work, and family).
* Avoiding feature creep after the scope has been agreed.
* Allowing sufficient time for review and revision after the draft is complete.

## Priority risks

The five risks most likely to affect your grade are:

1. **Weak experimental evidence** (insufficient evaluation or comparisons).
2. **Incomplete critical analysis** (reporting results without explaining them).
3. **Feature leakage or an invalid prediction setup**.
4. **Late report writing**, resulting in a technically good project but a weak presentation.
5. **Attempting to expand the scope instead of strengthening the current pipeline**.

Overall, your project scope is already well defined. The greatest remaining challenge is **demonstrating, through rigorous experiments and analysis, that the chosen pipeline effectively predicts future engagement and sentiment surges** while clearly communicating that evidence in the report.
