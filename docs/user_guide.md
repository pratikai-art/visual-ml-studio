# User guide

- **Ports and colors**: blue = dataset, green = split, graphite = trained model, amber = metrics, pink = chart.
- **Why preprocessing attaches to the split**: imputers/scalers/encoders learn from training rows only. Place them after *Train/Test Split*.
- **Fair comparison**: branch models from the same split/preprocessing node; *Compare Models* warns when test sets differ.
- **Cross-validation / search** (Advanced mode): on a model node set *Cross-validation folds* or *Hyperparameter search*. Tuning uses training data only; the test set is evaluated once.
- **Time series**: set *Time column* on the split node — rows are ordered and never shuffled.
- **Errors**: the node shows a plain-language message; *Errors & Warnings* has an expandable technical trace. Fix the failing node; upstream results stay available.
- **Reconnecting datasets**: projects reference uploaded files by content hash. If a dataset is missing on open, upload the same file again and it reconnects automatically.
