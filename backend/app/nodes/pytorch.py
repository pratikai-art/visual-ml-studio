"""Optional PyTorch neural-network nodes for tabular supervised workflows."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from ..utils.jsonable import df_preview, to_jsonable
from ..workflow.payloads import Trained
from ..workflow.spec import NodeError, NodeResult, NodeSpec, Param, Port, register
from .steps import build_pipeline


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise NodeError(
            "PyTorch is not installed. Install the optional deep-learning dependency with "
            "`pip install -e '.[deep-learning]'` and restart the backend.",
            str(exc),
        ) from exc
    return torch, nn


def _network(nn, architecture: str, n_features: int, n_outputs: int, hidden: int, layers: int, dropout: float):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.architecture = architecture
            if architecture == "cnn":
                self.body = nn.Sequential(
                    nn.Conv1d(1, hidden, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.AdaptiveAvgPool1d(1),
                )
                self.head = nn.Linear(hidden, n_outputs)
            elif architecture in ("lstm", "gru"):
                recurrent = nn.LSTM if architecture == "lstm" else nn.GRU
                self.body = recurrent(1, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
                self.head = nn.Linear(hidden, n_outputs)
            elif architecture == "transformer":
                heads = 4 if hidden % 4 == 0 else (2 if hidden % 2 == 0 else 1)
                layer = nn.TransformerEncoderLayer(
                    d_model=hidden, nhead=heads, dim_feedforward=hidden * 2,
                    dropout=dropout, batch_first=True, activation="gelu",
                )
                self.input_projection = nn.Linear(1, hidden)
                self.body = nn.TransformerEncoder(layer, num_layers=layers)
                self.head = nn.Linear(hidden, n_outputs)
            else:
                self.body = nn.Sequential(
                    nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
                    *sum(([nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout)] for _ in range(layers - 1)), []),
                )
                self.head = nn.Linear(hidden, n_outputs)

        def forward(self, x):
            if self.architecture == "cnn":
                return self.head(self.body(x.transpose(1, 2)).squeeze(-1))
            if self.architecture in ("lstm", "gru"):
                sequence, _ = self.body(x)
                return self.head(sequence[:, -1, :])
            if self.architecture == "transformer":
                return self.head(self.body(self.input_projection(x)).mean(dim=1))
            return self.head(self.body(x.flatten(1)))

    return Net()


class _TorchPredictor:
    def __init__(self, model: Any, architecture: str, task: str, classes: list | None):
        self.model = model
        self.architecture = architecture
        self.task = task
        self.classes_ = np.asarray(classes) if classes is not None else None

    def _output(self, X):
        torch, _ = _torch()
        values = np.array(X, dtype=np.float32, copy=True)
        tensor = torch.from_numpy(values[:, :, None])
        self.model.eval()
        with torch.no_grad():
            return self.model(tensor)

    def predict(self, X):
        torch, _ = _torch()
        output = self._output(X)
        if self.task == "classification":
            indices = output.argmax(dim=1).cpu().numpy()
            return self.classes_[indices]
        return output.squeeze(1).cpu().numpy()

    def predict_proba(self, X):
        torch, _ = _torch()
        if self.task != "classification":
            raise AttributeError("Regression models do not provide probabilities.")
        return torch.softmax(self._output(X), dim=1).cpu().numpy()


def _prepare(split, task):
    pre = build_pipeline(split.steps, task)
    try:
        X_train = pre.fit_transform(split.X_train, split.y_train)
        X_test = pre.transform(split.X_test)
    except Exception as exc:
        raise NodeError(
            "PyTorch models need numeric features with missing values handled first. "
            "Add an imputer and encoder when required.",
            str(exc),
        ) from exc
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
    bad = [str(c) for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])]
    if bad:
        raise NodeError(
            "PyTorch models need numeric features. Add an encoder before the neural-network node.",
            f"Non-numeric columns: {bad}",
        )
    if X_train.isna().any().any() or X_test.isna().any().any():
        raise NodeError("PyTorch models cannot train with missing values. Add an imputer first.")
    return pre, X_train.to_numpy(dtype=np.float32), X_test.to_numpy(dtype=np.float32), list(X_train.columns)


def run_pytorch(ctx, p, ins):
    torch, nn = _torch()
    split = ins["split"][0]
    task = split.task
    if task not in ("classification", "regression"):
        raise NodeError("PyTorch training requires a classification or regression target.")
    pre, x_train, x_test, feature_names = _prepare(split, task)
    torch.manual_seed(p["seed"])
    if task == "classification":
        classes, y_train = np.unique(split.y_train.to_numpy(), return_inverse=True)
        class_to_index = {value: index for index, value in enumerate(classes)}
        try:
            y_test = np.asarray([class_to_index[value] for value in split.y_test.to_numpy()])
        except KeyError as exc:
            raise NodeError("The test split contains a class that was not present in the training split.", str(exc)) from exc
        n_outputs = len(classes)
        target_train = torch.tensor(y_train, dtype=torch.long)
        target_test = torch.tensor(y_test, dtype=torch.long)
    else:
        classes = None
        n_outputs = 1
        target_train = torch.tensor(split.y_train.to_numpy(dtype=np.float32).reshape(-1, 1))
        target_test = torch.tensor(split.y_test.to_numpy(dtype=np.float32).reshape(-1, 1))
    model = _network(nn, p["architecture"], x_train.shape[1], n_outputs, p["hidden"], p["layers"], p["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"])
    loss_fn = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
    features = torch.tensor(x_train[:, :, None])
    t0 = time.perf_counter()
    model.train()
    for epoch in range(p["epochs"]):
        ctx.check_cancel()
        optimizer.zero_grad()
        output = model(features)
        loss = loss_fn(output, target_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % max(1, p["epochs"] // 5) == 0:
            ctx.log(f"Epoch {epoch + 1}/{p['epochs']} - loss {loss.item():.4f}")
    train_time = time.perf_counter() - t0
    predictor = _TorchPredictor(model, p["architecture"], task, classes.tolist() if classes is not None else None)
    t1 = time.perf_counter()
    y_pred = predictor.predict(x_test)
    infer_time = time.perf_counter() - t1
    y_pred_train = predictor.predict(x_train)
    y_proba = predictor.predict_proba(x_test) if task == "classification" else None
    y_proba_train = predictor.predict_proba(x_train) if task == "classification" else None
    kwargs = {key: p[key] for key in ("architecture", "epochs", "learning_rate", "hidden", "layers", "dropout", "weight_decay", "seed")}
    tr = Trained(
        predictor, split, f"pytorch_{p['architecture']}", f"PyTorch {p['architecture'].upper()}",
        task, kwargs, y_pred, y_proba, y_pred_train, y_proba_train, train_time, infer_time,
        classes=classes.tolist() if classes is not None else None, feature_names=feature_names,
    )
    pred = pd.DataFrame({"row_index": split.X_test.index, "actual": split.y_test.values, "predicted": y_pred})
    preview = {
        "kind": "model", "algorithm": tr.algo_name, "task": task, "params": to_jsonable(kwargs),
        "train_time": train_time, "infer_time": infer_time, "n_train": len(x_train),
        "n_test": len(x_test), "n_features": len(feature_names), "features": feature_names[:200],
        "predictions": df_preview(pred, 100),
    }
    return NodeResult({"trained": tr}, preview)


def code_pytorch(c, nid, p, ins):
    sv = ins["split"][0]
    var = c.out(nid, "trained")
    meta = c.meta[sv]
    c.imports |= {"import torch", "from torch import nn", "from sklearn.pipeline import Pipeline",
                  "from sklearn.preprocessing import FunctionTransformer"}
    architecture = p["architecture"]
    code = f"""
class TorchNet_{var}(nn.Module):
    def __init__(self, n_features, n_outputs):
        super().__init__()
        self.architecture = {architecture!r}
        hidden, layers, dropout = {p["hidden"]}, {p["layers"]}, {p["dropout"]!r}
        if self.architecture == "cnn":
            self.body = nn.Sequential(nn.Conv1d(1, hidden, 3, padding=1), nn.ReLU(), nn.Dropout(dropout), nn.AdaptiveAvgPool1d(1))
            self.head = nn.Linear(hidden, n_outputs)
        elif self.architecture in ("lstm", "gru"):
            recurrent = nn.LSTM if self.architecture == "lstm" else nn.GRU
            self.body = recurrent(1, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Linear(hidden, n_outputs)
        elif self.architecture == "transformer":
            heads = 4 if hidden % 4 == 0 else (2 if hidden % 2 == 0 else 1)
            layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads, dim_feedforward=hidden * 2, dropout=dropout, batch_first=True, activation="gelu")
            self.input_projection = nn.Linear(1, hidden)
            self.body = nn.TransformerEncoder(layer, num_layers=layers)
            self.head = nn.Linear(hidden, n_outputs)
        else:
            self.body = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout))
            self.head = nn.Linear(hidden, n_outputs)
    def forward(self, x):
        if self.architecture == "cnn":
            return self.head(self.body(x.transpose(1, 2)).squeeze(-1))
        if self.architecture in ("lstm", "gru"):
            sequence, _ = self.body(x)
            return self.head(sequence[:, -1, :])
        if self.architecture == "transformer":
            return self.head(self.body(self.input_projection(x)).mean(dim=1))
        return self.head(self.body(x.flatten(1)))


class TorchPredictor_{var}:
    def __init__(self, model, task, classes):
        self.model, self.task, self.classes_ = model, task, np.asarray(classes) if classes is not None else None
    def _output(self, X):
        with torch.no_grad():
            return self.model(torch.from_numpy(np.asarray(X, dtype=np.float32)[:, :, None]))
    def predict(self, X):
        out = self._output(X)
        return self.classes_[out.argmax(1).numpy()] if self.task == "classification" else out.squeeze(1).numpy()
    def predict_proba(self, X):
        return torch.softmax(self._output(X), dim=1).numpy()

pre_{var} = Pipeline(steps_{sv} or [("passthrough", FunctionTransformer(feature_names_out="one-to-one"))])
X_train_{var} = pre_{var}.fit_transform(X_train_{sv}, y_train_{sv})
X_test_{var} = pre_{var}.transform(X_test_{sv})
X_train_{var} = np.asarray(X_train_{var}, dtype=np.float32)
X_test_{var} = np.asarray(X_test_{var}, dtype=np.float32)
task_{var} = {meta["task"]!r}
"""
    if meta["task"] == "classification":
        code += f"""classes_{var}, y_train_idx_{var} = np.unique(np.asarray(y_train_{sv}), return_inverse=True)
y_test_idx_{var} = np.searchsorted(classes_{var}, np.asarray(y_test_{sv}))
n_outputs_{var} = len(classes_{var})
y_train_tensor_{var} = torch.tensor(y_train_idx_{var}, dtype=torch.long)
"""
    else:
        code += f"""classes_{var} = None
n_outputs_{var} = 1
y_train_tensor_{var} = torch.tensor(np.asarray(y_train_{sv}, dtype=np.float32).reshape(-1, 1))
"""
    code += f"""torch.manual_seed({p["seed"]})
model_{var} = TorchNet_{var}(X_train_{var}.shape[1], n_outputs_{var})
optimizer_{var} = torch.optim.Adam(model_{var}.parameters(), lr={p["learning_rate"]!r}, weight_decay={p["weight_decay"]!r})
loss_fn_{var} = nn.CrossEntropyLoss() if task_{var} == "classification" else nn.MSELoss()
features_{var} = torch.from_numpy(X_train_{var}[:, :, None])
model_{var}.train()
for epoch_{var} in range({p["epochs"]}):
    optimizer_{var}.zero_grad()
    output_{var} = model_{var}(features_{var})
    loss_{var} = loss_fn_{var}(output_{var}, y_train_tensor_{var})
    loss_{var}.backward()
    optimizer_{var}.step()
pipe_{var} = TorchPredictor_{var}(model_{var}, task_{var}, classes_{var})
with torch.no_grad():
    output_test_{var} = pipe_{var}._output(X_test_{var})
y_pred_{var} = pipe_{var}.predict(X_test_{var})
y_pred_train_{var} = pipe_{var}.predict(X_train_{var})
y_proba_{var} = pipe_{var}.predict_proba(X_test_{var}) if task_{var} == "classification" else None
y_proba_train_{var} = pipe_{var}.predict_proba(X_train_{var}) if task_{var} == "classification" else None
"""
    c.meta[var] = {
        "split": sv, "task": meta["task"], "label": f"PyTorch {architecture.upper()}",
        "key": f"pytorch_{architecture}", "has_proba": meta["task"] == "classification",
        "seed": meta["seed"],
    }
    c.cell(nid, f"Train: PyTorch {architecture.upper()}", "Trains a real PyTorch neural network on the training split.", code)
    return {"trained": var}


register(NodeSpec(
    "pytorch_neural_network", "PyTorch Neural Network", "Machine Learning",
    "Train a PyTorch CNN, LSTM, GRU, Transformer or MLP on a supervised tabular split.",
    [Port("split", "split", "Train/test data")],
    [Port("trained", "trained", "Trained model", required=False)],
    [
        Param("architecture", "Architecture", "select", "mlp", "", options=["mlp", "cnn", "lstm", "gru", "transformer"]),
        Param("epochs", "Epochs", "int", 30, "Training passes over the training data.", min=1, max=2000),
        Param("learning_rate", "Learning rate", "float", 0.001, "", min=0.000001, max=1, step=0.0001),
        Param("hidden", "Hidden units", "int", 32, "", min=4, max=512),
        Param("layers", "Hidden layers", "int", 1, "", min=1, max=8),
        Param("dropout", "Dropout", "float", 0.1, "", min=0, max=0.9, step=0.05),
        Param("weight_decay", "Weight decay", "float", 0.0001, "", min=0, max=1, step=0.0001),
        Param("seed", "Random seed", "int", 42, "", min=0),
    ],
    run_pytorch, code_pytorch, icon="brain",
    subcategory="Deep Learning",
    doc={"what": "Trains a real PyTorch neural network using the selected architecture.",
         "how": "Tabular features are treated as a sequence of scalar values for CNN, LSTM, GRU, and Transformer architectures.",
         "limitations": "This node is CPU-oriented and intended for small/medium tabular datasets. Image and raw text tensors are not supported by the current dataset contract."},
    keywords="pytorch torch deep learning neural network cnn lstm gru transformer mlp",
))
