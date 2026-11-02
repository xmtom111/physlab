"""Curve fits and a Markdown report.

``fit_models`` tries six shapes and sorts them by AIC. ``fit_custom``
evaluates an expression with a small parser: numbers, ``x``, parameter
names, ``+ - * / **``, parentheses, and ``sin cos exp log sqrt``. It does
not call ``eval``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from scipy.optimize import curve_fit

from physlab.core.errors import ConfigError
from physlab.experiments.results import RunResult

_FUNCS: dict[str, Callable[[float], float]] = {
    "sin": math.sin,
    "cos": math.cos,
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
}
_Node = tuple[object, ...]
_Fn = Callable[[float, dict[str, float]], float]
_Usable = Callable[[np.ndarray, np.ndarray], bool]
_Guess = Callable[[np.ndarray, np.ndarray], list[float]]


class ModelFit(BaseModel):
    """One fitted curve.

    Attributes:
        name: Candidate name, or ``custom``.
        formula_str: Expression in ``x`` and the parameter names.
        params: Best-fit values.
        param_std: Standard errors from the covariance diagonal.
        r2: Coefficient of determination.
        aic: Akaike information criterion. Lower is better.
        bic: Bayesian information criterion.
        residual_std: Residual standard deviation.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    name: str
    formula_str: str
    params: dict[str, float]
    param_std: dict[str, float]
    r2: float
    aic: float
    bic: float
    residual_std: float


def fit_models(
    frame: pd.DataFrame,
    x: str,
    y: str,
    candidates: tuple[str, ...] = (
        "linear",
        "power",
        "exponential",
        "logarithmic",
        "inverse",
        "quadratic",
    ),
) -> list[ModelFit]:
    """Fit each candidate and sort by AIC.

    Args:
        frame: Trial table.
        x: Independent column.
        y: Dependent column.
        candidates: Subset of the six built-in shapes.

    Returns:
        Fits that converged, lowest AIC first.

    Raises:
        ConfigError: A candidate name is unknown.
    """
    xs, ys = _columns(frame, x, y)
    fits = [_try_fit(name, xs, ys) for name in candidates]
    found = [item for item in fits if item is not None]
    return sorted(found, key=lambda item: item.aic)


def fit_custom(
    frame: pd.DataFrame,
    expr: str,
    x: str,
    y: str,
    params: list[str],
    p0: list[float] | None = None,
) -> ModelFit:
    """Fit ``expr`` with ``curve_fit``.

    Args:
        frame: Trial table.
        expr: Right-hand side in ``x`` and ``params``.
        x: Independent column.
        y: Dependent column.
        params: Parameter names, in order.
        p0: Initial guess. Ones are used when omitted.

    Returns:
        The custom fit.

    Raises:
        ConfigError: The expression or the fit is unusable.
    """
    xs, ys = _columns(frame, x, y)
    compiled = compile_expression(expr, params)
    guess = [1.0] * len(params) if p0 is None else list(p0)
    return _fit_callable("custom", expr, params, compiled, xs, ys, guess)


def compile_expression(expr: str, params: list[str]) -> _Fn:
    """Parse ``expr`` into a callable of ``(x, values)``.

    Args:
        expr: Expression text.
        params: Legal parameter names besides ``x``.

    Returns:
        Callable ``(x, mapping) -> float``. The return is untyped because the
        parser builds nested tuples.

    Raises:
        ConfigError: The expression contains a forbidden token.
    """
    parser = _Parser(expr, set(params))
    node = parser.parse()
    return lambda value, mapping: _evaluate(node, float(value), mapping)


def report(run: RunResult, fits: list[ModelFit], x: str, y: str, path: Path | None = None) -> Path:
    """Write a Markdown report with a fit table and a plot.

    Args:
        run: Finished run. Plots are saved under its ``media`` directory.
        fits: Models, ideally already sorted by AIC.
        x: Column plotted on the horizontal axis.
        y: Column plotted on the vertical axis.
        path: Markdown path. Defaults to ``report.md`` in the run directory.

    Returns:
        Path of the Markdown file.
    """
    destination = run.directory / "report.md" if path is None else path
    figure = _plot(run, fits, x, y)
    text = _markdown(run, fits, x, y, figure, destination)
    destination.write_text(text, encoding="utf-8")
    return destination


def _try_fit(name: str, xs: np.ndarray, ys: np.ndarray) -> ModelFit | None:
    spec = _SPECS.get(name)
    if spec is None:
        raise ConfigError("Unknown model", details={"name": name})
    if not spec.usable(xs, ys):
        return None
    guess = spec.guess(xs, ys)
    return _safe_fit(name, spec.formula, spec.names, spec.function, xs, ys, guess)


def _safe_fit(
    name: str,
    formula: str,
    names: list[str],
    function: _Fn,
    xs: np.ndarray,
    ys: np.ndarray,
    guess: list[float],
) -> ModelFit | None:
    try:
        return _fit_callable(name, formula, names, function, xs, ys, guess)
    except (RuntimeError, ValueError):
        return None


def _fit_callable(
    name: str,
    formula: str,
    names: list[str],
    function: _Fn,
    xs: np.ndarray,
    ys: np.ndarray,
    guess: list[float],
) -> ModelFit:
    def wrapped(xx: np.ndarray, *theta: float) -> np.ndarray:
        mapping = dict(zip(names, theta, strict=True))
        return np.array([function(float(value), mapping) for value in xx], dtype=np.float64)

    popt, pcov = curve_fit(wrapped, xs, ys, p0=guess, maxfev=8000)
    return _score(name, formula, names, wrapped, xs, ys, popt, pcov)


def _score(
    name: str,
    formula: str,
    names: list[str],
    wrapped: Callable[..., np.ndarray],
    xs: np.ndarray,
    ys: np.ndarray,
    popt: np.ndarray,
    pcov: np.ndarray,
) -> ModelFit:
    predicted = wrapped(xs, *popt)
    residual = ys - predicted
    rss = float(np.sum(residual**2))
    tss = float(np.sum((ys - np.mean(ys)) ** 2))
    count = int(ys.size)
    k = len(names)
    variance = max(rss / count, 1.0e-30)
    stderr = _stderr(pcov, k)
    return ModelFit(
        name=name,
        formula_str=formula,
        params={names[i]: float(popt[i]) for i in range(k)},
        param_std={names[i]: stderr[i] for i in range(k)},
        r2=1.0 - rss / tss if tss > 0.0 else 1.0,
        aic=count * math.log(variance) + 2 * k,
        bic=count * math.log(variance) + k * math.log(count),
        residual_std=math.sqrt(max(rss / max(count - k, 1), 0.0)),
    )


def _stderr(pcov: np.ndarray, count: int) -> list[float]:
    diagonal = np.diag(pcov) if pcov.shape == (count, count) else np.full(count, np.nan)
    return [_std(value) for value in diagonal]


def _columns(frame: pd.DataFrame, x: str, y: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(frame[x], dtype=np.float64),
        np.asarray(frame[y], dtype=np.float64),
    )


def _std(value: float) -> float:
    if np.isfinite(value) and value >= 0.0:
        return float(math.sqrt(value))
    return float("nan")


def _plot(run: RunResult, fits: list[ModelFit], x: str, y: str) -> Path | None:
    rows = [
        trial
        for trial in run.trials
        if trial.success and x in trial.values and y in trial.measurements
    ]
    if not rows:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = np.array([trial.values[x] for trial in rows])
    ys = np.array([trial.measurements[y] for trial in rows])
    figure, axis = plt.subplots()
    axis.scatter(xs, ys, label="trials")
    _draw_fit(axis, fits[0] if fits else None, xs)
    axis.set_xlabel(x)
    axis.set_ylabel(y)
    axis.legend()
    path = run.directory / "media" / f"{y}_vs_{x}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    return path


def _draw_fit(axis: object, fit: ModelFit | None, xs: np.ndarray) -> None:
    if fit is None or xs.size == 0:
        return
    compiled = compile_expression(fit.formula_str, list(fit.params))
    grid = np.linspace(float(np.min(xs)), float(np.max(xs)), 80)
    curve = [compiled(float(value), fit.params) for value in grid]
    axis.plot(grid, curve, label=fit.name)  # type: ignore[attr-defined]


def _markdown(
    run: RunResult,
    fits: list[ModelFit],
    x: str,
    y: str,
    figure: Path | None,
    destination: Path,
) -> str:
    lines = [
        f"# {run.protocol_id}",
        "",
        run.question,
        "",
        run.hypothesis,
        "",
        _trial_table(run),
        "",
        "## Models",
        "",
        _fit_table(fits),
        "",
    ]
    if figure is not None:
        lines.append(f"![{y} versus {x}]({_relative(destination, figure)})")
    lines.append("")
    return "\n".join(lines)


def _relative(document: Path, figure: Path) -> str:
    try:
        return figure.relative_to(document.parent).as_posix()
    except ValueError:
        return figure.name


def _trial_table(run: RunResult) -> str:
    header = "| order | success | reason |"
    rule = "| --- | --- | --- |"
    body = [f"| {trial.order} | {trial.success} | {trial.reason or ''} |" for trial in run.trials]
    return "\n".join([header, rule, *body])


def _fit_table(fits: list[ModelFit]) -> str:
    header = "| name | formula | aic | r2 |"
    rule = "| --- | --- | --- | --- |"
    body = [f"| {fit.name} | `{fit.formula_str}` | {fit.aic:.3f} | {fit.r2:.4f} |" for fit in fits]
    return "\n".join([header, rule, *body]) if body else "_no fit_"


class _Spec:
    def __init__(
        self,
        formula: str,
        names: list[str],
        function: _Fn,
        usable: _Usable,
        guess: _Guess,
    ) -> None:
        self.formula = formula
        self.names = names
        self.function = function
        self.usable = usable
        self.guess = guess


def _enough(xs: np.ndarray, ys: np.ndarray) -> bool:
    return bool(xs.size >= 2 and ys.size >= 2)


def _positive_y(xs: np.ndarray, ys: np.ndarray) -> bool:
    del xs
    return bool(ys.size >= 2 and np.all(ys > 0.0))


def _positive_x(xs: np.ndarray, ys: np.ndarray) -> bool:
    del ys
    return bool(xs.size >= 2 and np.all(xs > 0.0))


def _nonzero_x(xs: np.ndarray, ys: np.ndarray) -> bool:
    del ys
    return bool(xs.size >= 2 and np.all(xs != 0.0))


def _three(xs: np.ndarray, ys: np.ndarray) -> bool:
    del ys
    return bool(xs.size >= 3)


def _positive(xs: np.ndarray, ys: np.ndarray) -> bool:
    return bool(np.all(xs > 0.0) and np.all(ys > 0.0) and xs.size >= 2)


def _power_guess(xs: np.ndarray, ys: np.ndarray) -> list[float]:
    slope, intercept = np.polyfit(np.log(xs), np.log(ys), 1)
    return [float(np.exp(intercept)), float(slope)]


def _exp_guess(xs: np.ndarray, ys: np.ndarray) -> list[float]:
    slope, intercept = np.polyfit(xs, np.log(np.clip(ys, 1e-12, None)), 1)
    return [float(np.exp(intercept)), float(slope)]


def _log_guess(xs: np.ndarray, ys: np.ndarray) -> list[float]:
    slope, intercept = np.polyfit(np.log(xs), ys, 1)
    return [float(intercept), float(slope)]


def _linear_guess(xs: np.ndarray, ys: np.ndarray) -> list[float]:
    slope, intercept = np.polyfit(xs, ys, 1)
    return [float(intercept), float(slope)]


def _quad_guess(xs: np.ndarray, ys: np.ndarray) -> list[float]:
    coef = np.polyfit(xs, ys, 2)
    return [float(coef[2]), float(coef[1]), float(coef[0])]


def _inverse_guess(xs: np.ndarray, ys: np.ndarray) -> list[float]:
    slope, intercept = np.polyfit(1.0 / xs, ys, 1)
    return [float(intercept), float(slope)]


def _call(names: list[str], formula: str) -> _Fn:
    return compile_expression(formula, names)


_SPECS: dict[str, _Spec] = {}


def _install() -> None:
    rows = (
        ("linear", "a + b * x", ["a", "b"], _enough, _linear_guess),
        ("power", "a * x ** b", ["a", "b"], _positive, _power_guess),
        ("exponential", "a * exp(b * x)", ["a", "b"], _positive_y, _exp_guess),
        ("logarithmic", "a + b * log(x)", ["a", "b"], _positive_x, _log_guess),
        ("inverse", "a + b / x", ["a", "b"], _nonzero_x, _inverse_guess),
        ("quadratic", "a + b * x + c * x ** 2", ["a", "b", "c"], _three, _quad_guess),
    )
    for name, formula, names, usable, guess in rows:
        _SPECS[name] = _Spec(formula, names, _call(names, formula), usable, guess)


class _Parser:
    def __init__(self, text: str, params: set[str]) -> None:
        self._tokens = _tokenize(text)
        self._index = 0
        self._params = params

    def parse(self) -> _Node:
        node = self._expr()
        if self._index != len(self._tokens):
            raise ConfigError("Unexpected token", details={"token": self._tokens[self._index]})
        return node

    def _expr(self) -> _Node:
        node = self._term()
        while self._op() in {"+", "-"}:
            op = self._pop_op()
            node = ("bin", op, node, self._term())
        return node

    def _term(self) -> _Node:
        node = self._power()
        while self._op() in {"*", "/"}:
            op = self._pop_op()
            node = ("bin", op, node, self._power())
        return node

    def _power(self) -> _Node:
        node = self._unary()
        if self._op() == "**":
            self._pop_op()
            node = ("bin", "**", node, self._power())
        return node

    def _unary(self) -> _Node:
        if self._op() == "+":
            self._pop_op()
            return self._unary()
        if self._op() == "-":
            self._pop_op()
            return ("neg", self._unary())
        return self._primary()

    def _primary(self) -> _Node:
        token = self._pop()
        if token[0] == "num":
            return ("num", float(token[1]))
        if token[0] == "name":
            return self._name(str(token[1]))
        if token == ("op", "("):
            node = self._expr()
            if self._pop() != ("op", ")"):
                raise ConfigError("Missing closing parenthesis")
            return node
        raise ConfigError("Unexpected token", details={"token": token})

    def _name(self, name: str) -> _Node:
        if self._op() == "(" and name in _FUNCS:
            self._pop_op()
            arg = self._expr()
            if self._pop() != ("op", ")"):
                raise ConfigError("Missing closing parenthesis", details={"function": name})
            return ("call", name, arg)
        if name != "x" and name not in self._params:
            raise ConfigError("Unknown name", details={"name": name})
        return ("name", name)

    def _op(self) -> str | None:
        if self._index >= len(self._tokens):
            return None
        token = self._tokens[self._index]
        return str(token[1]) if token[0] == "op" else None

    def _pop_op(self) -> str:
        token = self._pop()
        return str(token[1])

    def _pop(self) -> tuple[str, str]:
        token = self._tokens[self._index]
        self._index += 1
        return token


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    index = 0
    while index < len(text):
        if text[index].isspace():
            index += 1
            continue
        index = _token(text, index, tokens)
    return tokens


def _token(text: str, index: int, tokens: list[tuple[str, str]]) -> int:
    if text.startswith("**", index):
        tokens.append(("op", "**"))
        return index + 2
    if text[index] in "+-*/(),":
        tokens.append(("op", text[index]))
        return index + 1
    if text[index].isdigit() or text[index] == ".":
        return _number(text, index, tokens)
    if text[index].isalpha():
        return _word(text, index, tokens)
    raise ConfigError("Forbidden token", details={"index": index, "text": text[index]})


def _number(text: str, index: int, tokens: list[tuple[str, str]]) -> int:
    end = index + 1
    while end < len(text) and (text[end].isdigit() or text[end] in ".eE+-"):
        if text[end] in "+-" and text[end - 1] not in "eE":
            break
        end += 1
    tokens.append(("num", text[index:end]))
    return end


def _word(text: str, index: int, tokens: list[tuple[str, str]]) -> int:
    end = index + 1
    while end < len(text) and (text[end].isalnum() or text[end] == "_"):
        end += 1
    tokens.append(("name", text[index:end]))
    return end


def _evaluate(node: _Node, x: float, params: dict[str, float]) -> float:
    kind = node[0]
    if kind == "num":
        return _number_value(node[1])
    if kind == "name":
        if node[1] == "x":
            return x
        return float(params[str(node[1])])
    if kind == "neg":
        return -_evaluate(_child(node, 1), x, params)
    if kind == "call":
        value = _FUNCS[str(node[1])](_evaluate(_child(node, 2), x, params))
        return value
    left = _evaluate(_child(node, 2), x, params)
    right = _evaluate(_child(node, 3), x, params)
    return _binary(str(node[1]), left, right)


def _number_value(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    raise ConfigError("Broken number")


def _child(node: _Node, index: int) -> _Node:
    child = node[index]
    if not isinstance(child, tuple):
        raise ConfigError("Broken expression")
    return child


def _binary(op: str, left: float, right: float) -> float:
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if op == "/":
        return left / right
    if op == "**":
        return cast(float, left**right)
    raise ConfigError("Unknown operator", details={"op": op})


_install()
