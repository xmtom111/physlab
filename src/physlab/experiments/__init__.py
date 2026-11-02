"""Experiment protocols, execution, fitting, and reports."""

from physlab.experiments.analysis import ModelFit, fit_custom, fit_models, report
from physlab.experiments.protocol import ExperimentProtocol, validate_protocol
from physlab.experiments.registry import SkillRegistry, experiment_registry
from physlab.experiments.results import RunResult, TrialRecord
from physlab.experiments.runner import ExperimentRunner

__all__ = [
    "ExperimentProtocol",
    "ExperimentRunner",
    "ModelFit",
    "RunResult",
    "SkillRegistry",
    "TrialRecord",
    "experiment_registry",
    "fit_custom",
    "fit_models",
    "report",
    "validate_protocol",
]
