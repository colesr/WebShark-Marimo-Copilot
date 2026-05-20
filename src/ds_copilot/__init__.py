"""DAG-aware AI copilot for reactive data-science notebooks."""

from ds_copilot.agent import LeakageAudit, Plan, ProposedCell, plan
from ds_copilot.decisions import DecisionEvent, DecisionLog, history
from ds_copilot.profiler import (
    ColumnProfile,
    LeakageHint,
    Profile,
    profile,
    profile_widget,
)
from ds_copilot.ui import PlannerWidget, planner_widget

__version__ = "0.0.1"

__all__ = [
    "ColumnProfile",
    "DecisionEvent",
    "DecisionLog",
    "LeakageAudit",
    "LeakageHint",
    "Plan",
    "PlannerWidget",
    "Profile",
    "ProposedCell",
    "__version__",
    "history",
    "plan",
    "planner_widget",
    "profile",
    "profile_widget",
]
