"""Risk layer — hard rules enforced by system, not modifiable by agent."""
from .rules import *
from .guards import check_deployment, check_strategy_health, check_factor_promotion
