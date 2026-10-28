"""bcpd-agent: an LLM agent that answers questions about regime changes in time series.

The agent never computes statistics itself. It calls tools (Python functions)
that load data and run exact Bayesian changepoint detection with the `bcpd`
package, and then explains the results. Guardrails check that every number in
the answer comes from a tool, and an evaluation suite measures how often it is
right.

See GUIDE.md for a full walkthrough.
"""

__version__ = "0.1.0"
