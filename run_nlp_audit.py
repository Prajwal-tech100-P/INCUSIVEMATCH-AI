"""
InclusiveMatch AI - Root module proxy for run_nlp_audit.
Ensures static analyzers (Pyrefly, Pyright, Pylance) and direct script executions
can import run_nlp_audit directly from workspace root or scratch directory.
"""
from scratch.run_nlp_audit import *  # noqa: F401, F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("scratch.run_nlp_audit", run_name="__main__")

