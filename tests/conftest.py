"""Shared pytest configuration. Prevents wandb from creating local run files."""

import os

os.environ["WANDB_MODE"] = "disabled"
