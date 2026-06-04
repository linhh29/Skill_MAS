#!/usr/bin/env python3
"""Skill-MAS DRB runner entry under deep_research_bench."""

from __future__ import annotations

try:
    from .skill_mas_agent_runner import main
except ImportError:
    from skill_mas_agent_runner import main


if __name__ == "__main__":
    main()
