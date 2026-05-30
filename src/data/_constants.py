"""Shared constants for the data package.

Single source of truth for action-to-text mappings and other shared lookups
that were previously duplicated across dataset and collator modules.
"""

# System-2 action encoding: discrete action code → human-readable text.
SYSTEM2_ACTION_TEXT: dict[int, str] = {
    0: "STOP",
    1: "↑",
    2: "←",
    3: "→",
    5: "↓",
}
