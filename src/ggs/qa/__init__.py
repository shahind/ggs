"""Automatic asset checks. Every asset passes QA before export (hard rule 5)."""

from ggs.qa.sprite import QaResult, SpriteQaRules, check_sprite

__all__ = ["QaResult", "SpriteQaRules", "check_sprite"]
