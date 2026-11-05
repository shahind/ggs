"""Spelling in the prompt box: unknown words get a red wavy underline, right-click offers fixes.

A typo goes straight into the model ("clawn" draws something, but not a clown), so the box shows it before the
user presses a button. Uses pyspellchecker (MIT, English word list) when installed; without it nothing is
underlined.
"""

from __future__ import annotations

import re
from functools import lru_cache

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QSyntaxHighlighter, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QPlainTextEdit

WORD = re.compile(r"[A-Za-z][A-Za-z']+")
# words that are fine in prompts but missing from a general dictionary
EXTRA = {"chibi", "pixelart", "sprite", "sprites", "spritesheet", "tileable", "isometric", "steampunk",
         "cyberpunk", "lowpoly", "ui", "hud", "vfx", "rpg", "npc", "npcs", "gui", "orc", "orcs", "goblin",
         "goblins", "katana", "dreadlocks", "hoodie", "cel", "unshaded", "outlines", "ggs"}


@lru_cache(maxsize=1)
def _checker():
    try:
        from spellchecker import SpellChecker
    except ImportError:
        return None
    sc = SpellChecker(distance=1)
    sc.word_frequency.load_words(EXTRA)
    return sc


def unknown_words(text: str) -> set[str]:
    sc = _checker()
    if sc is None:
        return set()
    words = {w.lower() for w in WORD.findall(text) if not any(c.isupper() for c in w[1:])}  # skip GGSINK, McX
    return {w for w in sc.unknown(words) if len(w) > 2}


def suggestions(word: str, limit: int = 5) -> list[str]:
    sc = _checker()
    if sc is None:
        return []
    best = sc.correction(word.lower())
    cands = sorted(sc.candidates(word.lower()) or [], key=lambda w: -sc.word_usage_frequency(w))
    out = ([best] if best and best != word.lower() else []) + [c for c in cands if c != best and c != word.lower()]
    if word[:1].isupper():
        out = [c.capitalize() for c in out]
    return out[:limit]


class SpellHighlighter(QSyntaxHighlighter):
    def __init__(self, doc):
        super().__init__(doc)
        self.fmt = QTextCharFormat()
        self.fmt.setUnderlineStyle(QTextCharFormat.UnderlineStyle.SpellCheckUnderline)
        self.fmt.setUnderlineColor(QColor("#ff5c5c"))

    def highlightBlock(self, text: str) -> None:
        bad = unknown_words(text)
        if not bad:
            return
        for m in WORD.finditer(text):
            if m.group(0).lower() in bad:
                self.setFormat(m.start(), m.end() - m.start(), self.fmt)


class PromptEdit(QPlainTextEdit):
    """A short prompt box with spelling marks. Enter sends (``submitted``), Shift+Enter makes a new line."""

    submitted = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighter = SpellHighlighter(self.document())
        self.setTabChangesFocus(True)

    def keyPressEvent(self, e) -> None:
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.submitted.emit()
            return
        super().keyPressEvent(e)

    def has_typos(self) -> list[str]:
        return sorted(unknown_words(self.toPlainText()))

    @staticmethod
    def _replacer(cursor: QTextCursor, word: str):
        return lambda _=False: cursor.insertText(word)

    def contextMenuEvent(self, e) -> None:
        menu = self.createStandardContextMenu()
        cur = self.cursorForPosition(e.pos())
        cur.select(QTextCursor.SelectionType.WordUnderCursor)
        word = cur.selectedText()
        if word and word.lower() in unknown_words(word):
            first = menu.actions()[0] if menu.actions() else None
            fixes = suggestions(word)
            for s in fixes:
                act = menu.addAction(s)
                act.triggered.connect(self._replacer(QTextCursor(cur), s))
                if first is not None:
                    menu.insertAction(first, act)
            if not fixes:
                none = menu.addAction("(no suggestions)")
                none.setEnabled(False)
                if first is not None:
                    menu.insertAction(first, none)
            if first is not None:
                menu.insertSeparator(first)
        menu.exec(e.globalPos())
