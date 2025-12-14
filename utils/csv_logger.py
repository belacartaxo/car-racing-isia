from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from stable_baselines3.common.logger import (
    Figure,
    FormatUnsupportedError,
    HParam,
    Image,
    KVWriter,
    Video,
    configure,
    filter_excluded_keys,
)


class AppendingCSVOutputFormat(KVWriter):
    """
    CSV logger que replica o formato do stdout do Stable-Baselines3:
    - append sem truncar
    - colunas criadas dinamicamente
    - células vazias para métricas ausentes
    - ordem: time → rollout → train
    - floats formatados com 3 algarismos significativos (exatamente como o SB3)
    """

    def __init__(self, filename: str, meta: Optional[Dict[str, Any]] = None):
        self.path = Path(filename)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.file = open(self.path, "a+", encoding="utf-8", newline="")
        self.separator = ";"
        self.quotechar = '"'

        self.meta = meta or {}
        self.keys: list[str] = []

        self._load_existing_header()

    # ------------------------------------------------------------------

    def _load_existing_header(self) -> None:
        self.file.seek(0)
        header = self.file.readline().strip()
        if header:
            self.keys = header.split(self.separator)
        self.file.seek(0, os.SEEK_END)

    # ------------------------------------------------------------------

    def _sort_keys(self, keys: list[str]) -> list[str]:
        def priority(k: str):
            if k.startswith("time/"):
                return (0, k)
            if k.startswith("rollout/"):
                return (1, k)
            if k.startswith("train/"):
                return (2, k)
            return (3, k)

        return sorted(keys, key=priority)

    # ------------------------------------------------------------------

    def _rewrite_with_new_keys(self, extra_keys: list[str]) -> None:
        extra_keys = self._sort_keys(extra_keys)
        self.keys.extend(extra_keys)

        self.file.seek(0)
        lines = self.file.readlines()

        self.file.seek(0)
        self.file.truncate()

        # header
        self.file.write(self.separator.join(self.keys) + "\n")

        # linhas antigas com células vazias
        for line in lines[1:]:
            line = line.rstrip("\n")
            self.file.write(
                line
                + self.separator
                + self.separator.join([""] * len(extra_keys))
                + "\n"
            )

    # ------------------------------------------------------------------

    def _format_float(self, value: float) -> str:
        """
        Formato idêntico ao stdout do Stable-Baselines3:
        3 algarismos significativos, notação científica quando necessário.
        """
        return format(value, ".3g")

    # ------------------------------------------------------------------

    def _write_value(self, value: Any) -> None:
        if isinstance(value, (Video, Figure, Image, HParam)):
            raise FormatUnsupportedError(["csv"], type(value).__name__)

        if isinstance(value, str):
            escaped = value.replace(self.quotechar, self.quotechar * 2)
            self.file.write(self.quotechar + escaped + self.quotechar)

        elif value is not None:
            if isinstance(value, bool):
                self.file.write("1" if value else "0")
            elif isinstance(value, int):
                self.file.write(str(value))
            elif isinstance(value, float):
                self.file.write(self._format_float(value))
            else:
                self.file.write(str(value))

    # ------------------------------------------------------------------

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, tuple[str, ...]],
        step: int = 0,
    ) -> None:
        filtered = filter_excluded_keys(key_values, key_excluded, "csv")
        merged: Dict[str, Any] = {**self.meta, **filtered}

        extra_keys = [k for k in merged.keys() if k not in self.keys]
        if extra_keys:
            self._rewrite_with_new_keys(extra_keys)

        for i, key in enumerate(self.keys):
            if i > 0:
                self.file.write(self.separator)
            self._write_value(merged.get(key))

        self.file.write("\n")
        self.file.flush()

    # ------------------------------------------------------------------

    def close(self) -> None:
        self.file.close()


# ======================================================================
# FUNÇÕES AUXILIARES
# ======================================================================

def attach_csv_logger(model, csv_path: Path, meta: Optional[Dict[str, Any]] = None):
    """
    Anexa o CSV logger ao modelo SB3 sem remover outros outputs.
    """
    if not hasattr(model, "_logger") or model.__dict__.get("_logger") is None:
        model.set_logger(configure(None, ["stdout"]))

    fmt = AppendingCSVOutputFormat(str(csv_path), meta=meta)
    model.logger.output_formats.append(fmt)
    return fmt


def read_last_metric(csv_path: Path, key: str) -> Optional[float]:
    """
    Lê o último valor válido de uma métrica no CSV.
    """
    if not csv_path.exists():
        return None

    import csv

    last_value = None
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        rows = list(reader)

        if not rows:
            return None

        header = rows[0]
        if key not in header:
            return None

        idx = header.index(key)

        for row in rows[1:]:
            if len(row) <= idx:
                continue
            cell = row[idx].strip()
            if cell == "":
                continue
            try:
                last_value = float(cell)
            except ValueError:
                pass

    return last_value
