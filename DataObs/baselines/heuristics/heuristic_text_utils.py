# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small RevThink text helpers used by the heuristic augmentation scripts."""

from __future__ import annotations

import re


def get_true_false(text: str) -> str:
    """Return the last true/false token in generated text, or N/A if absent."""
    if not text:
        return "N/A"
    match = re.findall(r"(true|false)", text, re.IGNORECASE)
    return match[-1].lower() if match else "N/A"


def remove_backward_answer(text: str) -> str:
    """Remove the original baseline's trailing answer annotation from a backward question."""
    pattern = r"The correct answer is \([A-Z]\)\."
    return re.sub(pattern, "", text).strip()
