"""Packed-coordinate chromosome backend for TE World 2.

The biological model and random-number calls remain in TESim.  This backend
changes only chromosome storage: element coordinates live in a NumPy array,
and immutable gene objects are shared when hosts are cloned.
"""

from __future__ import annotations

import numpy as np

import TESim as reference


class CompactTestChromosome2(reference.TestChromosome2):
    """A TestChromosome2 with packed coordinates and inexpensive gene clones."""

    def __init__(self, length=None, elements=None, starts=None, te_mask=None):
        chromosome_length = length or self.length
        if type(chromosome_length) not in (int, float):
            raise Exception(repr(chromosome_length))

        self.length = chromosome_length
        self.elements = [] if elements is None else elements
        if starts is None:
            input_starts = np.fromiter(
                (element.start for element in self.elements),
                dtype=np.int64,
                count=len(self.elements),
            )
        else:
            input_starts = np.asarray(starts, dtype=np.int64)

        if len(input_starts) != len(self.elements):
            raise ValueError("elements and starts must have the same length")

        if te_mask is None:
            input_te_mask = np.fromiter(
                (
                    isinstance(element, reference.SelectiveInsertTE)
                    for element in self.elements
                ),
                dtype=np.bool_,
                count=len(self.elements),
            )
        else:
            input_te_mask = np.asarray(te_mask, dtype=np.bool_)
        if len(input_te_mask) != len(self.elements):
            raise ValueError("elements and te_mask must have the same length")

        self._size = len(self.elements)
        capacity = max(16, 1 << max(0, self._size - 1).bit_length())
        self._starts = np.empty(capacity, dtype=np.int64)
        self._te_mask = np.empty(capacity, dtype=np.bool_)
        self._starts[: self._size] = input_starts
        self._te_mask[: self._size] = input_te_mask

        self.host = None
        for element in self.elements:
            if isinstance(element, reference.SelectiveInsertTE):
                element.chromosome = self
        self._recount_live_tes()

    def __setstate__(self, state):
        """Restore process-local identity indexes after unpickling."""
        self.__dict__.update(state)
        for element_index in np.flatnonzero(self._active_te_mask()):
            self.elements[element_index].chromosome = self
        self._recount_live_tes()

    def __getstate__(self):
        """Exclude unused array capacity from compact checkpoints."""
        state = self.__dict__.copy()
        state["_starts"] = self._active_starts().copy()
        state["_te_mask"] = self._active_te_mask().copy()
        return state

    def _active_starts(self):
        return self._starts[: self._size]

    def _active_te_mask(self):
        return self._te_mask[: self._size]

    def _ensure_capacity(self):
        if self._size < len(self._starts):
            return
        new_capacity = max(16, len(self._starts) * 2)
        new_starts = np.empty(new_capacity, dtype=np.int64)
        new_te_mask = np.empty(new_capacity, dtype=np.bool_)
        new_starts[: self._size] = self._active_starts()
        new_te_mask[: self._size] = self._active_te_mask()
        self._starts = new_starts
        self._te_mask = new_te_mask

    def _recount_live_tes(self):
        te_indices = np.flatnonzero(self._active_te_mask())
        self._element_ids = {id(self.elements[index]) for index in te_indices}
        self._max_element_length = reference.ProkGene1.length
        self._live_autonomous_tes = 0
        self._live_non_autonomous_tes = 0
        self._non_autonomous_parent = None
        for element_index in te_indices:
            element = self.elements[element_index]
            self._max_element_length = max(self._max_element_length, element.length)
            if element.dead:
                continue
            if element.autonomous:
                self._live_autonomous_tes += 1
            else:
                self._live_non_autonomous_tes += 1
                if self._non_autonomous_parent is None:
                    self._non_autonomous_parent = element

    def _find_index(self, coordinate):
        """Return the earliest-starting element covering coordinate."""
        earliest_start = coordinate - self._max_element_length + 1
        starts = self._active_starts()
        low = int(np.searchsorted(starts, earliest_start, side="left"))
        for element_index in range(low, len(self.elements)):
            start = int(starts[element_index])
            if start > coordinate:
                break
            element = self.elements[element_index]
            if start <= coordinate < start + element.length:
                return element_index
        return None

    def _set_visible_start(self, element_index):
        """Synchronize one compatibility object before returning it to TESim."""
        element = self.elements[element_index]
        start = int(self._starts[element_index])
        element.start = start
        element.end = start + element.length
        return element

    def _insert_at(self, element, start, *, shift_suffix):
        insertion_index = int(
            np.searchsorted(self._active_starts(), start, side="left")
        )
        self._ensure_capacity()
        if insertion_index < self._size:
            self._starts[insertion_index + 1 : self._size + 1] = self._starts[
                insertion_index : self._size
            ]
            self._te_mask[insertion_index + 1 : self._size + 1] = self._te_mask[
                insertion_index : self._size
            ]
            if shift_suffix:
                self._starts[insertion_index + 1 : self._size + 1] += element.length
        self._starts[insertion_index] = start
        self._te_mask[insertion_index] = isinstance(
            element, reference.SelectiveInsertTE
        )
        self._size += 1
        self.elements.insert(insertion_index, element)
        element.start = int(start)
        element.end = element.start + element.length
        element.chromosome = self
        self._track_inserted_element(element)
        return insertion_index

    def _remove_at(self, element_index):
        removed_element = self.elements.pop(element_index)
        if element_index < self._size - 1:
            self._starts[element_index : self._size - 1] = self._starts[
                element_index + 1 : self._size
            ]
            self._te_mask[element_index : self._size - 1] = self._te_mask[
                element_index + 1 : self._size
            ]
        self._size -= 1
        self._track_removed_element(removed_element)
        return removed_element

    def _identity_index(self, item):
        for element_index, element in enumerate(self.elements):
            if element is item:
                return element_index
        raise ValueError(f"{item!r} is not in chromosome")

    def place(self, element):
        collision_index = self._find_index(element.start)
        if collision_index is not None:
            whats_there = self.elements[collision_index]
            if isinstance(whats_there, reference.SelectiveInsertTE):
                self._remove_at(collision_index)
            else:
                raise reference.ElementDestroyed(element, whats_there)
        self._insert_at(element, element.start, shift_suffix=False)

    def insert(self, element):
        collision = 0
        collision_index = self._find_index(element.start)
        if collision_index is not None:
            whats_there = self.elements[collision_index]
            whats_there_start = int(self._starts[collision_index])
            if isinstance(whats_there, reference.SelectiveInsertTE):
                self._remove_at(collision_index)
                collision = 1
            elif element.start != whats_there_start:
                self.insert_anyway(element)
                raise reference.ElementDestroyed(element, whats_there)

        self.insert_anyway(element)
        return collision

    def insert_anyway(self, element):
        self._insert_at(element, element.start, shift_suffix=True)
        self.length += element.length

    def excise(self, element):
        element_index = self._identity_index(element)
        start = int(self._starts[element_index])
        starts = self._active_starts()
        starts[starts > start] -= element.length
        self._remove_at(element_index)
        self.length -= element.length

    def __getitem__(self, coordinate):
        element_index = self._find_index(coordinate)
        if element_index is None:
            return reference.JUNK
        return self._set_visible_start(element_index)

    def remove(self, item):
        self._remove_at(self._identity_index(item))

    def TEs(self, live=True, dead=True, autonomous=None):
        filtered_tes = []
        for element_index in np.flatnonzero(self._active_te_mask()):
            element = self.elements[element_index]
            if element.dead != dead and element.dead == live:
                continue
            if autonomous is not None and element.autonomous != autonomous:
                continue
            self._set_visible_start(element_index)
            filtered_tes.append(element)
        return filtered_tes

    def genes(self):
        genes = []
        for element_index in np.flatnonzero(~self._active_te_mask()):
            genes.append(self._set_visible_start(element_index))
        return genes

    def get_kidnapping_state(self):
        if self._non_autonomous_parent is None and self._live_non_autonomous_tes > 0:
            for element_index in np.flatnonzero(self._active_te_mask()):
                element = self.elements[element_index]
                if not element.dead and not element.autonomous:
                    self._non_autonomous_parent = element
                    break
        return (
            reference.parameters.Kidnapping_frequency(
                self._live_autonomous_tes,
                self._live_non_autonomous_tes,
            ),
            self._non_autonomous_parent,
        )

    def junk(self):
        return self.length - sum(element.length for element in self.elements)

    def iter_elements_with_starts(self):
        return zip(self.elements, self._active_starts())

    def trace_summary(self):
        live_autonomous = 0
        live_non_autonomous = 0
        dead_autonomous = 0
        dead_non_autonomous = 0
        live_te_starts = []
        for element_index in np.flatnonzero(self._active_te_mask()):
            element = self.elements[element_index]
            if element.dead:
                if element.autonomous:
                    dead_autonomous += 1
                else:
                    dead_non_autonomous += 1
            else:
                live_te_starts.append(int(self._starts[element_index]))
                if element.autonomous:
                    live_autonomous += 1
                else:
                    live_non_autonomous += 1
        return {
            "live_autonomous": live_autonomous,
            "live_non_autonomous": live_non_autonomous,
            "dead_autonomous": dead_autonomous,
            "dead_non_autonomous": dead_non_autonomous,
            "te_starts": live_te_starts,
            "gene_starts": self._active_starts()[~self._active_te_mask()].tolist(),
        }

    def copy(self, host):
        copied_elements = self.elements.copy()
        for element_index in np.flatnonzero(self._active_te_mask()):
            element = copied_elements[element_index]
            copied_elements[element_index] = reference.SelectiveInsertTE(
                int(self._starts[element_index]),
                element.dead,
                element.autonomous,
            )

        result = self.__class__(
            length=self.length,
            elements=copied_elements,
            starts=self._active_starts(),
            te_mask=self._active_te_mask(),
        )
        result.host = host
        return result

    def __repr__(self):
        serialized_elements = []
        for element, start in self.iter_elements_with_starts():
            start = int(start)
            if isinstance(element, reference.ProkGene1):
                serialized_elements.append(f"ProkGene1( {start!r} )")
            else:
                serialized_elements.append(
                    "SelectiveInsertTE( %s, %s, %s )"
                    % (start, element.dead, element.autonomous)
                )
        return "TestChromosome2( %s, [%s] )" % (
            repr(self.length),
            ", ".join(serialized_elements),
        )
