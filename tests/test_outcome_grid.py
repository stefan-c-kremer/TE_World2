import importlib.util
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "TEWorldCodeV2" / "TE-Simulations" / "CreateOutcomeGrid.py"
spec = importlib.util.spec_from_file_location("outcome_grid", SCRIPT)
outcome_grid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(outcome_grid)


class OutcomeGridTests(unittest.TestCase):
    def test_corrected_bits_map_into_low_excision_columns(self):
        self.assertEqual(outcome_grid.grid_position("HHHHHHH-Z"), (4, 0))
        self.assertEqual(outcome_grid.grid_position("HLLHHHH-Z"), (7, 0))
        self.assertEqual(outcome_grid.grid_position("LHHLLLL-Z"), (12, 15))
        self.assertEqual(outcome_grid.grid_position("LLLLLLL-Z"), (15, 15))

    def test_every_corrected_condition_maps_to_a_unique_low_excision_cell(self):
        import itertools

        positions = {
            outcome_grid.grid_position("".join(bits) + "-Z")
            for bits in itertools.product("HL", repeat=7)
        }
        self.assertEqual(len(positions), 128)
        self.assertTrue(all(column % 8 >= 4 for column, _ in positions))

    def test_first_experiment_transcription_has_six_slots_in_every_cell(self):
        positions = outcome_grid.first_experiment_positions()
        self.assertEqual(len(positions), 256)
        self.assertTrue(all(len(statuses) == 6 for statuses in positions.values()))

        totals = {
            status: sum(statuses.count(status) for statuses in positions.values())
            for status in outcome_grid.COLORS
        }
        self.assertEqual(totals["incomplete"], 768)
        self.assertEqual(totals["host_extinction"], 169)
        self.assertEqual(totals["maximum_generations"], 23)
        self.assertEqual(totals["te_extinction"], 576)

    def test_renderer_uses_requested_number_of_slots(self):
        positions = {(0, 0): ["incomplete", "host_extinction", "te_extinction"]}
        svg = outcome_grid.render_svg(
            positions,
            slots_per_cell=3,
            title="Title",
            subtitle="Subtitle",
            description="Description",
            note="Note",
        )
        self.assertEqual(svg.count('class="slot"'), 16 * 16 * 3)
        self.assertIn("Subtitle", svg)


if __name__ == "__main__":
    unittest.main()
