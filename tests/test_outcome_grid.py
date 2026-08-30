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


if __name__ == "__main__":
    unittest.main()
