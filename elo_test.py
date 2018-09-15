import elo
import unittest

class EloTests(unittest.TestCase):
    
    def test_expected_outcome_same_ranking(self):
        outcome = elo.expected_outcome(1000, 1000)
        self.assertEquals(outcome, 0.5)
        
    def test_expected_outcome_different_rankings(self):
        expected = 0.35993500019711494
        outcome = elo.expected_outcome(1000, 1050)

        self.assertAlmostEquals(outcome, expected)
        outcome = elo.expected_outcome(1050, 1000)
        self.assertAlmostEquals(outcome, 1-expected)
        
    def test_update_rating_sigmoid(self):
        change = elo.update_rating_sigmoid(1050, 1000, 1)
        self.assertAlmostEquals(change, 7.198700003942298)
        
        change = elo.update_rating_sigmoid(1000, 1050, 1)
        self.assertAlmostEquals(change, 12.801299996057702)
        
        change = elo.update_rating_sigmoid(1050, 1000, 0)
        self.assertAlmostEquals(change, -12.801299996057702)
        
        change = elo.update_rating_sigmoid(1000, 1050, 0)
        self.assertAlmostEquals(change, -7.198700003942298)
        
    def test_mean_regression(self):
        self.assertEquals(elo.mean_regression(1050), 1040.0)
        self.assertEquals(elo.mean_regression(950), 960.0)
        self.assertEquals(elo.mean_regression(1000), 1000)