import os
import unittest
import numpy as np
from tblup import BlupParallelEvaluator


geno = np.random.randint(0, 2, (100, 100))
pheno = np.random.rand(100)


class EvaluatorTest(unittest.TestCase):

    GENO_FILE = "./geno.npy"
    PHENO_FILE = "./pheno.npy"

    def setUp(self):
        np.save(self.GENO_FILE, geno)
        np.save(self.PHENO_FILE, pheno)

    def tearDown(self):
        os.remove(self.GENO_FILE)
        os.remove(self.PHENO_FILE)


class TestBlupParallelEvaluator(EvaluatorTest):
    def test_index_creation(self):
        evaluator = BlupParallelEvaluator(self.GENO_FILE, self.PHENO_FILE, 0.5)

        train = set(evaluator.training_indices)
        valid = set(evaluator.validation_indices)
        test = set(evaluator.testing_indices)

        self.assertEqual(len(train), len(evaluator.training_indices))
        self.assertEqual(len(valid), len(evaluator.validation_indices))
        self.assertEqual(len(test), len(evaluator.testing_indices))

        self.assertEqual(len(train.intersection(valid)), 0)
        self.assertEqual(len(train.intersection(test)), 0)
        self.assertEqual(len(valid.intersection(test)), 0)


if __name__ == '__main__':
    unittest.main()
