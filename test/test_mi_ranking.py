import unittest
from os import path
import csv
import typing
import pandas as pd

from d3m import container
from d3m.primitives.distil import MIRanking
from d3m.metadata import base as metadata_base


class MIRankingPrimitiveTestCase(unittest.TestCase):

    _dataset_path = path.abspath(path.join(path.dirname(__file__), 'dataset'))

    def test_discrete_target(self) -> None:
        dataframe = self._load_data()

        hyperparams_class = \
            MIRanking.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        hyperparams = hyperparams_class.defaults().replace(
            {
                'target_col_index': 1
            }
        )
        mi_ranking = MIRanking(hyperparams=hyperparams)
        result_dataframe = mi_ranking.produce(inputs=dataframe).value

        # verify the output
        self.assertListEqual(list(result_dataframe['idx']), [2, 5, 3])
        self.assertListEqual(list(result_dataframe['name']), ['bravo', 'echo', 'charlie'])
        expected_ranks = [1.405357, 0.562335, 0.042475]
        for i, r in enumerate(result_dataframe['rank']):
            self.assertAlmostEqual(r, expected_ranks[i], places=6)

    def test_continuous_target(self) -> None:
        dataframe = self._load_data()

        hyperparams_class = \
            MIRanking.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        hyperparams = hyperparams_class.defaults().replace(
            {
                'target_col_index': 2
            }
        )
        mi_ranking = MIRanking(hyperparams=hyperparams)
        result_dataframe = mi_ranking.produce(inputs=dataframe).value

        # verify the output
        self.assertListEqual(list(result_dataframe['idx']), [1, 5, 3])
        self.assertListEqual(list(result_dataframe['name']), ['alpha', 'echo', 'charlie'])
        expected_ranks = [1.405357, 0.422024, 0.0]
        for i, r in enumerate(result_dataframe['rank']):
            self.assertAlmostEqual(r, expected_ranks[i], places=6)

    def _load_data(cls) -> container.DataFrame:
        dataset_doc_path = path.join(cls._dataset_path, 'datasetDoc.json')

        # load the dataset and convert resource 0 to a dataframe
        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))
        dataframe = dataset['0']
        dataframe.metadata = dataframe.metadata.set_for_value(dataframe)

        # set the struct type
        dataframe.metadata = dataframe.metadata.update((metadata_base.ALL_ELEMENTS, 0),
                                                       {'structural_type': int})
        dataframe.metadata = dataframe.metadata.update((metadata_base.ALL_ELEMENTS, 1),
                                                       {'structural_type': int})
        dataframe.metadata = dataframe.metadata.update((metadata_base.ALL_ELEMENTS, 2),
                                                       {'structural_type': float})
        dataframe.metadata = dataframe.metadata.update((metadata_base.ALL_ELEMENTS, 3),
                                                       {'structural_type': int})
        dataframe.metadata = dataframe.metadata.update((metadata_base.ALL_ELEMENTS, 4),
                                                       {'structural_type': str})
        dataframe.metadata = dataframe.metadata.update((metadata_base.ALL_ELEMENTS, 5),
                                                       {'structural_type': int})

        # set the semantic type
        dataframe.metadata = dataframe.metadata.\
            add_semantic_type((metadata_base.ALL_ELEMENTS, 1),
                              'https://metadata.datadrivendiscovery.org/types/CategoricalData')
        dataframe.metadata = dataframe.metadata.\
            add_semantic_type((metadata_base.ALL_ELEMENTS, 2), 'http://schema.org/Float')
        dataframe.metadata = dataframe.metadata.\
            add_semantic_type((metadata_base.ALL_ELEMENTS, 3), 'http://schema.org/Boolean')
        dataframe.metadata = dataframe.metadata.\
            add_semantic_type((metadata_base.ALL_ELEMENTS, 4), 'http://schema.org/Text')
        dataframe.metadata = dataframe.metadata.\
            add_semantic_type((metadata_base.ALL_ELEMENTS, 5), 'http://schema.org/Integer')

        return dataframe

if __name__ == '__main__':
    unittest.main()
