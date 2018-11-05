import unittest
from os import path
import csv
import typing

from d3m import container
from d3m.primitives.distil import MIRanking
from d3m.metadata import base as metadata_base


class MIRankingPrimitiveTestCase(unittest.TestCase):

    _dataset_path = path.abspath(path.join(path.dirname(__file__), 'dataset'))

    def test_basic(self) -> None:
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
        self.assertTrue(True)

    def _load_data(cls) -> container.DataFrame:
        dataset_doc_path = path.join(cls._dataset_path, 'datasetDoc.json')

        # load the dataset and convert resource 0 to a dataframe
        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))
        dataframe = dataset['0']
        dataframe.metadata = dataframe.metadata.set_for_value(dataframe)

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
