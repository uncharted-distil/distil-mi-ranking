import typing
import os
import csv
import collections

import frozendict  # type: ignore
import pandas as pd  # type: ignore
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

from d3m import container, exceptions, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer
from common_primitives import utils

__all__ = ('MIRankingPrimitive',)


class Hyperparams(hyperparams.Hyperparams):
    target_col_index = hyperparams.Hyperparameter[typing.Optional[int]](
        default=None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Index of source vector column'
    )


class MIRankingPrimitive(transformer.TransformerPrimitiveBase[container.DataFrame,
                                                              container.DataFrame,
                                                              Hyperparams]):
    """
    Feature ranking based on a mutual information between features and a selected
    target.  Will rank any feature column with a semantic type of Float, Boolean,
    Integer or Categorical, and a corresponding structural type of int or float.
    """

    # allowable target column types
    _discrete_types = (
        'http://schema.org/Boolean',
        'http://schema.org/Integer',
        'https://metadata.datadrivendiscovery.org/types/CategoricalData'
    )

    _continous_types = (
        'http://schema.org/Float',
    )

    _structural_types = set((
        'int',
        'float'
    ))

    _semantic_types = set(_discrete_types).union(_continous_types)

    __author__ = 'Uncharted Software',
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'a31b0c26-cca8-4d54-95b9-886e23df8886',
            'version': '0.1.0',
            'name': 'Mutual Information Feature Ranking',
            'python_path': 'd3m.primitives.distil.MIFRanking',
            'keywords': ['vector', 'columns', 'dataframe'],
            'source': {
                'name': 'Uncharted Software',
                'contact': 'mailto:chris.bethune@uncharted.software'
            },
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://gitlab.com/unchartedsoftware/distil-mi-ranking.git@' +
                               '{git_commit}#egg=distil-mi-ranking'
                               .format(git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),),
            }],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.DATA_CONVERSION,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        }
    )

    @classmethod
    def _can_use_column(cls, inputs_metadata: metadata_base.DataMetadata, column_index: typing.Optional[int]) -> bool:

        column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

        valid_struct_type = column_metadata.get('structural_type', []) in cls._structural_types

        semantic_types = column_metadata.get('semantic_types', [])
        valid_semantic_type = len(set(cls._semantic_types).intersection(semantic_types)) > 0

        return valid_struct_type and valid_semantic_type

    @classmethod
    def _get_continous_features(cls, inputs: container.DataFrame) -> pd.Dataframe:
        col_indices = utils.list_columns_with_semantic_types(inputs.metadata, cls._continous_types)
        return inputs.values().iloc[:, col_indices]

    @classmethod
    def _get_discrete_features(cls, inputs: container.DataFrame) -> pd.Dataframe:
        col_indices = utils.list_columns_with_semantic_types(inputs.metadata, cls._discrete_types)
        return inputs.values().iloc[:, col_indices]

    def produce(self, *,
                inputs: container.DataFrame,
                timeout: float = None,
                iterations: int = None) -> base.CallResult[container.DataFrame]:

        # make sure the target column is of a valid type
        target_idx = self.hyperparams['target_col_index']
        if not self._can_use_column(inputs.metadata, target_idx):
            raise exceptions.InvalidArgumentValueError('column idx=' + str(target_idx) + ' from '
                                                       + str(inputs.columns)
                                                       + ' does not contain continuous or discreet type')

        # check if target is discrete or continuous
        semantic_types = inputs.metadata.query_column(target_idx)['semantic_types']
        discrete = len(set(semantic_types).intersection(self._discrete_types)) > 0

        # split out the target, continuous and discrete features
        target_df = inputs.values().iloc[:, target_idx]
        continuous_df = self._get_continous_features(inputs)
        discrete_df = self._get_discrete_features(inputs)

        # convert to numpy data types
        target_np = target_df.matrix()
        continuous_np = continuous_df.matrix()
        discrete_np = discrete_df.matrix()

        # compute mutual information for discrete and continuous inputs
        if discrete:
            mi_continuous_x = mutual_info_classif(continuous_np, target_np, discrete_features=False)
            mi_discrete_x = mutual_info_classif(continuous_np, target_np, discrete_features=True)
        else:
            mi_continuous_x = mutual_info_regression(continuous_np, target_np, discrete_features=False)
            mi_discrete_x = mutual_info_regression(continuous_np, target_np, discrete_features=True)

        # merge into a single list and sort

        # wrap as a D3M container - metadata should be auto generated
        return base.CallResult(inputs)

    @classmethod
    def can_accept(cls, *,
                   method_name: str,
                   arguments: typing.Dict[str, typing.Union[metadata_base.Metadata, type]],
                   hyperparams: Hyperparams) -> typing.Optional[metadata_base.DataMetadata]:
        output_metadata = super().can_accept(method_name=method_name, arguments=arguments, hyperparams=hyperparams)

        # If structural types didn't match, don't bother.
        if output_metadata is None:
            return None

        if method_name != 'produce':
            return output_metadata

        if 'inputs' not in arguments:
            return output_metadata

        inputs_metadata = typing.cast(metadata_base.DataMetadata, arguments['inputs'])

        # make sure there's a real vector column (search if unspecified)
        vector_col_index = hyperparams['vector_col_index']
        if vector_col_index is not None:
            can_use_column = cls._can_use_column(inputs_metadata, vector_col_index)
            if not can_use_column:
                return None
        else:
            inferred_index = cls._find_real_vector_column(inputs_metadata)
            if inferred_index is None:
                return None

        return inputs_metadata
