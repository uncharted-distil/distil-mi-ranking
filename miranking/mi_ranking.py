import typing
import os
import csv
import collections

import frozendict  # type: ignore
import pandas as pd  # type: ignore
import numpy as np
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
        description='Index of target feature to rank against.'
    )


class MIRankingPrimitive(transformer.TransformerPrimitiveBase[container.DataFrame,
                                                              container.DataFrame,
                                                              Hyperparams]):
    """
    Feature ranking based on a mutual information between features and a selected
    target.  Will rank any feature column with a semantic type of Float, Boolean,
    Integer or Categorical, and a corresponding structural type of int or float.
    A DataFrame containing (col_idx, col_name, score) tuples for each ranked feature
    will be returned to the caller.  Features that could not be ranked are excluded
    from the returned set.
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
        int,
        float
    ))

    _semantic_types = set(_discrete_types).union(_continous_types)

    _random_seed = 100

    __author__ = 'Uncharted Software',
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'a31b0c26-cca8-4d54-95b9-886e23df8886',
            'version': '0.1.0',
            'name': 'Mutual Information Feature Ranking',
            'python_path': 'd3m.primitives.distil.MIRanking',
            'keywords': ['vector', 'columns', 'dataframe'],
            'source': {
                'name': 'Uncharted Software',
                'contact': 'mailto:cbethune@uncharted.software'
            },
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://github.com/unchartedsoftware/distil-mi-ranking.git@' +
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
    def _append_rank_info(cls,
                          inputs: container.DataFrame,
                          result: typing.List[typing.Tuple[int, str, float]],
                          rank_np: np.array,
                          rank_df: pd.DataFrame) -> typing.List[typing.Tuple[int, str, float]]:
        for i, rank in enumerate(rank_np):
            col_name = rank_df.columns.values[i]
            result.append((inputs.columns.get_loc(col_name), col_name, rank))
        return result

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

        # split out the target feature
        target_df = inputs.iloc[:, target_idx]

        # drop features that are not compatible with ranking
        feature_indices = set(utils.list_columns_with_semantic_types(inputs.metadata, self._semantic_types))
        all_indices = set(range(0, inputs.shape[1]))
        skipped_indices = all_indices.difference(feature_indices)
        skipped_indices.add(target_idx)  # drop the target too
        feature_df = inputs
        for i, v in enumerate(skipped_indices):
            feature_df = feature_df.drop(inputs.columns[v], axis=1)

        # figure out the discrete and continuous feature indices and create an array
        # that flags them
        discrete_indices = utils.list_columns_with_semantic_types(inputs.metadata, self._discrete_types)
        discrete_flags = [False] * feature_df.shape[1]
        for v in discrete_indices:
            col_name = inputs.columns[v]
            if col_name in feature_df:
                col_idx = feature_df.columns.get_loc(col_name)
                discrete_flags[col_idx] = True

        # convert to numpy data types
        target_np = target_df.values
        feature_np = feature_df.values

        # compute mutual information for discrete or continuous target
        ranked_features_np = None
        if discrete:
            ranked_features_np = mutual_info_classif(feature_np,
                                                     target_np,
                                                     discrete_features=discrete_flags,
                                                     random_state=self._random_seed)
        else:
            ranked_features_np = mutual_info_regression(feature_np,
                                                        target_np,
                                                        discrete_features=discrete_flags,
                                                        random_state=self._random_seed)

        # merge back into a single list of col idx / rank value tuples
        data: typing.List[typing.Tuple[int, str, float]] = []
        data = self._append_rank_info(inputs, data, ranked_features_np, feature_df)

        cols = ['idx', 'name', 'rank']
        results = container.DataFrame(data=data, columns=cols)
        results = results.sort_values(by=['rank'], ascending=False).reset_index(drop=True)

        # wrap as a D3M container - metadata should be auto generated
        return base.CallResult(results)

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

        # make sure target column is discrete or continuous (search if unspecified)
        target_col_index = hyperparams['target_col_index']
        if target_col_index is not None:
            can_use_column = cls._can_use_column(inputs_metadata, target_col_index)

        if not can_use_column:
            return None

        return inputs_metadata
