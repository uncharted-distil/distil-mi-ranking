from distutils.core import setup

setup(
    name='DistilMIRanking',
    version='0.2.0',
    description='Mutual information ranking of features',
    packages=['miranking'],
    keywords=['d3m_primitive'],
    install_requires=[
        'pandas>=0.23.4',
        'frozendict>=1.2',
        'scikit-learn>=0.20.2',
        'd3m'
    ],
    entry_points={
        'd3m.primitives': [
            'feature_selection.mi_ranking.DistilMIRanking = miranking.mi_ranking:MIRankingPrimitive'
        ],
    }
)
