from distutils.core import setup

setup(
    name='DistilMIRanking',
    version='0.1.0',
    description='Mutual information ranking of features',
    packages=['miranking'],
    keywords=['d3m_primitive'],
    install_requires=[
        'pandas == 0.22.0',
        'frozendict==1.2',
        'scikit-learn',
        'd3m'
    ],
    entry_points={
        'd3m.primitives': [
            'distil.MIRanking = miranking.mi_ranking:MIRankingPrimitive'
        ],
    }
)
