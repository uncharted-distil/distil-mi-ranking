from distutils.core import setup

setup(
    name='DistilMIRanking',
    version='0.1.0',
    description='Mutual information ranking of features',
    packages=['miranking'],
    keywords=['d3m_primitive'],
    install_requires=[
        'pandas>=0.23.4',
        'frozendict>=1.2',
        'scikit-learn>=0.20.2',
        'd3m==2019.1.21'
    ],
    dependency_links=[
        'git+https://gitlab.com/datadrivendiscovery/common-primitives.git#egg=common_primitives',
    ],
    entry_points={
        'd3m.primitives': [
            'distil.MIRanking = miranking.mi_ranking:MIRankingPrimitive'
        ],
    }
)
