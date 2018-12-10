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
        'd3m',
        'common-primitives'
    ],
    dependency_links=[
        'git+https://gitlab.com/datadrivendiscovery/common-primitives.git@devel#egg=common_primitives',
        'git+https://gitlab.com/datadrivendiscovery/d3m.git@devel#egg=d3m'
    ],
    entry_points={
        'd3m.primitives': [
            'distil.MIRanking = miranking.mi_ranking:MIRankingPrimitive'
        ],
    }
)
