# distil-mi-ranking

Feature ranking based on a mutual information between features and a selected target.  Will rank any feature column with a semantic type of `Float`, `Boolean`, `Integer` or `Categorical`, and a corresponding structural type of `int` or `float`. A DataFrame containing `(col_idx, col_name, score)` tuples for each ranked feature will be returned to the caller.  Features that could not be ranked are excluded from the returned set.

Deployment:

```shell
pip install -e git+ssh://git@github.com/uncharted-distil/distil-mi-ranking.git#egg=DistilMIRanking --process-dependency-links
```

Development:

```shell
pip install -r requirements.txt
```
