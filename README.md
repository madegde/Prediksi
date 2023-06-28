# Prediksi
Age Distribution
![download](https://github.com/madegde/Prediksi/assets/73327109/6ad7ab08-92b5-4f97-a61a-de4ceca129e7)


Age Box Plot
![download](https://github.com/madegde/Prediksi/assets/73327109/b2f45fe1-3674-427d-9c9a-1c3fc7990c8f)


BMI Distribution
![download](https://github.com/madegde/Prediksi/assets/73327109/db7a6037-dc3c-4021-8778-54c6d4731709)


BMI Box Plot
![download](https://github.com/madegde/Prediksi/assets/73327109/c89ee55e-83d4-43fa-85cf-9eaad30f94dd)


hbA1c Level Distribution
![download](https://github.com/madegde/Prediksi/assets/73327109/c1ee9283-6ba7-4f6f-b0b7-cb99da8adf2a)


HbA1c_level Box Plot
![download](https://github.com/madegde/Prediksi/assets/73327109/08c37d28-5fc3-41a6-8122-be839efe6691)


Blood Glucose Level Distribution
![download](https://github.com/madegde/Prediksi/assets/73327109/6c9f6fe3-e78d-4f4c-8504-a76b430e6473)


Blood Glucose Level Box Plot
![download](https://github.com/madegde/Prediksi/assets/73327109/820aab5f-3d9f-44d7-a3d4-681276681ca2)


Gender Distribution
![download](https://github.com/madegde/Prediksi/assets/73327109/b656f4fd-5244-4b9a-88fa-aada77d0077e)


Hypertension Distribution
![download](https://github.com/madegde/Prediksi/assets/73327109/3a4fa185-f3c5-4e82-b8eb-d63bba1ffdc2)


Heart Disease Distribution
![download](https://github.com/madegde/Prediksi/assets/73327109/ae1a54e1-9222-4c49-b255-d46997fc3c7d)


Smoking History Distribution
![download](https://github.com/madegde/Prediksi/assets/73327109/84532cde-99f7-4797-8fa5-fe9fdf3de89a)


Diabetes Distribution
![download](https://github.com/madegde/Prediksi/assets/73327109/4d193b93-9f78-492a-9929-8580a3035ef0)


Diabetes and Age
![download](https://github.com/madegde/Prediksi/assets/73327109/77fa4496-8a69-4792-8102-b4771b5c3f7a)


Diabetes and BMI
![download](https://github.com/madegde/Prediksi/assets/73327109/a4ab3c45-ad25-48c5-a22f-44a8c998f299)


Diabetes and Gender
![download](https://github.com/madegde/Prediksi/assets/73327109/5336ca52-afd5-4bf0-be86-e4c6c04e7576)
![download](https://github.com/madegde/Prediksi/assets/73327109/769363c1-722c-40d3-9832-4eb60ff504ea)


Diabetes and HbA1c_level
![download](https://github.com/madegde/Prediksi/assets/73327109/ddb77d1f-4495-4e2f-9aab-59df82a49c77)
![download](https://github.com/madegde/Prediksi/assets/73327109/8829f2b7-b0e4-4aa2-9372-93e24cec4272)


Diabetes and Blood Glucose Level
![download](https://github.com/madegde/Prediksi/assets/73327109/cd3da135-8062-48c1-9f22-485e3c935c7c)
![download](https://github.com/madegde/Prediksi/assets/73327109/43701232-9f4b-4726-a449-c9e291d46407)


Pair Plot for Numeric Features
![download](https://github.com/madegde/Prediksi/assets/73327109/06754947-7e37-42f9-a784-c7fca9f03e2c)


Diabetes, BMI, and Age
![download](https://github.com/madegde/Prediksi/assets/73327109/95e497e8-fd9d-4a88-b923-2b767ba6a927)


Correlation Matrix
![download](https://github.com/madegde/Prediksi/assets/73327109/87ed2643-c20d-4b8f-80b2-809577faa1b1)


Correlation with Diabetes
![download](https://github.com/madegde/Prediksi/assets/73327109/0f6f782c-078e-4373-af7e-994bcb7e85a5)


**StandardScaler Parameter**
copy : bool, default=True
    If False, try to avoid a copy and do inplace scaling instead.
This is not guaranteed to always work inplace; e.g. if the data is
not a NumPy array or scipy.sparse CSR matrix, a copy may still be
returned.

with_mean : bool, default=True
    If True, center the data before scaling.
This does not work (and will raise an exception) when attempted on
sparse matrices, because centering them entails building a dense
matrix which in common use cases is likely to be too large to fit in
memory.

with_std : bool, default=True
    If True, scale the data to unit variance (or equivalently,
unit standard deviation).

categories : 'auto' or a list of array-like, default='auto'
    Categories (unique values) per feature:

**OneHotEncoder Parameter**
'auto' : Determine categories automatically from the training data.

list : categories[i] holds the categories expected in the ith
column. The passed categories should not mix strings and numeric
values within a single feature, and should be sorted in case of
numeric values.


    The used categories can be found in the categories_ attribute.

drop : {'first', 'if_binary'} or an array-like of shape (n_features,),             default=None
    Specifies a methodology to use to drop one of the categories per
feature. This is useful in situations where perfectly collinear
features cause problems, such as when feeding the resulting data
into an unregularized linear regression model.

    However, dropping one category breaks the symmetry of the original
representation and can therefore induce a bias in downstream models,
for instance for penalized linear classification or regression models.


None : retain all features (the default).

'first' : drop the first category in each feature. If only one
category is present, the feature will be dropped entirely.

'if_binary' : drop the first category in each feature with two
categories. Features with 1 or more than 2 categories are
left intact.

array : drop[i] is the category in feature X[:, i] that
should be dropped.


    When max_categories or min_frequency is configured to group
infrequent categories, the dropping behavior is handled after the
grouping.

sparse : bool, default=True
    Will return sparse matrix if set True else will return an array.

sparse_output : bool, default=True
    Will return sparse matrix if set True else will return an array.

dtype : number type, default=float
    Desired dtype of output.

handle_unknown : {'error', 'ignore', 'infrequent_if_exist'},                      default='error'
    Specifies the way unknown categories are handled during transform.


'error' : Raise an error if an unknown category is present during transform.

'ignore' : When an unknown category is encountered during
      transform, the resulting one-hot encoded columns for this feature
will be all zeros. In the inverse transform, an unknown category
will be denoted as None.

'infrequent_if_exist' : When an unknown category is encountered
      during transform, the resulting one-hot encoded columns for this
feature will map to the infrequent category if it exists. The
infrequent category will be mapped to the last position in the
encoding. During inverse transform, an unknown category will be
mapped to the category denoted 'infrequent' if it exists. If the
'infrequent' category does not exist, then transform and
inverse_transform will handle an unknown category as with
handle_unknown='ignore'. Infrequent categories exist based on
min_frequency and max_categories. Read more in the
User Guide <one_hot_encoder_infrequent_categories>.


min_frequency : int or float, default=None
    Specifies the minimum frequency below which a category will be
considered infrequent.


If int, categories with a smaller cardinality will be considered
      infrequent.


If float, categories with a smaller cardinality than
      min_frequency * n_samples  will be considered infrequent.



max_categories : int, default=None
    Specifies an upper limit to the number of output features for each input
feature when considering infrequent categories. If there are infrequent
categories, max_categories includes the category representing the
infrequent categories along with the frequent categories. If None,
there is no limit to the number of output features.

**traintestsplit parameter**
arrays : sequence of indexables with same length / shape[0]
    Allowed inputs are lists, numpy arrays, scipy-sparse
matrices or pandas dataframes.

test_size : float or int, default=None
    If float, should be between 0.0 and 1.0 and represent the proportion
of the dataset to include in the test split. If int, represents the
absolute number of test samples. If None, the value is set to the
complement of the train size. If train_size is also None, it will
be set to 0.25.

train_size : float or int, default=None
    If float, should be between 0.0 and 1.0 and represent the
proportion of the dataset to include in the train split. If
int, represents the absolute number of train samples. If None,
the value is automatically set to the complement of the test size.

random_state : int, RandomState instance or None, default=None
    Controls the shuffling applied to the data before applying the split.
Pass an int for reproducible output across multiple function calls.
See Glossary <random_state>.

shuffle : bool, default=True
    Whether or not to shuffle the data before splitting. If shuffle=False
then stratify must be None.

stratify : array-like, default=None
    If not None, data is split in a stratified fashion, using this as
the class labels.
Read more in the User Guide <stratification>.

**RandomForestClasifier Parameter**
n_estimators : int, default=100
    The number of trees in the forest.

criterion : {"gini", "entropy", "log_loss"}, default="gini"
    The function to measure the quality of a split. Supported criteria are
"gini" for the Gini impurity and "log_loss" and "entropy" both for the
Shannon information gain, see tree_mathematical_formulation.
    Note: This parameter is tree-specific.

max_depth : int, default=None
    The maximum depth of the tree. If None, then nodes are expanded until
all leaves are pure or until all leaves contain less than
min_samples_split samples.

min_samples_split : int or float, default=2
    The minimum number of samples required to split an internal node:


If int, then consider min_samples_split as the minimum number.

If float, then min_samples_split is a fraction and
      ceil(min_samples_split * n_samples) are the minimum
number of samples for each split.


min_samples_leaf : int or float, default=1
    The minimum number of samples required to be at a leaf node.
A split point at any depth will only be considered if it leaves at
least min_samples_leaf training samples in each of the left and
right branches.  This may have the effect of smoothing the model,
especially in regression.


If int, then consider min_samples_leaf as the minimum number.

If float, then min_samples_leaf is a fraction and
      ceil(min_samples_leaf * n_samples) are the minimum
number of samples for each node.


min_weight_fraction_leaf : float, default=0.0
    The minimum weighted fraction of the sum total of weights (of all
the input samples) required to be at a leaf node. Samples have
equal weight when sample_weight is not provided.

max_features : {"sqrt", "log2", None}, int or float, default="sqrt"
    The number of features to consider when looking for the best split:


If int, then consider max_features features at each split.

If float, then max_features is a fraction and
      max(1, int(max_features * n_features_in_)) features are considered at each
split.

If "auto", then max_features=sqrt(n_features).

If "sqrt", then max_features=sqrt(n_features).

If "log2", then max_features=log2(n_features).

If None, then max_features=n_features.


    Note: the search for a split does not stop until at least one
valid partition of the node samples is found, even if it requires to
effectively inspect more than max_features features.

max_leaf_nodes : int, default=None
    Grow trees with max_leaf_nodes in best-first fashion.
Best nodes are defined as relative reduction in impurity.
If None then unlimited number of leaf nodes.

min_impurity_decrease : float, default=0.0
    A node will be split if this split induces a decrease of the impurity
greater than or equal to this value.

    The weighted impurity decrease equation is the following:

N_t / N * (impurity - N_t_R / N_t * right_impurity
                    - N_t_L / N_t * left_impurity)
    where N is the total number of samples, N_t is the number of
samples at the current node, N_t_L is the number of samples in the
left child, and N_t_R is the number of samples in the right child.

    N, N_t, N_t_R and N_t_L all refer to the weighted sum,
if sample_weight is passed.

bootstrap : bool, default=True
    Whether bootstrap samples are used when building trees. If False, the
whole dataset is used to build each tree.

oob_score : bool, default=False
    Whether to use out-of-bag samples to estimate the generalization score.
Only available if bootstrap=True.

n_jobs : int, default=None
    The number of jobs to run in parallel. fit, predict,
decision_path and apply are all parallelized over the
trees. None means 1 unless in a joblib.parallel_backend
context. -1 means using all processors. See Glossary <n_jobs> for more details.

random_state : int, RandomState instance or None, default=None
    Controls both the randomness of the bootstrapping of the samples used
when building trees (if bootstrap=True) and the sampling of the
features to consider when looking for the best split at each node
(if max_features < n_features).
See Glossary <random_state> for details.

verbose : int, default=0
    Controls the verbosity when fitting and predicting.

warm_start : bool, default=False
    When set to True, reuse the solution of the previous call to fit
and add more estimators to the ensemble, otherwise, just fit a whole
new forest. See Glossary <warm_start> and
gradient_boosting_warm_start for details.

class_weight : {"balanced", "balanced_subsample"}, dict or list of dicts,             default=None
    Weights associated with classes in the form {class_label: weight}.
If not given, all classes are supposed to have weight one. For
multi-output problems, a list of dicts can be provided in the same
order as the columns of y.

    Note that for multioutput (including multilabel) weights should be
defined for each class of every column in its own dict. For example,
for four-class multilabel classification weights should be
[{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
[{1:1}, {2:5}, {3:1}, {4:1}].

    The "balanced" mode uses the values of y to automatically adjust
weights inversely proportional to class frequencies in the input data
as n_samples / (n_classes * np.bincount(y))

    The "balanced_subsample" mode is the same as "balanced" except that
weights are computed based on the bootstrap sample for every tree
grown.

    For multi-output, the weights of each column of y will be multiplied.

    Note that these weights will be multiplied with sample_weight (passed
through the fit method) if sample_weight is specified.

ccp_alpha : non-negative float, default=0.0
    Complexity parameter used for Minimal Cost-Complexity Pruning. The
subtree with the largest cost complexity that is smaller than
ccp_alpha will be chosen. By default, no pruning is performed. See
minimal_cost_complexity_pruning for details.

max_samples : int or float, default=None
    If bootstrap is True, the number of samples to draw from X
to train each base estimator.


If None (default), then draw X.shape[0] samples.

If int, then draw max_samples samples.

If float, then draw max_samples * X.shape[0] samples. Thus,
max_samples should be in the interval (0.0, 1.0].

**XGBClassifier Parameter**


Plot accuracy score for each algorithm
![download](https://github.com/madegde/Prediksi/assets/73327109/263800e9-9072-4d1d-a7f7-5360f9fcefa1)


Confusion Matrix RF
![download](https://github.com/madegde/Prediksi/assets/73327109/b6c2569d-298e-48f4-9e96-02a227d05da3)


Confusion Matrix XGB
![download](https://github.com/madegde/Prediksi/assets/73327109/8485a88a-da26-48b9-a9f3-cfae1bcff360)


Hyperparameters Tuning Results
![download](https://github.com/madegde/Prediksi/assets/73327109/7591aa6e-93ea-4f01-8a9c-d6462845f23e)


Confusion Matrix
![download](https://github.com/madegde/Prediksi/assets/73327109/6d6a90af-1bbc-4da5-aa30-0a4226a729d8)


Feature Importances
![download](https://github.com/madegde/Prediksi/assets/73327109/aba053cb-0612-4bbe-a271-ae9c6229aaa2)

$$ Yi = {xi−μ(x⃗ )} \over σ(x⃗) $$

#asd

$$ x = {-b \pm \sqrt{b^2-4ac} \over 2a} $$
