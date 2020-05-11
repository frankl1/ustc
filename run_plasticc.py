from sktime.classifiers.compose import ColumnEnsembleClassifier
from ust import utils as ust_utils
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ust.transformers import *
from ust.classifiers import UGaussianNB
from ust.u_number import *
from ust.models import build_ust_nb_model
from multiprocessing import Pool, cpu_count
import pandas as pd

def plasticc_model(distance, seed=None, cmp_type=None, time_limit_in_mins=1, predefined_ig_rejection_level=0.05):
    ust = ContractedUShapeletTransform(time_limit_in_mins=time_limit_in_mins, 
                                        remove_self_similar=True, 
                                         random_state=seed, 
                                         cmp_type=cmp_type,
                                         distance=distance,
                                         predefined_ig_rejection_level=predefined_ig_rejection_level)
    
    components = [
        ('ros', RandomOverSampler(seed)), 
        ('st', ust), 
        ('ss', StandardScaler()),
        ('f2u', Flat2UncertainTransformer),
        ('gnb', UGaussianNB())
    ]
    
    shapelet_clf = Pipeline(components)
    return shapelet_clf

bands = 'giruyz'
plasticc_folder = 'dataset/plasticc_5d_dataset'
seed = 5813
min_ig = 0.000001
time_limit_in_mins = 1

print(f'Min IG:{min_ig}, Time limit: {time_limit_in_mins}')

res = []
args = [('plassticc_5d_'+b, plasticc_folder) for b in bands]
NB_PROCESS = min(len(args), cpu_count())
with Pool(NB_PROCESS) as pool:
    res = pool.starmap(ust_utils.load_uncertain_dataset, args)

plasticc_train_X = pd.DataFrame()
plasticc_train_y = None 
plasticc_test_X = pd.DataFrame()
plasticc_test_y = None 
for (b, _), (trainX, trainy, testX, testy) in zip(args, res):
    plasticc_train_X[b] = trainX.dim_0
    plasticc_test_X[b] = testX.dim_0
plasticc_train_y = res[0][1]
plasticc_test_y = res[0][3]

print('Plasticc loaded')
print('Train shape:', plasticc_train_X.shape, plasticc_train_y.shape)
print('Test shape:', plasticc_test_X.shape, plasticc_test_y.shape)

plasticc_clf = ColumnEnsembleClassifier(estimators=[
    (
        f'ust_{b}', 
        plasticc_model(cmp_type=UNumber.INTERVAL_CMP, distance=UShapeletTransform.UED, seed=seed, time_limit_in_mins=time_limit_in_mins, predefined_ig_rejection_level=min_ig), 
        [i]
    ) for i, b in enumerate(bands)
])
plasticc_clf.fit(plasticc_train_X, plasticc_train_y)
score = plasticc_clf.score(plasticc_test_X, plasticc_test_y)
print('Plasticc score:', score)
