import time
import numpy as np
from . import utils as ust_utils
from .transformers import ContractedUShapeletTransform, UShapeletTransform, Flat2UncertainTransformer
from sktime.transformations.panel.shapelets import ContractedShapeletTransform
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .classifiers import UGaussianNB
from .u_number import UNumber


def build_ust_dt_model(distance, seed=None, cmp_type=None, time_limit_in_mins=1, predefined_ig_rejection_level=0.05):
	ust = ContractedUShapeletTransform(time_limit_in_mins=time_limit_in_mins, 
									 remove_self_similar=True, 
									 random_state=seed, 
									 cmp_type=cmp_type,
									 distance=distance,
									 predefined_ig_rejection_level=predefined_ig_rejection_level)

	dt = DecisionTreeClassifier(random_state=seed)

	shapelet_clf = Pipeline([('st', ust), ('ss', StandardScaler()), ('dt', dt)])
	return shapelet_clf

def build_ust_nb_model(distance, seed=None, cmp_type=None, time_limit_in_mins=1, predefined_ig_rejection_level=0.05, use_ugnb = True):
	ust = None
	if time_limit_in_mins == np.inf:
		ust = UShapeletTransform(remove_self_similar=True, 
								 random_state=seed, 
								 cmp_type=cmp_type,
								 distance=distance,
								 predefined_ig_rejection_level=predefined_ig_rejection_level)
	else: 
		ust = ContractedUShapeletTransform(time_limit_in_mins=time_limit_in_mins, 
										 remove_self_similar=True, 
										 random_state=seed, 
										 cmp_type=cmp_type,
										 distance=distance,
										 predefined_ig_rejection_level=predefined_ig_rejection_level)

	components = [('st', ust), ('ss', StandardScaler())]
		
	if use_ugnb and distance == UShapeletTransform.UED:
		gnb = UGaussianNB()
		f2u = Flat2UncertainTransformer
		components.append(('f2u', f2u))
	else:
		gnb = GaussianNB()

	components.append(('gnb', gnb))
	shapelet_clf = Pipeline(components)
	return shapelet_clf

def build_st_dt_model(seed=None):
	st = ContractedShapeletTransform(time_limit_in_mins=0.5, remove_self_similar=True, random_state=seed)

	dt = DecisionTreeClassifier(random_state=seed)

	shapelet_clf = Pipeline([('st', st), ('ss', StandardScaler()), ('dt', dt)])
	return shapelet_clf

def build_st_nb_model(seed=None):
	st = ContractedShapeletTransform(time_limit_in_mins=0.5, remove_self_similar=True, random_state=seed)

	gnb = GaussianNB()

	shapelet_clf = Pipeline([('st', st), ('ss', StandardScaler()), ('gnb', gnb)])
	return shapelet_clf

def build_and_run_model(dataset_name, dataset_folder, seed, cmp_type=UNumber.SIMPLE_CMP, time_limit_in_mins=1, 
	distance=UShapeletTransform.UED, use_ugnb=True, return_model=False):
	try:
		print(dataset_name, 'Started...')
		start = time.time()
		train_X, train_y, test_X, test_y = ust_utils.load_uncertain_dataset(dataset_name, dataset_folder)
		ust_clf = build_ust_nb_model(cmp_type=cmp_type, distance=distance, seed=seed, time_limit_in_mins=time_limit_in_mins, use_ugnb=use_ugnb)
		ust_clf.fit(train_X, train_y)
		cp1 = time.time()
		score = ust_clf.score(test_X, test_y)
		cp2 = time.time()
		train_duration = round(cp1 - start, 2)
		test_duration = round(cp2 - cp1, 2)
		print(dataset_name, f'Finished (took {train_duration + test_duration} seconds)')
		if return_model:
			return ust_clf, score, train_duration, test_duration
		return score, train_duration, test_duration
	except Exception as e:
		print(f"UST failed: dataset = {dataset_name}, cmp = {cmp_type}, distance = {distance}, folder={dataset_folder}", e)
		return None, None, None