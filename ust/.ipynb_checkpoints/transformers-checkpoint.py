import numpy as np
import sys
import time
import warnings
import numpy as np
import pandas as pd
from . import utils as ust_utils
from .u_number import *
from operator import itemgetter
from scipy.stats import norm
from itertools import zip_longest
from sklearn.utils import check_random_state
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.utils.multiclass import class_distribution
from sklearn.preprocessing import FunctionTransformer
from sktime.transformers.shapelets import Shapelet, ShapeletTransform, ContractedShapeletTransform, ShapeletPQ

def flat2UncertainTransformer(X):
    """Convert the input to an uncertain dataset
    Input
        X: a pandas dataframe of shape (n, 2k)
    Output:
        U_X: a numpy array of shape (n, k, 2) such that U_X[i, j] = [X[i, j], X[i, j+k]]
    """
    nc = X.shape[1]//2
    m = X[:, 0:nc]
    e = X[:, nc:]
    U_X = np.array([ [[m[i, j], e[i, j]] for j in range(nc)] for i in range(X.shape[0])])
    return U_X

Flat2UncertainTransformer = FunctionTransformer(func=flat2UncertainTransformer)

class UShapeletTransform(ShapeletTransform):
    DUST_UNIFORM = "dust_uniform"
    DUST_NORMAL = "dust_normal"
    FOTS = 'fots'
    UED = "ued"
    ED = "ed"
    def __init__(self,
             min_shapelet_length=3,
             max_shapelet_length=np.inf,
             max_shapelets_to_store_per_class=200,
             random_state=None,
             verbose=0,
             remove_self_similar=True,
             cmp_type = None,
             distance = "ued",
             predefined_ig_rejection_level=0.05):
        
        super().__init__(min_shapelet_length, 
                         max_shapelet_length, 
                         max_shapelets_to_store_per_class, 
                         random_state, 
                         verbose, 
                         remove_self_similar)

        self.predefined_ig_rejection_level = predefined_ig_rejection_level
        self.distance = distance
        self.cmp_type = cmp_type


    def fit(self, X, y=None):
        if type(self) is ContractedShapeletTransform and self.time_limit_in_mins <= 0:
            raise ValueError("Error: time limit cannot be equal to or less than 0")

        X_lens = np.array([len(X.iloc[r,0]) for r in range(len(X))]) # note, assumes all dimensions of a case are the same length. A shapelet would not be well defined if indices do not match!
        X = np.array([[X.iloc[r,c].values for c in range(len(X.columns))] for r in range(len(X))]) # may need to pad with nans here for uneq length, look at later

        num_ins = len(y)
        distinct_class_vals = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]

        candidates_evaluated = 0

        num_series_to_visit = num_ins

        shapelet_heaps_by_class = {i: ShapeletPQ() for i in distinct_class_vals}

        self.random_state = check_random_state(self.random_state)

        # Here we establish the order of cases to sample. We need to sample x cases and y shapelets from each (where x = num_cases_to_sample
        # and y = num_shapelets_to_sample_per_case). We could simply sample x cases without replacement and y shapelets from each case, but
        # the idea is that if we are using a time contract we may extract all y shapelets from each x candidate and still have time remaining.
        # Therefore, if we get a list of the indices of the series and shuffle them appropriately, we can go through the list again and extract
        # another y shapelets from each series (if we have time).

        # We also want to ensure that we visit all classes so we will visit in round-robin order. Therefore, the code below extracts the indices
        # of all series by class, shuffles the indices for each class independently, and then combines them in alternating order. This results in
        # a shuffled list of indices that are in alternating class order (e.g. 1,2,3,1,2,3,1,2,3,1...)

        def _round_robin(*iterables):
            sentinel = object()
            return (a for x in zip_longest(*iterables, fillvalue=sentinel) for a in x if a != sentinel)

        case_ids_by_class = {i: np.where(y == i)[0] for i in distinct_class_vals}

        # if transform is contract then shuffle the data initially when determining which cases to visit
        if type(self) is ContractedShapeletTransform:
            for i in range(len(distinct_class_vals)):
                self.random_state.shuffle(case_ids_by_class[distinct_class_vals[i]])

        num_train_per_class = {i : len(case_ids_by_class[i]) for i in case_ids_by_class}
        round_robin_case_order = _round_robin(*[list(v) for k, v in case_ids_by_class.items()])
        cases_to_visit = [(i, y[i]) for i in round_robin_case_order]
        # this dictionary will be used to store all possible starting positions and shapelet lengths for a give series length. This
        # is because we enumerate all possible candidates and sample without replacement when assessing a series. If we have two series
        # of the same length then they will obviously have the same valid shapelet starting positions and lengths (especially in standard
        # datasets where all series are equal length) so it makes sense to store the possible candidates and reuse, rather than
        # recalculating each time

        # Initially the dictionary will be empty, and each time a new series length is seen the dict will be updated. Next time that length
        # is used the dict will have an entry so can simply reuse
        possible_candidates_per_series_length = {}

        # a flag to indicate if extraction should stop (contract has ended)
        time_finished = False

        # max time calculating a shapelet
        # for timing the extraction when contracting
        start_time = time.time()
        time_taken = lambda: time.time() - start_time
        max_time_calc_shapelet = -1
        time_last_shapelet = time_taken()

        # for every series
        case_idx = 0
        slen = 0
        while case_idx < len(cases_to_visit):

            series_id = cases_to_visit[case_idx][0]
            this_class_val = cases_to_visit[case_idx][1]

            # minus 1 to remove this candidate from sums
            binary_ig_this_class_count = num_train_per_class[this_class_val]-1
            binary_ig_other_class_count = num_ins-binary_ig_this_class_count-1

            if self.verbose:
                print("visiting series: " + str(series_id) + " (#" + str(case_idx + 1) + ")")

            this_series_len = len(X[series_id][0])

            # The bound on possible shapelet lengths will differ series-to-series if using unequal length data.
            # However, shapelets cannot be longer than the series, so set to the minimum of the series length
            # and max shapelet length (which is inf by default)
            if self.max_shapelet_length == -1:
                this_shapelet_length_upper_bound = this_series_len
            else:
                this_shapelet_length_upper_bound = min(this_series_len, self.max_shapelet_length)

            # all possible start and lengths for shapelets within this series (calculates if series length is new, a simple look-up if not)
            # enumerate all possible candidate starting positions and lengths.

            # First, try to reuse if they have been calculated for a series of the same length before.
            candidate_starts_and_lens = possible_candidates_per_series_length.get(this_series_len)
            # else calculate them for this series length and store for possible use again
            if candidate_starts_and_lens is None:
                candidate_starts_and_lens = [
                    [start, length] for start in range(0, this_series_len - self.min_shapelet_length + 1)
                    for length in range(self.min_shapelet_length, this_shapelet_length_upper_bound + 1) if start + length <= this_series_len]
                possible_candidates_per_series_length[this_series_len] = candidate_starts_and_lens

            # default for full transform
            candidates_to_visit = candidate_starts_and_lens
            num_candidates_per_case = len(candidate_starts_and_lens)

            # limit search otherwise:
            if hasattr(self,"num_candidates_to_sample_per_case"):
                num_candidates_per_case = min(self.num_candidates_to_sample_per_case, num_candidates_per_case)
                cand_idx = list(self.random_state.choice(list(range(0, len(candidate_starts_and_lens))), num_candidates_per_case, replace=False))
                candidates_to_visit = [candidate_starts_and_lens[x] for x in cand_idx]

            for candidate_idx in range(num_candidates_per_case):

                # if shapelet heap for this class is not full yet, set entry criteria to be the predetermined IG threshold
                ig_cutoff = self.predefined_ig_rejection_level
                # otherwise if we have max shapelets already, set the threshold as the IG of the current 'worst' shapelet we have
                if shapelet_heaps_by_class[this_class_val].get_size() >= self.max_shapelets_to_store_per_class:
                    ig_cutoff = max(shapelet_heaps_by_class[this_class_val].peek()[0], ig_cutoff)

                cand_start_pos = candidates_to_visit[candidate_idx][0]
                cand_len = candidates_to_visit[candidate_idx][1]

                candidate = self.zscore(X[series_id][:,cand_start_pos: cand_start_pos + cand_len])

                # now go through all other series and get a distance from the candidate to each
                orderline = []

                # initialise here as copy, decrease the new val each time we evaluate a comparison series
                num_visited_this_class = 0
                num_visited_other_class = 0

                candidate_rejected = False

                for comparison_series_idx in range(len(cases_to_visit)):
                    i = cases_to_visit[comparison_series_idx][0]

                    if y[i] != cases_to_visit[comparison_series_idx][1]:
                        raise ValueError("class match sanity test broken")

                    if i == series_id:
                        # don't evaluate candidate against own series
                        continue

                    if y[i]==this_class_val:
                        num_visited_this_class += 1
                        binary_class_identifier = 1 # positive for this class
                    else:
                        num_visited_other_class += 1
                        binary_class_identifier = -1 # negative for any other class

                    bsf_dist = UNumber(np.inf, 0.0, cmp_type=self.cmp_type)

                    start_left = cand_start_pos
                    start_right = cand_start_pos+1

                    if X_lens[i]==cand_len:
                        start_left = 0
                        start_right = 0

                    for num_cals in range(max(1,int(np.ceil((X_lens[i]-cand_len)/2)))): # max used to force iteration where series len == candidate len
                        if start_left < 0:
                            start_left = X_lens[i]-1-cand_len

                        comparison = self.zscore(X[i][:,start_left: start_left+ cand_len])
                        
                        dist_left = self.compute_uncertain_distance(candidate,comparison)
                        bsf_dist = min(dist_left, bsf_dist)

                        # for odd lengths
                        if start_left == start_right:
                            continue

                        # right
                        if start_right == X_lens[i]-cand_len+1:
                            start_right = 0
                        comparison = self.zscore(X[i][:,start_right: start_right + cand_len])
                        dist_right = self.compute_uncertain_distance(candidate, comparison)
                        bsf_dist = min(dist_right, bsf_dist)

                        start_left-=1
                        start_right+=1

                    orderline.append((bsf_dist,binary_class_identifier))
                    # sorting required after each add for early IG abandon.
                    # timsort should be efficient as array is almost in order - insertion-sort like behaviour in this case.
                    # Can't use heap as need to traverse in order multiple times, not just access root
                    orderline.sort()

                    if len(orderline) > 2:
                        ig_upper_bound = super().calc_early_binary_ig(orderline, num_visited_this_class, num_visited_other_class, binary_ig_this_class_count-num_visited_this_class, binary_ig_other_class_count-num_visited_other_class)
                        # print("upper: "+str(ig_upper_bound))
                        if ig_upper_bound <= ig_cutoff:
                            candidate_rejected = True
                            break

                candidates_evaluated += 1
                if self.verbose > 3 and candidates_evaluated % 100 == 0:
                    print("candidates evaluated: " + str(candidates_evaluated))

                # only do if candidate was not rejected
                if candidate_rejected is False:
                    final_ig = super().calc_binary_ig(orderline, binary_ig_this_class_count, binary_ig_other_class_count)
                    accepted_candidate = Shapelet(series_id, cand_start_pos, cand_len, final_ig, candidate)

                    # add to min heap to store shapelets for this class
                    shapelet_heaps_by_class[this_class_val].push(accepted_candidate)

                    # informal, but extra 10% allowance for self similar later
                    if shapelet_heaps_by_class[this_class_val].get_size() > self.max_shapelets_to_store_per_class*3:
                        shapelet_heaps_by_class[this_class_val].pop()

                # Takes into account the use of the MAX shapelet calculation time to not exceed the time_limit (not exact, but likely a good guess).
                if hasattr(self,'time_limit_in_mins') and self.time_limit_in_mins > 0:
                    time_now = time_taken()
                    time_this_shapelet = (time_now - time_last_shapelet)
                    if time_this_shapelet > max_time_calc_shapelet:
                        max_time_calc_shapelet = time_this_shapelet
                    time_last_shapelet = time_now
                    if (time_now + max_time_calc_shapelet) > self.time_limit_in_mins * 60:
                        if self.verbose > 0:
                            print("No more time available! It's been {0:02d}:{1:02}".format(int(round(time_now / 60, 3)), int((round(time_now / 60, 3) - int(round(time_now / 60, 3))) * 60)))
                        time_finished = True
                        break
                    else:
                        if self.verbose > 0:
                            if candidate_rejected is False:
                                print("Candidate finished. {0:02d}:{1:02} remaining".format(int(round(self.time_limit_in_mins - time_now / 60, 3)),
                                                                                            int((round(self.time_limit_in_mins - time_now / 60, 3) - int(round(self.time_limit_in_mins - time_now / 60, 3))) * 60)))
                            else:
                                print("Candidate rejected. {0:02d}:{1:02} remaining".format(int(round(self.time_limit_in_mins - time_now / 60, 3)),
                                                                                            int((round(self.time_limit_in_mins - time_now / 60, 3) - int(round(self.time_limit_in_mins - time_now / 60, 3))) * 60)))

            # stopping condition: in case of iterative transform (i.e. num_cases_to_sample have been visited)
            #                     in case of contracted transform (i.e. time limit has been reached)
            case_idx += 1

            if case_idx >= num_series_to_visit:
                if hasattr(self,'time_limit_in_mins') and time_finished is not True:
                    case_idx = 0
            elif case_idx >= num_series_to_visit or time_finished:
                if self.verbose > 0:
                    print("Stopping search")
                break

        # remove self similar here
        # for each class value
        #       get list of shapelets
        #       sort by quality
        #       remove self similar

        self.shapelets = []
        for class_val in distinct_class_vals:
            by_class_descending_ig = sorted(shapelet_heaps_by_class[class_val].get_array(), key=itemgetter(0), reverse=True)

            if self.remove_self_similar and len(by_class_descending_ig) > 0:
                by_class_descending_ig = super().remove_self_similar_shapelets(by_class_descending_ig)
            else:
                # need to extract shapelets from tuples
                by_class_descending_ig = [x[2] for x in by_class_descending_ig]

            # if we have more than max_shapelet_per_class, trim to that amount here
            if len(by_class_descending_ig) > self.max_shapelets_to_store_per_class:
                by_class_descending_ig = by_class_descending_ig[:self.max_shapelets_to_store_per_class]

            self.shapelets.extend(by_class_descending_ig)

        # final sort so that all shapelets from all classes are in descending order of information gain
        self.shapelets.sort(key=lambda x:x.info_gain, reverse=True)
        self.is_fitted_ = True

        # warn the user if fit did not produce any valid shapelets
        if len(self.shapelets) == 0:
            warnings.warn("No valid shapelets were extracted from this dataset and calling the transform method "
                          "will raise an Exception. Please re-fit the transform with other data and/or "
                          "parameter options.")
            
    def zscore(self, a, axis=0, ddof=0):
        zscored = []
        for i, u_candidate in enumerate(a):
            """computer the z-score of the observation and make the error relative if not"""
            
            can = np.array([[o[0], o[1]] for o in u_candidate])
            
            j = can[:, 0]

            # save error in relative
            relative_errors = can[:, 1] 
            relative_errors[j > 0] /= j[j > 0]
            
            sstd = np.std(j, axis=axis, ddof=ddof)

            # special case - if shapelet is a straight line (i.e. no variance), zscore ver should be np.zeros(len(a))
            if sstd == 0:
                zscored.append(np.array(pd.Series([(0,0)]*j.shape[0])))
            else:
                mns = np.mean(j, axis=axis)
                if axis and mns.ndim < j.ndim:
                    score = ((j - np.expand_dims(mns, axis=axis)) /
                                    np.expand_dims(sstd, axis=axis))
                else:
                    score = (j - mns) / sstd

                errors = np.abs(np.multiply(relative_errors, score)) # convert the error back to absolute
                zscored.append(pd.Series(zip(score, errors)).values)
        zscored = np.array(zscored)
        return zscored 
    
    def transform(self, X, **transform_params):
        if self.is_fitted_ is False:
            raise Exception("fit has not been called . Please call fit before using the transform method.")
        elif len(self.shapelets) == 0:
            raise Exception("No shapelets were extracted in fit that exceeded the minimum information gain threshold. Please retry with other data and/or parameter settings.")

        X = np.array([[X.iloc[r, c].values for c in range(len(X.columns))] for r in range(len(X))])  # may need to pad with nans here for uneq length, look at later

        nb_shapelet = len(self.shapelets)
        output = np.zeros([len(X), nb_shapelet*2], dtype=np.float32, )

        # for the i^th series to transform
        for i in range(0, len(X)):
            this_series = X[i]

            # get the s^th shapelet
            for s in range(0, nb_shapelet):
                # find distance between this series and each shapelet
                min_dist = UNumber(np.inf, 0.0, cmp_type=self.cmp_type)
                this_shapelet_length = self.shapelets[s].length

                for start_pos in range(0, len(this_series[0]) - this_shapelet_length + 1):
                    comparison = self.zscore(this_series[:, start_pos:start_pos + this_shapelet_length])

                    dist = self.compute_uncertain_distance(self.shapelets[s].data, comparison)
                    dist.value = 1.0/this_shapelet_length*dist.value
                    min_dist = min(min_dist, dist)

                    if not np.isfinite(min_dist.value) or not np.isfinite(min_dist.err):
                        print("Not finite", min_dist)

                    output[i][s] = min_dist.value
                    output[i][s + nb_shapelet] = min_dist.err

        return pd.DataFrame(output)
    
    def compute_uncertain_distance(self, uSeries1, uSeries2):
        x = np.array([[o[0], o[1]] for o in uSeries1[0]])
        y = np.array([[o[0], o[1]] for o in uSeries2[0]])
        if self.distance == UShapeletTransform.DUST_UNIFORM:
            return self.dust_uniform(x, y)
        if self.distance == UShapeletTransform.DUST_NORMAL:
            return self.dust_normal(x, y)
        if self.distance == UShapeletTransform.FOTS:
        	return self.fots(x[:,0], y[:,0])
        if self.distance == UShapeletTransform.UED:
            return self.uncertain_euclidean_distance(x, y)
        return self.euclidean_distance(x[:, 0], y[:,0])

    
    def uncertain_euclidean_distance(self, uSeries1, uSeries2):
        dist, err = 0, 0
        for u, v in zip(uSeries1, uSeries2):
            tmp = np.abs(u[0]-v[0])
            dist += (tmp**2)
            err += (tmp * (u[1] + v[1]))
            # print(u[1], v[1])

        return UNumber(dist, err, cmp_type=self.cmp_type)

    def euclidean_distance(self, series1, series2):
        dist = np.sum((series1 - series2)**2)
            # print(u[1], v[1])

        return UNumber(dist, 0, cmp_type=self.cmp_type)

    def autocoravariance_matrix(self, X, w, t, m = None, pad=0):
        if m is None:
            m = w
        start = t - m + 1
        gamma_t = np.zeros((w, w))
        X_copy = X.copy()
        pw = 0
        if start < 0:
            pw = -start
            # padding with 0
            X_copy = np.pad(X, pad_width=(pw, 0), constant_values=pad)
        for tau in range(start-1, t):
            s = tau + pw
            gamma_t += np.outer(X_copy[s:s+w], X_copy[s:s+w])
        return gamma_t

    def fots(self, X, Y, t = None, w = None, k=4, m=None, pad=0):
        if w is None:
            w = len(X) // 2
        if t is None:
            t = len(X) // 2
        if t+w > len(X):
            w = t
        gamma_Xt = self.autocoravariance_matrix(X, w, t, m, pad)
        gamma_Yt = self.autocoravariance_matrix(Y, w, t, m, pad)
        _, eigVectorX = np.linalg.eigh(gamma_Xt)
        _, eigVectorY = np.linalg.eigh(gamma_Yt)
        k = min(k, min(eigVectorX.shape[1], eigVectorY.shape[1]))
        return UNumber(np.linalg.norm(eigVectorX[:,-k:] - eigVectorY[:,-k:], ord='fro'), err=0, cmp_type=self.cmp_type)
    
    def dust_uniform(self, x, y):
        err_std = np.max([y[:, 1], x[:, 1]], axis=0)
        err_std[err_std==0] = 0.5
        return UNumber(np.sqrt(np.sum((np.abs(x[:, 0] - y[:, 0])/(2*err_std))**2)), err=0, cmp_type=self.cmp_type)

    def dust_normal(self, x, y):
        err_std = np.max([y[:, 1], x[:, 1]], axis=0)
        err_std[err_std==0] = 0.4238 # an approximation solution of 2x(1 + x^2) = 1
        return UNumber(np.sqrt(np.sum((np.abs(x[:, 0] - y[:, 0]) / (2 * err_std * ( 1 + err_std**2)))**2)), err=0, cmp_type=self.cmp_type)

class ContractedUShapeletTransform(UShapeletTransform):

    def __init__(
            self,
            min_shapelet_length = 3,
            max_shapelet_length = np.inf,
            max_shapelets_to_store_per_class = 200,
            time_limit_in_mins=60,
            num_candidates_to_sample_per_case = 20,
            random_state = None,
            verbose = 0,
            remove_self_similar = True,
            cmp_type = None,
            distance = "ued",
            predefined_ig_rejection_level = 0.05
    ):

        super().__init__(min_shapelet_length, max_shapelet_length, max_shapelets_to_store_per_class, random_state,
                         verbose, remove_self_similar)

        self.predefined_ig_rejection_level = predefined_ig_rejection_level
        self.distance = distance
        self.cmp_type = cmp_type
        self.num_candidates_to_sample_per_case = num_candidates_to_sample_per_case
        self.time_limit_in_mins = time_limit_in_mins
        self.shapelets = None
    
if __name__ == "__main__":
    u1 = [[(5, 0.2), (2, 0.01)]]
    u2 = [[(4, 0.1), (1, 0.3)]]
    print(u1, u2)
    print(f"ued(u1, u2)", UShapeletTransform().uncertain_euclidean_distance(u1, u2))
    print(f"dust_uniform(u1, u2)", UShapeletTransform().dust_uniform(u1, u2))
    print(f"dust_normal(u1, u2)", UShapeletTransform().dust_normal(u1, u2))

    X = np.random.randint(-6, 5, size=(10, 6))
    X = pd.DataFrame(data=X, columns=[f'c{i}' for i in range(X.shape[1])])
    nc = X.shape[1]//2
    m = X.iloc[:, 0:nc]
    e = X.iloc[:, nc:]
    U_X = np.array([ [[m.iloc[i,j], e.iloc[i,j]] for j in range(nc)] for i in range(X.shape[0])])
    print("Original:\n",X)
    print("Transfomed:\n", Flat2UncertainTransformer.transform(X))
