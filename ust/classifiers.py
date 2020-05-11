import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.multiclass import _check_partial_fit_first_call
from sklearn.utils.validation import check_array, check_X_y

class UGaussianNB(GaussianNB):
    """ Based on the paper:
    Qin, B., Xia, Y., Wang, S., & Du, X. (2011). A novel Bayesian classification for uncertain data. 
    Knowledge-Based Systems, 24(8), 1151–1158. https://doi.org/10.1016/j.knosys.2011.04.011
    """
    @staticmethod
    def _update_mean_variance(n_past, mu, var, X, sample_weight=None):
        if X.shape[0] == 0:
            return mu, var

        # Compute (potentially weighted) mean and variance of new datapoints
        c = X[:, :, 0]
        e = X[:, :, 1]
        lb = c - e # the lower bound of each attribute
        ub = c + e # the upper bound of each attribute
        if sample_weight is not None:
            n_new = float(sample_weight.sum())
            new_mu = np.average((lb + ub) / 2, axis=0, weights=sample_weight)
            new_var = np.average(((lb + ub) / 2)**2, axis=0, weights=sample_weight)
            new_var -= np.average((lb + ub) / 2, axis=0, weights=sample_weight)**2
            new_var += np.average(((ub - lb) / 6)**2, axis=0, weights=sample_weight)
        else:
            n_new = X.shape[0]
            new_mu = np.mean((lb + ub) / 2, axis=0)
            new_var = np.mean(((lb + ub) / 2)**2, axis=0)
            new_var -= np.mean((lb + ub) / 2, axis=0)**2
            new_var += np.mean(((ub - lb) / 6)**2, axis=0)

        if n_past == 0:
            return new_mu, new_var

        n_total = float(n_past + n_new)

        # Combine mean of old and new data, taking into consideration
        # (weighted) number of observations
        total_mu = (n_new * new_mu + n_past * mu) / n_total

        # Combine variance of old and new data, taking into consideration
        # (weighted) number of observations. This is achieved by combining
        # the sum-of-squared-differences (ssd)
        old_ssd = n_past * var
        new_ssd = n_new * new_var
        total_ssd = (old_ssd + new_ssd +
                     (n_new * n_past / n_total) * (mu - new_mu) ** 2)
        total_var = total_ssd / n_total

        return total_mu, total_var
    
    def _check_X(self, X):
        return check_array(X, accept_sparse='csr', allow_nd=True)
    
    def _check_X_y(self, X, y):
        return check_X_y(X, y, accept_sparse='csr', allow_nd=True)
    
    def _partial_fit(self, X, y, classes=None, _refit=False,
                     sample_weight=None):
        """Actual implementation of Gaussian NB fitting.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target values.
        classes : array-like, shape (n_classes,), optional (default=None)
            List of all the classes that can possibly appear in the y vector.
            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.
        _refit : bool, optional (default=False)
            If true, act as though this were the first time we called
            _partial_fit (ie, throw away any past fitting and start over).
        sample_weight : array-like, shape (n_samples,), optional (default=None)
            Weights applied to individual samples (1. for unweighted).
        Returns
        -------
        self : object
        """
        X, y = self._check_X_y(X, y)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        # If the ratio of data variance between dimensions is too small, it
        # will cause numerical errors. To address this, we artificially
        # boost the variance by epsilon, a small fraction of the standard
        # deviation of the largest dimension.
        self.epsilon_ = self.var_smoothing * np.var(X[:, :, 0], axis=0).max()

        if _refit:
            self.classes_ = None

        if _check_partial_fit_first_call(self, classes):
            # This is the first call to partial_fit:
            # initialize various cumulative counters
            n_features = X.shape[1]
            n_classes = len(self.classes_)
            self.theta_ = np.zeros((n_classes, n_features))
            self.sigma_ = np.zeros((n_classes, n_features))

            self.class_count_ = np.zeros(n_classes, dtype=np.float64)

            # Initialise the class prior
            # Take into account the priors
            if self.priors is not None:
                priors = np.asarray(self.priors)
                # Check that the provide prior match the number of classes
                if len(priors) != n_classes:
                    raise ValueError('Number of priors must match number of'
                                     ' classes.')
                # Check that the sum is 1
                if not np.isclose(priors.sum(), 1.0):
                    raise ValueError('The sum of the priors should be 1.')
                # Check that the prior are non-negative
                if (priors < 0).any():
                    raise ValueError('Priors must be non-negative.')
                self.class_prior_ = priors
            else:
                # Initialize the priors to zeros for each class
                self.class_prior_ = np.zeros(len(self.classes_),
                                             dtype=np.float64)
        else:
            if X.shape[1] != self.theta_.shape[1]:
                msg = "Number of features %d does not match previous data %d."
                raise ValueError(msg % (X.shape[1], self.theta_.shape[1]))
            # Put epsilon back in each time
            self.sigma_[:, :] -= self.epsilon_

        classes = self.classes_

        unique_y = np.unique(y)
        unique_y_in_classes = np.in1d(unique_y, classes)

        if not np.all(unique_y_in_classes):
            raise ValueError("The target label(s) %s in y do not exist in the "
                             "initial classes %s" %
                             (unique_y[~unique_y_in_classes], classes))

        for y_i in unique_y:
            i = classes.searchsorted(y_i)
            X_i = X[y == y_i, :]

            if sample_weight is not None:
                sw_i = sample_weight[y == y_i]
                N_i = sw_i.sum()
            else:
                sw_i = None
                N_i = X_i.shape[0]

            new_theta, new_sigma = self._update_mean_variance(
                self.class_count_[i], self.theta_[i, :], self.sigma_[i, :],
                X_i, sw_i)

            self.theta_[i, :] = new_theta
            self.sigma_[i, :] = new_sigma
            self.class_count_[i] += N_i

        self.sigma_[:, :] += self.epsilon_

        # Update if only no priors is provided
        if self.priors is None:
            # Empirical prior, with sample_weight taken into account
            self.class_prior_ = self.class_count_ / self.class_count_.sum()

        return self
    
    def predict(self, X):
        # The paper does not give the solution of the integral during prediction.
        # We decide to predict based on the center of the interval
        return super().predict(X[:, :, 0])

    def predict_proba(self, X):
        # The paper does not give the solution of the integral during prediction.
        # We decide to predict based on the center of the interval
        return super().predict_proba(X[:, :, 0])

if __name__ == '__main__':
    # example from the original paper
    X = np.array([
        [[115, 5], [115, 5]],
        [[110, 10], [115, 5]],
        [[72.5, 12.5], [115, 5]],
        [[127.5, 17.5], [115, 5]],
        [[115, 5], [115, 5]],
        [[65, 15], [115, 5]],
        [[210, 40], [115, 5]],
        [[92.5, 7.5], [115, 5]],
        [[90, 10], [115, 5]],
        [[132.5, 12.5], [115, 5]],
        [[115, 10], [115, 5]],
        [[87.5, 7.5], [115, 5]]
    ])
    y = np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0])

    c = X[:, :, 0]
    e = X[:, :, 1]
    lb = c - e
    ub = c + e

    # print("Lower bound:", lb,"Upper bound:", ub, sep='\n')
    new_mu = np.average((lb + ub) / 2, axis=0)

    new_var = np.average(((lb + ub) / 2)**2, axis=0)
    new_var -= np.average((lb + ub) / 2, axis=0)**2
    new_var += np.average(((ub - lb) / 6)**2, axis=0)

    print(f"New mu: {new_mu}, \nNew var: {new_var}")

    ugnb = UGaussianNB()
    ugnb.fit(X, y)
    score = ugnb.score(X, y)
    print("Score:", score)
    print(ugnb.predict(X))