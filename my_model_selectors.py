import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError


    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)



        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN

    In statistics, the Bayesian information criterion (BIC) or Schwarz criterion (also SBC, SBIC) is a criterion for model selection
    among a finite set of models; the model with the lowest BIC is preferred. It is based, in part, on the likelihood function and it
    is closely related to the Akaike information criterion (AIC). When fitting models, it is possible to increase the likelihood by
    adding parameters, but doing so may result in overfitting. Both BIC and AIC attempt to resolve this problem by introducing a penalty
    term for the number of parameters in the model; the penalty term is larger in BIC than in AIC.The BIC was developed by Gideon
    E. Schwarz and published in a 1978 paper,[1] where he gave a Bayesian argument for adopting it.

    Properties of BIC :
        - It is independent of the prior or the prior is "vague" (a constant).
        - It can measure the efficiency of the parameterized model in terms of predicting the data.
        - It penalizes the complexity of the model where complexity refers to the number of parameters in the model.
        - It is approximately equal to the minimum description length criterion but with negative sign.
        - It can be used to choose the number of clusters according to the intrinsic complexity present in a particular dataset.
        - It is closely related to other penalized likelihood criteria such as RIC[clarification needed] and the Akaike information criterion.
        - It is used to scoring model topologies by balancing fit and complexity within the training set for each word


    BIC Equation:

    BIC = -2 * log L + p * log N

    - where "L" is likelihood of "fitted" model
    - where "p" is the qty of free parameters in model (aka model "complexity"). Reference [2][3]
    - where "p * log N" is the "penalty term" (increases with higher "p" to penalise "complexity" and avoid "overfitting")
    - where "N" is qty of data points (size of data set)

    Lower the BIC score the "better" the model.

    References:
        [1] - http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
        [2] - https://en.wikipedia.org/wiki/Bayesian_information_criterion
        """



    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)




        best_score = float('+inf')
        best_model = None
        try:
            n_components = range(self.min_n_components, self.max_n_components + 1)
            for n in n_components:
                model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                LogL = model.score(self.X, self.lengths)
                N = len(self.X)
                params = (n * n) + (2 * n * len(self.X[0])) - 1
                BIC_score = ((-2) * LogL) + (params * math.log(N))
                if BIC_score < best_score:
                    best_score = BIC_score
                    best_model = model
        except:
            return best_model
        return best_model



class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))

    The deviance information criterion (DIC) is a hierarchical modeling generalization of the Akaike information criterion (AIC)
    and the Bayesian information criterion (BIC). It is particularly useful in Bayesian model selection problems where the posterior
    distributions of the models have been obtained by Markov chain Monte Carlo (MCMC) simulation. Like AIC and BIC, DIC is an
    asymptotic approximation as the sample size becomes large. It is only valid when the posterior distribution is approximately
    multivariate normal.

    DIC Equation:
        DIC = log(P(X(i)) - 1/(M - 1) * sum(log(P(X(all but i))

        = log likelihood of the data belonging to model
              - avg of anti log likelihood of data X and model M
            = log(P(original word)) - average(log(P(other words)))

        where anti log likelihood means likelihood of data X and model M belonging to competing categories
        where log(P(X(i))) is the log-likelihood of the fitted model for the current word
        where where "L" is likelihood of data fitting the model ("fitted" model)
        where X is input training data given in the form of a word dictionary
        where X(i) is the current word being evaluated

    Higher the DIC, better the model

     References:
        [1] - https://en.wikipedia.org/wiki/Deviance_information_criterion
        [2] - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
        [3] - https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
'''



    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)




        max_score = float("-inf")
        max_model = None

        n_components = range(self.min_n_components, self.max_n_components + 1)
        for n in n_components :
            try:
                sum_score = 0.0
                wc = 0.0
                model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                for word in self.hwords:
                    if word != self.this_word:
                        X, lengths = self.hwords[word]
                        sum_score += model.score(X, lengths)
                        wc +=1

                DIC_score = model.score(self.X, self.lengths) - (sum_score/wc)

                if DIC_score > max_score:
                    max_score = DIC_score
                    max_model = model
            except:
                pass

        return max_model



class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    Cross Validation is used to assess the predictive performance of the models
    and and to judge how they perform outside the sample to a new data set also
    known as test data
    The motivation to use cross validation techniques is that when we fit a model,
    we are fitting it to a training dataset. Without cross validation we only have
    information on how does our model perform to our in-sample data. Ideally we would
    like to see how does the model perform when we have a new data in terms of accuracy
    of its predictions. In science, theories are judged by its predictive performance.

    K-Folds cross-validator
    Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (without shuffling by default).
    Each fold is then used once as a validation while the k - 1 remaining folds form the training set.

    Higher the CV score, better the model
        [1] - http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
        [2] - https://www.researchgate.net/post/What_is_the_purpose_of_performing_cross-validation
        [3] - https://en.wikipedia.org/wiki/Cross-validation_(statistics)



    '''



    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.warn("deprecated", DeprecationWarning)



        best_score = float("-inf")
        best_model = None
        logs_array = []
        n_components = range(self.min_n_components, self.max_n_components + 1)

        try:
            # Number of folds cannot be less than number of sequences, default is 3
            split_method = KFold(min(3,len(self.sequences)))
            for n in n_components:
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                        try:
                            X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                            model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
                            X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                            LogL = model.score(X_test, lengths_test)
                            logs_array.append(LogL)
                        except:
                            pass
                mean_score = np.mean(logs_array)
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
        except:
            pass

        return best_model
