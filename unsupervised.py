import numpy as np





class SVD:
    def __init__(self, n_components=None):
        self.n_components = n_components
    
    def fit(self, X):
        X = np.array(X)
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        self.components_ = Vt[:self.n_components].T if self.n_components else Vt.T
        self.explained_variance_ = (s ** 2) / (X.shape[0] - 1)
        self.singular_values_ = s
        self.U = U
        self.s = s
        self.Vt = Vt
        return U, s, Vt 
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def transform(self, X):
        X = np.array(X)
        return np.dot(X, self.components_)
    
    def inverse_transform(self, X, k=None):
        if k is None:
            k = self.n_components
        X = np.array(X)
        X_transformed = np.dot(X, self.components_[:k, :].T)
        return X_transformed
    



class PCA_U:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # center the data
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # compute the covariance matrix
        cov = np.cov(X, rowvar=False)

        # compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # sort the eigenvalues and eigenvectors in decreasing order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # store the first n_components eigenvectors as the principal components
        self.components = eigenvectors[:, : self.n_components]

    def transform(self, X):
        # center the data
        X = X - self.mean

        # project the data onto the principal components
        X_transformed = np.dot(X, self.components)

        return X_transformed 




 
    

class TSNE_U:
    def __init__(self, n_components=2, target_perplexity=12.0, learning_rate=12, n_iter=1000, seed=42, momentum=0.9):
        self.n_components = n_components
        self.target_perplexity = target_perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.seed = seed
        self.tse = True
        self.momentum = momentum


    def neg_squared_euc_dists(self, X):
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        return -D

    def softmax(self, X, diag_zero=True):
        e_x = np.exp(X - np.max(X, axis=1).reshape([-1, 1]))
        if diag_zero:
            np.fill_diagonal(e_x, 0.)
        e_x = e_x + 1e-8
        return e_x / e_x.sum(axis=1).reshape([-1, 1])

    def calc_prob_matrix(self, distances, sigmas=None):
        if sigmas is not None:
            two_sig_sq = 2. * np.square(sigmas.reshape((-1, 1)))
            return self.softmax(distances / two_sig_sq)
        else:
            return self.softmax(distances)

    def binary_search(self, eval_fn, target, tol=1e-10, max_iter=10000,
                      lower=1e-20, upper=1000.):
        for i in range(max_iter):
            guess = (lower + upper) / 2.
            val = eval_fn(guess)
            if val > target:
                upper = guess
            else:
                lower = guess
            if np.abs(val - target) <= tol:
                break
        return guess

    def calc_perplexity(self, prob_matrix):
        entropy = -np.sum(prob_matrix * np.log2(prob_matrix), 1)
        perplexity = 2 ** entropy
        return perplexity

    def perplexity(self, distances, sigmas):
        return self.calc_perplexity(self.calc_prob_matrix(distances, sigmas))
    
    def find_optimal_sigmas(self, distances, target_perplexity):
        sigmas = []
        for i in range(distances.shape[0]):
            eval_fn = lambda sigma: \
                self.perplexity(distances[i:i + 1, :], np.array(sigma))
            correct_sigma = self.binary_search(eval_fn, target_perplexity)
            sigmas.append(correct_sigma)
        return np.array(sigmas)

    def p_conditional_to_joint(self, P):
        return (P + P.T) / (2. * P.shape[0])

    def p_joint(self, X, target_perplexity):
        distances = self.neg_squared_euc_dists(X)
        sigmas = self.find_optimal_sigmas(distances, target_perplexity)
        p_conditional = self.calc_prob_matrix(distances, sigmas)
        P = self.p_conditional_to_joint(p_conditional)
        return P

    def q_joint(self, Y):
        distances = self.neg_squared_euc_dists(Y)
        exp_distances = np.exp(distances)
        np.fill_diagonal(exp_distances, 0.)
        return exp_distances / np.sum(exp_distances), None
    

    def symmetric_sne_grad(self, P, Q, Y, _):
        pq_diff = P - Q
        pq_expanded = np.expand_dims(pq_diff, 2)
        y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)
        grad = 4. * (pq_expanded * y_diffs).sum(1)
        return grad

    def q_tsne(self, Y):
        distances = self.neg_squared_euc_dists(Y)
        inv_distances = np.power(1. - distances, -1)
        np.fill_diagonal(inv_distances, 0.)
        return inv_distances / np.sum(inv_distances), inv_distances

    def tsne_grad(self, P, Q, Y, inv_distances):
        pq_diff = P - Q
        pq_expanded = np.expand_dims(pq_diff, 2)
        y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)
        distances_expanded = np.expand_dims(inv_distances, 2)
        y_diffs_wt = y_diffs * distances_expanded
        grad = 4. * (pq_expanded * y_diffs_wt).sum(1)
        return grad

    def estimate_sne(self, X, P, rng, num_iters, q_fn, grad_fn, learning_rate, momentum):
        Y = rng.normal(0., 0.0001, [X.shape[0], 2])
        if momentum:
            Y_m2 = Y.copy()
            Y_m1 = Y.copy()
        for i in range(num_iters):
            Q, distances = q_fn(Y)
            grads = grad_fn(P, Q, Y, distances)
            Y = Y - learning_rate * grads
            if momentum:
                Y += momentum * (Y_m1 - Y_m2)
                Y_m2 = Y_m1.copy()
                Y_m1 = Y.copy()
        return Y

    def fit(self, X):
        pass

    def fit_transform(self, X):
        P = self.p_joint(X, self.target_perplexity)
        rng = np.random.RandomState(self.seed)
        Y = self.estimate_sne(X, P, rng, self.n_iter,
                                q_fn=self.q_tsne if self.tse else self.q_joint,
                                grad_fn=self.tsne_grad if self.tse else self.symmetric_sne_grad,
                                learning_rate=self.learning_rate,
                                momentum=self.momentum)
        return Y

    def transform(self, X):
            pass    