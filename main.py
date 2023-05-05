import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from tqdm import tqdm


def coulomb_potential(x, y):
    return 1 / (np.abs(x - y) + 0.1)

class Genetic_Algorithm:
    def __init__(self, N, l, w, beta, lambda_star):
        self.N = N
        self.l = l
        self.active_set_indices = np.arange(0, N, 1)
        self.beta = beta
        self.lambda_star = lambda_star
        self.cost_matrix = w
        self.columns = self.initialize_A()
        self.current_cost = self.initialize_cost_lambda(self.columns)

    def initialize_A(self):
        columns = np.eye(self.l) * self.N
        for i in range(self.l * (self.beta - 1)):
            # vector with length l with N ones in random positions
            v = np.zeros(self.l)
            random_indices = np.random.choice(self.l, self.N, replace=False)
            v[random_indices] = 1
            columns = np.vstack((columns, v))
        return columns / self.N

    def run(self, max_iter, max_samples=100):
        alpha_active = None
        samples = 0
        gain = -1
        for _ in tqdm(range(max_iter)):
            if samples >= max_samples:
                break
            alpha_active = self.solve_RMP(alpha_active)
            self.active_set_indices = np.where(alpha_active > 0)[0]
            self.y_star = self.solve_DRMP()
            while gain < 0:
                parent_index = np.random.choice(self.active_set_indices)
                parent = self.columns[parent_index]
                child = self.get_child(parent)
                cost_child = self.get_cost_lambda(child)
                gain = child.dot(self.y_star) - cost_child
                samples += 1
            gain = -1
            self.columns = np.vstack((self.columns, child))
            self.current_cost = np.append(self.current_cost, cost_child)
            if self.columns.shape[0] >= self.l * (self.beta + 1):
                # clear oldest l non active columns
                non_active_indices = np.setdiff1d(
                    np.arange(0, self.columns.shape[0], 1),
                    self.active_set_indices)[:self.l]
                self.columns = np.delete(self.columns, non_active_indices,
                                         axis=0)
                alpha_active = np.delete(alpha_active, non_active_indices,
                                         axis=0)
                self.current_cost = np.delete(self.current_cost,
                                              non_active_indices, axis=0)

    def get_child(self, parent):
        # we subtract 1 to a random position in the parent vector and add 1 to a random position in the parent vector
        child = parent.copy()
        # choose a non zero position in the parent vector
        non_zero_indices = np.where(parent > 0)[0]
        random_index = np.random.choice(non_zero_indices)
        # subtract 1 from the random position
        child[random_index] -= 1 / self.N
        # toss a coin to decide where to add 1
        coin = np.random.randint(0, 2, 1)[0]
        # print("coin: ", coin)
        # either add 1 in the position to the right or to the left of the random position
        if coin == 0:
            if random_index == 0:
                child[-1] += 1 / self.N
            else:
                child[random_index - 1] += 1 / self.N
        else:
            if random_index == self.l - 1:
                child[0] += 1 / self.N
            else:
                child[random_index + 1] += 1 / self.N
        # print("child: ", child.sum())
        return child

    def plot(self):
        plt.scatter(range(100), self.y_star)
        plt.title("y star")
        plt.show()
        plt.close()
        marginal = self.compute_2_marginal_matrix()
        print("marginal: ", marginal.shape)
        marginal = np.sum(marginal, axis=0)
        print("marginal: ", marginal.shape)
        plt.imshow(marginal)
        plt.title("marginal")
        plt.show()
        plt.close()

    def solve_RMP(self, alpha_active):
        if alpha_active is None:
            alpha = sp.optimize.linprog(c=self.current_cost,
                                        A_eq=self.columns.T,
                                        b_eq=self.lambda_star, method="highs",
                                        bounds=(0, None))
        # minimize current_cost*alpha subject to alpha*columns = lambda_star and alpha >= 0
        else:
            alpha_active = np.append(alpha_active, 0)
            alpha = sp.optimize.linprog(c=self.current_cost,
                                        A_eq=self.columns.T,
                                        b_eq=self.lambda_star, method="highs",
                                        bounds=(0, None), x0=alpha_active)
        print("cost", alpha.fun)
        return alpha.x

    def solve_DRMP(self):
        # maximize y*lambda_star subject to columns*y <= current_cost
        y = sp.optimize.linprog(c=-self.lambda_star, A_ub=self.columns,
                                b_ub=self.current_cost, method="highs")
        return y.x

    def initialize_cost_lambda(self, column):
        cost_lambda = np.array([(N ** 2 / 2) * i.dot(self.cost_matrix).dot(
            i) - (N / 2) * np.diag(self.cost_matrix).dot(i) for i in column])
        return cost_lambda

    def get_cost_lambda(self, column):
        cost_lambda = (N ** 2 / 2) * column.dot(self.cost_matrix).dot(
            column) - (N / 2) * np.diag(self.cost_matrix).dot(column)
        return cost_lambda

    def compute_2_marginal_matrix(self):
        marginal = np.array([self.N / (self.N - 1) * np.tensordot(column,
                                                                  column,
                                                                  axes=0) - 1 / (
                                         self.N - 1) * np.diag(column) for
                             column in self.columns])
        print("marginal: ", marginal)
        return marginal


if __name__ == '__main__':
    x = np.linspace(0, 1, 100)
    N = 10  # number of marginals
    l = 100  # number of sites
    w = np.array(
        [[coulomb_potential(x[i], x[j]) for i in range(l)] for j in range(l)])

    beta = 5  # hyperparameter
    lambda_star = 0.2 + np.sin(np.pi * np.arange(0, l, 1) / (l + 1)) ** 2
    lambda_star = lambda_star / np.sum(lambda_star)
    max_iter = 10000
    ga = Genetic_Algorithm(N=N, l=l, w=w, beta=beta, lambda_star=lambda_star)
    ga.run(max_iter, max_samples=30000)
    ga.plot()

