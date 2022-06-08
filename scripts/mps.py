import numpy as np
from ncon import ncon
import scipy.sparse.linalg as ln

class MPS:

	def __init__(self, N, fm="random"):

		self.N = N
		self.fm = fm
		
		self.sigmaX = np.array([[0, 1], [1, 0]], dtype=np.cfloat)
		self.sigmaY = np.array([[0, -1j], [1j, 0]], dtype=np.cfloat)
		self.sigmaZ = np.array([[1, 0], [0, -1]], dtype=np.cfloat)
		self.id = np.eye(2)

		self.initMPS()

	def initMPS(self):

		if self.fm=="up":
			self.A = [np.array([1, 0]).reshape(1, 2, 1) for i in range(self.N)]
			self.B = [np.array([1, 0]).reshape(1, 2, 1) for i in range(self.N)]
		elif self.fm=="+":
			self.A = [1/np.sqrt(2)*np.array([1, 1]).reshape(1, 2, 1) for i in range(self.N)]
			self.B = [1/np.sqrt(2)*np.array([1, 1]).reshape(1, 2, 1) for i in range(self.N)]
		elif self.fm=="down":
			self.A = [np.array([0, 1]).reshape(1, 2, 1) for i in range(self.N)]
			self.B = [np.array([0, 1]).reshape(1, 2, 1) for i in range(self.N)]
		elif self.fm=="-":
			self.A = [1/np.sqrt(2)*np.array([1, -1]).reshape(1, 2, 1) for i in range(self.N)]
			self.B = [1/np.sqrt(2)*np.array([1, -1]).reshape(1, 2, 1) for i in range(self.N)]
		elif self.fm=="random":
			self.A = [np.random.rand(2).reshape(1, 2, 1) for i in range(self.N)]
			self.B = [np.random.rand(2).reshape(1, 2, 1) for i in range(self.N)]
		self.weight = [np.eye(1, dtype=np.cfloat) for i in range(self.N+1)]
    
	def getChi(self):
		return [self.B[i].shape[2] for i in range(self.N-1)]

	def spin(self, i):

		legLinks = [[1, 2], [2, 4, 6], [4, 5], [1, 3], [3, 5, 6]]
		spinX = ncon([self.weight[i], self.B[i], self.sigmaX, self.weight[i], np.conjugate(self.B[i])], legLinks)
		spinY = ncon([self.weight[i], self.B[i], self.sigmaY, self.weight[i], np.conjugate(self.B[i])], legLinks)
		spinZ = ncon([self.weight[i], self.B[i], self.sigmaZ, self.weight[i], np.conjugate(self.B[i])], legLinks)

		return np.real_if_close(spinX), np.real_if_close(spinY), np.real_if_close(spinZ)

	def entropy(self, l):

		S = np.zeros(len(l))

		for i in range(len(l)):
			s = np.diag(self.weight[l[i]])
			S[i] = -np.sum(s**2 * np.log(s**2))

		return S

	def correlationLength(self):

		legLinks = [[1, 3, -3], [2, 3, -4], [-1, 1], [-2, 2]]
		T = ncon([self.B[self.N//2], np.conjugate(self.B[self.N//2]), self.weight[self.N//2], self.weight[self.N//2]], legLinks)
		T = T.reshape(T.shape[0]**2, T.shape[0]**2)

		e = ln.eigen.arpack.eigs(T, k=2, which='LM', return_eigenvectors=False)

		xi = self.N / np.log(abs(e[1])/abs(e[0]))

		return xi

	def parity(self):
		
		one3 = np.array([1]).reshape(1, 1, 1)
		sigmaZ = self.sigmaZ.reshape(1, 2, 1, 2)

		legLinks = [[1, 2, 3], [1, 4, -1], [2, 4, -2, 5], [3, 5, -3]]
		parity = ncon([one3, self.B[0], sigmaZ, np.conjugate(self.B[0])], legLinks)
		
		for i in range(1, self.N):
			legLinks = [[1, 2, 3], [1, 4, -1], [2, 4, -2, 5], [3, 5, -3]]
			parity = ncon([parity, self.B[i], sigmaZ, np.conjugate(self.B[i])], legLinks)

		legLinks = [[1, 2, 3], [1, 2, 3]]
		parity = ncon([parity, one3], legLinks)

		return np.real_if_close(parity)