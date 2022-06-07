import numpy as np
from ncon import ncon
import scipy.sparse.linalg as ln

class DMRGEngine:

	def __init__(self, mpo, mps, N, chi, d, precVar, precSing, precEig, dimKrylov, numSweepsMin, numSweepsMax, useArpack, k, printEnergies):

		self.mpo = mpo
		self.mps = mps
		self.N = N
		self.chi = chi
		self.d = d
		self.precVar = precVar
		self.precSing = precSing
		self.precEig = precEig
		self.dimKrylov = dimKrylov
		self.numSweepsMin = numSweepsMin
		self.numSweepsMax = numSweepsMax
		self.useArpack = useArpack
		self.k = k
		self.printEnergies = printEnergies

		self.E = [[] for i in range(self.k)]
		self.saveEnergies = False
		self.update = True
		self.endedWarmup = False
		self.stop = False
		self.numSweeps = 0

	def run(self):

		self.initEnvironment()

		useArpack_ = self.useArpack
		self.useArpack = False

		while (not self.stop or self.numSweeps<self.numSweepsMin or not self.endedWarmup) and self.numSweeps<self.numSweepsMax:

			if self.printEnergies:
				print("\n -----> R")
			self.rightSweep() 

			if self.printEnergies:
				print("\n -----> L")
			self.leftSweep() 

			var = self.computeVariance()
			if var<self.precVar:
				self.stop = True
			
			if self.endedWarmup:
				self.numSweeps += 1
				print(f"Ended sweep {self.numSweeps} with variance {var}")
			else:
				print(f"Ended sweep with variance {var}")
				
			if not self.endedWarmup and (max(self.mps.getChi())==self.chi or self.stop):
				self.useArpack = useArpack_
				self.endedWarmup = True
				self.saveEnergies = True
				print(f"--- Ended warmup")

		self.update = False
		self.rightSweep() 
		self.leftSweep() 

		print(f"\n--------------------------------------------------------- DONE")

		return self.mps, self.E

	def initEnvironment(self):

		self.L = [None for i in range(self.N)]
		self.R = [None for i in range(self.N)]

		chiL = self.mps.B[0].shape[0]
		chiR = self.mps.B[-1].shape[2]
		mpoDim = self.mpo.mpo[0].shape[0]

		L_ = np.zeros([chiL, mpoDim, chiL], dtype=np.cfloat)
		R_ = np.zeros([chiR, mpoDim, chiR], dtype=np.cfloat)
		L_[:, mpoDim-1, :] = np.eye(chiL)
		R_[:, 0, :] = np.eye(chiR)
		self.L[0] = L_
		self.R[-1] = R_

		for i in range(self.N-1, 0, -1):
			
			legLinks = [[-2, 1, 2, 3], [4, 2, 5], [-1, 1, 4], [-3, 3, 5]]
			self.R[i-1] = ncon([self.mpo.mpo[i], self.R[i], self.mps.B[i], np.conj(self.mps.B[i])], legLinks)

	def rightSweep(self):

		for i in range(self.N):

			if i==(self.N-1):

				chiL = self.mps.B[i].shape[0]
				chiR = self.mps.B[i].shape[2]

				legLinks = [[-1, 1], [1, -2, -3]]
				M = ncon([self.mps.weight[i], self.mps.B[i]], legLinks).reshape(chiL*self.d, chiR)
				u, s, v = np.linalg.svd(M, full_matrices=False)
				self.mps.A[i] = u.reshape(chiL, self.d, chiR)
				self.mps.weight[i+1] = np.diag(s)@v

			else:

				chiL = self.mps.B[i].shape[0]
				chiR = self.mps.B[i+1].shape[2]

				legLinks = [[-1, 1], [1, -2, 2], [2, -3, -4]]
				gs = ncon([self.mps.weight[i], self.mps.B[i], self.mps.B[i+1]], legLinks).reshape(chiL*self.d*self.d*chiR)
				
				if self.update:

					if self.useArpack:
						
						lin = Heff(self.L[i], self.mpo.mpo[i], self.mpo.mpo[i+1], self.R[i+1])
						e, v = ln.eigen.arpack.eigsh(lin, k=self.k, which='SA', return_eigenvectors=True, v0=gs, tol=self.precEig)
						gs = v[:, 0]
						e = np.sort(e)
						
						if self.saveEnergies:
							for j in range(self.k):
								self.E[j].append(e[j])

					else:

						gs = self.lanczos(gs, twoSiteMPOAppliedOnMPS, (self.L[i].copy(), self.mpo.mpo[i], self.mpo.mpo[i+1], self.R[i+1].copy()))

					if self.printEnergies and self.saveEnergies:
						print(self.E[0][-1])

				u, s, v = np.linalg.svd(gs.reshape(chiL*self.d, self.d*chiR), full_matrices=False)
				chiTrunc = min(self.chi, np.sum(s>self.precSing)) 

				self.mps.A[i] = u[:, :chiTrunc].reshape(chiL, self.d, chiTrunc)
				self.mps.B[i+1] = v[:chiTrunc, :].reshape(chiTrunc, self.d, chiR)
				self.mps.weight[i+1] = np.diag(s[:chiTrunc]) / np.linalg.norm(s[:chiTrunc])

				legLinks = [[1, 2, 4], [2, 3, -2, 5], [1, 3, -1], [4, 5, -3]]
				self.L[i+1] = ncon([self.L[i], self.mpo.mpo[i], self.mps.A[i], np.conjugate(self.mps.A[i])], legLinks)

	def leftSweep(self):

		for i in range(self.N-1, -1, -1):

			if i==0:

				chiL = self.mps.A[i].shape[0]
				chiR = self.mps.A[i].shape[2]

				legLinks = [[-1, -2, 1], [1, -3]]
				M = ncon([self.mps.A[i], self.mps.weight[i+1]], legLinks).reshape(chiL, self.d*chiR)
				u, s, v = np.linalg.svd(M, full_matrices=False)
				self.mps.B[i] = v.reshape(chiL, self.d, chiR)
				self.mps.weight[i] = u@np.diag(s)

			else:

				chiL = self.mps.A[i-1].shape[0]
				chiR = self.mps.A[i].shape[2]

				legLinks = [[-1, -2, 1], [1, -3, 2], [2, -4]]
				gs = ncon([self.mps.A[i-1], self.mps.A[i], self.mps.weight[i+1]], legLinks).reshape(chiL*self.d*self.d*chiR)
				
				if self.update:

					if self.useArpack:
						
						lin = Heff(self.L[i-1], self.mpo.mpo[i-1], self.mpo.mpo[i], self.R[i])
						e, v = ln.eigen.arpack.eigsh(lin, k=self.k, which='SA', return_eigenvectors=True, v0=gs, tol=self.precEig)
						gs = v[:, 0]
						e = np.sort(e)

						if self.saveEnergies:
							for j in range(self.k):
								self.E[j].append(e[j])

					else:

						gs = self.lanczos(gs, twoSiteMPOAppliedOnMPS, (self.L[i-1].copy(), self.mpo.mpo[i-1], self.mpo.mpo[i], self.R[i].copy()))

					if self.printEnergies and self.saveEnergies:
						print(self.E[0][-1])

				u, s, v = np.linalg.svd(gs.reshape(chiL*self.d, self.d*chiR), full_matrices=False)
				chiTrunc = min(self.chi, np.sum(s>self.precSing)) 

				self.mps.A[i-1] = u[:, :chiTrunc].reshape(chiL, self.d, chiTrunc)
				self.mps.B[i] = v[:chiTrunc, :].reshape(chiTrunc, self.d, chiR)
				self.mps.weight[i] = np.diag(s[:chiTrunc]) / np.linalg.norm(s[:chiTrunc])

				legLinks = [[-2, 3, 2, 5], [1, 2, 4], [-1, 3, 1], [-3, 5, 4]]
				self.R[i-1] = ncon([self.mpo.mpo[i], self.R[i], self.mps.B[i], np.conjugate(self.mps.B[i])], legLinks)

	def lanczos(self, psi, f, fArgs):

		K = np.zeros([len(psi), self.dimKrylov+1], dtype=np.cfloat)
		Hk = np.zeros([self.dimKrylov, self.dimKrylov], dtype=np.cfloat)

		K[:, 0] = psi / max(np.linalg.norm(psi), 1e-10)

		for i in range(1, self.dimKrylov+1):
		
			K[:, i] = f(K[:, i-1], *fArgs)

			for k in range(i):
				
				a = np.vdot(K[:, k], K[:, i])

				Hk[i-1, k] = a
				Hk[k, i-1] = np.conj(a)

			for k in range(i):

				a = np.vdot(K[:, k], K[:, i])
				K[:, i] -= a*K[:, k]

			K[:, i] /= max(np.linalg.norm(K[:, i]), 1e-10)

		[vals, vecs] = np.linalg.eigh(Hk)
		psi = K[:, :self.dimKrylov]@vecs[:, 0]

		psi /= max(np.linalg.norm(psi), 1e-10)

		if self.saveEnergies:
			for i in range(self.k):
				self.E[i].append(vals[i])

		return psi

	def computeVariance(self):

		Hl = self.L[0]
		Hr = self.R[-1]

		legLinks = [[1, 3, -1], [1, 2, 4], [2, 3, -2, 5], [4, 6, 7], [6, 5, -3, 8], [7, 8, -4]]
		hSquared = ncon([self.mps.B[0], Hl, self.mpo.mpo[0], Hl, self.mpo.mpo[0], np.conjugate(self.mps.B[0])], legLinks)
		legLinks = [[1, 3, -1], [1, 2, 4], [2, 3, -2, 5], [4, 5, -3]]
		squaredH = ncon([self.mps.B[0], Hl, self.mpo.mpo[0], np.conjugate(self.mps.B[0])], legLinks)

		for i in range(1, self.N-1):
			legLinks = [[1, 2, 4, 6], [1, 3, -1], [2, 3, -2, 5], [4, 5, -3, 7], [6, 7, -4]]
			hSquared = ncon([hSquared, self.mps.B[i], self.mpo.mpo[i], self.mpo.mpo[i], np.conjugate(self.mps.B[i])], legLinks)
			legLinks = [[1, 3, -1], [1, 2, 4], [2, 3, -2, 5], [4, 5, -3]]
			squaredH = ncon([self.mps.B[i], squaredH, self.mpo.mpo[i], np.conjugate(self.mps.B[i])], legLinks)

		legLinks = [[9, 10, 11, 12], [9, 3, 1], [10, 3, 2, 5], [1, 2, 4], [11, 5, 6, 8], [4, 6, 7], [12, 8, 7]]
		hSquared = ncon([hSquared, self.mps.B[-1], self.mpo.mpo[-1], Hr, self.mpo.mpo[-1], Hr, np.conjugate(self.mps.B[-1])], legLinks)
		legLinks = [[6, 3, 1], [6, 7, 8], [7, 3, 2, 5], [1, 2, 4], [8, 5, 4]]
		squaredH = ncon([self.mps.B[-1], squaredH, self.mpo.mpo[-1], Hr, np.conjugate(self.mps.B[-1])], legLinks)
		
		var = hSquared - squaredH**2

		return np.real_if_close(var)

class Heff(ln.LinearOperator):

	def __init__(self, L, Wl, Wr, R):
		self.L = L
		self.R = R
		self.Wl = Wl
		self.Wr = Wr
		self.chiL = L.shape[0]
		self.chiR = R.shape[0]
		self.d = Wl.shape[1]

		self.dtype = Wr.dtype
		self.shape = (self.chiL*self.d*self.d*self.chiR, self.chiL*self.d*self.d*self.chiR)

	def _matvec(self, mps):
		
		mps = mps.reshape(self.chiL, self.d, self.d, self.chiR)
		
		legLinks = [[1, 3, 5, 7], [1, 2, -1], [2, 3, 4, -2], [4, 5, 6, -3], [7, 6, -4]]
		res = ncon([mps, self.L, self.Wl, self.Wr, self.R], legLinks).reshape(self.chiL*self.d*self.d*self.chiR)

		return res

def twoSiteMPOAppliedOnMPS(mps, L, Hl, Hr, R):

	mps = mps.reshape(L.shape[0], Hl.shape[1], Hr.shape[1], R.shape[0])

	legLinks = [[1, 3, 5, 7], [1, 2, -1], [2, 3, 4, -2], [4, 5, 6, -3], [7, 6, -4]]
	res = ncon([mps, L, Hl, Hr, R], legLinks).reshape(L.shape[0]*Hl.shape[1]*Hr.shape[1]*R.shape[0])
	
	return res