import numpy as np
from ncon import ncon

class MPO:

	def __init__(self, N, J, h, lambdaI, lambda3, lambdaC, bc, pin, parity):      

		self.N = N
		self.J = J
		self.h = h
		self.lambdaI = lambdaI
		self.lambda3 = lambda3
		self.lambdaC = lambdaC
		self.bc = bc
		self.pin = pin
		self.parity = parity

		self.sigmaX = np.array([[0, 1], [1, 0]], dtype=np.cfloat)
		self.sigmaY = np.array([[0, -1j], [1j, 0]], dtype=np.cfloat)
		self.sigmaZ = np.array([[1, 0], [0, -1]], dtype=np.cfloat)
		self.id = np.eye(2)

		if self.bc=="ff":
			self.initOpenMPO()
		elif self.bc=="++": 
			self.initOpenMPO(pinL=self.pin, pinR=self.pin)
		elif self.bc=="--": 
			self.initOpenMPO(pinL=-self.pin, pinR=-self.pin)
		elif self.bc=="+-": 
			self.initOpenMPO(pinL=self.pin, pinR=-self.pin)
		elif self.bc=="-+": 
			self.initOpenMPO(pinL=-self.pin, pinR=self.pin)
		elif self.bc=="+f": 
			self.initOpenMPO(pinL=self.pin, pinR=0)
		elif self.bc=="f+": 
			self.initOpenMPO(pinL=0, pinR=self.pin)
		elif self.bc=="-f": 
			self.initOpenMPO(pinL=-self.pin, pinR=0)
		elif self.bc=="f-": 
			self.initOpenMPO(pinL=0, pinR=-self.pin)
		elif self.bc=="pbc": 
			self.initPeriodicMPO()
		elif self.bc=="apbc": 
			self.initAntiperiodicMPO()

	def initOpenMPO(self, pinL=0, pinR=0):

		H = np.zeros([7, 2, 7, 2], dtype=np.cfloat)

		H[0, :, 0, :] = self.id
		H[1, :, 0, :] = self.sigmaX
		H[2, :, 0, :] = self.sigmaZ
		H[5, :, 0, :] = self.sigmaY

		H[3, :, 1, :] = self.sigmaX
		H[4, :, 2, :] = self.sigmaX

		H[6, :, 0, :] = -2*self.h*self.lambdaI*self.sigmaZ
		H[6, :, 1, :] = -2*self.J*self.lambdaI*self.sigmaX-self.lambdaC*self.sigmaY
		H[6, :, 3, :] = self.lambda3*self.sigmaZ
		H[6, :, 4, :] = self.lambda3*self.sigmaX
		H[6, :, 5, :] = self.lambdaC*self.sigmaX
		H[6, :, 6, :] = self.id

		self.mpo = [H.copy() for i in range(self.N)]

		H[6, :, 0, :] = -2*self.h*self.lambdaI*self.sigmaZ + pinL*self.sigmaX
		self.mpo[0] = H.copy()

		H[6, :, 0, :] = -2*self.h*self.lambdaI*self.sigmaZ + pinR*self.sigmaX
		self.mpo[-1] = H.copy()

	def initPeriodicMPO(self):

		H = np.zeros([15, 2, 15, 2], dtype=np.cfloat)

		H[0, :, 0, :] = self.id
		H[1, :, 0, :] = self.sigmaX
		H[2, :, 0, :] = self.sigmaZ

		H[3, :, 1, :] = self.id
		H[4, :, 2, :] = self.id
		H[7, :, 3, :] = self.sigmaX
		H[8, :, 4, :] = self.sigmaX
		H[11, :, 7, :] = self.id
		H[12, :, 8, :] = self.id

		H[14, :, 0, :] = -2*self.h*self.lambdaI*self.sigmaZ
		H[14, :, 3, :] = -2*self.J*self.lambdaI*self.sigmaX
		H[14, :, 11, :] = self.lambda3*self.sigmaZ
		H[14, :, 12, :] = self.lambda3*self.sigmaX
		H[14, :, 14, :] = self.id

		self.mpo = [H.copy() for i in range(self.N)]

		H0 = H.copy()
		H0[14, :, 1, :] = -2*self.J*self.lambdaI*self.sigmaX
		H0[14, :, 6, :] = self.lambda3*self.sigmaX
		H0[14, :, 7, :] = self.lambda3*self.sigmaZ
		H0[14, :, 8, :] = self.lambda3*self.sigmaX
		H0[14, :, 13, :] = self.lambda3*self.sigmaX
		self.mpo[0] = H0.copy()

		H1 = H.copy()
		H1[6, :, 2, :] = self.sigmaX
		H1[13, :, 1, :] = self.sigmaZ
		self.mpo[1] = H1.copy()

		H2 = H.copy()
		H2[5, :, 1, :] = self.sigmaX
		H2[6, :, 2, :] = self.sigmaX
		H2[13, :, 1, :] = self.sigmaZ
		H2[14, :, 1, :] = -2*self.J*self.lambdaI*self.sigmaX
		self.mpo[-2] = H2.copy()
		
		H3 = H.copy()
		H3[9, :, 5, :] = self.id
		H3[10, :, 6, :] = self.id
		H3[14, :, 5, :] = self.lambda3*self.sigmaZ
		H3[14, :, 13, :] = self.lambda3*self.sigmaX
		self.mpo[-3] = H3.copy()

		H4 = H.copy()
		H4[14, :, 9, :] = self.lambda3*self.sigmaZ
		H4[14, :, 10, :] = self.lambda3*self.sigmaX
		self.mpo[-4] = H4.copy()

		if self.parity!=0:

			parity = np.zeros([2, 2, 2, 2], dtype=np.cfloat)

			parity[0, :, 0, :] = self.id
			parity[1, :, 1, :] = self.sigmaZ
			
			self.par = [parity.copy() for i in range(self.N)]

			parity0 = parity.copy()
			parity0[1, :, 0, :] = self.id
			parity0[1, :, 1, :] = self.parity*self.sigmaZ
			self.par[0] = 0.5*parity0.copy()

			parity1 = parity.copy()
			parity1[1, :, 0, :] = self.sigmaZ
			self.par[-1] = parity1.copy()

			for i in range(self.N):
				legLinks = [[-1, -4, -5, 1], [-2, 1, -6, 2], [-3, 2, -7, -8]]
				shape = (self.par[i].shape[0]*self.mpo[i].shape[0]*self.par[i].shape[0], self.par[i].shape[1], self.par[i].shape[0]*self.mpo[i].shape[0]*self.par[i].shape[0], self.par[i].shape[1])
				self.mpo[i] = ncon([self.par[i].copy(), self.mpo[i].copy(), self.par[i].copy()], legLinks).reshape(shape)

	def initAntiperiodicMPO(self):

		H = np.zeros([15, 2, 15, 2], dtype=np.cfloat)

		H[0, :, 0, :] = self.id
		H[1, :, 0, :] = self.sigmaX
		H[2, :, 0, :] = self.sigmaZ

		H[3, :, 1, :] = self.id
		H[4, :, 2, :] = self.id
		H[7, :, 3, :] = self.sigmaX
		H[8, :, 4, :] = self.sigmaX
		H[11, :, 7, :] = self.id
		H[12, :, 8, :] = self.id

		H[14, :, 0, :] = -2*self.h*self.lambdaI*self.sigmaZ
		H[14, :, 3, :] = -2*self.J*self.lambdaI*self.sigmaX
		H[14, :, 11, :] = self.lambda3*self.sigmaZ
		H[14, :, 12, :] = self.lambda3*self.sigmaX
		H[14, :, 14, :] = self.id

		self.mpo = [H.copy() for i in range(self.N)]

		H0 = H.copy()
		H0[14, :, 1, :] = 2*self.J*self.lambdaI*self.sigmaX
		H0[14, :, 6, :] = -self.lambda3*self.sigmaX
		H0[14, :, 7, :] = self.lambda3*self.sigmaZ
		H0[14, :, 8, :] = -self.lambda3*self.sigmaX
		H0[14, :, 13, :] = self.lambda3*self.sigmaX
		self.mpo[0] = H0.copy()

		H1 = H.copy()
		H1[6, :, 2, :] = self.sigmaX
		H1[13, :, 1, :] = self.sigmaZ
		self.mpo[1] = H1.copy()

		H2 = H.copy()
		H2[5, :, 1, :] = self.sigmaX
		H2[6, :, 2, :] = self.sigmaX
		H2[13, :, 1, :] = self.sigmaZ
		H2[14, :, 1, :] = -2*self.J*self.lambdaI*self.sigmaX
		self.mpo[-2] = H2.copy()
		
		H3 = H.copy()
		H3[9, :, 5, :] = self.id
		H3[10, :, 6, :] = self.id
		H3[14, :, 5, :] = self.lambda3*self.sigmaZ
		H3[14, :, 13, :] = self.lambda3*self.sigmaX
		self.mpo[-3] = H3.copy()

		H4 = H.copy()
		H4[14, :, 9, :] = self.lambda3*self.sigmaZ
		H4[14, :, 10, :] = self.lambda3*self.sigmaX
		self.mpo[-4] = H4.copy()

		if self.parity!=0:
			
			parity = np.zeros([2, 2, 2, 2], dtype=np.cfloat)

			parity[0, :, 0, :] = self.id
			parity[1, :, 1, :] = self.sigmaZ
			
			self.par = [parity.copy() for i in range(self.N)]

			parity0 = parity.copy()
			parity0[1, :, 0, :] = self.id
			parity0[1, :, 1, :] = self.parity*self.sigmaZ
			self.par[0] = 0.5*parity0.copy()

			parity1 = parity.copy()
			parity1[1, :, 0, :] = self.sigmaZ
			self.par[-1] = parity1.copy()

			for i in range(self.N):
				legLinks = [[-1, -4, -5, 1], [-2, 1, -6, 2], [-3, 2, -7, -8]]
				shape = (self.par[i].shape[0]*self.mpo[i].shape[0]*self.par[i].shape[0], self.par[i].shape[1], self.par[i].shape[0]*self.mpo[i].shape[0]*self.par[i].shape[0], self.par[i].shape[1])
				self.mpo[i] = ncon([self.par[i].copy(), self.mpo[i].copy(), self.par[i].copy()], legLinks).reshape(shape)
