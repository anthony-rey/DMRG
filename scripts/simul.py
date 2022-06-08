import dmrg as dmrg
import mpo as mpo
import mps as mps
import numpy as np
from scipy.optimize import curve_fit

dataFolder = "../data/"

# simul = "energies"
# simul = "entropies"
# simul = "gaps"
# simul = "phase"
# simul = "transition"
# simul = "through"
# simul = "spin"
simul = "conformal"
# simul = "pinning"
# simul = "edge"
# simul = "ratios"
# simul = "degeneracy"

if simul=="energies":

	N = 10
	chi = 30
	d = 2
	useArpack = True
	k = 3
	precVar = 1e-5
	precSing = 1e-10
	precEig = 1e-5
	dimKrylov = 4
	numSweepsMin = 1
	numSweepsMax = 2
	printEnergies = False

	J = 1
	h = 1
	lambdaI = 1
	lambda3 = 0.9
	lambdaC = 0

	pin = 0
	parity = -1

	dataFilename = f"L={N}_chi={chi}_J={J}_h={h}_i={lambdaI}_3={lambda3}_c={lambdaC}.dat"

	H = mpo.MPO(N, J, h, lambdaI, lambda3, lambdaC, "pbc", pin, parity)
	psi = mps.MPS(N, "random")

	engine = dmrg.DMRGEngine(mpo=H, mps=psi, N=N, chi=chi, d=d, useArpack=useArpack, k=k,
		precVar=precVar, precSing=precSing, precEig=precEig, dimKrylov=dimKrylov,
		numSweepsMin=numSweepsMin, numSweepsMax=numSweepsMax, printEnergies=printEnergies)

	print(f"\n================= Start DMRG for " + dataFilename[:-4] + "\n")

	psi, E = engine.run()

	print(min(E[0]))
	print(min(E[1]))
	print(min(E[2]))
	print(psi.parity())

	data = np.array(E[0]).reshape(len(E[0]), 1)
	for j in range(1, k):
		data = np.concatenate((data, np.array(E[j]).reshape(len(E[j]), 1)), axis=1)
	# np.savetxt(dataFolder + f"{simul}/" + dataFilename, data, fmt='%.15f')

if simul=="entropies":

	N = np.array([140, 160]).astype(int)
	# N = np.array([50]).astype(int)
	chi = np.array([150, 150]).astype(int)
	d = 2
	useArpack = True
	k = 1
	precVar = 1e-8
	precSing = 1e-10
	precEig = 1e-5
	dimKrylov = 4
	numSweepsMin = 1
	numSweepsMax = 2
	printEnergies = False

	J = 1
	h = 1
	lambdaI = 1
	lambda3 = 0.856
	lambdaC = 0

	pin = 100
	parity = 0
	
	for i in range(len(N)):

		dataFilename = f"L={N[i]}_chi={chi[i]}_J={J}_h={h}_i={lambdaI}_3={lambda3}_c={lambdaC}.dat"

		H = mpo.MPO(N[i], J, h, lambdaI, lambda3, lambdaC, "--", pin, parity)
		psi = mps.MPS(N[i], "random")
		engine = dmrg.DMRGEngine(mpo=H, mps=psi, N=N[i], chi=chi[i], d=d, useArpack=useArpack, k=k,
			precVar=precVar, precSing=precSing, precEig=precEig, dimKrylov=dimKrylov,
			numSweepsMin=numSweepsMin, numSweepsMax=numSweepsMax, printEnergies=printEnergies)

		print(f"\n================= Start DMRG for " + dataFilename[:-4] + "\n")

		psi, E = engine.run()

		l = np.arange(int(N[i]/2-0.17*N[i])+1, int(N[i]/2+0.17*N[i])+1).astype(int)
		S = psi.entropy(l)

		data = np.concatenate((l.reshape(len(l), 1), S.reshape(len(l), 1)), axis=1)
		np.savetxt(dataFolder + f"{simul}/" + dataFilename, data, fmt='%.15f')

if simul=="gaps":

	N = np.array([60, 80, 100, 120]).astype(int)
	chi = 200
	d = 2
	useArpack = True
	k = 2
	precVar = 1e-8
	precSing = 1e-10
	precEig = 1e-5
	dimKrylov = 4
	numSweepsMin = 1
	numSweepsMax = 10
	printEnergies = False

	J = 1
	h = 1
	lambdaI = 1
	lambda3 = 0
	lambdaC = 0
	
	for i in range(len(N)):

		dataFilename = f"L={N[i]}_chi={chi}_J={J}_h={h}_i={lambdaI}_3={lambda3}_c={lambdaC}.dat"

		H = mpo.MPO(N[i], J, h, lambdaI, lambda3, lambdaC)
		psi = mps.MPS(N[i], "up")
		engine = dmrg.DMRGEngine(mpo=H, mps=psi, N=N[i], chi=chi, d=d, useArpack=useArpack, k=k,
			precVar=precVar, precSing=precSing, precEig=precEig, dimKrylov=dimKrylov,
			numSweepsMin=numSweepsMin, numSweepsMax=numSweepsMax, printEnergies=printEnergies)

		print(f"\n================= Start DMRG for " + dataFilename[:-4] + "\n")

		psi, E = engine.run()

		data = np.array(E[0]).reshape(len(E[0]), 1)
		for j in range(1, k):
			data = np.concatenate((data, np.array(E[j]).reshape(len(E[j]), 1)), axis=1)
		np.savetxt(dataFolder + f"{simul}/" + dataFilename, data, fmt='%.15f')

if simul=="phase":

	N = np.array([20, 30, 40, 50, 70, 100]).astype(int)
	# N = np.array([100]).astype(int)
	chi = np.array([40, 50, 60, 80, 100, 100]).astype(int)
	# chi = np.array([170]).astype(int)
	d = 2
	useArpack = True
	k = 1
	precVar = 1e-3
	precSing = 1e-10
	precEig = 1e-5
	dimKrylov = 4
	numSweepsMin = 1
	numSweepsMax = 2
	printEnergies = False

	J = 1
	h = 1
	lambdaI = 1
	lambda3 = np.array([0.75, 0.83, 0.86])
	lambdaC = 0

	pin = 100

	for j in range(len(lambda3)):
		
		# dataFilename = f"chi={max(chi)}_J={J}_h={h}_i={lambdaI}_3={lambda3[j]}_c={lambdaC}.dat"
		dataFilename = f"J={J}_h={h}_i={lambdaI}_3={lambda3[j]}_c={lambdaC}.dat"
		
		c = np.zeros(len(N))
		cErr = np.zeros(len(N))

		for i in range(len(N)):

			H = mpo.MPO(N[i], J, h, lambdaI, lambda3[j], lambdaC, "pbc", pin)
			psi = mps.MPS(N[i], "random")
			engine = dmrg.DMRGEngine(mpo=H, mps=psi, N=N[i], chi=chi[i], d=d, useArpack=useArpack, k=k,
				precVar=precVar, precSing=precSing, precEig=precEig, dimKrylov=dimKrylov,
				numSweepsMin=numSweepsMin, numSweepsMax=numSweepsMax, printEnergies=printEnergies)

			print(f"\n================= Start DMRG for L={N[i]}_" + dataFilename[:-4] + "\n")

			psi, E = engine.run()

			l = np.arange(int(N[i]/2-0.17*N[i])+1, int(N[i]/2+0.17*N[i])+1).astype(int)
			S = psi.entropy(l)

			L = N[i]
			def f(l, c, const):
				# return c/6 * np.log(2*L/np.pi * np.sin(np.pi*l/L)) + const
				return c/3 * np.log(L/np.pi * np.sin(np.pi*l/L)) + const

			popt, pcov = curve_fit(f, l, S)
			perr = np.sqrt(np.diag(pcov))

			c[i] = popt[0]
			cErr[i] = perr[0]

		data = np.concatenate((N.reshape(len(N), 1), c.reshape(len(N), 1), cErr.reshape(len(N), 1)), axis=1)
		np.savetxt(dataFolder + f"{simul}/" + dataFilename, data, fmt='%.15f')

if simul=="transition":

	N = 50
	chi = 50
	d = 2
	useArpack = True
	k = 3
	precVar = 1e-8
	precSing = 1e-10
	precEig = 1e-5
	dimKrylov = 4
	numSweepsMin = 1
	numSweepsMax = 5
	printEnergies = False

	J = 1
	h = np.linspace(0.5, 1.5, 15)
	lambdaI = 0.5
	lambda3 = 0
	lambdaC = 0

	for i in range(len(h)):

		dataFilename = f"L={N}_chi={chi}_J={J}_h={h[i]}_i={lambdaI}_3={lambda3}_c={lambdaC}.dat"
		
		H = mpo.MPO(N, J, h[i], lambdaI, lambda3, lambdaC, "open")
		psi = mps.MPS(N, "up")
		engine = dmrg.DMRGEngine(mpo=H, mps=psi, N=N, chi=chi, d=d, useArpack=useArpack, k=k,
			precVar=precVar, precSing=precSing, precEig=precEig, dimKrylov=dimKrylov,
			numSweepsMin=numSweepsMin, numSweepsMax=numSweepsMax, printEnergies=printEnergies)

		print(f"\n================= Start DMRG for " + dataFilename[:-4] + "\n")

		psi, E = engine.run()

		data = np.array(E[0]).reshape(len(E[0]), 1)
		for j in range(1, k):
			data = np.concatenate((data, np.array(E[j]).reshape(len(E[j]), 1)), axis=1)
		np.savetxt(dataFolder + f"{simul}/" + dataFilename, data, fmt='%.15f')

if simul=="through":

	N = 20
	chi = 50
	d = 2
	useArpack = True
	k = 1
	precVar = 1e-5
	precSing = 1e-10
	precEig = 1e-5
	dimKrylov = 3
	numSweepsMin = 3
	numSweepsMax = 3
	printEnergies = False

	J = 1
	h = 1
	lambdaI = 0
	# lambda3 = np.array([0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.856, 0.86, 0.87, 0.88, 0.89, 0.90, 0.92, 0.95, 1])
	lambda3 = np.array([1])
	lambdaC = 0

	pin = 0
	parity = 0

	for i in range(len(lambda3)):

		dataFilename = f"L={N}_chi={chi}_J={J}_h={h}_i={lambdaI}_3={lambda3[i]}_c={lambdaC}.dat"

		H = mpo.MPO(N, J, h, lambdaI, lambda3[i], lambdaC, "pbc", pin, parity)
		psi = mps.MPS(N, "random")
		engine = dmrg.DMRGEngine(mpo=H, mps=psi, N=N, chi=chi, d=d, useArpack=useArpack, k=k,
			precVar=precVar, precSing=precSing, precEig=precEig, dimKrylov=dimKrylov,
			numSweepsMin=numSweepsMin, numSweepsMax=numSweepsMax, printEnergies=printEnergies)

		print(f"\n================= Start DMRG for " + dataFilename[:-4] + "\n")

		psi, E = engine.run()

		l = np.arange(int(N/2-0.17*N)+1, int(N/2+0.17*N)+1).astype(int)
		S = psi.entropy(l)

		data = np.concatenate((l.reshape(len(l), 1), S.reshape(len(l), 1)), axis=1)
		np.savetxt(dataFolder + f"{simul}/" + dataFilename, data, fmt='%.15f')

if simul=="spin":

	save = 1
	saveGS = 0

	N = 30
	chi = 50
	d = 2
	useArpack = True
	k = 1
	precVar = 1e-3
	precSing = 1e-10
	precEig = 1e-5
	dimKrylov = 4
	numSweepsMin = 1
	numSweepsMax = 2
	printEnergies = False

	J = 1
	h = np.linspace(1, 1, 1)
	lambdaI = 1
	lambda3 = 1
	lambdaC = 0

	pin = 0
	parity = 0
	
	for i in range(len(h)):
		
		dataFilename = f"L={N}_chi={chi}_J={J}_h={h[i]}_i={lambdaI}_3={lambda3}_c={lambdaC}.dat"

		H = mpo.MPO(N, J, h[i], lambdaI, lambda3, lambdaC, "pbc", pin, parity)
		psi = mps.MPS(N, "-")
		engine = dmrg.DMRGEngine(mpo=H, mps=psi, N=N, chi=chi, d=d, useArpack=useArpack, k=k,
			precVar=precVar, precSing=precSing, precEig=precEig, dimKrylov=dimKrylov,
			numSweepsMin=numSweepsMin, numSweepsMax=numSweepsMax, printEnergies=printEnergies)

		print(f"\n================= Start DMRG for " + dataFilename[:-4] + "\n")

		psi, E = engine.run()

		if saveGS:
			np.save(dataFolder + f"{simul}/B_" + dataFilename[:-4], np.array(psi.B, dtype=object))
			np.save(dataFolder + f"{simul}/weigths_" + dataFilename[:-4], np.array(psi.weight, dtype=object))

		spinX = [psi.spin(i)[0] for i in range(N)]
		spinY = [psi.spin(i)[1] for i in range(N)]
		spinZ = [psi.spin(i)[2] for i in range(N)]

		print(spinX)
		print(min(E[0]))

		if save:
			data = np.concatenate((np.array(spinX).reshape(len(spinX), 1), np.array(spinY).reshape(len(spinY), 1), np.array(spinZ).reshape(len(spinZ), 1)), axis=1)
			np.savetxt(dataFolder + f"{simul}/" + dataFilename, data, fmt='%.15f')

if simul=="conformal":

	N = np.array([30, 40, 50, 60, 70]).astype(int)
	chi = 50
	d = 2
	useArpack = True
	k = 10
	precVar = 1e-6
	precSing = 1e-10
	precEig = 1e-8
	dimKrylov = 4
	numSweepsMin = 1
	numSweepsMax = 2
	printEnergies = False

	J = 1
	h = 1
	lambdaI = 1
	lambda3 = 0.856
	lambdaC = 0
	
	pin = 100
	parity = 0

	for i in range(len(N)):

		dataFilename = f"L={N[i]}_chi={chi}_J={J}_h={h}_i={lambdaI}_3={lambda3}_c={lambdaC}.dat"

		H = mpo.MPO(N[i], J, h, lambdaI, lambda3, lambdaC, "ff", pin, parity)
		psi = mps.MPS(N[i], "random")
		engine = dmrg.DMRGEngine(mpo=H, mps=psi, N=N[i], chi=chi, d=d, useArpack=useArpack, k=k,
			precVar=precVar, precSing=precSing, precEig=precEig, dimKrylov=dimKrylov,
			numSweepsMin=numSweepsMin, numSweepsMax=numSweepsMax, printEnergies=printEnergies)

		print(f"\n================= Start DMRG for " + dataFilename[:-4] + "\n")

		psi, E = engine.run()

		data = np.array(E[0]).reshape(len(E[0]), 1)
		for j in range(1, k):
			data = np.concatenate((data, np.array(E[j]).reshape(len(E[j]), 1)), axis=1)
		np.savetxt(dataFolder + f"{simul}/" + dataFilename, data, fmt='%.15f')

if simul=="pinning":

	N = np.array([20, 30, 40, 50]).astype(int)
	chi = 100
	d = 2
	useArpack = True
	k = 1
	precVar = 1e-8
	precSing = 1e-10
	precEig = 1e-5
	dimKrylov = 4
	numSweepsMin = 1
	numSweepsMax = 2
	printEnergies = False

	J = 1
	h = 1
	lambdaI = 1
	lambda3 = 0.856
	lambdaC = 0

	pin = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])

	for j in range(len(N)):
		
		dataFilename = f"L={N[j]}_chi={chi}_J={J}_h={h}_i={lambdaI}_3={lambda3}_c={lambdaC}.dat"
		
		c = np.zeros(len(pin))
		cErr = np.zeros(len(pin))

		for i in range(len(pin)):

			H = mpo.MPO(N[j], J, h, lambdaI, lambda3, lambdaC, "--", pin[i])
			psi = mps.MPS(N[j], "random")
			engine = dmrg.DMRGEngine(mpo=H, mps=psi, N=N[j], chi=chi, d=d, useArpack=useArpack, k=k,
				precVar=precVar, precSing=precSing, precEig=precEig, dimKrylov=dimKrylov,
				numSweepsMin=numSweepsMin, numSweepsMax=numSweepsMax, printEnergies=printEnergies)

			print(f"\n================= Start DMRG for pin={pin[i]}_" + dataFilename[:-4] + "\n")

			psi, E = engine.run()

			l = np.arange(int(N[j]/2-0.17*N[j])+1, int(N[j]/2+0.17*N[j])+1).astype(int)
			S = psi.entropy(l)

			L = N[j]
			def f(l, c, const):
				return c/6 * np.log(2*L/np.pi * np.sin(np.pi*l/L)) + const

			popt, pcov = curve_fit(f, l, S)
			perr = np.sqrt(np.diag(pcov))

			c[i] = popt[0]
			cErr[i] = perr[0]

		data = np.concatenate((pin.reshape(len(pin), 1), c.reshape(len(pin), 1), cErr.reshape(len(pin), 1)), axis=1)
		np.savetxt(dataFolder + f"{simul}/" + dataFilename, data, fmt='%.15f')

if simul=="edge":

	N = 30
	chi = 50
	d = 2
	useArpack = True
	k = 1
	precVar = 1e-8
	precSing = 1e-10
	precEig = 1e-5
	dimKrylov = 4
	numSweepsMin = 1
	numSweepsMax = 2
	printEnergies = False

	J = 1
	h = 1
	lambdaI = 1
	lambda3 = 0.856
	lambdaC = 0

	pin = np.array([0, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25, 50, 100])
		
	leftfree = np.zeros(len(pin))
	leftup = np.zeros(len(pin))
	leftzero = np.zeros(len(pin))
	leftmix = np.zeros(len(pin))
	rightfree = np.zeros(len(pin))
	rightup = np.zeros(len(pin))
	rightzero = np.zeros(len(pin))
	rightmix = np.zeros(len(pin))

	for i in range(len(pin)):
		
		dataFilename = f"L={N}_chi={chi}_J={J}_h={h}_i={lambdaI}_3={lambda3}_c={lambdaC}.dat"

		H = mpo.MPO(N, J, h, lambdaI, lambda3, lambdaC, "ff", pin[i])
		psi = mps.MPS(N, "random")
		engine = dmrg.DMRGEngine(mpo=H, mps=psi, N=N, chi=chi, d=d, useArpack=useArpack, k=k,
			precVar=precVar, precSing=precSing, precEig=precEig, dimKrylov=dimKrylov,
			numSweepsMin=numSweepsMin, numSweepsMax=numSweepsMax, printEnergies=printEnergies)

		print(f"pin = {pin[i]}")
		print(f"\n================= Start DMRG for " + dataFilename[:-4] + "\n")

		psi, E = engine.run()

		spinX = [psi.spin(i)[0] for i in range(N)]

		leftfree[i] = spinX[0]
		rightfree[i] = spinX[-1]

		H = mpo.MPO(N, J, h, lambdaI, lambda3, lambdaC, "--", pin[i])
		psi = mps.MPS(N, "random")
		engine = dmrg.DMRGEngine(mpo=H, mps=psi, N=N, chi=chi, d=d, useArpack=useArpack, k=k,
			precVar=precVar, precSing=precSing, precEig=precEig, dimKrylov=dimKrylov,
			numSweepsMin=numSweepsMin, numSweepsMax=numSweepsMax, printEnergies=printEnergies)

		print(f"\n================= Start DMRG for " + dataFilename[:-4] + "\n")

		psi, E = engine.run()

		spinX = [psi.spin(i)[0] for i in range(N)]

		leftup[i] = spinX[0]
		rightup[i] = spinX[-1]

		H = mpo.MPO(N, J, h, lambdaI, lambda3, lambdaC, "+-", pin[i])
		psi = mps.MPS(N, "random")
		engine = dmrg.DMRGEngine(mpo=H, mps=psi, N=N, chi=chi, d=d, useArpack=useArpack, k=k,
			precVar=precVar, precSing=precSing, precEig=precEig, dimKrylov=dimKrylov,
			numSweepsMin=numSweepsMin, numSweepsMax=numSweepsMax, printEnergies=printEnergies)

		print(f"\n================= Start DMRG for " + dataFilename[:-4] + "\n")

		psi, E = engine.run()

		spinX = [psi.spin(i)[0] for i in range(N)]

		leftzero[i] = spinX[0]
		rightzero[i] = spinX[-1]

		H = mpo.MPO(N, J, h, lambdaI, lambda3, lambdaC, "f+", pin[i])
		psi = mps.MPS(N, "random")
		engine = dmrg.DMRGEngine(mpo=H, mps=psi, N=N, chi=chi, d=d, useArpack=useArpack, k=k,
			precVar=precVar, precSing=precSing, precEig=precEig, dimKrylov=dimKrylov,
			numSweepsMin=numSweepsMin, numSweepsMax=numSweepsMax, printEnergies=printEnergies)

		print(f"\n================= Start DMRG for " + dataFilename[:-4] + "\n")

		psi, E = engine.run()

		spinX = [psi.spin(i)[0] for i in range(N)]

		leftmix[i] = spinX[0]
		rightmix[i] = spinX[-1]

	data = np.concatenate((np.array(pin).reshape(len(pin), 1), np.array(leftfree).reshape(len(leftfree), 1), np.array(rightfree).reshape(len(rightfree), 1), np.array(leftup).reshape(len(leftup), 1), np.array(rightup).reshape(len(rightup), 1), np.array(leftzero).reshape(len(leftzero), 1), np.array(rightzero).reshape(len(rightzero), 1), np.array(leftmix).reshape(len(leftmix), 1), np.array(rightmix).reshape(len(rightmix), 1)), axis=1)
	np.savetxt(dataFolder + f"{simul}/" + dataFilename, data, fmt='%.15f')

if simul=="ratios":

	# N = np.array([10, 15, 20, 25, 30, 35, 40]).astype(int)
	# N = np.array([10, 15, 20]).astype(int)
	# N = np.array([10, 20, 30, 40]).astype(int)
	N = np.array([50]).astype(int)
	chi = 100
	d = 2
	useArpack = True
	k = 2
	precVar = 1e-3
	precSing = 1e-10
	precEig = 1e-5
	dimKrylov = 4
	numSweepsMin = 1
	numSweepsMax = 2
	printEnergies = False

	J = 1
	h = 1
	lambdaI = 1
	lambda3 = np.array([0, 0.8, 0.855, 0.856, 0.857, 0.87])
	lambdaC = 0
	
	pin = 0
	parity = np.array([1, -1])

	for l in range(len(lambda3)):

		en = np.zeros((len(N), k*2+1))
		for i in range(len(N)):

			for j in range(len(parity)):

				dataFilename = f"J={J}_h={h}_i={lambdaI}_3={lambda3[l]}_c={lambdaC}.dat"

				H = mpo.MPO(N[i], J, h, lambdaI, lambda3[l], lambdaC, "pbc", pin, parity[j])
				psi = mps.MPS(N[i], "random")
				engine = dmrg.DMRGEngine(mpo=H, mps=psi, N=N[i], chi=chi, d=d, useArpack=useArpack, k=k,
					precVar=precVar, precSing=precSing, precEig=precEig, dimKrylov=dimKrylov,
					numSweepsMin=numSweepsMin, numSweepsMax=numSweepsMax, printEnergies=printEnergies)

				print(f"\n================= Start DMRG for " + dataFilename[:-4])
				print(f"PBCs -- parity = {parity[j]}, L = {N[i]} \n")

				psi, E = engine.run()
				
				print(np.round(psi.parity()))

				en[i, j] = min(E[0])
				en[i, j+2] = min(E[1])

			H = mpo.MPO(N[i], J, h, lambdaI, lambda3[l], lambdaC, "apbc", pin, -1)
			psi = mps.MPS(N[i], "random")
			engine = dmrg.DMRGEngine(mpo=H, mps=psi, N=N[i], chi=chi, d=d, useArpack=useArpack, k=1,
				precVar=precVar, precSing=precSing, precEig=precEig, dimKrylov=dimKrylov,
				numSweepsMin=numSweepsMin, numSweepsMax=numSweepsMax, printEnergies=printEnergies)

			print(f"\n================= Start DMRG for " + dataFilename[:-4])
			print(f"ABCs -- parity = {parity[j]}, L = {N[i]} \n")

			psi, E = engine.run()

			print(np.round(psi.parity()))

			en[i, 4] = min(E[0])

		data = np.concatenate((np.array(N).reshape(len(N), 1), en), axis=1)
		np.savetxt(dataFolder + f"{simul}/" + dataFilename, data, fmt='%.15f')

if simul=="degeneracy":

	# N = np.array([10, 15, 20, 25, 30, 35, 40]).astype(int)
	# N = np.array([10, 15, 20, 25]).astype(int)
	# N = np.array([10, 20, 30, 40]).astype(int)
	N = np.array([30]).astype(int)
	chi = 50
	d = 2
	useArpack = True
	k = 2
	precVar = 1e-3
	precSing = 1e-10
	precEig = 1e-8
	dimKrylov = 4
	numSweepsMin = 1
	numSweepsMax = 2
	printEnergies = False

	J = 1
	h = 1
	lambdaI = 1
	lambda3 = 0.9
	lambdaC = 0
	
	pin = 0
	parity = np.array([1, -1])

	en = np.zeros((len(N), k*2+1))
	for i in range(len(N)):

		for j in range(len(parity)):

			dataFilename = f"J={J}_h={h}_i={lambdaI}_3={lambda3}_c={lambdaC}.dat"

			H = mpo.MPO(N[i], J, h, lambdaI, lambda3, lambdaC, "pbc", pin, parity[j])
			psi = mps.MPS(N[i], "random")
			engine = dmrg.DMRGEngine(mpo=H, mps=psi, N=N[i], chi=chi, d=d, useArpack=useArpack, k=k,
				precVar=precVar, precSing=precSing, precEig=precEig, dimKrylov=dimKrylov,
				numSweepsMin=numSweepsMin, numSweepsMax=numSweepsMax, printEnergies=printEnergies)

			print(f"\n================= Start DMRG for " + dataFilename[:-4])
			print(f"PBCs -- parity = {parity[j]}, L = {N[i]} \n")

			psi, E = engine.run()
			
			print(np.round(psi.parity()))

			en[i, j] = min(E[0])
			en[i, j+2] = min(E[1])
			
			# data = np.array(E[0]).reshape(len(E[0]), 1)
			# for j in range(1, k):
			# 	data = np.concatenate((data, np.array(E[j]).reshape(len(E[j]), 1)), axis=1)
			# np.savetxt(dataFolder + f"energies/L={N[i]}_chi={chi}_" + dataFilename, data, fmt='%.15f')
		
		H = mpo.MPO(N[i], J, h, lambdaI, lambda3, lambdaC, "apbc", pin, -1)
		psi = mps.MPS(N[i], "random")
		engine = dmrg.DMRGEngine(mpo=H, mps=psi, N=N[i], chi=chi, d=d, useArpack=useArpack, k=1,
			precVar=precVar, precSing=precSing, precEig=precEig, dimKrylov=dimKrylov,
			numSweepsMin=numSweepsMin, numSweepsMax=numSweepsMax, printEnergies=printEnergies)

		print(f"\n================= Start DMRG for " + dataFilename[:-4])
		print(f"ABCs -- parity = {parity[j]}, L = {N[i]} \n")

		psi, E = engine.run()

		print(np.round(psi.parity()))

		en[i, 4] = min(E[0])

	data = np.concatenate((np.array(N).reshape(len(N), 1), en), axis=1)
	np.savetxt(dataFolder + f"{simul}/" + dataFilename, data, fmt='%.15f')
