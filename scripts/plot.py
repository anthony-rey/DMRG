import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from cycler import cycler
from uncertainties import ufloat
import math
import os as os
import re as re

colors = [[0.8, 0, 0.5],[0, 0.5, 0.8],[0.8, 0.5, 0]]
peach = [0.8, 0.3, 0.2]

plt.rcParams.update({
	'text.usetex': True,
	'font.family': 'serif',
	'font.size': 14,
	'legend.facecolor': peach,
	'legend.framealpha': 0.1,
	'legend.fontsize': 14,
	'savefig.dpi': 250,
	'savefig.bbox': 'tight',
	})

def magnitude(x):
    return int(math.floor(math.log10(abs(x))))

def r2(y, fit):
	ss_res = np.sum((y-fit)**2)
	ss_tot = np.sum((y-np.mean(y))**2)
	return (1 - ss_res/ss_tot)


dataFolder = "../data/"
graphFolder = "../graphs/"
saving = 1

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

# simul = "exponent"

if simul=="energies":

	filename = []
	for file in sorted(os.scandir(dataFolder + f"{simul}/"), key=lambda e: e.name):
		filename.append(file.name)

	for file in filename:

		expression = f"(?:L=)(?P<L>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_J=)(?P<J>[0-9e\.-]+)(?:_h=)(?P<h>[0-9e\.-]+)(?:_i=)(?P<lambdaI>[0-9e\.-]+)(?:_3=)(?P<lamba3>[0-9e\.-]+)(?:_c=)(?P<lambdac>[0-9e\.-]+)(?:\.dat)"
		m = re.match(expression, file)
		d = m.groupdict()

		for key in d.keys():
			d[key] = float(d[key])

		E =  np.loadtxt(dataFolder + f"{simul}/" + file)
		L, chi, J, h, lambdaI, lambda3, lambdaC = d.values()

		itr = range(1, len(E[:, 0])+1)

		fig, ax = plt.subplots(figsize=(6, 5))

		for j in range(len(E[0, :])):
			ax.plot(itr, E[:, j], label=f"$E_{j}$")
			print(min(E[:, j]))

		# ax.set_ylim(min(E[:, 0])-0.1, min(E[:, -1])+0.6)

		plt.xlabel("iteration")
		plt.ylabel("$E$")
		ax.minorticks_on()
		ax.grid(which='minor', linewidth=0.2)
		ax.grid(which='major', linewidth=0.6)
		ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)

		plt.draw()

		if saving:
			fig.savefig(graphFolder + f"{simul}/energies_L={L}_chi={chi}_J={J}_h={h}_i={lambdaI}_3={lambda3}_c={lambdaC}" + ".png")

	plt.show()

if simul=="entropies":

	c = []
	cErr = []
	N = []
	fig, ax = plt.subplots(figsize=(6, 5))

	filename = []
	for file in sorted(os.scandir(dataFolder + f"{simul}/"), key=lambda e: e.name):
		filename.append(file.name)

	for file in filename:

		expression = f"(?:L=)(?P<L>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_J=)(?P<J>[0-9e\.-]+)(?:_h=)(?P<h>[0-9e\.-]+)(?:_i=)(?P<lambdaI>[0-9e\.-]+)(?:_3=)(?P<lamba3>[0-9e\.-]+)(?:_c=)(?P<lambdac>[0-9e\.-]+)(?:\.dat)"
		m = re.match(expression, file)
		d = m.groupdict()

		for key in d.keys():
			d[key] = float(d[key])

		l, S =  np.loadtxt(dataFolder + f"{simul}/" + file, unpack=True)
		L, chi, J, h, lambdaI, lambda3, lambdaC = d.values()

		# length = len(l)
		# l = l[length//2-2:length//2+3]
		# S = S[length//2-2:length//2+3]

		def f(l, c, const):
			return c/6 * np.log(2*L/np.pi * np.sin(np.pi*l/L)) + const
			# return c/3 * np.log(L/np.pi * np.sin(np.pi*l/L)) + const

		popt, pcov = curve_fit(f, l, S)
		perr = np.sqrt(np.diag(pcov))
		
		c.append(popt[0])
		cErr.append(perr[0])
		N.append(L)

	c = np.array(c)	
	cErr = np.array(cErr)	
	N = np.array(N)

	x = np.linspace(min(1/N), max(1/N), 10)
	
	def g(inv, a, b):
		return a*inv + b

	popt, pcov = curve_fit(g, 1/N, c)
	perr = np.sqrt(np.diag(pcov))

	rSquared = r2(c, g(1/N, *popt))
	print(rSquared)

	label=fr"$c =  ({popt[1]:.3f} \pm {perr[1]:0.3f}) + ({popt[0]:.1f} \pm {perr[0]:0.1f}) \frac{{1}}{{L}}$"

	ax.scatter(1/N, c, marker='x', color=colors[0])
	ax.plot(x, g(x, *popt), linestyle=':', color=colors[1], label=label)

	plt.xlabel(r"$\frac{1}{L}$")
	plt.ylabel("$c$")
	ax.minorticks_on()
	ax.grid(which='minor', linewidth=0.2)
	ax.grid(which='major', linewidth=0.6)
	ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0)

	plt.draw()

	if saving:
		fig.savefig(graphFolder + f"{simul}/calabrese_chi={chi}_J={J}_h={h}_i={lambdaI}_3={lambda3}_c={lambdaC}" + ".png")
	
	c = []
	N = []
	fig, ax = plt.subplots(figsize=(6, 5))

	filename = []
	for file in sorted(os.scandir(dataFolder + f"{simul}/"), key=lambda e: e.name):
		filename.append(file.name)

	for file in filename:

		expression = f"(?:L=)(?P<L>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_J=)(?P<J>[0-9e\.-]+)(?:_h=)(?P<h>[0-9e\.-]+)(?:_i=)(?P<lambdaI>[0-9e\.-]+)(?:_3=)(?P<lamba3>[0-9e\.-]+)(?:_c=)(?P<lambdac>[0-9e\.-]+)(?:\.dat)"
		m = re.match(expression, file)
		d = m.groupdict()

		for key in d.keys():
			d[key] = float(d[key])

		l, S =  np.loadtxt(dataFolder + f"{simul}/" + file, unpack=True)
		L, chi, J, h, lambdaI, lambda3, lambdaC = d.values()
		
		# length = len(l)
		# l = l[length//2-2:length//2+3]
		# S = S[length//2-2:length//2+3]

		c.append(6*(S[l==L//2-1]-S[l==L//2])/np.log(np.cos(np.pi/L)))
		# c.append(3*(S[l==L//2-1]-S[l==L//2])/np.log(np.cos(np.pi/L)))
		N.append(L)

	c = np.array(c).reshape(len(c))
	N = np.array(N)

	x = np.linspace(min(1/N), max(1/N), 10)
	
	def g(inv, a, b):
		return a*inv + b

	popt, pcov = curve_fit(g, 1/N, c)
	perr = np.sqrt(np.diag(pcov))

	label=fr"$c =  ({popt[1]:.3f} \pm {perr[1]:0.3f}) + ({popt[0]:.1f} \pm {perr[0]:0.1f}) \frac{{1}}{{L}}$"

	ax.scatter(1/N, c, marker='x', color=colors[0])
	ax.plot(x, g(x, *popt), linestyle=':', color=colors[1], label=label)

	plt.xlabel(r"$\frac{1}{L}$")
	plt.ylabel("$c$")
	ax.minorticks_on()
	ax.grid(which='minor', linewidth=0.2)
	ax.grid(which='major', linewidth=0.6)
	ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0)

	plt.draw()

	if saving:
		fig.savefig(graphFolder + f"{simul}/other_chi={chi}_J={J}_h={h}_i={lambdaI}_3={lambda3}_c={lambdaC}" + ".png")
		
	filename = []
	for file in sorted(os.scandir(dataFolder + f"{simul}/"), key=lambda e: e.name):
		filename.append(file.name)

	for file in filename:
		
		fig, ax = plt.subplots(figsize=(6, 5))

		expression = f"(?:L=)(?P<L>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_J=)(?P<J>[0-9e\.-]+)(?:_h=)(?P<h>[0-9e\.-]+)(?:_i=)(?P<lambdaI>[0-9e\.-]+)(?:_3=)(?P<lamba3>[0-9e\.-]+)(?:_c=)(?P<lambdac>[0-9e\.-]+)(?:\.dat)"
		m = re.match(expression, file)
		d = m.groupdict()

		for key in d.keys():
			d[key] = float(d[key])

		l, S =  np.loadtxt(dataFolder + f"{simul}/" + file, unpack=True)
		L, chi, J, h, lambdaI, lambda3, lambdaC = d.values()

		# length = len(l)
		# l = l[length//2-2:length//2+3]
		# S = S[length//2-2:length//2+3]

		# l = l-4

		def f(l, c, const):
			# return c/6 * np.log(2*L/np.pi * np.sin(np.pi*l/L)) + const
			return c/3 * np.log(L/np.pi * np.sin(np.pi*l/L)) + const

		popt, pcov = curve_fit(f, l, S)
		perr = np.sqrt(np.diag(pcov))

		label=fr"$S = \frac{{({popt[0]:.5f} \pm {perr[0]:0.5f})}}{{6}} \ln\left[\frac{{2L}}{{\pi}}\sin\frac{{\pi \ell}}{{L}} \right] + ({popt[1]:.5f} \pm {perr[1]:0.5f})$"

		ax.scatter(l, S, marker='x', color=colors[0])
		ax.plot(l, f(l, *popt), linestyle=':', color=colors[1], label=label)

		plt.xlabel(r"$\ell$")
		plt.ylabel("$S$")
		ax.minorticks_on()
		ax.grid(which='minor', linewidth=0.2)
		ax.grid(which='major', linewidth=0.6)
		ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0)

		plt.draw()

		if saving:
			fig.savefig(graphFolder + f"{simul}/alone_L={L}_chi={chi}_J={J}_h={h}_i={lambdaI}_3={lambda3}_c={lambdaC}" + ".png")
		
	plt.show()

if simul=="gaps":

	# filename = []
	# for file in sorted(os.scandir(dataFolder + f"{simul}/"), key=lambda e: e.name):
	# 	filename.append(file.name)

	# for file in filename:
		
	# 	fig, ax = plt.subplots(figsize=(6, 5))
		
	# 	expression = f"(?:L=)(?P<L>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_J=)(?P<J>[0-9e\.-]+)(?:_h=)(?P<h>[0-9e\.-]+)(?:_i=)(?P<lambdaI>[0-9e\.-]+)(?:_3=)(?P<lamba3>[0-9e\.-]+)(?:_c=)(?P<lambdac>[0-9e\.-]+)(?:\.dat)"
	# 	m = re.match(expression, file)
	# 	d = m.groupdict()

	# 	for key in d.keys():
	# 		d[key] = float(d[key])

	# 	E =  np.loadtxt(dataFolder + f"{simul}/" + file)
	# 	L, chi, J, h, lambdaI, lambda3, lambdaC = d.values()

	# 	itr = range(1, len(E[:, 0])+1)

	# 	for j in range(len(E[0, :])):
	# 		ax.plot(itr, E[:, j], label=f"$E_{j}$")

	# 	plt.xlabel("iteration")
	# 	plt.ylabel("$E$")
	# 	ax.minorticks_on()
	# 	ax.grid(which='minor', linewidth=0.2)
	# 	ax.grid(which='major', linewidth=0.6)
	# 	ax.legend(loc="upper left")

	# 	plt.draw()
		
	# 	if saving:
	# 		fig.savefig(graphFolder + f"{simul}/energies_L={L}_chi={chi}_J={J}_h={h}_i={lambdaI}_3={lambda3}_c={lambdaC}" + ".png")

	gap = []
	N = []
	fig, ax = plt.subplots(figsize=(6, 5))
	
	filename = []
	for file in sorted(os.scandir(dataFolder + f"{simul}/"), key=lambda e: e.name):
		filename.append(file.name)

	for file in filename:
		
		expression = f"(?:L=)(?P<L>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_J=)(?P<J>[0-9e\.-]+)(?:_h=)(?P<h>[0-9e\.-]+)(?:_i=)(?P<lambdaI>[0-9e\.-]+)(?:_3=)(?P<lamba3>[0-9e\.-]+)(?:_c=)(?P<lambdac>[0-9e\.-]+)(?:\.dat)"
		m = re.match(expression, file)
		d = m.groupdict()

		for key in d.keys():
			d[key] = float(d[key])

		E =  np.loadtxt(dataFolder + f"{simul}/" + file)
		L, chi, J, h, lambdaI, lambda3, lambdaC = d.values()

		gap.append(abs(min(E[:, 1])-min(E[:, 0])))
		N.append(L)

	gap = np.array(gap).reshape(len(gap))/2
	N = np.array(N)

	x = np.linspace(min(1/N), max(1/N), 10)
	
	def g(inv, a, b):
		return a*inv + b

	popt, pcov = curve_fit(g, 1/N, gap)
	perr = np.sqrt(np.diag(pcov))

	mag = magnitude(popt[1])
	const = popt[1]*10**(-mag)
	consterr = perr[1]*10**(-mag)

	label=fr"$|E_1-E_0| = ({const:.1f} \pm {consterr:.1f})\cdot 10^{{{mag}}} + ({popt[0]:.3f} \pm {perr[0]:0.3f}) \frac{{1}}{{L}}$"

	ax.scatter(1/N, gap, marker='x', color=colors[0])
	ax.plot(x, g(x, *popt), linestyle=':', color=colors[1], label=label)

	plt.xlabel(r"$\frac{1}{L}$")
	plt.ylabel("$|E_1 - E_0|$")
	ax.minorticks_on()
	ax.grid(which='minor', linewidth=0.2)
	ax.grid(which='major', linewidth=0.6)
	ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0)

	plt.draw()
	
	if saving:
		fig.savefig(graphFolder + f"{simul}/gap_chi={chi}_J={J}_h={h}_i={lambdaI}_3={lambda3}_c={lambdaC}" + ".png")

	plt.show()

if simul=="phase":

	plt.rcParams.update({
	'axes.prop_cycle': (cycler(color=plt.cm.magma(np.linspace(0, 0.9, 11))))
	})

	fig, ax = plt.subplots(figsize=(6, 5))

	filename = []
	for file in sorted(os.scandir(dataFolder + f"{simul}/"), key=lambda e: e.name):
		filename.append(file.name)

	for file in filename:

		expression = f"(?:J=)(?P<J>[0-9e\.-]+)(?:_h=)(?P<h>[0-9e\.-]+)(?:_i=)(?P<lambdaI>[0-9e\.-]+)(?:_3=)(?P<lamba3>[0-9e\.-]+)(?:_c=)(?P<lambdac>[0-9e\.-]+)(?:\.dat)"
		# expression = f"(?:chi=)(?P<chi>[0-9e\.-]+)(?:_J=)(?P<J>[0-9e\.-]+)(?:_h=)(?P<h>[0-9e\.-]+)(?:_i=)(?P<lambdaI>[0-9e\.-]+)(?:_3=)(?P<lamba3>[0-9e\.-]+)(?:_c=)(?P<lambdac>[0-9e\.-]+)(?:\.dat)"
		m = re.match(expression, file)
		d = m.groupdict()

		for key in d.keys():
			d[key] = float(d[key])

		L, c, cErr = np.loadtxt(dataFolder + f"{simul}/" + file, unpack=True)
		J, h, lambdaI, lambda3, lambdaC = d.values()

		# ax.plot(1/L, c, marker='.', alpha=0.75, label=fr"$\lambda_3 = {lambda3} \cdot \lambda_I$")
		ax.plot(1/L**2, c, marker='.', alpha=0.75, label=fr"$\lambda_3 = {lambda3} \cdot \lambda_I$")

	# ax.hlines(0.5, 0, 0.01, color='k', alpha=0.4, linestyle="--")
	# ax.hlines(0.7, 0, 0.01, color='k', alpha=0.4, linestyle="--")

	# ax.set_xlim(0, max(1/L))
	ax.set_xlim(0, max(1/L**2+0.0001))
	# ax.set_ylim(0, 0.73)

	# plt.xlabel(r"$\frac{1}{L}$")
	plt.xlabel(r"$\frac{1}{L^2}$")
	plt.ylabel("$c$")
	ax.minorticks_on()
	ax.grid(which='minor', linewidth=0.2)
	ax.grid(which='major', linewidth=0.6)
	ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0)

	plt.draw()

	if saving:
		# fig.savefig(graphFolder + f"{simul}/chi={chi}_J={J}_h={h}_i={lambdaI}_c={lambdaC}" + ".png")
		fig.savefig(graphFolder + f"{simul}/J={J}_h={h}_i={lambdaI}_c={lambdaC}" + ".png")

	plt.show()

if simul=="transition":
	
	# filename = []
	# for file in sorted(os.scandir(dataFolder + f"{simul}/"), key=lambda e: e.name):
	# 	filename.append(file.name)

	# for file in filename:

	# 	expression = f"(?:L=)(?P<L>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_J=)(?P<J>[0-9e\.-]+)(?:_h=)(?P<h>[0-9e\.-]+)(?:_i=)(?P<lambdaI>[0-9e\.-]+)(?:_3=)(?P<lamba3>[0-9e\.-]+)(?:_c=)(?P<lambdac>[0-9e\.-]+)(?:\.dat)"
	# 	m = re.match(expression, file)
	# 	d = m.groupdict()

	# 	for key in d.keys():
	# 		d[key] = float(d[key])

	# 	E =  np.loadtxt(dataFolder + f"{simul}/" + file)
	# 	L, chi, J, h, lambdaI, lambda3, lambdaC = d.values()

	# 	itr = range(1, len(E[:, 0])+1)

	# 	fig, ax = plt.subplots(figsize=(6, 5))

	# 	for j in range(len(E[0, :])):
	# 		ax.plot(itr, E[:, j], label=f"$E_{j}$")

	# 	plt.xlabel("iteration")
	# 	plt.ylabel("$E$")
	# 	ax.minorticks_on()
	# 	ax.grid(which='minor', linewidth=0.2)
	# 	ax.grid(which='major', linewidth=0.6)
	# 	ax.legend(loc="upper left")

	# 	plt.draw()

	# 	if saving:
	# 		fig.savefig(graphFolder + f"{simul}/energies_L={L}_chi={chi}_J={J}_h={h}_i={lambdaI}_3={lambda3}_c={lambdaC}" + ".png")
	
	h1 = []
	h2 = []
	diff1 = []
	diff2 = []
	fig, ax = plt.subplots(figsize=(6, 5))
	
	filename = []
	for file in sorted(os.scandir(dataFolder + f"{simul}/"), key=lambda e: e.name):
		filename.append(file.name)

	for file in filename:

		expression = f"(?:L=)(?P<L>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_J=)(?P<J>[0-9e\.-]+)(?:_h=)(?P<h>[0-9e\.-]+)(?:_i=)(?P<lambdaI>[0-9e\.-]+)(?:_3=)(?P<lamba3>[0-9e\.-]+)(?:_c=)(?P<lambdac>[0-9e\.-]+)(?:\.dat)"
		m = re.match(expression, file)
		d = m.groupdict()

		for key in d.keys():
			d[key] = float(d[key])

		E =  np.loadtxt(dataFolder + f"{simul}/" + file)
		L, chi, J, h, lambdaI, lambda3, lambdaC = d.values()

		L = int(L)
		E0 = min(E[-L//2-L//4:-L//2+L//4, 0])
		E1 = min(E[-L//2-L//4:-L//2+L//4, 1])
		E2 = min(E[-L//2-L//4:-L//2+L//4, 2])

		h1.append(h)
		diff1.append(abs(E1-E0))
		if h<0.95:
			h2.append(h)
			diff2.append(abs(E2-E0))

	ax.scatter(h1, diff1, marker='o', label="$|E_1-E_0|$")
	ax.scatter(h2, diff2, marker='o', label="$|E_2-E_0|$")

	ax.set_ylim(-0.1, min(max(diff1), max(diff2))+0.1)

	plt.xlabel("$h$")
	plt.ylabel("gap")
	ax.minorticks_on()
	ax.grid(which='minor', linewidth=0.2)
	ax.grid(which='major', linewidth=0.6)
	ax.legend()

	plt.draw()

	if saving:
		fig.savefig(graphFolder + f"{simul}/gap_L={L}_chi={chi}_J={J}_i={lambdaI}_3={lambda3}_c={lambdaC}" + ".png")

	plt.show()

if simul=="through":

	c = []
	param = []
	fig, ax = plt.subplots(figsize=(6, 5))

	filename = []
	for file in sorted(os.scandir(dataFolder + f"{simul}/"), key=lambda e: e.name):
		filename.append(file.name)

	for file in filename:

		expression = f"(?:L=)(?P<L>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_J=)(?P<J>[0-9e\.-]+)(?:_h=)(?P<h>[0-9e\.-]+)(?:_i=)(?P<lambdaI>[0-9e\.-]+)(?:_3=)(?P<lamba3>[0-9e\.-]+)(?:_c=)(?P<lambdac>[0-9e\.-]+)(?:\.dat)"
		m = re.match(expression, file)
		d = m.groupdict()

		for key in d.keys():
			d[key] = float(d[key])

		l, S =  np.loadtxt(dataFolder + f"{simul}/" + file, unpack=True)
		L, chi, J, h, lambdaI, lambda3, lambdaC = d.values()

		def f(l, c, const):
			# return c/6 * np.log(2*L/np.pi * np.sin(np.pi*l/L)) + const
			return c/3 * np.log(L/np.pi * np.sin(np.pi*l/L)) + const

		popt, pcov = curve_fit(f, l, S)
		perr = np.sqrt(np.diag(pcov))

		c.append(popt[0])
		param.append(lambda3)

	ax.scatter(param, c, marker='o', color=colors[0])
	ax.axvline(0.856, linestyle="-", color=colors[2], label=fr"$\lambda_3 = 0.856 \cdot \lambda_I$")
	ax.axhline(0.5, linestyle="--", color='k', alpha=0.4)
	ax.axhline(0.7, linestyle="--", color='k', alpha=0.4)

	plt.xlabel(r"$\lambda_3$")
	plt.ylabel("$c$")
	ax.minorticks_on()
	ax.grid(which='minor', linewidth=0.2)
	ax.grid(which='major', linewidth=0.6)
	ax.legend()

	plt.draw()

	if saving:
		fig.savefig(graphFolder + f"{simul}/L={L}_chi={chi}_J={J}_h={h}_i={lambdaI}_c={lambdaC}" + ".png")
	
	# filename = []
	# for file in sorted(os.scandir(dataFolder + f"{simul}/"), key=lambda e: e.name):
	# 	filename.append(file.name)

	# for file in filename:
		
	# 	fig, ax = plt.subplots(figsize=(6, 5))

	# 	expression = f"(?:L=)(?P<L>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_J=)(?P<J>[0-9e\.-]+)(?:_h=)(?P<h>[0-9e\.-]+)(?:_i=)(?P<lambdaI>[0-9e\.-]+)(?:_3=)(?P<lamba3>[0-9e\.-]+)(?:_c=)(?P<lambdac>[0-9e\.-]+)(?:\.dat)"
	# 	m = re.match(expression, file)
	# 	d = m.groupdict()

	# 	for key in d.keys():
	# 		d[key] = float(d[key])

	# 	l, S =  np.loadtxt(dataFolder + f"{simul}/" + file, unpack=True)
	# 	L, chi, J, h, lambdaI, lambda3, lambdaC = d.values()

	# 	def f(l, c, const):
	# 		# return c/6 * np.log(2*L/np.pi * np.sin(np.pi*l/L)) + const
	# 		return c/3 * np.log(L/np.pi * np.sin(np.pi*l/L)) + const

	# 	popt, pcov = curve_fit(f, l, S)
	# 	perr = np.sqrt(np.diag(pcov))

	# 	label=fr"$S = \frac{{({popt[0]:.5f} \pm {perr[0]:0.5f})}}{{6}} \ln\left[\frac{{2L}}{{\pi}}\sin\frac{{\pi l}}{{L}} \right] + ({popt[1]:.5f} \pm {perr[1]:0.5f})$"
	# 	# label=fr"$S = \frac{{({popt[0]:.4f} \pm {perr[0]:0.4f})}}{{6}} \ln\left[\frac{{2L}}{{\pi}}\sin\frac{{\pi l}}{{L}} \right] + ({popt[1]:.4f} \pm {perr[1]:0.4f})$"

	# 	ax.scatter(l, S, marker='x', color=colors[0])
	# 	ax.plot(l, f(l, *popt), linestyle=':', color=colors[1], label=label)

	# 	plt.xlabel("$l$")
	# 	plt.ylabel("$S$")
	# 	ax.minorticks_on()
	# 	ax.grid(which='minor', linewidth=0.2)
	# 	ax.grid(which='major', linewidth=0.6)
	# 	ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0)

	# 	plt.draw()

	# 	if saving:
	# 		fig.savefig(graphFolder + f"{simul}/alone_L={L}_chi={chi}_J={J}_h={h}_i={lambdaI}_3={lambda3}_c={lambdaC}" + ".png")
		
	plt.show()

if simul=="spin":
	
	field = []
	x = []
	y = []
	z = []
	fig, ax = plt.subplots(figsize=(6, 5))

	filename = []
	for file in sorted(os.scandir(dataFolder + f"{simul}/"), key=lambda e: e.name):
		filename.append(file.name)

	for file in filename:
		
		expression = f"(?:L=)(?P<L>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_J=)(?P<J>[0-9e\.-]+)(?:_h=)(?P<h>[0-9e\.-]+)(?:_i=)(?P<lambdaI>[0-9e\.-]+)(?:_3=)(?P<lamba3>[0-9e\.-]+)(?:_c=)(?P<lambdac>[0-9e\.-]+)(?:\.dat)"
		m = re.match(expression, file)
		d = m.groupdict()

		for key in d.keys():
			d[key] = float(d[key])

		spinX, spinY, spinZ =  np.loadtxt(dataFolder + f"{simul}/" + file, unpack=True)
		L, chi, J, h, lambdaI, lambda3, lambdaC = d.values()

		L = int(L)
		field.append(h)
		x.append(spinX[L//2])
		y.append(spinY[L//2])
		z.append(spinZ[L//2])

	field = np.array(field)
	x = np.array(x)
	y = np.array(y)
	z = np.array(z)

	ax.scatter(field, x, marker='o', label=r"$\alpha = x$")
	ax.scatter(field, z, marker='o', label=r"$\alpha = z$")

	plt.xlabel("$h$")
	plt.ylabel(r"$\langle \sigma_{L/2}^\alpha \rangle$")
	ax.minorticks_on()
	ax.grid(which='minor', linewidth=0.2)
	ax.grid(which='major', linewidth=0.6)
	ax.legend()

	plt.draw()
	
	if saving:
		fig.savefig(graphFolder + f"{simul}/L={L}_chi={chi}_J={J}_i={lambdaI}_3={lambda3}_c={lambdaC}" + ".png")

	plt.show()

if simul=="conformal":
	
	# plt.rcParams.update({
	# 'axes.prop_cycle': (cycler(color=plt.cm.copper(np.linspace(0, 1, 10))))
	# })

	# filename = []
	# for file in sorted(os.scandir(dataFolder + f"{simul}/"), key=lambda e: e.name):
	# 	filename.append(file.name)

	# for file in filename:

	# 	expression = f"(?:L=)(?P<L>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_J=)(?P<J>[0-9e\.-]+)(?:_h=)(?P<h>[0-9e\.-]+)(?:_i=)(?P<lambdaI>[0-9e\.-]+)(?:_3=)(?P<lamba3>[0-9e\.-]+)(?:_c=)(?P<lambdac>[0-9e\.-]+)(?:\.dat)"
	# 	m = re.match(expression, file)
	# 	d = m.groupdict()

	# 	for key in d.keys():
	# 		d[key] = float(d[key])

	# 	E =  np.loadtxt(dataFolder + f"{simul}/" + file)
	# 	L, chi, J, h, lambdaI, lambda3, lambdaC = d.values()

	# 	itr = range(1, len(E[:, 0])+1)

	# 	fig, ax = plt.subplots(figsize=(6, 5))

	# 	for j in range(len(E[0, :])):
	# 		ax.plot(itr, E[:, j], label=f"$E_{j}$")

	# 	ax.set_ylim(min(E[:, 0])-0.1, min(E[:, -1])+0.6)

	# 	plt.xlabel("iteration")
	# 	plt.ylabel("$E$")
	# 	ax.minorticks_on()
	# 	ax.grid(which='minor', linewidth=0.2)
	# 	ax.grid(which='major', linewidth=0.6)
	# 	ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0)

	# 	plt.draw()

	# 	if saving:
	# 		fig.savefig(graphFolder + f"{simul}/energies_L={L}_chi={chi}_J={J}_h={h}_i={lambdaI}_3={lambda3}_c={lambdaC}" + ".png")

	# ---
	# TFI id+epsilon
	# qi = [2, 3, 4]
	# muli = [1, 1, 2]
	# qe = [1/2, 3/2, 5/2, 7/2, 9/2]
	# mule = [1, 1, 1, 1, ""]
	# vCFT = 1/2
	# scale = 1/48
	# confDim = 0

	# TFI id
	# q = [2, 3, 4, 5, 6]
	# mul = [1, 1, 2, 2, 3]
	# vCFT = 1/2
	# scale = 1/48
	# confDim = 0

	# TFI epsilon
	# q = [1, 2, 3, 4, 5, 6]
	# mul = [1, 1, 1, 2, 2, ""]
	# vCFT = 1/2
	# scale = 1/48
	# confDim = 1/2

	# TFI sigma
	# q = [1, 2, 3, 4, 5]
	# mul = [1, 1, 2, 2, 3]
	# vCFT = 1/2
	# scale = 1/48
	# confDim = 1/16

	# TCI -+
	# qi = [1, 3, 4, 5]
	# muli = [1, 1, 1, 2]
	# qe = [3/2, 5/2, 7/2]
	# mule = [1, 1, 2]
	# vCFT = 1
	# scale = 7/240
	# confDim = 0

	# TCI --
	# q = [2, 3, 4, 5, 6]
	# mul = [1, 1, 2, 2, 3]
	# vCFT = 7/10
	# scale = 7/240
	# confDim = 0

	# TCI ff
	q = [1, 2, 3, 4]
	mul = [1, 1, 2, 3]
	vCFT = 7/10
	scale = 7/240
	confDim = 0
	# ---

	# E0 = []
	# N = []
	# fig, ax = plt.subplots(figsize=(6, 5))
	
	# filename = []
	# for file in sorted(os.scandir(dataFolder + f"{simul}/"), key=lambda e: e.name):
	# 	filename.append(file.name)	

	# for file in filename:
		
	# 	expression = f"(?:L=)(?P<L>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_J=)(?P<J>[0-9e\.-]+)(?:_h=)(?P<h>[0-9e\.-]+)(?:_i=)(?P<lambdaI>[0-9e\.-]+)(?:_3=)(?P<lamba3>[0-9e\.-]+)(?:_c=)(?P<lambdac>[0-9e\.-]+)(?:\.dat)"
	# 	m = re.match(expression, file)
	# 	d = m.groupdict()

	# 	for key in d.keys():
	# 		d[key] = float(d[key])

	# 	E =  np.loadtxt(dataFolder + f"{simul}/" + file)
	# 	L, chi, J, h, lambdaI, lambda3, lambdaC = d.values()

	# 	E0.append(min(E[:, 0]))
	# 	N.append(L)

	# E0 = np.array(E0).reshape(len(E0))
	# N = np.array(N)

	# x = np.linspace(min(1/N**2), max(1/N**2), 10)

	# def f(n, e0, e1, v):
	# 	return e0*n + e1 - np.pi*v/n * (-scale + confDim)

	# popt, pcov = curve_fit(f, N, E0)
	# perr = np.sqrt(np.diag(pcov))

	# v = abs(popt[2])
	# vErr = perr[2]

	# y = (E0-popt[1])/N - popt[0]

	# def g(inv2, a, b):
	# 	return a*inv2 + b
	
	# popt, pcov = curve_fit(g, 1/N**2, y)
	# perr = np.sqrt(np.diag(pcov))

	# mag = magnitude(popt[1])
	# const = popt[1]*10**(-mag)
	# consterr = perr[1]*10**(-mag)

	# label=fr"$\frac{{E_0-\varepsilon_1}}{{L}} - \varepsilon_0$" + "\n" + fr"$= \frac{{\pi({v:.4f} \pm {vErr:0.4f})}}{{48}}\frac{{1}}{{L^2}}$"+ fr"$+ ({const:.0f} \pm {consterr:.0f})\cdot 10^{{{mag}}}$"

	# ax.scatter(1/N**2, y, marker='x', color=colors[0])
	# ax.plot(x, g(x, *popt), linestyle=':', color=colors[1], label=label)

	# plt.xlabel(r"$\frac{1}{L^2}$")
	# plt.ylabel(r"$\frac{E_0-\varepsilon_1}{L} - \varepsilon_0$")
	# ax.minorticks_on()
	# ax.grid(which='minor', linewidth=0.2)
	# ax.grid(which='major', linewidth=0.6)
	# # ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0)
	# # ax.legend(loc="upper left")
	# ax.legend(loc="lower left")

	# plt.draw()
	
	# if saving:
	# 	fig.savefig(graphFolder + f"{simul}/gs_chi={chi}_J={J}_h={h}_i={lambdaI}_3={lambda3}_c={lambdaC}" + ".png")
	
	# ----
	# k = len(E[0, :])
	k = 10
	# ----

	Es = [[] for i in range(k)]
	N = []
	fig, ax = plt.subplots(figsize=(6, 5))
	
	filename = []
	for file in sorted(os.scandir(dataFolder + f"{simul}/"), key=lambda e: e.name):
		filename.append(file.name)
	
	for file in filename:
		
		expression = f"(?:L=)(?P<L>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_J=)(?P<J>[0-9e\.-]+)(?:_h=)(?P<h>[0-9e\.-]+)(?:_i=)(?P<lambdaI>[0-9e\.-]+)(?:_3=)(?P<lamba3>[0-9e\.-]+)(?:_c=)(?P<lambdac>[0-9e\.-]+)(?:\.dat)"
		m = re.match(expression, file)
		d = m.groupdict()

		for key in d.keys():
			d[key] = float(d[key])

		E =  np.loadtxt(dataFolder + f"{simul}/" + file)
		L, chi, J, h, lambdaI, lambda3, lambdaC = d.values()

		for i in range(k):
			Es[i].append(min(E[:, i])/4)
		N.append(L)

	Es = np.array(Es).reshape(k, len(filename))
	N = np.array(N)

	ax.set_xlim(0, max(1/N)+0.001)
	# ax.set_ylim(0, max(qi)+1)

	n = np.zeros((k-1, len(filename)))
	for i in range(k-1):
		n[i, :] = N*(Es[i+1, :]-Es[0, :])/(np.pi*vCFT)

	# for i in range(len(qi)):
	# 	ax.axhline(qi[i], color=colors[0], lw=1)
	# 	plt.annotate(f"{muli[i]}", xy=(0.01, qi[i]+0.05), color=colors[0], fontsize=10, xycoords=('axes fraction', 'data'))
	# for i in range(len(qe)):
	# 	ax.axhline(qe[i], color=colors[1], lw=1)
	# 	plt.annotate(f"{mule[i]}", xy=(0.01, qe[i]+0.05), color=colors[1], fontsize=10, xycoords=('axes fraction', 'data'))

	for i in range(len(q)):
		ax.axhline(q[i], color=colors[2], lw=1)
		plt.annotate(f"{mul[i]}", xy=(0.01, q[i]+0.05), color=colors[2], fontsize=10, xycoords=('axes fraction', 'data'))

	for i in range(k-1):
		ax.scatter(1/N, n[i, :], marker='o', color='k', s=10)

	plt.xlabel(r"$\frac{1}{L}$")
	plt.ylabel(r"$\frac{E_n - E_0}{\pi v} L$")
	ax.minorticks_on()
	ax.grid(which='minor', linewidth=0.2)
	ax.grid(which='major', linewidth=0.6)

	plt.draw()
	
	if saving:
		fig.savefig(graphFolder + f"{simul}/towers_chi={chi}_J={J}_h={h}_i={lambdaI}_3={lambda3}_c={lambdaC}" + ".png")

	plt.show()

if simul=="pinning":

	plt.rcParams.update({
	'axes.prop_cycle': (cycler(color=plt.cm.tab10(np.linspace(0, 1, 10))))
	})

	central = []
	N = []
	fig, ax = plt.subplots(figsize=(6, 5))

	filename = []
	for file in sorted(os.scandir(dataFolder + f"{simul}/"), key=lambda e: e.name):
		filename.append(file.name)
	
	for file in filename:

		expression = f"(?:L=)(?P<L>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_J=)(?P<J>[0-9e\.-]+)(?:_h=)(?P<h>[0-9e\.-]+)(?:_i=)(?P<lambdaI>[0-9e\.-]+)(?:_3=)(?P<lamba3>[0-9e\.-]+)(?:_c=)(?P<lambdac>[0-9e\.-]+)(?:\.dat)"
		m = re.match(expression, file)
		d = m.groupdict()

		for key in d.keys():
			d[key] = float(d[key])

		pin, c, cErr = np.loadtxt(dataFolder + f"{simul}/" + file, unpack=True)
		L, chi, J, h, lambdaI, lambda3, lambdaC = d.values()

		central.append(c)
		N.append(L)

	central = np.array(central).T
	N = np.array(N)

	idx = np.argsort(N)
	N = N[idx]
	for i in range(len(pin)):
		ax.plot(1/N, central[i, :][idx], marker='.', alpha=0.75, label=fr"$h_\mathrm{{pin}} = -{pin[i]}$")

	ax.set_xlim(0, max(1/N)+0.001)
	ax.set_ylim(np.min(central)-0.01, 0.7)

	plt.xlabel(r"$\frac{1}{L}$")
	plt.ylabel("$c$")
	ax.minorticks_on()
	ax.grid(which='minor', linewidth=0.2)
	ax.grid(which='major', linewidth=0.6)
	ax.legend(loc="upper right")

	plt.draw()

	if saving:
		fig.savefig(graphFolder + f"{simul}/chi={chi}_J={J}_h={h}_i={lambdaI}_3={lambda3}_c={lambdaC}" + ".png")

	plt.show()

if simul=="edge":

	filename = []
	for file in sorted(os.scandir(dataFolder + f"{simul}/"), key=lambda e: e.name):
		filename.append(file.name)

	for file in filename:
	
		fig, ax = plt.subplots(figsize=(6, 5))
		
		expression = f"(?:L=)(?P<L>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_J=)(?P<J>[0-9e\.-]+)(?:_h=)(?P<h>[0-9e\.-]+)(?:_i=)(?P<lambdaI>[0-9e\.-]+)(?:_3=)(?P<lamba3>[0-9e\.-]+)(?:_c=)(?P<lambdac>[0-9e\.-]+)(?:\.dat)"
		m = re.match(expression, file)
		d = m.groupdict()

		for key in d.keys():
			d[key] = float(d[key])

		pin, leftfree, rightfree, leftup, rightup, leftzero, rightzero, leftmix, rightmix =  np.loadtxt(dataFolder + f"{simul}/" + file, unpack=True)
		L, chi, J, h, lambdaI, lambda3, lambdaC = d.values()

		ax.scatter(pin, leftfree, marker=4, s=50, lw=2, label=r"$\ell =$ left with $[ff]$")
		ax.scatter(pin, rightfree, marker=5, s=50, lw=2, label=r"$\ell =$ right with $[ff]$")
		ax.scatter(pin, leftup, marker=4, s=50, lw=2, label=r"$\ell =$ left with $[++]$")
		ax.scatter(pin, rightup, marker=5, s=50, lw=2, label=r"$\ell =$ right with $[++]$")
		ax.scatter(pin, leftzero, marker=4, s=50, lw=2, label=r"$\ell =$ left with $[-+]$")
		ax.scatter(pin, rightzero, marker=6, s=50, lw=2, label=r"$\ell =$ right with $[-+]$")
		ax.scatter(pin, leftmix, marker=6, s=50, lw=2, label=r"$\ell =$ left with $[f-]$")
		ax.scatter(pin, rightmix, marker=5, s=50, lw=2, label=r"$\ell =$ right with $[f-]$")

		plt.xlabel("$h_{\mathrm{pin}}$")
		plt.ylabel(r"$\langle \sigma_{\ell}^x \rangle$")
		ax.minorticks_on()
		ax.grid(which='minor', linewidth=0.2)
		ax.grid(which='major', linewidth=0.6)
		ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0)

		plt.draw()
		
		if saving:
			fig.savefig(graphFolder + f"{simul}/L={L}_chi={chi}_J={J}_h={h}_i={lambdaI}_3={lambda3}_c={lambdaC}" + ".png")

	plt.show()

if simul=="ratios":

	filename = []
	for file in sorted(os.scandir(dataFolder + f"{simul}/"), key=lambda e: e.name):
		filename.append(file.name)

	r1 = []
	r2 = []
	r3 = []
	
	l3 = []

	for file in filename:

		expression = f"(?:J=)(?P<J>[0-9e\.-]+)(?:_h=)(?P<h>[0-9e\.-]+)(?:_i=)(?P<lambdaI>[0-9e\.-]+)(?:_3=)(?P<lamba3>[0-9e\.-]+)(?:_c=)(?P<lambdac>[0-9e\.-]+)(?:\.dat)"
		m = re.match(expression, file)
		d = m.groupdict()

		for key in d.keys():
			d[key] = float(d[key])

		L, p0p, p0m, p1p, p1m, a0m =  np.loadtxt(dataFolder + f"{simul}/" + file, unpack=True)
		J, h, lambdaI, lambda3, lambdaC = d.values()

		r1.append((a0m-p0p)/(p1p-p0p))
		r2.append((p0m-p0p)/(p1p-p0p))
		r3.append((p1m-p0p)/(p1p-p0p))

		l3.append(lambda3)

	fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5), tight_layout=True)

	for i in range(len(filename)):
		ax1.scatter(1/L, r1[i], s=15, marker='o', label=fr"$\lambda_3 = {l3[i]} \cdot \lambda_I$")
		ax2.scatter(1/L, r2[i], s=15, marker='o')
		ax3.scatter(1/L, r3[i], s=15, marker='o')

	ax1.axhline(7/2, color=colors[0], lw=1)
	ax1.axhline(1/2, color=colors[1], lw=1)
	ax2.axhline(3/8, color=colors[0], lw=1)
	ax2.axhline(1/8, color=colors[1], lw=1)
	ax3.axhline(35/8, color=colors[0], lw=1)
	ax3.axhline(9/8, color=colors[1], lw=1)
	
	ax1.annotate(r"$\frac{7}{2}$", xy=(0.03, 7/2+0.16), color=colors[0], fontsize=12, xycoords=('axes fraction', 'data'))
	ax1.annotate(r"$\frac{1}{2}$", xy=(0.03, 1/2+0.16), color=colors[1], fontsize=12, xycoords=('axes fraction', 'data'))
	ax2.annotate(r"$\frac{3}{8}$", xy=(0.03, 3/8+0.01), color=colors[0], fontsize=12, xycoords=('axes fraction', 'data'))
	ax2.annotate(r"$\frac{1}{8}$", xy=(0.03, 1/8+0.01), color=colors[1], fontsize=12, xycoords=('axes fraction', 'data'))
	ax3.annotate(r"$\frac{35}{8}$", xy=(0.03, 35/8+0.18), color=colors[0], fontsize=12, xycoords=('axes fraction', 'data'))
	ax3.annotate(r"$\frac{9}{8}$", xy=(0.03, 9/8+0.18), color=colors[1], fontsize=12, xycoords=('axes fraction', 'data'))

	ax1.set_xlabel(r"$\frac{1}{L}$")
	ax2.set_xlabel(r"$\frac{1}{L}$")
	ax3.set_xlabel(r"$\frac{1}{L}$")
	ax1.set_ylabel(r"$R_1$")
	ax2.set_ylabel(r"$R_2$")
	ax3.set_ylabel(r"$R_3$")

	ax1.set_xlim(0, max(1/L)+0.01)
	ax2.set_xlim(0, max(1/L)+0.01)
	ax3.set_xlim(0, max(1/L)+0.01)

	ax1.minorticks_on()
	ax2.minorticks_on()
	ax3.minorticks_on()
	ax1.grid(which='minor', linewidth=0.2)
	ax2.grid(which='minor', linewidth=0.2)
	ax3.grid(which='minor', linewidth=0.2)
	ax1.grid(which='major', linewidth=0.6)
	ax2.grid(which='major', linewidth=0.6)
	ax3.grid(which='major', linewidth=0.6)

	fig.legend(bbox_to_anchor=(0.5, 0.97), ncol=3, loc='lower center')

	plt.draw()
		
	if saving:
		fig.savefig(graphFolder + f"{simul}/J={J}_h={h}_i={lambdaI}_c={lambdaC}" + ".png")

	# plt.show()

if simul=="degeneracy":

	filename = []
	for file in sorted(os.scandir(dataFolder + f"{simul}/"), key=lambda e: e.name):
		filename.append(file.name)

	r1 = []
	r2 = []
	r3 = []
	# r4 = []
	
	l3 = []

	for file in filename:

		expression = f"(?:J=)(?P<J>[0-9e\.-]+)(?:_h=)(?P<h>[0-9e\.-]+)(?:_i=)(?P<lambdaI>[0-9e\.-]+)(?:_3=)(?P<lamba3>[0-9e\.-]+)(?:_c=)(?P<lambdac>[0-9e\.-]+)(?:\.dat)"
		m = re.match(expression, file)
		d = m.groupdict()

		for key in d.keys():
			d[key] = float(d[key])

		L, p0p, p0m, p1p, p1m, a0m =  np.loadtxt(dataFolder + f"{simul}/" + file, unpack=True)
		J, h, lambdaI, lambda3, lambdaC = d.values()

		r1.append(abs(p1p-p0p))
		r2.append(abs(p0m-p0p))
		r3.append(abs(p1m-p0p))
		# r4.append(abs(p1m-p0p))

		l3.append(lambda3)

	r1 = np.array(r1)
	r2 = np.array(r2)
	r3 = np.array(r3)
	# r4 = np.array(r4)

	l3 = np.array(l3)

	x = np.linspace(0, max(1/L))

	fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5), tight_layout=True)
	# fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 5), tight_layout=True)
	
	def g(inv, a, b):
		return a*inv + b

	for i in range(len(filename)):

		popt1, pcov1 = curve_fit(g, 1/L, r1[i])
		perr1 = np.sqrt(np.diag(pcov1))
		popt2, pcov2 = curve_fit(g, 1/L, r2[i])
		perr2 = np.sqrt(np.diag(pcov2))
		popt3, pcov3 = curve_fit(g, 1/L, r3[i])
		perr3 = np.sqrt(np.diag(pcov3))

		print(popt1)
		print(popt2)
		print(popt3)

		ax1.plot(1/L, r1[i], marker='o', label=fr"$\lambda_3 = {l3[i]} \cdot \lambda_I$")
		ax2.plot(1/L, r2[i], marker='o')
		ax3.plot(1/L, r3[i], marker='o')
		# ax4.plot(1/L, r4[i], marker='o')

		# ax1.plot(x, g(x, *popt1), linestyle=':')
		# ax2.plot(x, g(x, *popt2), linestyle=':')
		# ax3.plot(x, g(x, *popt3), linestyle=':')

	ax1.set_xlabel(r"$\frac{1}{L}$")
	ax2.set_xlabel(r"$\frac{1}{L}$")
	ax3.set_xlabel(r"$\frac{1}{L}$")
	# ax4.set_xlabel(r"$\frac{1}{L}$")
	ax1.set_ylabel(r"$|P_1^+ - P_0^+|$")
	ax2.set_ylabel(r"$|P_0^- - P_0^+|$")
	ax3.set_ylabel(r"$|P_1^- - P_0^+|$")
	# ax4.set_ylabel(r"$|P_1^- - P_0^+|$")

	ax1.set_xlim(0, max(1/L)+0.01)
	ax2.set_xlim(0, max(1/L)+0.01)
	ax3.set_xlim(0, max(1/L)+0.01)
	# ax4.set_xlim(0, max(1/L)+0.01)

	ax1.set_ylim(-0.01)
	ax2.set_ylim(-0.005)
	ax3.set_ylim(-0.05)
	# ax4.set_ylim(-0.05)

	# ax1.set_xscale('log')
	# ax2.set_xscale('log')
	# ax3.set_xscale('log')
	# ax1.set_yscale('log')
	# ax2.set_yscale('log')
	# ax3.set_yscale('log')

	ax1.minorticks_on()
	ax2.minorticks_on()
	ax3.minorticks_on()
	# ax4.minorticks_on()
	ax1.grid(which='minor', linewidth=0.2)
	ax2.grid(which='minor', linewidth=0.2)
	ax3.grid(which='minor', linewidth=0.2)
	# ax4.grid(which='minor', linewidth=0.2)
	ax1.grid(which='major', linewidth=0.6)
	ax2.grid(which='major', linewidth=0.6)
	ax3.grid(which='major', linewidth=0.6)
	# ax4.grid(which='major', linewidth=0.6)

	fig.legend(bbox_to_anchor=(0.5, 0.97), ncol=3, loc='lower center')

	plt.draw()
		
	if saving:
		fig.savefig(graphFolder + f"{simul}/J={J}_h={h}_i={lambdaI}_c={lambdaC}" + ".png")

	plt.show()
