from symbolic import *


def process(sum, term_gen=wignerTerm):
	return apply(sum, flattenSum, (replaceRhoWithQuasiprobability, term_gen),
		flattenSum,
		derivativesToFront,
		flattenSum,
		sortFactors,
		(dropHighOrderDerivatives, 2),
		groupTerms
	)

def hermitianConjugate(term):
	assert term.isSimple()

	new_factors = []
	for factor in reversed(term.factors):
		if factor.isConstant():
			new_factors.append(factor)
		else:
			new_factors.append(factor.conjugate())

	return Term(term.coeff, new_factors)

def lossesOperator(term):
	return Sum([
		Term(2, [term, rho, hermitianConjugate(term)]),
		Term(-1, [hermitianConjugate(term), term, rho]),
		Term(-1, [rho, hermitianConjugate(term), term])
		])

def showSorted(sum):
	assert sum.isFlat()
	terms = {}

	for term in sum.terms:
		diffs = []
		others = []

		for factor in term.factors:
			if factor.isDifferential():
				diffs.append(factor)
			else:
				others.append(factor)

		key = str(Term(1, diffs))

		if key in terms:
			terms[key].terms.append(Term(term.coeff, others))
		else:
			terms[key] = Sum([Term(term.coeff, others)])

	for key in sorted(terms.keys()):
		print key + ": " + str(terms[key])

losses111 = Term(0.5, [k111, lossesOperator(Term(1, [Psi1, Psi1, Psi1]))])
losses12 = Term(0.5, [k12, lossesOperator(Term(1, [Psi1, Psi2]))])
losses22 = Term(0.5, [k22, lossesOperator(Term(1, [Psi2, Psi2]))])

# interaction part of the hamiltonian
h = Sum([Term(1, [Vhf, Psi2_plus, Psi2]),
	Term(0.5, [Omega, Psi1_plus, Psi2]),
	Term(0.5, [Omega, Psi2_plus, Psi1]),
	Term(0.5, [Omega_star, Psi1, Psi2_plus]),
	Term(0.5, [Omega_star, Psi2, Psi1_plus]),
	Term(0.5, [U11, Psi1_plus, Psi1_plus, Psi1, Psi1]),
	Term(0.5, [U22, Psi2_plus, Psi2_plus, Psi2, Psi2]),
	Term(0.5, [U12, Psi1_plus, Psi2_plus, Psi1, Psi2]),
	Term(0.5, [U12, Psi2_plus, Psi1_plus, Psi2, Psi1])])

h_comm = Sum([Term(1, [h, rho]), Term(-1, [rho, h])])
losses = Sum([losses111, losses12, losses22])

s = process(Sum([h_comm, losses]), term_gen=wignerTerm)
showSorted(s)
