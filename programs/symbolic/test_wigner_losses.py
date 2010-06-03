from symbolic import *


def process(sum, term_gen=wignerTerm):
	return apply(sum, flattenSum, (replaceRhoWithQuasiprobability, term_gen),
		flattenSum, derivativesToFront, flattenSum,
		sortFactors, (dropHighOrderDerivatives, 2), groupTerms)

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
		Term(2, [term, RHO, hermitianConjugate(term)]),
		Term(-1, [hermitianConjugate(term), term, RHO]),
		Term(-1, [RHO, hermitianConjugate(term), term])
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


losses111 = Term(0.5, [KAPPA111, lossesOperator(Term(1, [PSI1_OP, PSI1_OP, PSI1_OP]))])
losses12 = Term(0.5, [KAPPA12, lossesOperator(Term(1, [PSI1_OP, PSI2_OP]))])
losses21 = Term(0.5, [KAPPA12, lossesOperator(Term(1, [PSI2_OP, PSI1_OP]))])
losses22 = Term(0.5, [KAPPA22, lossesOperator(Term(1, [PSI2_OP, PSI2_OP]))])

# interaction part of the hamiltonian
h = Sum([Term(0.5, [GAMMA11, PSI1_OP_PLUS, PSI1_OP_PLUS, PSI1_OP, PSI1_OP]),
	Term(0.5, [GAMMA22, PSI2_OP_PLUS, PSI2_OP_PLUS, PSI2_OP, PSI2_OP]),
	Term(0.5, [GAMMA12, PSI1_OP_PLUS, PSI2_OP_PLUS, PSI1_OP, PSI2_OP]),
	Term(0.5, [GAMMA12, PSI2_OP_PLUS, PSI1_OP_PLUS, PSI2_OP, PSI1_OP])])

h_comm = Sum([Term(-1j, [h, RHO]), Term(1j, [RHO, h])])
full = Sum([h_comm, losses111, losses12, losses21, losses22])

s = process(h_comm)
showSorted(s)
