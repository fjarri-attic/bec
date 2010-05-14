from copy import deepcopy
from itertools import product

# Factors
A = "a"
A_PLUS = "a+"
RHO = "rho"
K = "K" # sum of kinetic and potential energy
ALPHA = "alpha"
ALPHA_STAR = "alpha*"
D_ALPHA = "d/d_alpha" # partial derivative d/da
D_ALPHA_STAR = "d/d_alpha*"

GAMMA = "G"

CONSTANTS = [GAMMA]

DERIVATIVES = {
	D_ALPHA: ALPHA,
	D_ALPHA_STAR: ALPHA_STAR
}


class Term:

	def __init__(self, coeff, factors):
		self.coeff = coeff
		self.factors = deepcopy(factors)

	def isSimple(self):
		for factor in self.factors:
			if isinstance(factor, Term) or isinstance(factor, Sum):
				return False

		return True

	def isProduct(self):
		for factor in self.factors:
			if isinstance(factor, Sum):
				return False
			elif isinstance(factor, Term) and not factor.isProduct():
				return False

		return True

	def firstPostfixDerivative(self):
		assert self.isSimple()

		for i in xrange(len(self.factors) - 1):
			if self.factors[i] not in DERIVATIVES and self.factors[i + 1] in DERIVATIVES:
				return i + 1

		return -1

	def hasDerivativesInFront(self):
		return self.firstPostfixDerivative() == -1

	def __str__(self):
		if len(self.factors) == 0:
			return str(self.coeff)
		else:
			if self.coeff == 1:
				coeff_str = ""
			elif self.coeff == -1:
				coeff_str = "-"
			else:
				coeff_str = str(self.coeff) + " "

			return coeff_str + " ".join([str(factor) for factor in self.factors])


class Sum:

	def __init__(self, terms):
		self.terms = deepcopy(terms)

	def isFlat(self):
		for term in self.terms:
			if not isinstance(term, Term) or not term.isSimple():
				return False

		return True

	def hasDerivativesInFront(self):
		if not self.isFlat():
			return False

		for term in self.terms:
			if not term.hasDerivativesInFront():
				return False

		return True

	def __str__(self):
		return "(" + " + ".join([str(term) for term in self.terms]) + ")"


def simplifyTerm(term):
	"""Product of atoms and terms -> Product of atoms"""
	assert term.isProduct()

	coeff = term.coeff
	factors = []

	for factor in term.factors:
		if isinstance(factor, Term):
			factor = simplifyTerm(factor)
			coeff *= factor.coeff
			factors += factor.factors
		else:
			factors.append(factor)

	return Term(coeff, factors)

def termToFlatSum(term):
	"""Arbitrary term -> Sum of simple terms (flat sum)"""
	assert isinstance(term, Term)

	if term.isSimple():
		return Sum([term])

	# get (term + ...) * (term + ...) * ...
	sums = []
	for factor in term.factors:
		if isinstance(factor, Sum):
			sums.append(flattenSum(factor))
		elif isinstance(factor, Term):
			sums.append(termToFlatSum(factor))
		else:
			sums.append(Sum([Term(1, [factor])]))

	# multiply
	terms = []
	sums = tuple([sum.terms for sum in sums])
	for tup in product(*sums):
		factors = []
		coeff = term.coeff
		for t in tup:
			t = simplifyTerm(t)
			factors += t.factors
			coeff *= t.coeff
		terms.append(Term(coeff, factors))

	return Sum(terms)

def flattenSum(sum):
	"""Arbitrary sum -> Sum of simple terms (flat sum)"""
	assert isinstance(sum, Sum)

	if sum.isFlat():
		return deepcopy(sum)

	sums = []
	for term in sum.terms:
		if isinstance(term, Term):
			sums.append(termToFlatSum(term))
		elif isinstance(term, Sum):
			sums.append(flattenSum(term))
		else:
			sums.append(Sum([Term(1, [term])]))

	terms = []
	for sum in sums:
		terms += sum.terms

	return Sum(terms)

def derivativesToFront(obj):
	"""
	Simple term -> Arbitrary term
	Flat sum -> Arbitrary sum
	"""
	if isinstance(obj, Sum):
		assert obj.isFlat()
		return Sum([derivativesToFront(term) for term in obj.terms])

	term = obj
	assert term.isSimple()

	pos = term.firstPostfixDerivative()

	if pos == -1:
		return Term(term.coeff, term.factors)

	der = term.factors[pos]
	var = term.factors[pos - 1]

	if var == DERIVATIVES[der]:
		new_factors = term.factors[:pos-1] + \
			[Sum([Term(1, [der, var]), Term(-1, [])])] + \
			term.factors[pos+1:]
	else:
		new_factors = term.factors[:pos-1] + [der, var] + term.factors[pos+1:]

	return Term(term.coeff, new_factors)

def replaceRhoWithW(obj):
	"""
	Simple term -> Arbitrary term
	Flat sum -> Arbitrary sum
	"""
	if isinstance(obj, Sum):
		assert obj.isFlat()
		return Sum([replaceRhoWithW(term) for term in obj.terms])

	term = obj
	assert term.isSimple()

	new_factors = []
	rho_pos = term.factors.index(RHO)

	WIGNER = {
		(A, RHO): Sum([ALPHA, Term(0.5, [D_ALPHA_STAR])]),
		(A_PLUS, RHO): Sum([ALPHA_STAR, Term(-0.5, [D_ALPHA])]),
		(RHO, A): Sum([ALPHA, Term(-0.5, [D_ALPHA_STAR])]),
		(RHO, A_PLUS): Sum([ALPHA_STAR, Term(0.5, [D_ALPHA])])
	}

	for i in xrange(rho_pos):
		if term.factors[i] == A:
			new_factors.append(WIGNER[(A, RHO)])
		elif term.factors[i] == A_PLUS:
			new_factors.append(WIGNER[(A_PLUS, RHO)])
		else:
			new_factors.append(term.factors[i])

	for i in xrange(len(term.factors) - 1, rho_pos, -1):
		if term.factors[i] == A:
			new_factors.append(WIGNER[(RHO, A)])
		elif term.factors[i] == A_PLUS:
			new_factors.append(WIGNER[(RHO, A_PLUS)])
		else:
			new_factors.append(term.factors[i])

	return Term(term.coeff, new_factors)

def sortFactors(obj):
	"""
	Simple term with derivatives in front -> Simple term with derivatives in front
	Flat sum -> Flat sum
	"""
	if isinstance(obj, Sum):
		assert obj.isFlat()
		return Sum([sortFactors(term) for term in obj.terms])

	term = obj
	assert term.hasDerivativesInFront()

	def key1(elem):
		if elem in DERIVATIVES:
			return sorted(DERIVATIVES).index(elem) / 10.0

		if elem in CONSTANTS:
			return 1 + CONSTANTS.index(elem) / 10.0

		if elem in [ALPHA, ALPHA_STAR, K]:
			return 2

	def key2(elem):
		if elem in DERIVATIVES or elem in CONSTANTS:
			return 0

		if elem == ALPHA:
			return 1

		if elem == ALPHA_STAR:
			return 2

	factors = sorted(term.factors, key=key1)

	if K in factors:
		k_pos = factors.index(K)
		factors = sorted(factors[:k_pos], key=key2) + [K] + sorted(factors[k_pos+1:], key=key2)
	else:
		factors = sorted(factors, key=key2)

	return Term(term.coeff, factors)

def groupTerms(sum):
	"""Flat sum -> Flat sum"""
	assert sum.isFlat()

	memo = {}

	for term in sum.terms:
		key = tuple(term.factors)
		if key in memo:
			memo[key] += term.coeff
		else:
			memo[key] = term.coeff

	return Sum([Term(memo[key], list(key)) for key in memo if abs(memo[key]) > 0])

def dropHighOrderDerivatives(sum, cutoff):
	"""Flat sum -> Flat sum"""
	assert self.hasDerivativesInFront()

	new_terms = []
	for term in sum.terms:
		order = 0
		for factor in term.factors:
			if factor in DERIVATIVES:
				order += 1
			else:
				break

		if order <= cutoff:
			new_terms.append(term)

	return Sum(new_terms)

def process(sum):
	return groupTerms(sortFactors(flattenSum(derivativesToFront(flattenSum(replaceRhoWithW(flattenSum(sum)))))))

x1 = Term(2, [A, RHO, A_PLUS])
x2 = Term(-1, [A_PLUS, A, RHO])
x3 = Term(-1, [RHO, A_PLUS, A])
s = Sum([x1, x2, x3])
print process(s)
