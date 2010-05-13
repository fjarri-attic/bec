from copy import deepcopy
from itertools import product


class Term:
	def __init__(self, coeff=1, factors=[]):
		self.coeff = coeff
		self.factors = deepcopy(factors)

	def isSimple(self):
		for factor in self.factors:
			if isinstance(factor, Term) or isinstance(factor, Sum):
				return False

		return True

	def simplify(self):
		if self.isSimple():
			return deepcopy(self)

		coeff = 1
		factors = []

		for factor in self.factors:
			assert not isinstance(factor, Sum) # selftest

			if isinstance(factor, Term):
				factor = factor.simplify()
				coeff *= factor.coeff
				factors += factor.factors
			else:
				factors.append(factor)

		return Term(coeff=coeff, factors=factors)

	def flatten(self):
		if self.isSimple():
			return Sum([self])

		# get (term + ...) * (term + ...) * ...
		sums = []
		for factor in self.factors:
			if isinstance(factor, Term) or isinstance(factor, Sum):
				sums.append(factor.flatten())
			else:
				sums.append(Sum([factor]))

		# multiply
		terms = []
		sums = tuple([sum.terms for sum in sums])
		for tup in product(*sums):
			factors = []
			coeff = self.coeff
			for term in tup:
				term = term.simplify()
				factors += term.factors
				coeff *= term.coeff
			terms.append(Term(coeff=coeff, factors=factors))

		return Sum(terms)

	def __str__(self):
		factors = []
		for factor in self.factors:
			if isinstance(factor, Term):
				factors.append(str(factor))
			else:
				factors.append("(" + str(factor) + ")")

		return str(self.coeff) + (" * " if len(self.factors) > 0 else "") + " * ".join([str(factor) for factor in self.factors])

	def replace(self, rules):
		factors = []
		for factor in self.factors:
			if isinstance(factor, Term) or isinstance(factor, Sum):
				factors.append(factor.replace(rules))
			else:
				for old, new in rules:
					if old == factor:
						factor = new
						break

				factors.append(factor)

		return Term(coeff=self.coeff, factors=factors)

	def _firstPostfixDerivative(self):
		for i in xrange(len(self.factors) - 1):
			if self.factors[i] not in DERIVATIVES and self.factors[i + 1] in DERIVATIVES:
				return i + 1

		return -1

	def hasDerivativesInFront(self):
		return self._firstPostfixDerivative() == -1

	def derivativesToFront(self):
		pos = self._firstPostfixDerivative()

		if pos == -1:
			return self

		der = self.factors[pos]
		var = self.factors[pos - 1]

		if var == DERIVATIVES[der]:
			new_factors = self.factors[:pos-1] + \
				[Sum([Term(coeff=1, factors=[der, var]), Term(coeff=-1)])] + \
				self.factors[pos+1:]
		else:
			new_factors = self.factors[:pos-1] + [der, var] + self.factors[pos+1:]

		return Term(coeff=self.coeff, factors=new_factors)

	def replaceRhoWithW(self):
		assert self.isSimple()

		factors = []
		rho_pos = self.factors.index(RHO)

		WIGNER = {
			(A, RHO): Sum([ALPHA, Term(coeff=0.5, factors=[D_ALPHA_STAR])]),
			(A_PLUS, RHO): Sum([ALPHA_STAR, Term(coeff=-0.5, factors=[D_ALPHA])]),
			(RHO, A): Sum([ALPHA, Term(coeff=-0.5, factors=[D_ALPHA_STAR])]),
			(RHO, A_PLUS): Sum([ALPHA_STAR, Term(coeff=0.5, factors=[D_ALPHA])])
		}

		for i in xrange(rho_pos):
			if self.factors[i] == A:
				factors.append(WIGNER[(A, RHO)])
			elif self.factors[i] == A_PLUS:
				factors.append(WIGNER[(A_PLUS, RHO)])
			else:
				factors.append(self.factors[i])

		for i in xrange(len(self.factors) - 1, rho_pos, -1):
			if self.factors[i] == A:
				factors.append(WIGNER[(RHO, A)])
			elif self.factors[i] == A_PLUS:
				factors.append(WIGNER[(RHO, A_PLUS)])
			else:
				factors.append(self.factors[i])

		return Term(coeff=self.coeff, factors=factors)

	def sortFactors(self):
		assert self.isSimple()
		assert self.hasDerivativesInFront()

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

		factors = sorted(self.factors, key=key1)

		if K in factors:
			k_pos = factors.index(K)
			factors = sorted(factors[:k_pos], key=key2) + [K] + sorted(factors[k_pos+1:], key=key2)
		else:
			factors = sorted(factors, key=key2)

		return Term(coeff=self.coeff, factors=factors)


class Sum:
	def __init__(self, terms):
		self.terms = deepcopy(terms) # can be list of Sum or Term objects

	def flatten(self):
		sums = []
		for term in self.terms:
			if isinstance(term, Term) or isinstance(term, Sum):
				sums.append(term.flatten())
			else:
				sums.append(Sum([Term(factors=[term])]))

		terms = []
		for sum in sums:
			terms += sum.terms

		# selftest
		for term in terms:
			assert term.isSimple()

		return Sum(terms)

	def isFlat(self):
		for term in self.terms:
			if not isinstance(term, Term) or not term.isSimple():
				return False

		return True

	def __str__(self):
		return "(" + " + ".join([str(term) for term in self.terms]) + ")"

	def replace(self, rules):
		terms = []
		for term in self.terms:
			if isinstance(term, Term) or isinstance(term, Sum):
				terms.append(term.replace(rules))
			else:
				for old, new in rules:
					if old == term:
						term = new
						break

				terms.append(term)

		return Sum(terms)

	def hasDerivativesInFront(self):
		for term in self.terms:
			if not term.hasDerivativesInFront():
				return False

		return True

	def derivativesToFront(self):
		assert self.isFlat()

		sum = deepcopy(self)
		while not sum.hasDerivativesInFront():
			terms = [term.derivativesToFront() for term in sum.terms]
			sum = Sum(terms).flatten()

		return sum

	def replaceRhoWithW(self):
		assert self.isFlat()
		return Sum([term.replaceRhoWithW() for term in self.terms]).flatten()

	def group(self):
		assert self.isFlat()

		memo = {}

		for term in self.terms:
			key = tuple(term.factors)
			if key in memo:
				memo[key] += term.coeff
			else:
				memo[key] = term.coeff

		return Sum([Term(coeff=memo[key], factors=list(key)) for key in memo if abs(memo[key]) > 0])

	def sortFactors(self):
		assert self.isFlat()

		return Sum([term.sortFactors() for term in self.terms])


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

x1 = Term(coeff=2, factors=[A, RHO, A_PLUS])
x2 = Term(coeff=-1, factors=[A_PLUS, A, RHO])
x3 = Term(coeff=-1, factors=[RHO, A_PLUS, A])
s = Sum([x1, x2, x3])

s = s.replaceRhoWithW().flatten().derivativesToFront().sortFactors().group()

print str(s)
