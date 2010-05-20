from copy import deepcopy
from itertools import product

class Atom:

	def __init__(self, name, constant=False, operator=False, conjugate=False):
		self._name = name
		self._constant = constant
		self._operator = operator

		self._differential_of = None
		self._differential = None
		self._corresponding_atom = None
		self._conjugate_atom = None
		self._conjugate = conjugate

	@staticmethod
	def declareDifferential(atom, differential):
		atom._differential = differential
		differential._differential_of = atom

	@staticmethod
	def declareCorrespondence(operator, function):
		operator._corresponding_atom = function
		function._corresponding_atom = operator

	@staticmethod
	def declareConjugate(atom, other):
		assert not atom.isConjugate()
		assert other.isConjugate()
		atom._conjugate_atom = other
		other._conjugate_atom = atom

	def correspondence(self):
		assert self._corresponding_atom is not None
		return self._corresponding_atom

	def conjugate(self):
		assert self._conjugate_atom is not None
		return self._conjugate_atom

	def differential(self):
		assert self._differential is not None
		return self._differential

	def differentialOf(self):
		assert self.isDifferential()
		return self._differential_of

	def isConjugate(self):
		return self._conjugate

	def isDifferential(self):
		return self._differential_of is not None

	def isOperator(self):
		return self._operator

	def isConstant(self):
		return self._constant

	def __str__(self):
		return self._name

	def __cmp__(self, other):
		return cmp(self._name, other._name)

	def __hash__(self):
		return hash(self._name)


# Atoms
A = Atom("a", operator=True)
A_PLUS = Atom("a+", operator=True, conjugate=True)
RHO = Atom("rho", operator=True)

PSI1_OP = Atom("psi1^", operator=True)
PSI1_OP_PLUS = Atom("psi1^+", operator=True, conjugate=True)
PSI2_OP = Atom("psi2^", operator=True)
PSI2_OP_PLUS = Atom("psi2^+", operator=True, conjugate=True)

ALPHA = Atom("alpha")
ALPHA_STAR = Atom("alpha*", conjugate=True)
PSI1 = Atom("psi1")
PSI1_STAR = Atom("psi1*", conjugate=True)
PSI2 = Atom("psi2")
PSI2_STAR = Atom("psi2*", conjugate=True)

D_ALPHA = Atom("d/dalpha")
D_ALPHA_STAR = Atom("d/dalpha*", conjugate=True)
D_PSI1 = Atom("d/dpsi1")
D_PSI1_STAR = Atom("d/dpsi1*", conjugate=True)
D_PSI2 = Atom("d/dpsi2")
D_PSI2_STAR = Atom("d/dpsi2*", conjugate=True)

GAMMA = Atom("G", constant=True)
GAMMA1_SMALL = Atom("g1", constant=True)
GAMMA2_SMALL = Atom("g2", constant=True)
GAMMA3_SMALL = Atom("g3", constant=True)

Atom.declareCorrespondence(A, ALPHA)
Atom.declareCorrespondence(A_PLUS, ALPHA_STAR)
Atom.declareCorrespondence(PSI1_OP, PSI1)
Atom.declareCorrespondence(PSI1_OP_PLUS, PSI1_STAR)
Atom.declareCorrespondence(PSI2_OP, PSI2)
Atom.declareCorrespondence(PSI2_OP_PLUS, PSI2_STAR)

Atom.declareDifferential(ALPHA, D_ALPHA)
Atom.declareDifferential(ALPHA_STAR, D_ALPHA_STAR)
Atom.declareDifferential(PSI1, D_PSI1)
Atom.declareDifferential(PSI1_STAR, D_PSI1_STAR)
Atom.declareDifferential(PSI2, D_PSI2)
Atom.declareDifferential(PSI2_STAR, D_PSI2_STAR)

Atom.declareConjugate(A, A_PLUS)
Atom.declareConjugate(PSI1_OP, PSI1_OP_PLUS)
Atom.declareConjugate(PSI2_OP, PSI2_OP_PLUS)
Atom.declareConjugate(ALPHA, ALPHA_STAR)
Atom.declareConjugate(PSI1, PSI1_STAR)
Atom.declareConjugate(PSI2, PSI2_STAR)


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

	def firstPostfixDifferential(self):
		assert self.isSimple()

		for i in xrange(len(self.factors) - 1):
			if not self.factors[i].isDifferential() and self.factors[i + 1].isDifferential():
				return i + 1

		return -1

	def hasDifferentialsInFront(self):
		return self.firstPostfixDifferential() == -1

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

	def hasDifferentialsInFront(self):
		if not self.isFlat():
			return False

		for term in self.terms:
			if not term.hasDifferentialsInFront():
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
			sums.append(Sum([Term(1.0, [factor])]))

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
			sums.append(Sum([Term(1.0, [term])]))

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

	pos = term.firstPostfixDifferential()

	if pos == -1:
		return Term(term.coeff, term.factors)

	dif = term.factors[pos]
	var = term.factors[pos - 1]

	if dif.differentialOf() == var:
		new_factors = term.factors[:pos-1] + \
			[Sum([Term(1.0, [dif, var]), Term(-1, [])])] + \
			term.factors[pos+1:]
	else:
		new_factors = term.factors[:pos-1] + [dif, var] + term.factors[pos+1:]

	return derivativesToFront(termToFlatSum(Term(term.coeff, new_factors)))

def wignerTerm(op, prefix):
	assert op.isOperator()

	is_conj = op.isConjugate()
	conj_op = op.conjugate()

	func = op.correspondence()
	conj_func = conj_op.correspondence()

	if prefix:
		return Sum([func, Term(0.5 * (-1 if is_conj else 1), [conj_func.differential()])])
	else:
		return Sum([func, Term(0.5 * (1 if is_conj else -1), [conj_func.differential()])])

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

	for i in xrange(rho_pos):
		if term.factors[i].isOperator():
			new_factors.append(wignerTerm(term.factors[i], True))
		else:
			new_factors.append(term.factors[i])

	for i in xrange(len(term.factors) - 1, rho_pos, -1):
		if term.factors[i].isOperator():
			new_factors.append(wignerTerm(term.factors[i], False))
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
	assert term.hasDifferentialsInFront()

	def key1(elem):
		if elem.isDifferential():
			return "1" + str(elem)

		if elem.isConstant():
			return "2" + str(elem)

		return "3" + str(elem)

	factors = sorted(term.factors, key=key1)
	return Term(term.coeff, factors)

def groupTerms(sum):
	"""Flat sum -> Flat sum"""
	assert sum.isFlat()

	memo = {}
	factors = {}

	for term in sum.terms:
		key = tuple([str(factor) for factor in term.factors])
		factors[key] = term.factors
		if key in memo:
			memo[key] += term.coeff
		else:
			memo[key] = term.coeff

	return Sum([Term(memo[key], factors[key]) for key in memo if abs(memo[key]) > 0])

def dropHighOrderDerivatives(sum, cutoff):
	"""Flat sum -> Flat sum"""
	assert sum.hasDifferentialsInFront()

	new_terms = []
	for term in sum.terms:
		order = 0
		for factor in term.factors:
			if factor.isDifferential():
				order += 1
			else:
				break

		if order <= cutoff:
			new_terms.append(term)

	return Sum(new_terms)

def replace(obj, old, new):
	if isinstance(obj, Sum):
		new_terms = []
		for term in obj.terms:
			new_terms.append(replace(term, old, new))
		return Sum(new_terms)

	elif isinstance(obj, Term):
		new_factors = []
		for factor in obj.factors:
			new_factors.append(replace(factor, old, new))

		return Term(obj.coeff, new_factors)

	else:
		if obj == old:
			return new
		else:
			return obj

def apply(x, *args, **kwds):
	if 'verbose' in kwds:
		verbose = kwds['verbose']
	else:
		verbose = False

	for arg in args:
		if isinstance(arg, tuple):
			func = arg[0]
			func_args = list(arg[1:])
		else:
			func = arg
			func_args = []

		func_args = tuple([x] + func_args)
		x = func(*func_args)

		if verbose:
			print "--- " + str(func)
			print x

	return x
