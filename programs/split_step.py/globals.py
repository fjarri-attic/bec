from pycuda.autoinit import device
from pycuda.driver import device_attribute
from pycuda.tools import DeviceMemoryPool

mempool = DeviceMemoryPool()

def log2(x):
	"""Auxiliary function, calculating binary logarithm for integer"""
	pows = [1]
	while x > 2 ** pows[-1]:
		pows.append(pows[-1] * 2)

	res = 0
	for pow in reversed(pows):
		if x >= (2 ** pow):
			x >>= pow
			res += pow
	return res

def getExecutionParameters(func, elements):
	max_block_size = device.get_attribute(device_attribute.MAX_BLOCK_DIM_X)
	max_registers = device.get_attribute(device_attribute.MAX_REGISTERS_PER_BLOCK)

	block_size = 2 ** log2(max_registers / func.num_regs)
	block_size = min(max_block_size, block_size)
	if block_size == 0:
		raise Exception("Too much registers used by kernel")

	max_grid_x = device.get_attribute(device_attribute.MAX_GRID_DIM_X)
	max_grid_y = device.get_attribute(device_attribute.MAX_GRID_DIM_Y)
	blocks_num_x = min(max_grid_x, elements / block_size)
	blocks_num_y = 1 if blocks_num_x <= elements else elements / blocks_num_x
	if blocks_num_y > max_grid_y:
		raise Exception("Insufficient grid size to handle all the elements")

	return (block_size, 1, 1), (blocks_num_x, blocks_num_y)
