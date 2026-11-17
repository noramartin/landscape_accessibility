import numpy as np 
from functions import *
from functools import partial

def look_up_fitness_or_generate_sRMF(g, genotype_vs_fitness, p, delta, a):
	key = int('1'+ g)
	L = len(g)
	try:
		return genotype_vs_fitness[key]
	except KeyError:
		additive_comp = L + 1 - Hamming_dist(g, '0'*len(g)) - delta*p
		genotype_vs_fitness[key] = float(np.random.uniform(-1*a, a)) + random.choices([delta, 0], weights=[p, 1-p])[0] + additive_comp
		return genotype_vs_fitness[key]

def look_up_fitness_or_generate_HoC(g, genotype_vs_fitness):
	key = int('1'+ g)
	try:
		return genotype_vs_fitness[key]
	except KeyError:
		genotype_vs_fitness[key] = np.random.uniform(1, 2)
		return genotype_vs_fitness[key]

def sample_genotype_from_shell(L, K, d):
	mutated_pos = np.random.choice(L, size=d, replace=False)
	assert len(set(mutated_pos)) == d
	return ''.join(['0' if i not in mutated_pos else str(np.random.choice(np.arange(1, K))) for i in range(L)])


def get_fitness_increasing_neighbours(g, genotype_vs_fitness, alphabet, look_up_fitness_or_generate):
	fit_g = look_up_fitness_or_generate(g, genotype_vs_fitness)
	return [g2 for g2 in neighbours_seq(g, alphabet=alphabet) if look_up_fitness_or_generate(g2, genotype_vs_fitness) > fit_g]

def run_evolution_on_the_fly(n_current, alphabet, genotype_vs_fitness, look_up_fitness_or_generate, termination='1'):
	assert termination in ['peak', '1']
	L= len(n_current)
	ref_seq = '0'*L
	d_init = Hamming_dist(n_current, '0'*L)
	fitness_increasing_neighbours = get_fitness_increasing_neighbours(n_current, genotype_vs_fitness, alphabet, look_up_fitness_or_generate)
	while True:
		if termination == 'peak':
			if len(fitness_increasing_neighbours) == 0:
				return n_current
		else:
			if len(fitness_increasing_neighbours) == 0: # peak transition
				return 0
			elif d_init > Hamming_dist(n_current, '0'*L) + 0.5:
				return 1
		n_current = random.choice(fitness_increasing_neighbours)
		fitness_increasing_neighbours = get_fitness_increasing_neighbours(n_current, genotype_vs_fitness, alphabet, look_up_fitness_or_generate)


def simulate_landscape_on_the_fly(L, K, number_samples_shell, n_it, n_it_giveng, params={}, type_landscape='sRMF', shell_exit = True):
	if L < 10:
		step_size = 1
	elif L < 20:
		step_size = 2
	else:
		step_size = 5
	ref_seq = '0'*L
	alphabet = [str(i) for i in range(K)]
	genotype_vs_fitness = {}
	if type_landscape == 'sRMF':
		look_up_fitness_or_generate = partial(look_up_fitness_or_generate_sRMF, p=params['p'], delta=params['delta'], a=params['a'])
	else:
		look_up_fitness_or_generate = partial(look_up_fitness_or_generate_HoC)
	if shell_exit:
		shells_to_check = [d for d in np.arange(L + step_size, step=step_size) if d <= L]
		shell_vs_exit_prob = {d: [] for d in shells_to_check}
		for d in shells_to_check:
			print('d', d)
			for sample_g_id in range(number_samples_shell):
				g = sample_genotype_from_shell(L, K, d)
				assert Hamming_dist(g, '0'*len(g)) == d
				fitness_increasing_neighbours = get_fitness_increasing_neighbours(g, genotype_vs_fitness, alphabet, look_up_fitness_or_generate)
				assert len(fitness_increasing_neighbours) <= (K-1)*L
				if len(fitness_increasing_neighbours) > 0:
					for rep in range(n_it_giveng):
						shell_vs_exit_prob[d].append(run_evolution_on_the_fly(g, alphabet, genotype_vs_fitness, look_up_fitness_or_generate, termination='1'))
	###
	peak_heights_reached, peak_heights_reached_antipode, peak_heights_reference = [], [], []
	for sample_g_id in range(n_it//10):
		g = ''.join([str(np.random.choice(K)) for i in range(L)])
		peak = run_evolution_on_the_fly(g, alphabet, genotype_vs_fitness, look_up_fitness_or_generate, termination='peak')
		peak_heights_reached.append(genotype_vs_fitness[int('1' + peak)])
		######
		g = sample_genotype_from_shell(L, K, d=L)
		peak = run_evolution_on_the_fly(g, alphabet, genotype_vs_fitness, look_up_fitness_or_generate, termination='peak')
		peak_heights_reached_antipode.append(genotype_vs_fitness[int('1' + peak)])
	for sample_g_id in range(n_it):
		g = ''.join([str(np.random.choice(K)) for i in range(L)])	
		if len(get_fitness_increasing_neighbours(g, genotype_vs_fitness, alphabet, look_up_fitness_or_generate)) == 0:
			peak_heights_reference.append(genotype_vs_fitness[int('1' + g)])
	if shell_exit:
		return peak_heights_reached, peak_heights_reached_antipode, peak_heights_reference, shell_vs_exit_prob 
	else:
		return peak_heights_reached, peak_heights_reached_antipode, peak_heights_reference


def basin_approx(L, K, number_samples_shell, n_it, params={}):
	alphabet = [str(i) for i in range(K)]
	genotype_vs_fitness = {}
	look_up_fitness_or_generate = partial(look_up_fitness_or_generate_sRMF, p=params['p'], delta=params['delta'], a=params['a'])
	####
	peak_vs_reached = {}
	for rep in range(n_it):
		if rep%1000 == 0:
			print('rep', rep)
		g = ''.join([str(np.random.choice(K)) for i in range(L)])
		peak = run_evolution_on_the_fly(g, alphabet, genotype_vs_fitness, look_up_fitness_or_generate, termination='peak')	
		try:
			peak_vs_reached[peak] += 1/n_it	
		except KeyError:
			peak_vs_reached[peak] = 1/n_it	
	peak_vs_reached = {p: peak_vs_reached[p] for p in peak_vs_reached if peak_vs_reached[p] > 4.5/n_it}
	print('number of peaks', len(peak_vs_reached))
	#####
	d_vs_peaks = {d: [g for g in peak_vs_reached if d == Hamming_dist(g, '0'*L)] for d in set([Hamming_dist(g, '0'*L) for g in peak_vs_reached])}
	peaks_forbasin = [g for d, peak_l in d_vs_peaks.items() for g in peak_l[:min(10, len(peak_l))]]
	#####
	peak_vs_basin_size_lower, peak_vs_fitness = {}, {}
	for g in peaks_forbasin:
		print('peak', g)
		peak_vs_basin_size_lower[g] = 0
		for rep in range(n_it):
			g2 = ''.join([str(np.random.choice(K)) for i in range(L)])
			if g2 == g:
				peak_vs_basin_size_lower[g] += 1/n_it
			elif is_fitness_increasing_MF(g2, g):
				if has_increasing_path_direct(g2, g, genotype_vs_fitness, look_up_fitness_or_generate):
					peak_vs_basin_size_lower[g] += 1/n_it
		peak_vs_fitness[g] = genotype_vs_fitness[int('1' + g)]
	return peak_vs_basin_size_lower, {p: peak_vs_reached[p] for p in peak_vs_fitness}, peak_vs_fitness

def has_increasing_path_direct(g2, g, genotype_vs_fitness, look_up_fitness_or_generate):
	#edited from Claude code  
	diff_positions = [i for i in range(len(g)) if (g[i] == '0' and  g2[i] != '0') or (g[i] != '0' and  g2[i] != g[i] and g2[i] != '0')] # all except the single 0 -> 1 transition
	visited_dead_ends = set()  # frozensets of positions already known to fail
	assert len(diff_positions) == Hamming_dist(g2, g) or len(diff_positions) == Hamming_dist(g2, g) - 1
	def dfs(current_g, mutated_so_far):
		if mutated_so_far == frozenset(diff_positions):
			assert Hamming_dist(current_g, g) <= 1
			if look_up_fitness_or_generate(g, genotype_vs_fitness) >= look_up_fitness_or_generate(current_g, genotype_vs_fitness):
				return True
			else:
				return False
		if mutated_so_far in visited_dead_ends:
			return False
		fit_current = look_up_fitness_or_generate(current_g, genotype_vs_fitness)
		for pos in diff_positions:
			if pos in mutated_so_far:
				continue
			next_g = current_g[:pos] + g[pos] + current_g[pos+1:]
			fit_next = look_up_fitness_or_generate(next_g, genotype_vs_fitness)
			if fit_next >= fit_current:
				if dfs(next_g, mutated_so_far | {pos}):
					return True
		visited_dead_ends.add(mutated_so_far)
		return False

	return dfs(g2, frozenset())

def is_fitness_increasing_MF(g2, g):
	if len([i for i, c in enumerate(g) if g[i] != '0' and g2[i] == '0']) <= 1:
		return True
	else:
		return False