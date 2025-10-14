import copy
import random
import numpy as np
from tqdm import tqdm

class IterativeEvolutionFinder:
    def __init__(self, efficiency_predictor, accuracy_predictor, **kwargs):
        self.efficiency_predictor = efficiency_predictor
        self.accuracy_predictor = accuracy_predictor

        # evolution hyper-parameters
        self.arch_mutate_prob = kwargs.get("arch_mutate_prob", 0.1)
        self.resolution_mutate_prob = kwargs.get("resolution_mutate_prob", 0.5)
        self.population_size = kwargs.get("population_size", 100)
        self.max_time_budget = kwargs.get("max_time_budget", 500)
        self.parent_ratio = kwargs.get("parent_ratio", 0.25)
        self.mutation_ratio = kwargs.get("mutation_ratio", 0.5)
        self.passthrough = kwargs.get("passthrough", 1.0)

        self.mode = None

    @property
    def arch_manager(self):
        return self.accuracy_predictor.arch_encoder

    def update_hyper_params(self, new_param_dict):
        self.__dict__.update(new_param_dict)

    def random_valid_sample(self, constraint):
        while True:
            sample = self.arch_manager.random_sample_arch()
            efficiency = self.efficiency_predictor.get_efficiency(sample)
            if efficiency >= constraint['minimum_weights'] and efficiency <= constraint['maximum_weights']:
                return sample, efficiency

    def random_valid_sample_module(self, constraint, parent):
        while True:
            sample = self.arch_manager.random_sample_arch()

            if self.mode:
                sample['ks'][:20] = parent['ks'][:20]
                sample['e'][:20] = parent['e'][:20]
                sample['d'][:4] = parent['d'][:4]
            else:
                sample['ks'][20:] = parent['ks'][20:]
                sample['e'][20:] = parent['e'][20:]
                sample['d'][4:] = parent['d'][4:]

            efficiency = self.efficiency_predictor.get_efficiency(sample)
            if efficiency >= constraint['minimum_weights'] and efficiency <= constraint['maximum_weights']:
                return sample, efficiency

    def mutate_sample(self, sample, constraint):
        while True:
            new_sample = copy.deepcopy(sample)

            self.arch_manager.mutate_resolution(new_sample, self.resolution_mutate_prob)
            self.arch_manager.mutate_arch(new_sample, self.arch_mutate_prob)

            efficiency = self.efficiency_predictor.get_efficiency(new_sample)
            if efficiency >= constraint['minimum_weights'] and efficiency <= constraint['maximum_weights']:
                return new_sample, efficiency

    def mutate_sample_module(self, sample, constraint, parent):
        while True:
            new_sample = copy.deepcopy(sample)

            self.arch_manager.mutate_resolution(new_sample, self.resolution_mutate_prob)
            self.arch_manager.mutate_arch(new_sample, self.arch_mutate_prob)

            if self.mode:
                new_sample['ks'][:20] = parent['ks'][:20]
                new_sample['e'][:20] = parent['e'][:20]
                new_sample['d'][:4] = parent['d'][:4]
            else:
                new_sample['ks'][20:] = parent['ks'][20:]
                new_sample['e'][20:] = parent['e'][20:]
                new_sample['d'][4:] = parent['d'][4:]

            efficiency = self.efficiency_predictor.get_efficiency(new_sample)
            if efficiency >= constraint['minimum_weights'] and efficiency <= constraint['maximum_weights']:
                return new_sample, efficiency

    def crossover_sample(self, sample1, sample2, constraint):
        while True:
            new_sample = copy.deepcopy(sample1)
            for key in new_sample.keys():
                if not isinstance(new_sample[key], list):
                    new_sample[key] = random.choice([sample1[key], sample2[key]])
                else:
                    for i in range(len(new_sample[key])):
                        new_sample[key][i] = random.choice(
                            [sample1[key][i], sample2[key][i]]
                        )

            efficiency = self.efficiency_predictor.get_efficiency(new_sample)
            if efficiency >= constraint['minimum_weights'] and efficiency <= constraint['maximum_weights']:
                return new_sample, efficiency

    def crossover_sample_module(self, sample1, sample2, constraint, parent):
        while True:
            new_sample = copy.deepcopy(sample1)
            for key in new_sample.keys():
                if not isinstance(new_sample[key], list):
                    new_sample[key] = random.choice([sample1[key], sample2[key]])
                else:
                    for i in range(len(new_sample[key])):
                        new_sample[key][i] = random.choice(
                            [sample1[key][i], sample2[key][i]]
                        )
            if self.mode:
                new_sample['ks'][:20] = parent['ks'][:20]
                new_sample['e'][:20] = parent['e'][:20]
                new_sample['d'][:4] = parent['d'][:4]
            else:
                new_sample['ks'][20:] = parent['ks'][20:]
                new_sample['e'][20:] = parent['e'][20:]
                new_sample['d'][4:] = parent['d'][4:]

            efficiency = self.efficiency_predictor.get_efficiency(new_sample)
            if efficiency >= constraint['minimum_weights'] and efficiency <= constraint['maximum_weights']:
                return new_sample, efficiency

    def run_evolution_search(self, constraint, epochs, verbose=False, **kwargs):
        """Run a single roll-out of regularized evolution to a fixed time budget."""
        self.update_hyper_params(kwargs)

        mutation_numbers = int(round(self.mutation_ratio * self.population_size))
        parents_size = int(round(self.parent_ratio * self.population_size))
        passthrough_size = int(round(self.passthrough * self.population_size))

        best_valids = [-100]
        best_info = None
        memory = [None, None] # (backbone, neck/head) populations

        for epoch in range(epochs):
            population = []  # (validation, sample, latency) tuples
            child_pool = []
            efficiency_pool = []

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}")
            self.mode = epoch % 2

            if memory[self.mode] is None:
                if verbose:
                    print("Generate random population...")

                sample, efficiency = self.random_valid_sample(constraint)
                child_pool.append(sample)
                efficiency_pool.append(efficiency)

                for _ in range(self.population_size - 1):
                    sample, efficiency = self.random_valid_sample_module(constraint, sample)
                    child_pool.append(sample)
                    efficiency_pool.append(efficiency)

                accs = self.accuracy_predictor.predict_acc(child_pool)
                for i in range(self.population_size):
                    population.append((accs[i].item(), child_pool[i], efficiency_pool[i]))
            else:
                if verbose:
                    print(f"Loading {passthrough_size} population...")

                child_pool = memory[self.mode][:passthrough_size]
                parent = memory[1 - self.mode][0]
                dropped = []

                for i in range(passthrough_size):
                    if self.mode:
                        child_pool[i]['ks'][:20] = parent['ks'][:20]
                        child_pool[i]['e'][:20] = parent['e'][:20]
                        child_pool[i]['d'][:4] = parent['d'][:4]
                    else:
                        child_pool[i]['ks'][20:] = parent['ks'][20:]
                        child_pool[i]['e'][20:] = parent['e'][20:]
                        child_pool[i]['d'][4:] = parent['d'][4:]

                    efficiency = self.efficiency_predictor.get_efficiency(child_pool[i])
                    if efficiency >= constraint['minimum_weights'] and efficiency <= constraint['maximum_weights']:
                        efficiency_pool.append(efficiency)
                    else:
                        dropped.append(i)

                if verbose:
                    print(f"drop {len(dropped)} population...")
                for i in sorted(dropped, reverse=True):
                    child_pool.pop(i)

                if verbose:
                    print(f"Generate {self.population_size - len(child_pool)} population...")

                if child_pool:
                    sample = child_pool[-1]
                else:
                    sample, efficiency = self.random_valid_sample(constraint)
                    child_pool.append(sample)
                    efficiency_pool.append(efficiency)

                for i in range(self.population_size - len(child_pool)):
                    sample, efficiency = self.random_valid_sample_module(constraint, sample)
                    child_pool.append(sample)
                    efficiency_pool.append(efficiency)

                accs = self.accuracy_predictor.predict_acc(child_pool)
                for i in range(self.population_size):
                    population.append((accs[i].item(), child_pool[i], efficiency_pool[i]))

            if verbose:
                print("Start Evolution...")
            # After the population is seeded, proceed with evolving the population.
            with tqdm(
                total=self.max_time_budget,
                desc="Searching with constraint (%s)" % constraint,
                disable=(not verbose),
            ) as t:
                for i in range(self.max_time_budget):
                    parents = sorted(population, key=lambda x: x[0])[::-1][:parents_size]
                    acc = parents[0][0]
                    t.set_postfix({"acc": parents[0][0]})
                    if not verbose and (i + 1) % 100 == 0:
                        print("Iter: {} Acc: {}".format(i + 1, parents[0][0]))

                    if acc > best_valids[-1]:
                        best_valids.append(acc)
                        best_info = parents[0]
                    else:
                        best_valids.append(best_valids[-1])

                    population = parents
                    child_pool = []
                    efficiency_pool = []

                    for j in range(mutation_numbers):
                        par_sample = population[np.random.randint(parents_size)][1]
                        # Mutate
                        new_sample, efficiency = self.mutate_sample_module(par_sample, constraint, par_sample)
                        child_pool.append(new_sample)
                        efficiency_pool.append(efficiency)

                    for j in range(self.population_size - mutation_numbers):
                        par_sample1 = population[np.random.randint(parents_size)][1]
                        par_sample2 = population[np.random.randint(parents_size)][1]
                        # Crossover
                        new_sample, efficiency = self.crossover_sample_module(
                            par_sample1, par_sample2, constraint, par_sample1
                        )
                        child_pool.append(new_sample)
                        efficiency_pool.append(efficiency)

                    accs = self.accuracy_predictor.predict_acc(child_pool)
                    for j in range(self.population_size):
                        population.append(
                            (accs[j].item(), child_pool[j], efficiency_pool[j])
                        )

                    t.update(1)
            parents = sorted(population, key=lambda x: x[0])[::-1]
            memory[self.mode] = [x[1] for x in parents]
        return best_valids, best_info