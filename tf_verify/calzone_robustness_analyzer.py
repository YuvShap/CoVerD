import dataclasses
import math
import time
import random
import numpy as np
from itertools import chain
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.stats import norm
import coverd_finite_geometries

def decreasing_exp(x, a, b):
    return 1 - np.exp(a*(x-b))


class LZeroRobustnessAnalyzer:
    def __init__(self, image_index, image, gpu_workers, cpu_workers, label, t, sampling, timeout, dataset):
        self.__image_index = image_index
        self.__image = image
        self.__label = label
        self.__gpu_workers = gpu_workers
        self.__cpu_workers = cpu_workers
        self.__t = t
        self.__sampling = sampling
        self.__timeout = timeout
        if dataset == 'cifar10':
            self.__number_of_pixels = 1024
        else:
            self.__number_of_pixels = 784
        self.__refinement_db_max_v = 200
        self.__refinement_db_max_covering_size = 500000

    def analyze(self):
        analyzer_start_time = time.time()
        print('*******************************************************')
        print(f'Starting to analyze image {self.__image_index}')

        print('Starting to estimate by sampling')
        estimation_start_time = time.time()
        p_vector, w_vector = self.__estimate_p_and_w()
        estimation_duration = time.time() - estimation_start_time
        print(f'Estimation took {estimation_duration:.3f}')

        print('Choosing strategy')
        choosing_strategy_start_time = time.time()
        covering_sizes, fraction_robust = self.__load_covering_sizes_and_approximate_fraction_robust(p_vector)
        chosen_geometry, chosen_time, T_k, geometries_data = self.__choose_strategy(covering_sizes, fraction_robust, w_vector)
        choosing_strategy_duration = time.time() - choosing_strategy_start_time
        estimated_verification_time = chosen_time / len(self.__gpu_workers)
        print(f'Chosen geometry is {chosen_geometry}, mean block size is {self.__number_of_pixels * chosen_geometry.block_size() / chosen_geometry.number_of_points():.3f}, estimated verification time is {estimated_verification_time:.3f} sec')
        refinement_strategy = {key: value[1] for key, value in T_k.items()}

        self.__release_workers(chosen_geometry, refinement_strategy)

        gpupoly_stats_by_size = {size: {'runs': 0, 'successes': 0, 'total_duration': 0} for size in range(self.__t, self.__refinement_db_max_v + 1)}
        waiting_adversarial_example_suspects = set()
        innocent_adversarial_example_suspects = set()

        results = {'verified': True, 'timed_out': False, 'p_vector': p_vector.tolist(), 'w_vector': w_vector.tolist(),
                   'estimation_duration': estimation_duration,
                   'fraction_robust': fraction_robust,
                   'T_k': T_k,
                   'geometries_data': geometries_data,
                   'chosen_geometry': dataclasses.asdict(chosen_geometry),
                   'refinement_strategy': refinement_strategy,
                   'choosing_strategy_duration': choosing_strategy_duration,
                   'estimated_verification_time': estimated_verification_time.item(),
                   'gpupoly_stats_by_size': gpupoly_stats_by_size,
                   'time_waiting_for_milp_after_covering': 0}

        iterating_covering_start_time = time.time()
        timed_out, found_adversarial, adversarial_pixels, adversarial_example, adversarial_label = self.__wait_while_iterating_covering(
            analyzer_start_time, gpupoly_stats_by_size, waiting_adversarial_example_suspects,
            innocent_adversarial_example_suspects, chosen_geometry.number_of_blocks())
        iterating_covering_duration = time.time() - iterating_covering_start_time
        results['iterating_covering_duration'] = iterating_covering_duration

        if timed_out:
            results['verified'] = False
            results['timed_out'] = True
            results['proved_with_milp'] = len(innocent_adversarial_example_suspects)
            results['groups_waiting_for_milp'] = len(waiting_adversarial_example_suspects)
            return results

        if found_adversarial:
            results['verified'] = False
            results['adversarial_pixels'] = adversarial_pixels
            results['adversarial_example'] = adversarial_example
            results['adversarial_label'] = adversarial_label
            results['proved_with_milp'] = len(innocent_adversarial_example_suspects)
            results['groups_waiting_for_milp'] = len(waiting_adversarial_example_suspects)
            return results

        timed_out, found_adversarial, adversarial_pixels, adversarial_example, adversarial_label, waiting = self.__wait_for_cpu_workers_if_needed(
            analyzer_start_time,
            innocent_adversarial_example_suspects, waiting_adversarial_example_suspects)
        results['time_waiting_for_milp_after_covering'] = waiting

        if timed_out:
            results['verified'] = False
            results['timed_out'] = True
            results['proved_with_milp'] = len(innocent_adversarial_example_suspects)
            results['groups_waiting_for_milp'] = len(waiting_adversarial_example_suspects)
            return results

        if found_adversarial:
            results['verified'] = False
            results['adversarial_pixels'] = adversarial_pixels
            results['adversarial_example'] = adversarial_example
            results['adversarial_label'] = adversarial_label
            results['proved_with_milp'] = len(innocent_adversarial_example_suspects)
            results['groups_waiting_for_milp'] = len(waiting_adversarial_example_suspects)
            return results

        print('\nVerified!')
        print('*******************************************************')
        results['proved_with_milp'] = len(innocent_adversarial_example_suspects)
        results['groups_waiting_for_milp'] = len(waiting_adversarial_example_suspects)
        return results

    def __estimate_p_and_w(self):
        sampling_lower_bound = self.__t
        sampling_upper_bound = self.__refinement_db_max_v
        repetitions = math.ceil(self.__sampling / len(self.__gpu_workers))
        for gpu_worker in self.__gpu_workers:
            gpu_worker.send((self.__image, self.__label, sampling_lower_bound,
                             sampling_upper_bound, repetitions))

        sampling_counts_vector = np.zeros(sampling_upper_bound - sampling_lower_bound + 1)
        sampling_successes_vector = np.zeros(sampling_upper_bound - sampling_lower_bound + 1)
        sampling_time_vector = np.zeros(sampling_upper_bound - sampling_lower_bound + 1)
        for gpu_worker in self.__gpu_workers:
            sampling_counts, sampling_successes, sampling_time = gpu_worker.recv()
            sampling_counts_vector += np.array(sampling_counts)
            sampling_successes_vector += np.array(sampling_successes)
            sampling_time_vector += np.array(sampling_time)

        success_ratio_vector = sampling_successes_vector / sampling_counts_vector
        average_time_vector = sampling_time_vector / sampling_counts_vector

        p_vector = savgol_filter(success_ratio_vector, 15, 2)
        w_vector = np.copy(average_time_vector)

        indexes = sorted([index for index, sample in enumerate(success_ratio_vector) if sample < 0.97])
        stop_index = min(indexes[0] + 1, len(success_ratio_vector)) if len(indexes) > 0 else len(success_ratio_vector)
        if stop_index > 1:
            popt, pcov = curve_fit(decreasing_exp, range(sampling_lower_bound, sampling_lower_bound + stop_index), p_vector[:stop_index], maxfev=5000)
            print(f'a={popt[0]}, b={popt[1]}')
            p_vector[:stop_index] = decreasing_exp(np.asarray(range(sampling_lower_bound, sampling_lower_bound + stop_index)), *popt)

        return p_vector, w_vector

    def __load_covering_sizes_and_approximate_fraction_robust(self, p_vector):
        fraction_robust = dict()
        covering_table_file = np.genfromtxt(f'refinement_coverings/{self.__t}-table-refinement.csv', delimiter=',')
        covering_sizes = dict()
        for v in range(self.__t, self.__refinement_db_max_v + 1):
            fraction_robust[v] = dict()
            for k in range(self.__t, v):
                estimated_value = (p_vector[k - self.__t] - p_vector[v - self.__t]) / (1 - p_vector[v - self.__t])
                fraction_robust[v][k] = min(1, max(estimated_value, 0))
                covering_sizes[(v, k)] = covering_table_file[v - self.__t + 1][k - self.__t + 1]
        v = self.__number_of_pixels
        fraction_robust[v] = dict()
        for k in range(self.__t, self.__refinement_db_max_v + 1):
            fraction_robust[v][k] = min(1, max(p_vector[k - self.__t], 0))

        covering_sizes = {key: value for key, value in covering_sizes.items() if value < self.__refinement_db_max_covering_size}

        return covering_sizes, fraction_robust

    def __choose_strategy(self, covering_sizes, fraction_robust, w_vector):
        T_k = dict()
        T_k[self.__t] = (0, None)
        for v in range(self.__t + 1, self.__refinement_db_max_v + 1):
            best_k = None
            best_k_value = None
            for k in range(self.__t, min(v, self.__refinement_db_max_v + 1)):
                if (v, k) not in covering_sizes:
                    continue
                k_value = covering_sizes[(v, k)] * (w_vector[k - self.__t] + (1 - fraction_robust[v][k]) * T_k[k][0])
                if best_k_value is None or k_value < best_k_value:
                    best_k = k
                    best_k_value = k_value
            T_k[v] = (best_k_value, best_k)

        best_geometry = None
        best_time = None
        geometries = []
        for geometry in coverd_finite_geometries.get_possible_geometries(self.__number_of_pixels, self.__t):
            current_time = 0
            pseudo_k = geometry.block_size() * self.__number_of_pixels / geometry.number_of_points()
            std_dev = math.sqrt(pseudo_k * (1 - pseudo_k + (geometry.block_size() - 1) * (self.__number_of_pixels - 1) / (geometry.number_of_points() - 1)))
            total_number_of_blocks = geometry.number_of_blocks()

            if (1 - norm.cdf(self.__refinement_db_max_v, pseudo_k, std_dev)) * total_number_of_blocks > 1e-2:
                continue

            cdf_before = norm.cdf(self.__t - 0.5, pseudo_k, std_dev)
            for block_size in range(self.__t, self.__refinement_db_max_v + 1):
                cdf_after = norm.cdf(block_size + 0.5, pseudo_k, std_dev)
                block_size_probability = cdf_after - cdf_before
                cdf_before = cdf_after
                expected_number_of_blocks_of_current_size = block_size_probability * total_number_of_blocks
                number_of_blocks_of_current_size = math.floor(expected_number_of_blocks_of_current_size)
                if random.random() < expected_number_of_blocks_of_current_size - number_of_blocks_of_current_size:
                    number_of_blocks_of_current_size += 1
                current_time += number_of_blocks_of_current_size * (w_vector[block_size - self.__t] + (1 - fraction_robust[self.__number_of_pixels][block_size]) * T_k[block_size][0])

            geometries.append((dataclasses.asdict(geometry), current_time.item(), pseudo_k))
            if best_time is None or current_time < best_time:
                best_time = current_time
                best_geometry = geometry
        return best_geometry, best_time, T_k, geometries

    def __release_workers(self, best_geometry, refinement_strategy):
        chosen_points = random.sample(range(best_geometry.number_of_points()), self.__number_of_pixels)
        random.shuffle(chosen_points)
        for worker_index, gpu_worker in enumerate(self.__gpu_workers):
            gpu_worker.send((worker_index, len(self.__gpu_workers), self.__image, self.__label, best_geometry, chosen_points, refinement_strategy))
        for cpu_worker in self.__cpu_workers:
            cpu_worker.send((self.__image, self.__label))

    def __wait_while_iterating_covering(self, analyzer_start_time, statistics_by_size,
                                        waiting_adversarial_example_suspects,
                                        innocent_adversarial_example_suspects, initial_covering_size):
        number_of_groups_finished_from_initial_covering = 0
        next_cpu_worker = 0
        done_count = 0
        last_print_time = time.time()
        while done_count < len(self.__gpu_workers):
            if time.time() - analyzer_start_time >= self.__timeout:
                print('\nTimed out!')
                print('*******************************************************')
                self.__stop_workers()
                return True, False, None, None, None
            last_print_time = self.__print_progress(statistics_by_size, initial_covering_size,
                                                    innocent_adversarial_example_suspects,
                                                    last_print_time, number_of_groups_finished_from_initial_covering,
                                                    waiting_adversarial_example_suspects)

            found_adversarial, pixels, adversarial_example, adversarial_label = self.__pool_cpu_workers_messages(
                innocent_adversarial_example_suspects, waiting_adversarial_example_suspects)

            if found_adversarial:
                return False, found_adversarial, pixels, adversarial_example, adversarial_label

            for gpu_worker in self.__gpu_workers:
                if gpu_worker.poll():
                    message = gpu_worker.recv()
                    if message == 'adversarial-example-suspect':
                        adversarial_example_suspect = gpu_worker.recv()
                        if adversarial_example_suspect not in innocent_adversarial_example_suspects \
                                and adversarial_example_suspect not in waiting_adversarial_example_suspects:
                            waiting_adversarial_example_suspects.add(adversarial_example_suspect)
                            cpu_worker = self.__cpu_workers[next_cpu_worker]
                            cpu_worker.send(adversarial_example_suspect)
                            next_cpu_worker = (next_cpu_worker + 1) % len(self.__cpu_workers)

                    elif message == 'done':
                        done_count += 1

                    elif message == 'next':
                        number_of_groups_finished_from_initial_covering += 1

                    else:
                        verified, number_of_pixels, duration = message
                        size_statistics = statistics_by_size[number_of_pixels]
                        size_statistics['runs'] += 1
                        size_statistics['total_duration'] += duration
                        if verified:
                            size_statistics['successes'] += 1
        return False, False, None, None, None

    def __print_progress(self, gpu_statistics_by_size, initial_covering_size, innocent_adversarial_example_suspects,
                         last_print_time, number_of_groups_finished_from_initial_covering,
                         waiting_adversarial_example_suspects):
        current = time.time()
        if current - last_print_time < 2:
            return last_print_time
        initial_covering_string = f'progress: ' \
                                  f'{number_of_groups_finished_from_initial_covering}/{int(initial_covering_size)}=' \
                                  f'{100 * number_of_groups_finished_from_initial_covering / initial_covering_size:.3f}%'
        sizes_and_stats = ((size, size_stat['runs'], size_stat['successes'], size_stat['total_duration']) for
                           (size, size_stat) in sorted(gpu_statistics_by_size.items(), reverse=True))
        sizes_string = 'gpupoly: ' + '. '.join(
            f'size={size}, {succ}/{runs}={100 * succ / runs:.3f}%, {1000 * duration / runs:.1f} ms'
            for (size, runs, succ, duration) in sizes_and_stats if runs > 0)
        MILP_string = f'MILP: {len(innocent_adversarial_example_suspects)} verified, ' \
                      f'{len(waiting_adversarial_example_suspects)} waiting'
        print('\r' + initial_covering_string, sizes_string, MILP_string, sep='; ', end='')
        return current

    def __pool_cpu_workers_messages(self, innocent_adversarial_example_suspects, waiting_adversarial_example_suspects):
        for cpu_worker in self.__cpu_workers:
            if cpu_worker.poll():
                pixels, verified, adversarial_example, adversarial_label, timeout = cpu_worker.recv()
                if verified:
                    waiting_adversarial_example_suspects.remove(pixels)
                    innocent_adversarial_example_suspects.add(pixels)
                elif not timeout:
                    self.__handle_adversarial(pixels, adversarial_example, adversarial_label)
                    return True, list(map(int, pixels)), adversarial_example, adversarial_label
        return False, None, None, None

    def __handle_adversarial(self, pixels, adversarial_example, adversarial_label):
        self.__stop_workers()

        print(f'\nAdversarial example found, changing label "{self.__label}" to "{adversarial_label}" '
              f'while perturbing pixels: {pixels}')
        print(adversarial_example)
        print('*******************************************************')

    def __wait_for_cpu_workers_if_needed(self, analyzer_start_time, innocent_adversarial_example_suspects,
                                         waiting_adversarial_example_suspects):
        waiting = 0
        if len(waiting_adversarial_example_suspects) > 0:
            start_time = time.time()
            while len(waiting_adversarial_example_suspects) > 0:
                if time.time() - analyzer_start_time >= self.__timeout:
                    print('\nTimed out!')
                    print('*******************************************************')
                    self.__stop_workers()
                    return True, False, None, None, None, time.time() - start_time

                found_adversarial, pixels, adversarial_example, adversarial_label = self.__pool_cpu_workers_messages(
                    innocent_adversarial_example_suspects, waiting_adversarial_example_suspects)

                if found_adversarial:
                    return False, found_adversarial, pixels, adversarial_example, adversarial_label, time.time() - start_time
            waiting = time.time() - start_time

        self.__stop_workers()
        return False, False, None, None, None, waiting

    def __stop_workers(self):
        for worker in chain(self.__gpu_workers, self.__cpu_workers):
            worker.send('stop')
        for worker in chain(self.__gpu_workers, self.__cpu_workers):
            message = worker.recv()
            while message != 'stopped':
                message = worker.recv()
