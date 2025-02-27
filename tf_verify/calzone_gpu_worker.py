import time
import math
from multiprocessing.connection import Client, Listener
from random import shuffle, sample
import numpy as np
from calzone_utils import normalize
from subprocess import Popen


class LZeroGpuWorker:
    def __init__(self, port, network, means, stds, is_conv, dataset):
        self.__port = port
        self.__network = network
        self.__means = means
        self.__stds = stds
        self.__is_conv = is_conv
        self.__dataset = dataset
        if dataset == 'cifar10':
            self.__number_of_pixels = 1024
        else:
            self.__number_of_pixels = 784

    def work(self):
        address = ('localhost', self.__port)
        with Client(address) as conn:
            # Every iteration of this loop is one image
            message = conn.recv()
            while message != 'terminate':
                image, label, sampling_lower_bound, sampling_upper_bound, repetitions = message
                sampling_counts, sampling_successes, sampling_time = self.__sample(image, label, sampling_lower_bound, sampling_upper_bound, repetitions)
                conn.send((sampling_counts, sampling_successes, sampling_time))
                worker_index, number_of_workers, image, label, geometry, chosen_points, refinement_strategy = conn.recv()
                self.__prove(conn, worker_index, number_of_workers, image, label, geometry, chosen_points, refinement_strategy)
                message = conn.recv()

    def __sample(self, image, label, sampling_lower_bound, sampling_upper_bound, repetitions):
        population = list(range(0, self.__number_of_pixels))
        sampling_counts = [0] * (sampling_upper_bound - sampling_lower_bound + 1)
        sampling_successes = [0] * (sampling_upper_bound - sampling_lower_bound + 1)
        sampling_time = [0] * (sampling_upper_bound - sampling_lower_bound + 1)
        num_zeros = 0
        for size in range(sampling_lower_bound, sampling_upper_bound + 1):
            sampling_counts[size - sampling_lower_bound] = repetitions
            for i in range(0, repetitions):
                pixels = sample(population, size)
                start = time.time()
                verified = self.verify_group(image, label, pixels)
                duration = time.time() - start
                sampling_time[size - sampling_lower_bound] += duration
                if verified:
                    sampling_successes[size - sampling_lower_bound] += 1
            if sampling_successes[size - sampling_lower_bound] == 0:
                num_zeros += 1
            if num_zeros == 10:
                repetitions = 3

        return sampling_counts, sampling_successes, sampling_time

    def __load_refinement_coverings(self, t, refinement_strategy):
        refinement_coverings = dict()
        for size in range(t+1, max(refinement_strategy) + 1):
            covering = []
            with open(f'refinement_coverings/({size},{refinement_strategy[size]},{t}).txt',
                      'r') as coverings_file:
                for line in coverings_file:
                    block = tuple(int(item) for item in line.split(','))
                    covering.append(block)
                refinement_coverings[size] = covering
        return refinement_coverings

    def __prove(self, conn, worker_index, number_of_workers, image, label, geometry, chosen_points, refinement_strategy):
        t = geometry.t
        address = ('localhost', 0)
        with Listener(address) as blocks_listener:
            port = blocks_listener.address[1]
            chosen_string = ','.join((str(point) for point in chosen_points))
            blocks_generator_process = Popen(
                ["../../finite-geometry-coverings-construction/venv/bin/python3",
                 "../../finite-geometry-coverings-construction/main.py",
                 "--q", str(geometry.q),
                 "--m", str(geometry.m),
                 "--t", str(t),
                 '--chosen_points_indices_str', chosen_string,
                 '--num_of_parts', str(number_of_workers),
                 '--part_index', str(worker_index),
                 "--port", str(port)
                 ])
            refinement_coverings = self.__load_refinement_coverings(t, refinement_strategy)
            num_of_blocks = math.floor(geometry.number_of_blocks() / number_of_workers)
            if worker_index < geometry.number_of_blocks() % number_of_workers:
                num_of_blocks += 1
            with blocks_listener.accept() as blocks_conn:
                for _ in range(num_of_blocks):
                    if conn.poll() and conn.recv() == 'stop':
                        blocks_generator_process.kill()
                        conn.send('stopped')
                        return
                    pixels = blocks_conn.recv()
                    pixels = tuple(pixels)
                    if len(pixels) < t:
                        conn.send('next')
                        continue
                    start = time.time()
                    verified = self.verify_group(image, label, pixels)
                    duration = time.time() - start
                    if verified:
                        conn.send((True, len(pixels), duration))
                    else:
                        conn.send((False, len(pixels), duration))
                        if len(pixels) == t:
                            conn.send('adversarial-example-suspect')
                            conn.send(pixels)
                        else:
                            groups_to_verify = self.__break_failed_group(pixels, refinement_coverings[len(pixels)])
                            while len(groups_to_verify) > 0:
                                if conn.poll() and conn.recv() == 'stop':
                                    blocks_generator_process.kill()
                                    conn.send('stopped')
                                    return
                                group_to_verify = groups_to_verify.pop(0)
                                start = time.time()
                                verified = self.verify_group(image, label, group_to_verify)
                                duration = time.time() - start
                                if verified:
                                    conn.send((True, len(group_to_verify), duration))
                                else:
                                    conn.send((False, len(group_to_verify), duration))
                                    if len(group_to_verify) == t:
                                        conn.send('adversarial-example-suspect')
                                        conn.send(group_to_verify)
                                    else:
                                        groups_to_verify = self.__break_failed_group(group_to_verify, refinement_coverings[len(group_to_verify)]) + groups_to_verify
                    conn.send('next')
                blocks_conn.send('done')
                message = blocks_conn.recv()
                if message != 'done':
                    raise Exception('This should not happen')
        conn.send(f"done")
        message = conn.recv()
        if message != 'stop':
            raise Exception('This should not happen')
        conn.send('stopped')

    def __break_failed_group(self, pixels, covering):
        permutation = list(pixels)
        shuffle(permutation)
        return [tuple(sorted(permutation[item] for item in block)) for block in covering]

    def verify_group(self, image, label, pixels_group):
        specLB = np.copy(image)
        specUB = np.copy(image)
        for pixel_index in self.get_indexes_from_pixels(pixels_group):
            specLB[pixel_index] = 0
            specUB[pixel_index] = 1
        normalize(specLB, self.__means, self.__stds, self.__dataset, 'gpupoly', self.__is_conv)
        normalize(specUB, self.__means, self.__stds, self.__dataset, 'gpupoly', self.__is_conv)

        return self.__network.test(specLB, specUB, label)

    def get_indexes_from_pixels(self, pixels_group):
        if self.__dataset != 'cifar10':
            return pixels_group
        indexes = []
        for pixel in pixels_group:
            indexes.append(pixel * 3)
            indexes.append(pixel * 3 + 1)
            indexes.append(pixel * 3 + 2)
        return indexes