from dataclasses import dataclass


def q_binomial_coefficient(a, b, q):
    if b == 1 or a == b + 1:
        return sum(q ** i for i in range(a))
    return q_binomial_coefficient(a-1, b, q) * (q ** b) + q_binomial_coefficient(a-1, b-1, q)


@dataclass
class Geometry:
    q: int
    m: int
    t: int

    def number_of_points(self) -> int:
        return int((self.q ** (self.m + 1) - 1) / (self.q - 1))

    def block_size(self) -> int:
        return int((self.q ** self.t - 1) / (self.q - 1))

    def number_of_blocks(self) -> int:
        return q_binomial_coefficient(self.m + 1, self.t, self.q)


def get_possible_geometries(num_of_pixels, t):
    with open('coverd_primes.txt', 'r') as f:
        for line in f:
            q = int(line)
            if num_of_pixels / q < 1:
                break

            m = t
            while num_of_pixels * (q ** t - 1) / (q ** (m+1) - 1) >= t:
                geometry = Geometry(q, m, t)
                if geometry.number_of_points() >= num_of_pixels:
                    yield geometry
                m += 1

