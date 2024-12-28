import pandas as pd
import numpy as np
import random
from typing import List, Tuple, Dict



def load_all_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Data containers
    metadata = {}
    city_coordinates = {}
    items_in_cities = []

    # Locate sections
    node_section_start = None
    item_section_start = None

    for idx, line in enumerate(lines):
        if "NODE_COORD_SECTION" in line:
            node_section_start = idx + 1
        elif "ITEMS SECTION" in line:
            item_section_start = idx + 1

    # Parse metadata
    for line in lines[:node_section_start - 1]:
        if ":" in line:
            key, value = line.split(":", 1)
            if key in ['DIMENSION','NUMBER OF ITEMS','CAPACITY OF KNAPSACK','MIN SPEED','MAX SPEED','RENTING RATIO']:
                metadata[key.strip().lower().replace(" ", "_")] = float(value.strip())
            else:
                metadata[key.strip().lower().replace(" ", "_")] = value.strip()

    # Parse city coordinates
    for line in lines[node_section_start:item_section_start - 1]:
        parts = line.strip().split()
        city_id = int(parts[0])
        x, y = map(float, parts[1:])
        city_coordinates[city_id] = (x, y)

    # Parse items
    for line in lines[item_section_start:]:
        parts = line.strip().split()
        # if parts is empty then continue
        if not parts:
            continue
        item_id = int(parts[0])
        value = int(parts[1])  # Value (Profit)
        weight = int(parts[2])  # Weight
        city_id = int(parts[3])  # Related City ID

        items_in_cities.append([item_id,city_id, value, weight])


    return metadata, city_coordinates, items_in_cities

class NSGA2:
    def __init__(self, metadata: Dict, coordinates: pd.DataFrame, items: pd.DataFrame):
        self.metadata = metadata
        self.coordinates = coordinates
        self.items = items
        self.dimension = int(metadata['dimension'])
        self.capacity = metadata['capacity_of_knapsack']
        self.min_speed = metadata['min_speed']
        self.max_speed = metadata['max_speed']
        self.renting_ratio = metadata['renting_ratio']

        # NSGA-II parameters
        self.pop_size = 50
        self.generations = 100
        self.mutation_rate = 0.1
        self.tournament_size = 3

    def calculate_distance(self, city1: int, city2: int) -> float:
        """Calculate Euclidean distance between two cities"""
        x1, y1 = self.coordinates.loc[city1, ['x', 'y']]
        x2, y2 = self.coordinates.loc[city2, ['x', 'y']]
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def calculate_speed(self, current_weight: float) -> float:
        """Calculate current speed based on weight"""
        normalized_weight = current_weight / self.capacity
        return max(self.min_speed, self.max_speed - normalized_weight * (self.max_speed - self.min_speed))

    def evaluate_solution(self, tour: List[int], picking_plan: List[bool]) -> Tuple[float, float]:
        """Evaluate a complete TTP solution and return profit and time"""
        total_profit = 0
        current_weight = 0
        total_time = 0
        current_city = tour[0]

        picked_items = {}
        for i, (_, item) in enumerate(self.items.iterrows()):
            if item['city_id'] not in picked_items:
                picked_items[item['city_id']] = []
            picked_items[item['city_id']].append((picking_plan[i], item))

        for next_city in tour[1:] + [tour[0]]:  # Return to start
            if current_city in picked_items:
                for pick, item in picked_items[current_city]:
                    if (current_weight + item['weight']) > self.capacity:
                        picked_items[current_city][0] = (False,) + picked_items[current_city][0][1:]
                    else:
                        current_weight += item['weight']
                        total_profit += item['value']

            distance = self.calculate_distance(current_city, next_city)
            speed = self.calculate_speed(current_weight)
            total_time += distance / speed

            current_city = next_city

        return total_profit, total_time
    def dominated(self,a1,b1,a2,b2) -> bool:
        return a1 >= b1 and a2 <= b2 and (a1 > b1 or a2 < b2)

    def fast_nondominated_sort(self, population: List[Tuple[List[int], List[bool]]], fitnesses: List[Tuple[float, float]]):
        """Perform fast non-dominated sorting and return Pareto fronts"""
        fronts = [[]]
        domination_count = [0] * len(population)
        dominated_solutions = [[] for _ in range(len(population))]

        for p in range(len(population)):
            for q in range(len(population)):
                if self.dominated(fitnesses[p][0],fitnesses[q][0],fitnesses[p][1],fitnesses[q][1]):
                    dominated_solutions[p].append(q)
                elif self.dominated(fitnesses[q][0],fitnesses[p][0],fitnesses[q][1],fitnesses[p][1]):
                    domination_count[p] += 1

            if domination_count[p] == 0:
                fronts[0].append(p)

        while len(fronts[-1]) > 0:
            next_front = []
            for p in fronts[-1]:
                for q in dominated_solutions[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        next_front.append(q)
            fronts.append(next_front)

        return fronts[:-1]

    def calculate_crowding_distance(self, front: List[int], fitnesses: List[Tuple[float, float]]) -> List[float]:
        """Calculate crowding distance for a Pareto front"""
        distance = [[0] * len(front)]*2
        final_distance = [0] * len(front)
        num_objectives = len(fitnesses[0])

        for i in range(num_objectives):
            sorted_front = sorted(range(len(front)), key=lambda x: fitnesses[front[x]][i])
            distance[i][sorted_front[0]] = float('inf')
            distance[i][sorted_front[-1]] = float('inf')

            for j in range(1, len(sorted_front) - 1):
                distance[i][sorted_front[j]] += (fitnesses[front[sorted_front[j + 1]]][i] - fitnesses[front[sorted_front[j - 1]]][i])
        for i in range(len(final_distance)):
            final_distance[i] = distance[0][i] + distance[1][i]

        return final_distance

    def tournament_selection(self, population: List, fitnesses: List[Tuple[float, float]], ranks: List[int], distances: List[float]) -> Tuple:
        """Select parent using Pareto rank and crowding distance"""
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        sorted_indices = sorted(tournament_indices, key=lambda x: (ranks[x], -distances[x]))
        return population[sorted_indices[0]]

    def calculate_Euclidean_distence(self,front,fitness):
        distance = [0] * len(front)
        for i,idx in enumerate(front):
            distance[i]=np.linalg.norm([-fitness[idx][0],fitness[idx][1]])
        return distance

    def calculate_the_best_solution(self,front,population,fitness,best_tour,best_picking,best_profit,best_time):
        distances = self.calculate_Euclidean_distence(front, fitness)
        sorted_distances = sorted(range(len(front)), key=lambda x: distances[x])
        sorted_front = [front[i] for i in sorted_distances]
        idx = sorted_front[0]
        profit = fitness[idx][0]
        time = fitness[idx][1]
        tour = population[idx][0]
        picking = population[idx][1]
        return (tour,picking,profit,time) if self.dominated(profit,best_profit,time,best_time) else (best_tour,best_picking,best_profit,best_time)

    def solve(self) -> List[Tuple[float, float]]:
        """Main NSGA-II algorithm loop"""
        population = self.create_initial_population()
        best_idx= 0
        best_tour = population[best_idx][0]
        best_picking = population[best_idx][1]
        best_profit = 0
        best_time = float('inf')

        a = []
        b = []
        for generation in range(self.generations):
            fitnesses = [self.evaluate_solution(tour, picking) for tour, picking in population]
            fronts = self.fast_nondominated_sort(population, fitnesses)

            ranks = [0] * len(population)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank

            distances = [0] * len(population)
            for front in fronts:
                front_distances = self.calculate_crowding_distance(front, fitnesses)
                for i, idx in enumerate(front):
                    distances[idx] = front_distances[i]

            best_tour,best_picking,best_profit,best_time = self.calculate_the_best_solution(
                fronts[0], population,fitnesses,best_tour,best_picking,best_profit,best_time
            )

            new_population = []
            new_population.append(population[fronts[0][0]])

            while len(new_population) < self.pop_size:
                parent1 = self.tournament_selection(population, fitnesses, ranks, distances)
                parent2 = self.tournament_selection(population, fitnesses, ranks, distances)

                offspring_tour = self.crossover_tour(parent1[0], parent2[0])
                offspring_picking = self.crossover_picking(parent1[1], parent2[1])

                offspring_tour = self.mutate_tour(offspring_tour)
                offspring_picking = self.mutate_picking(offspring_picking)

                new_population.append((offspring_tour, offspring_picking))

            new_population_fitnesses = [self.evaluate_solution(tour, picking) for tour, picking in new_population]
            new_fronts = self.fast_nondominated_sort(new_population, new_population_fitnesses)

            next_population = []
            for front in new_fronts:
                if len(next_population) + len(front) <= self.pop_size:
                    next_population.extend(front)
                else:
                    distances = self.calculate_crowding_distance(front, new_population_fitnesses)
                    sorted_distances = sorted(range(len(front)), key=lambda x: -distances[x])
                    sorted_front = [front[i] for i in sorted_distances]
                    next_population.extend(sorted_front[:self.pop_size - len(next_population)])
                    break

            population = [new_population[i] for i in next_population]

            if generation % 10 == 0:
                best_tour, best_picking, best_profit, best_time = self.calculate_the_best_solution(
                    fronts[0], population, fitnesses, best_tour, best_picking, best_profit, best_time
                )
                print(f"Generation {generation} complete,Best profit:{best_profit},Best time:{best_time}")


        # final_fitnesses = [self.evaluate_solution(tour, picking) for tour, picking in population]
        # return final_fitnesses
        return best_tour, best_picking, best_profit, best_time

    def create_initial_population(self) -> List[Tuple[List[int], List[bool]]]:
        """Create initial population of tours and picking plans"""
        population = []
        cities = list(range(2, self.dimension + 1))

        for _ in range(self.pop_size):
            tour = cities.copy()
            random.shuffle(tour)
            tour = [1] + tour
            picking_plan = [random.random() < 0.5 for _ in range(len(self.items))]
            population.append((tour, picking_plan))

        return population

    def crossover_tour(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Order crossover (OX) for tour"""
        p1_rest = parent1[1:]
        p2_rest = parent2[1:]
        size = len(p1_rest)
        start, end = sorted(random.sample(range(size), 2))

        offspring = [0] * size
        offspring[start:end] = p1_rest[start:end]

        current_pos = end
        for city in p2_rest[end:] + p2_rest[:end]:
            if city not in offspring:
                offspring[current_pos] = city
                current_pos = (current_pos + 1) % size

        return [1] + offspring

    def crossover_picking(self, parent1: List[bool], parent2: List[bool]) -> List[bool]:
        """Uniform crossover for picking plan"""
        return [p1 if random.random() < 0.5 else p2 for p1, p2 in zip(parent1, parent2)]

    def mutate_tour(self, tour: List[int]) -> List[int]:
        """Swap mutation for tour"""
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(1, len(tour)), 2)
            tour[i], tour[j] = tour[j], tour[i]
        return tour

    def mutate_picking(self, picking_plan: List[bool]) -> List[bool]:
        """Bit flip mutation for picking plan"""
        return [not bit if random.random() < self.mutation_rate else bit for bit in picking_plan]


# Example usage
if __name__ == "__main__":
    file_path = "data_resources/a280-n279.txt"
    metadata, city_coords, city_items = load_all_data(file_path)
    city_coords = pd.DataFrame.from_dict(city_coords, orient='index', columns=['x', 'y'])
    items_info = pd.DataFrame(city_items, columns=['item_id', 'city_id', 'value', 'weight'])

    solver = NSGA2(metadata, city_coords, items_info)
    best_tour, best_picking, best_profit, best_time = solver.solve()

    print(f"Best profit:{best_profit},Best time:{best_time}")
    # print("\nFinal Pareto front:")
    # for profit, time in final_front:
    #     print(f"Profit: {profit}, Time: {time}")
