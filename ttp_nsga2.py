"""
This is a code using NSGA-II to solve this problem.

Code by Draeson.
"""
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import re
from typing import List, Tuple, Dict
import random

from torch.distributed.tensor import empty


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


# File path
file_path = "data_resources/test-example-n4.txt"

# Load the data from the file
metadata, city_coords, city_items = load_all_data(file_path)

# Print results
print("Metadata:", metadata)
# print("\nCity Coordinates:", city_coords)
city_coords = pd.DataFrame.from_dict(city_coords,orient='index',columns=['x','y'])
print(city_coords)
# print("\nItems in Cities:", city_items)
items_info = pd.DataFrame(city_items,columns=['item_id','city_id','value','weight'])
print(items_info)


class NEGA2:
    def __init__(self, metadata: Dict, coordinates: pd.DataFrame, items: pd.DataFrame):
        self.metadata = metadata
        self.coordinates = coordinates
        self.items = items
        self.dimension = int(metadata['dimension'])
        self.capacity = metadata['capacity_of_knapsack']
        self.min_speed = metadata['min_speed']
        self.max_speed = metadata['max_speed']
        self.renting_ratio = metadata['renting_ratio']

        # GA parameters
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
        return max(self.min_speed,
                   self.max_speed - normalized_weight * (self.max_speed - self.min_speed))

    def evaluate_solution(self, tour: List[int], picking_plan: List[bool]) -> float:
        """Evaluate a complete TTP solution"""
        total_profit = 0
        current_weight = 0
        total_time = 0
        current_city = tour[0]

        # Create a mapping of cities to items and their picking status
        picked_items = {}
        for i, (_, item) in enumerate(self.items.iterrows()):
            if item['city_id'] not in picked_items:
                picked_items[item['city_id']] = []
            picked_items[item['city_id']].append((picking_plan[i], item))

        for next_city in tour[1:] + [tour[0]]:  # Return to start
            # Pick items at current city
            if current_city in picked_items:
                for pick, item in picked_items[current_city]:
                    if pick:
                        current_weight += item['weight']
                        total_profit += item['value']

                        # Check if weight exceeds capacity
                        if current_weight > self.capacity:
                            return float('-inf')  # Invalid solution

            # Travel to next city
            distance = self.calculate_distance(current_city, next_city)
            speed = self.calculate_speed(current_weight)
            total_time += distance / speed

            current_city = next_city

        # Calculate objective value
        objective = total_profit - (self.renting_ratio * total_time)
        return objective

    def create_initial_population(self) -> List[Tuple[List[int], List[bool]]]:
        """Create initial population of tours and picking plans"""
        population = []
        cities = list(range(2, self.dimension + 1))

        for _ in range(self.pop_size):
            # Generate random tour
            tour = cities.copy()
            random.shuffle(tour)
            tour = [1] + tour
            # Generate random picking plan
            picking_plan = [random.random() < 0.5 for _ in range(len(self.items))]

            population.append((tour, picking_plan))

        return population

    def tournament_selection(self, population: List, fitness_values: List[float]) -> Tuple:
        """Select parent using tournament selection"""
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx]

    def crossover_tour(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Order crossover (OX) for tour"""
        p1_rest = parent1[1:]
        p2_rest = parent2[1:]
        size = len(p1_rest)
        # Select substring
        start, end = sorted(random.sample(range(size), 2))

        # Initialize offspring with parent1 substring
        offspring = [0] * size
        off_condition = [True] * size
        offspring[start:end] = p1_rest[start:end]

        # Fill remaining positions from parent2
        current_pos = end
        for city in p2_rest[end:] + p2_rest[:end]:
            if city not in offspring:
                offspring[current_pos] = city
                current_pos = (current_pos + 1) % size
                off_condition[current_pos] = False
                if current_pos == start:
                    break

        return [1] + offspring

    def crossover_picking(self, parent1: List[bool], parent2: List[bool]) -> List[bool]:
        """Uniform crossover for picking plan"""
        return [p1 if random.random() < 0.5 else p2 for p1, p2 in zip(parent1, parent2)]

    def mutate_tour(self, tour: List[int]) -> List[int]:
        """Swap mutation for tour"""
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(1,len(tour)), 2)
            tour[i], tour[j] = tour[j], tour[i]
        return tour

    def mutate_picking(self, picking_plan: List[bool]) -> List[bool]:
        """Bit flip mutation for picking plan"""
        return [not bit if random.random() < self.mutation_rate else bit
                for bit in picking_plan]

    def solve(self) -> Tuple[List[int], List[bool], float]:
        """Main NEGA2 algorithm loop"""
        # Initialize population
        population = self.create_initial_population()
        best_solution = None
        best_fitness = float('-inf')

        for generation in range(self.generations):
            # Evaluate current population
            fitness_values = [self.evaluate_solution(tour, picking)
                              for tour, picking in population]

            # Update best solution
            max_idx = np.argmax(fitness_values)
            if fitness_values[max_idx] > best_fitness:
                best_fitness = fitness_values[max_idx]
                best_solution = population[max_idx]

            # Create new population
            new_population = []

            while len(new_population) < self.pop_size:
                # Select parents
                parent1_tour, parent1_picking = self.tournament_selection(
                    population, fitness_values)
                parent2_tour, parent2_picking = self.tournament_selection(
                    population, fitness_values)

                # Create offspring
                offspring_tour = self.crossover_tour(parent1_tour, parent2_tour)
                offspring_picking = self.crossover_picking(
                    parent1_picking, parent2_picking)

                # Mutate offspring
                offspring_tour = self.mutate_tour(offspring_tour)
                offspring_picking = self.mutate_picking(offspring_picking)

                new_population.append((offspring_tour, offspring_picking))

            population = new_population

            if generation % 10 == 0:
                print(f"Generation {generation}, Best fitness: {best_fitness}")

        return best_solution[0], best_solution[1], best_fitness

# Create solver and solve
solver = NEGA2(metadata, city_coords, items_info)
best_tour, best_picking, best_fitness = solver.solve()

print("\nBest solution found:")
print(f"Tour: {best_tour}")
print("Picked items:")
for i, (picked, item) in enumerate(zip(best_picking, items_info.itertuples())):
    if picked:
        print(f"Item {item.item_id} at city {item.city_id}: value={item.value}, weight={item.weight}")
print(f"Total fitness: {best_fitness}")