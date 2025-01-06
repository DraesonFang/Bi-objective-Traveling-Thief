import pandas as pd
import numpy as np
import random
import os
from typing import List, Tuple, Dict

def load_all_data(file_path: str):
    """
    从 TTP 格式文件里加载元数据、城市坐标、物品列表。
    返回 (metadata, city_coordinates, items_in_cities) 三个结构。
    """
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
        line_upper = line.upper()
        if "NODE_COORD_SECTION" in line_upper:
            node_section_start = idx + 1
        elif "ITEMS SECTION" in line_upper:
            item_section_start = idx + 1

    # Parse metadata
    for line in lines[:node_section_start - 1]:
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().lower().replace(" ", "_")
            if key in ['dimension','number_of_items','capacity_of_knapsack','min_speed','max_speed','renting_ratio']:
                metadata[key] = float(value.strip())
            else:
                # 其他文字类信息
                metadata[key] = value.strip()

    # Parse city coordinates
    for line in lines[node_section_start:item_section_start - 1]:
        parts = line.strip().split()
        if not parts:
            continue
        city_id = int(parts[0])
        x, y = map(float, parts[1:])
        city_coordinates[city_id] = (x, y)

    # Parse items
    for line in lines[item_section_start:]:
        parts = line.strip().split()
        if not parts:
            continue
        item_id = int(parts[0])
        value = int(parts[1])   # Profit
        weight = int(parts[2])  # Weight
        city_id = int(parts[3]) # Related City ID
        items_in_cities.append([item_id, city_id, value, weight])

    return metadata, city_coordinates, items_in_cities

class NSGA2_TTP:
    def __init__(
        self,
        metadata: Dict,
        coordinates: pd.DataFrame,
        items: pd.DataFrame,
        pop_size: int = 100,
        generations: int = 10,
        mutation_rate: float = 0.2,
        tournament_size: int = 3
    ):
        self.metadata = metadata
        self.coordinates = coordinates
        self.items = items
        self.dimension = int(metadata['dimension'])
        self.capacity = metadata['capacity_of_knapsack']
        self.min_speed = metadata['min_speed']
        self.max_speed = metadata['max_speed']
        self.renting_ratio = metadata['renting_ratio']

        # NSGA-II parameters
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

        # 预计算城市间距离
        self.distance_matrix = np.zeros((self.dimension + 1, self.dimension + 1))
        coords = np.array([
            [self.coordinates.loc[i, 'x'], self.coordinates.loc[i, 'y']] 
            for i in range(1, self.dimension + 1)
        ])
        for i in range(self.dimension):
            dists = np.sqrt(np.sum((coords - coords[i])**2, axis=1))
            # 注意城市编号从1开始
            self.distance_matrix[i+1, 1:] = dists
            self.distance_matrix[1:, i+1] = dists

        # 生成每个城市最近邻列表（可用于贪心或邻域操作）
        self.nearest_neighbors = {i: [] for i in range(1, self.dimension + 1)}
        for i in range(1, self.dimension + 1):
            dist_row = self.distance_matrix[i, 1:]
            # 取前10个最近的城市（去掉自己）
            sorted_idx = np.argsort(dist_row)
            # 这里 sorted_idx 的值是 0-based，而城市是 1-based
            nn = []
            for idx_val in sorted_idx:
                c = idx_val + 1
                if c != i:
                    nn.append(c)
                if len(nn) >= 10:
                    break
            self.nearest_neighbors[i] = nn

        # 预处理物品信息
        self.items_by_city = {}
        self.item_weights = np.zeros(len(items))
        self.item_values = np.zeros(len(items))
        self.item_cities = np.zeros(len(items), dtype=int)
        self.value_weight_ratios = np.zeros(len(items))

        total_value = 0
        total_weight = 0
        for i, row in items.iterrows():
            cid = row['city_id']
            val = row['value']
            w = row['weight']
            self.item_cities[i] = cid
            self.item_values[i] = val
            self.item_weights[i] = w
            self.value_weight_ratios[i] = (val / w) if w > 0 else 0
            if cid not in self.items_by_city:
                self.items_by_city[cid] = []
            self.items_by_city[cid].append(i)
            total_value += val
            total_weight += w

        # 平均价值密度（仅做一个参考）
        if total_weight == 0:
            total_weight = 1e-8
        self.avg_value_density = total_value / total_weight

        # 对每个城市的物品按照价值密度排序（供贪心拾取时参考）
        for cid in self.items_by_city:
            self.items_by_city[cid].sort(
                key=lambda idx: self.value_weight_ratios[idx],
                reverse=True
            )

    def calculate_distance(self, c1: int, c2: int) -> float:
        return self.distance_matrix[c1, c2]

    def calculate_speed(self, current_weight: float) -> float:
        """
        根据背包当前重量算出移动速度。
        """
        w_ratio = current_weight / self.capacity
        return max(
            self.min_speed,
            self.max_speed - w_ratio * (self.max_speed - self.min_speed)
        )

    def evaluate_solution(self, tour: List[int], picking_plan: List[bool]) -> Tuple[float, float]:
        """
        评估TTP解：给定路线和物品拾取计划
        返回 (profit, time)
           profit 越大越好
           time   越小越好
        """
        total_profit = 0.0
        current_weight = 0.0
        total_time = 0.0
        current_city = tour[0]

        # 找到被选中的物品
        picked_indices = np.where(picking_plan)[0]
        if len(picked_indices) == 0:
            # 若无物品，则时间 = 线路距离 / max_speed
            dist = 0.0
            for i in range(len(tour) - 1):
                dist += self.distance_matrix[tour[i], tour[i+1]]
            dist += self.distance_matrix[tour[-1], tour[0]]
            return (0.0, dist / self.max_speed)

        # 将被选中的物品按城市进行分组
        city_items_map = {}
        for idx in picked_indices:
            cid = self.item_cities[idx]
            if cid not in city_items_map:
                city_items_map[cid] = []
            city_items_map[cid].append(idx)

        # 模拟走一圈
        for next_city in tour[1:] + [tour[0]]:
            # 到达城市后，尝试拾取物品
            if current_city in city_items_map:
                remain_cap = self.capacity - current_weight
                if remain_cap > 0:
                    # 这些物品
                    item_list = city_items_map[current_city]
                    # 简单按价值密度或事先排序都可，这里直接顺序
                    # 若要严格挑选就 sort 一下
                    for idx in item_list:
                        w = self.item_weights[idx]
                        v = self.item_values[idx]
                        if w <= remain_cap:
                            total_profit += v
                            current_weight += w
                            remain_cap -= w

            # 移动到下一城市
            dist = self.distance_matrix[current_city, next_city]
            speed = self.calculate_speed(current_weight)
            total_time += dist / speed
            current_city = next_city

        return (total_profit, total_time)

    # ---------------------- 非支配排序 & 拥挤距离 ----------------------
    def dominates(self, a: Tuple[float, float], b: Tuple[float, float]) -> bool:
        """
        a是否支配b？(profit, time):
           profit越大越好, time越小越好
        a支配b条件:
           a.profit >= b.profit
           a.time   <= b.time
           且至少一个严格不等
        """
        return (
            (a[0] >= b[0]) and
            (a[1] <= b[1]) and
            ((a[0] > b[0]) or (a[1] < b[1]))
        )

    def fast_nondominated_sort(
        self,
        pop: List[Tuple[List[int], List[bool]]],
        fitnesses: List[Tuple[float, float]]
    ) -> List[List[int]]:
        """
        标准的快速非支配排序。返回一个列表 fronts，
        每个front是一组个体的索引。
        """
        n = len(pop)
        dominated_solutions = [[] for _ in range(n)]  # S(i)
        domination_count = [0]*n  # n(i)

        # 计算支配关系
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self.dominates(fitnesses[i], fitnesses[j]):
                    dominated_solutions[i].append(j)
                elif self.dominates(fitnesses[j], fitnesses[i]):
                    domination_count[i] += 1

        # 第一层：找出那些未被任何解支配的
        fronts = [[]]
        for i in range(n):
            if domination_count[i] == 0:
                fronts[0].append(i)

        # 依次生成后续各层
        k = 0
        while len(fronts[k]) > 0:
            next_front = []
            for i in fronts[k]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            k += 1
            fronts.append(next_front)

        return fronts[:-1]

    def calculate_crowding_distance(
        self,
        front_indices: List[int],
        fitnesses: List[Tuple[float, float]]
    ) -> List[float]:
        """
        计算给定非支配层(由一组索引表示)的拥挤距离。只考虑(profit, time)两个目标。
        返回与front_indices等长的列表，对应每个个体的距离值。
        """
        if len(front_indices) == 0:
            return []
        if len(front_indices) == 1:
            return [float('inf')]

        # 分别对 profit, time 排序
        # 我们的目标：profit越大越好，所以排序时是升序，但要注意计算时区分
        # time越小越好，所以也是常规升序。
        distances = [0.0]*len(front_indices)

        # --- profit ---
        sorted_by_profit = sorted(
            range(len(front_indices)),
            key=lambda i: fitnesses[front_indices[i]][0]  # profit
        )
        min_profit = fitnesses[front_indices[sorted_by_profit[0]]][0]
        max_profit = fitnesses[front_indices[sorted_by_profit[-1]]][0]
        profit_range = max_profit - min_profit if max_profit != min_profit else 1e-9

        # 端点设为无限
        distances[sorted_by_profit[0]] = float('inf')
        distances[sorted_by_profit[-1]] = float('inf')
        for i in range(1, len(front_indices) - 1):
            prev_idx = front_indices[sorted_by_profit[i-1]]
            next_idx = front_indices[sorted_by_profit[i+1]]
            cur_idx = front_indices[sorted_by_profit[i]]
            distances[sorted_by_profit[i]] += (
                (fitnesses[next_idx][0] - fitnesses[prev_idx][0]) / profit_range
            )

        # --- time ---
        sorted_by_time = sorted(
            range(len(front_indices)),
            key=lambda i: fitnesses[front_indices[i]][1]  # time
        )
        min_time = fitnesses[front_indices[sorted_by_time[0]]][1]
        max_time = fitnesses[front_indices[sorted_by_time[-1]]][1]
        time_range = max_time - min_time if max_time != min_time else 1e-9

        distances[sorted_by_time[0]] = float('inf')
        distances[sorted_by_time[-1]] = float('inf')
        for i in range(1, len(front_indices) - 1):
            prev_idx = front_indices[sorted_by_time[i-1]]
            next_idx = front_indices[sorted_by_time[i+1]]
            cur_idx = front_indices[sorted_by_time[i]]
            distances[sorted_by_time[i]] += (
                (fitnesses[next_idx][1] - fitnesses[prev_idx][1]) / time_range
            )

        return distances

    # ---------------------- NSGA-II流程 ----------------------
    def tournament_selection(
        self,
        population: List[Tuple[List[int], List[bool]]],
        fitnesses: List[Tuple[float, float]],
        ranks: List[int],
        crowding: List[float]
    ) -> Tuple[List[int], List[bool]]:
        """
        锦标赛选择：在随机选出的若干个体中，先比较rank，再比较拥挤距离
        rank越低越好，距离越大越好。
        """
        candidates = random.sample(range(len(population)), self.tournament_size)
        # 按 (rank, -distance) 排序
        candidates.sort(key=lambda idx: (ranks[idx], -crowding[idx]))
        return population[candidates[0]]

    def crossover_tour(self, p1: List[int], p2: List[int]) -> List[int]:
        """
        对路线做次序交叉 (Order Crossover, OX)。
        注意去掉首城市 1 后再做，最后再加回 1。
        """
        parent1 = p1[1:]  # 除去 depot
        parent2 = p2[1:]
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        offspring = [None]*size
        # copy the slice from parent1
        offspring[start:end] = parent1[start:end]
        # fill from parent2
        pos = end
        for city in parent2[end:] + parent2[:end]:
            if city not in offspring:
                offspring[pos] = city
                pos = (pos + 1) % size
        return [1] + offspring

    def crossover_picking(self, p1: List[bool], p2: List[bool]) -> List[bool]:
        """
        对拾取计划做一致交叉(Uniform crossover)。
        """
        offspring = []
        for x, y in zip(p1, p2):
            if random.random() < 0.5:
                offspring.append(x)
            else:
                offspring.append(y)
        return offspring

    def mutate_tour(self, tour: List[int]) -> List[int]:
        """
        对路线做一定的扰动。
        包括 2-opt 翻转、插入突变、与最近邻交换等多种可能。
        """
        if random.random() < self.mutation_rate:
            r = random.random()
            if r < 0.4:
                # 2-opt
                i, j = sorted(random.sample(range(1, len(tour)), 2))
                tour[i:j] = reversed(tour[i:j])
            elif r < 0.7:
                # 插入突变
                i, j = random.sample(range(1, len(tour)), 2)
                city = tour.pop(i)
                tour.insert(j, city)
            else:
                # 最近邻交换
                i = random.randint(1, len(tour)-1)
                city = tour[i]
                nn_list = self.nearest_neighbors[city]
                if nn_list:
                    neighbor_city = random.choice(nn_list)
                    if neighbor_city in tour:
                        j = tour.index(neighbor_city)
                        tour[i], tour[j] = tour[j], tour[i]
        return tour

    def mutate_picking(self, picking: List[bool]) -> List[bool]:
        """
        对拾取计划做一定的扰动：城市批量翻转 or 价值密度导向突变。
        """
        if random.random() < self.mutation_rate:
            r = random.random()
            if r < 0.5:
                # 选一个城市，将其中物品翻转
                cid = random.randint(1, self.dimension)
                for i in range(len(picking)):
                    if self.item_cities[i] == cid:
                        picking[i] = not picking[i]
            else:
                # 价值密度导向
                for i in range(len(picking)):
                    if random.random() < 0.1:  # 10%的概率尝试翻转
                        if self.value_weight_ratios[i] >= self.avg_value_density*0.8:
                            picking[i] = True
                        else:
                            picking[i] = False
        return picking

    # ---------------------- 2-opt 优化 ----------------------
    def apply_2opt(self, tour: List[int], max_iterations: int = 5) -> List[int]:
        """对给定路径进行简单2-opt搜索。"""
        best_tour = tour[:]
        best_dist = self.calculate_tour_distance(tour)
        for _ in range(max_iterations):
            improved = False
            for i in range(1, len(best_tour) - 2):
                for j in range(i+1, len(best_tour) - 1):
                    if j - i == 1:
                        continue
                    new_tour = best_tour[:i] + list(reversed(best_tour[i:j])) + best_tour[j:]
                    new_dist = self.calculate_tour_distance(new_tour)
                    if new_dist < best_dist:
                        best_tour = new_tour
                        best_dist = new_dist
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return best_tour

    def calculate_tour_distance(self, tour: List[int]) -> float:
        dist = 0.0
        for i in range(len(tour)-1):
            dist += self.distance_matrix[tour[i], tour[i+1]]
        dist += self.distance_matrix[tour[-1], tour[0]]
        return dist

    # ---------------------- 主流程：solve ----------------------
    def solve(self):
        """
        标准 NSGA-II 主循环：
          1) 初始化种群 P0
          2) For each generation:
             - 根据 P_t 选择、交叉、变异 => Q_t
             - R_t = P_t ∪ Q_t
             - fast_nondominated_sort(R_t) + calculate_crowding_distance
             - 选出 P_{t+1}
        返回一个最终解 (tour, picking, profit, time) 仅作示范。
        """
        # Step 0: 初始种群
        population = self.create_initial_population()
        fitnesses = [self.evaluate_solution(t, p) for (t, p) in population]

        # 记录进化过程
        best_records = []

        for gen in range(self.generations):
            # 1) 用当前种群产生后代
            offspring = []
            while len(offspring) < self.pop_size:
                # 锦标赛选俩
                ranks, crowding = self.get_ranks_and_crowding(population, fitnesses)
                parent1 = self.tournament_selection(population, fitnesses, ranks, crowding)
                parent2 = self.tournament_selection(population, fitnesses, ranks, crowding)

                # 交叉
                child_tour = self.crossover_tour(parent1[0], parent2[0])
                child_picking = self.crossover_picking(parent1[1], parent2[1])
                # 变异
                child_tour = self.mutate_tour(child_tour)
                child_picking = self.mutate_picking(child_picking)
                offspring.append((child_tour, child_picking))
            offspring_fitnesses = [self.evaluate_solution(t, p) for (t, p) in offspring]

            # 2) 合并种群
            combined_pop = population + offspring
            combined_fit = fitnesses + offspring_fitnesses

            # 3) 非支配排序
            fronts = self.fast_nondominated_sort(combined_pop, combined_fit)

            # 4) 依次填充下一代
            new_population = []
            new_fitnesses = []
            front_idx = 0
            while front_idx < len(fronts) and len(new_population) + len(fronts[front_idx]) <= self.pop_size:
                for idx in fronts[front_idx]:
                    new_population.append(combined_pop[idx])
                    new_fitnesses.append(combined_fit[idx])
                front_idx += 1

            if front_idx < len(fronts) and len(new_population) < self.pop_size:
                # 还需要从下一个front里截取一部分
                remaining_spots = self.pop_size - len(new_population)
                # 计算这个front里各解的拥挤距离
                front_crowding = self.calculate_crowding_distance(fronts[front_idx], combined_fit)

                # 将 front[front_idx] 的索引按照拥挤距离从大到小排序
                # 注意 sorted() 里要用 front_crowding 的值
                # front_crowding[i] 对应 fronts[front_idx][i] 的距离
                # 因此先把 (解索引, 拥挤距离) 做映射
                index_and_dist = list(zip(fronts[front_idx], front_crowding))
                # 按距离逆序排
                index_and_dist.sort(key=lambda x: x[1], reverse=True)

                for i in range(remaining_spots):
                    pick_idx = index_and_dist[i][0]
                    new_population.append(combined_pop[pick_idx])
                    new_fitnesses.append(combined_fit[pick_idx])

            # 更新population
            population = new_population
            fitnesses = new_fitnesses

            if gen % 10 == 0:
                current_fitness = [fitnesses[i] for i,_ in enumerate(new_population)]
                print(f"Generation {gen}")
                print(f"Number of solutions in first front: {len(fronts[0])}")
                print(f"Best profit: {max(f[0] for f in current_fitness)}")
                print(f"Best time: {min(-f[1] for f in current_fitness)}")

        # Get final Pareto front
        final_fitness = [self.evaluate_solution(t, p) for (t, p) in population]
        final_fronts = self.fast_nondominated_sort(population, final_fitness)

        return population,final_fitness,final_fronts

    def create_initial_population(self) -> List[Tuple[List[int], List[bool]]]:
        """
        构造初始种群，一部分使用最近邻+贪心的启发式，一部分纯随机。
        """
        population = []
        # 城市编号(除去1)
        other_cities = list(range(2, self.dimension+1))

        # 用最近邻/贪心生成约 1/3
        for _ in range(self.pop_size // 3):
            tour = [1]
            unvisited = set(other_cities)
            current = 1

            while unvisited:
                # 从当前城市最近邻里选一个未访问的
                candidates = [c for c in self.nearest_neighbors[current] if c in unvisited]
                if candidates:
                    nxt = random.choice(candidates)
                else:
                    # 如果最近邻都访问了，则从unvisited里选最近的
                    nxt = min(unvisited, key=lambda c: self.distance_matrix[current, c])
                tour.append(nxt)
                unvisited.remove(nxt)
                current = nxt

            # 对tour做一个小的2-opt
            if len(tour) > 3:
                tour = self.apply_2opt(tour, max_iterations=5)

            # 贪心 picking
            picking_plan = np.zeros(len(self.items), dtype=bool)
            capacity_left = self.capacity
            # 先按价值密度排序
            all_items = list(range(len(self.items)))
            all_items.sort(key=lambda i: self.value_weight_ratios[i], reverse=True)
            for i in all_items:
                if self.value_weight_ratios[i] >= 0.8 * self.avg_value_density:
                    if self.item_cities[i] in tour:
                        w = self.item_weights[i]
                        if w <= capacity_left:
                            picking_plan[i] = True
                            capacity_left -= w

            population.append((tour, picking_plan))

        # 其余人口用随机生成
        while len(population) < self.pop_size:
            # 随机路线
            random_tour = [1] + random.sample(other_cities, len(other_cities))
            if len(random_tour) > 3:
                random_tour = self.apply_2opt(random_tour, max_iterations=3)

            # 随机 picking
            picking_plan = np.zeros(len(self.items), dtype=bool)
            # 可以选择 0~30% 的物品
            n_items = int(0.3 * len(self.items))
            candidate_items = random.sample(range(len(self.items)), n_items)
            for i in candidate_items:
                if random.random() < 0.5:
                    picking_plan[i] = True

            population.append((random_tour, picking_plan))

        return population

    def get_ranks_and_crowding(self,
        population: List[Tuple[List[int], List[bool]]],
        fitnesses: List[Tuple[float, float]]
    ):
        """
        对当前population做非支配排序，然后给出每个个体的rank和拥挤距离。
        返回 ranks, crowding 两个列表，与population等长。
        """
        fronts = self.fast_nondominated_sort(population, fitnesses)
        ranks = [0]*len(population)
        crowding = [0]*len(population)

        for rank_idx, front in enumerate(fronts):
            if len(front) == 1:
                # 单个解，距离设为inf
                ranks[front[0]] = rank_idx
                crowding[front[0]] = float('inf')
            else:
                # 计算这个front的crowding
                dists = self.calculate_crowding_distance(front, fitnesses)
                for i, idx in enumerate(front):
                    ranks[idx] = rank_idx
                    crowding[idx] = dists[i]
        return ranks, crowding

    def get_best_in_population(
        self,
        population: List[Tuple[List[int], List[bool]]],
        fitnesses: List[Tuple[float, float]]
    ) -> Tuple[int, Tuple[float, float]]:
        """
        在给定群体里找一个最优解并返回 (解的索引, (profit, time)).
        这里的策略是：在(-profit, time)空间上，找欧几里得距离最小者。
        你也可以换成别的多目标度量。
        """
        best_idx = 0
        best_dist = float('inf')
        for i, (profit, time_) in enumerate(fitnesses):
            # 将 profit 视为负、time 为正
            # 计算到(0,0)距离 => profit越大=> -profit越小 => 距离越小
            d = np.sqrt(( -profit )**2 + ( time_ )**2)
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx, fitnesses[best_idx]


# ------------------ main 例子 ------------------
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # file_path = os.path.join(current_dir, "..", "data_resoures", "a280-n279.txt")
    file_path = "data_resources/a280-n279.txt"
    metadata, city_coords, city_items = load_all_data(file_path)

    # 转成 DataFrame
    city_coords_df = pd.DataFrame.from_dict(city_coords, orient='index', columns=['x','y'])
    items_df = pd.DataFrame(city_items, columns=['item_id','city_id','value','weight'])

    # 创建并运行NSGA2
    solver = NSGA2_TTP(metadata, city_coords_df, items_df,
                       pop_size=100, generations=3,
                       mutation_rate=0.2, tournament_size=3)

    population,final_fitness,final_fronts = solver.solve()
    pareto_front = [(population[idx][0], population[idx][1], final_fitness[idx])
                    for idx in final_fronts[0]]
    profit_fronts = []
    time_fronts = []
    front_ind = []
    result_datas = []
    for i, (tour, picking, (profit, neg_time)) in enumerate(pareto_front):
        print("\n============================")
        print(f"\nSolution {i + 1}:")
        print(f"Profit: {profit}")
        print(f"Travel time: {neg_time}")  # Convert back to positive time
        profit_fronts.append(profit)
        time_fronts.append(neg_time)
        front_ind.append(0)
        result_data = {
            'tour': [tour],
            'picking_plan': list(picking),
            'profit': profit,
            'time': neg_time
        }
        result_data['picking_plan'] = [1 if pick else 0 for pick in result_data['picking_plan']]

        result_datas.append(result_data)


    for ind,_ in enumerate(final_fronts[1:]):
        other_front = [(population[idx][0], population[idx][1], final_fitness[idx])
                        for idx in _]
        for i, (tour, picking, (profit, neg_time)) in enumerate(other_front):
            print("\n============================")
            print(f"\nSolution {i + 1}:")
            print(f"Profit: {profit}")
            print(f"Travel time: {neg_time}")  # Convert back to positive time
            profit_fronts.append(profit)
            time_fronts.append(neg_time)
            front_ind.append(ind+1)

    # 创建结果目录
    result_dir = os.path.join(current_dir, "results")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)


    # 1. 保存进化过程记录
    df_records = pd.DataFrame({
        'front': front_ind,
        'profit': profit_fronts,
        'time': time_fronts
    })
    df_records.to_csv(os.path.join(result_dir, "nsga2_ttp_generation_records.csv"), index=False)
    print(f"\nGeneration records saved to: {os.path.join(result_dir, 'nsga2_ttp_generation_records.csv')}")


    #2. 保存pareto front 的结果数据
    result_record = pd.DataFrame(result_datas)
    result_record.to_csv(os.path.join(result_dir, "result_data.csv"), index=False)
    print(f"\nfinal result records saved to: {os.path.join(result_dir, 'result_data.csv')}")