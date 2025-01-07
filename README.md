# Bi-objective-Traveling-Thief
Team GG-Bond Solves Bi-objective Traveling Thief Problem using Nature-Inspired Algorithms

Here is the Github Link [GGBond Project](https://github.com/DraesonFang/Bi-objective-Traveling-Thief/tree/main)

## Start
#### Final_algorithm.py is our final code, you can go to 'if __name__ == "__main__":' This is where our program starts. 
#### Then change the value of 'file_path' with your file path.   
```
E.g.(like using file 'a280-n279.txt') the file is one of dataset given by the competition

file_path = "tpye your file path here"
```
#### And then, find the 'solver', adjust the required parameters(e.g.solver = NSGA2_TTP(metadata, city_coords_df, items_df, pop_size=100, generations=3, mutation_rate=0.2, tournament_size=3))

```
E.g. Adjust the parameters you want.

solver = NSGA2_TTP(metadata, city_coords_df, items_df,
                       pop_size=100, generations=3,
                       mutation_rate=0.2, tournament_size=3)
```


#### Then Congratulations! you can get the result in the new-created results folder.

## Results illustration
#### There are 2 file in the results folder。
#### First one is 'nsga2_ttp_generation_records.csv' contain the front each solution belong, and the time and profit of each solution.

```
E.g.

front,  profit,    time
0,      417605.0,  7206.796890208479
```

#### Second one is 'result_data',there are many lines in it, each line represent a solution's the tour,picking_plan,profit,time cost.

```
E.g. each line will be like below:

tour,             picking_plan,      profit,    time
"[[1,2,3,4,5]]",  "[1,0,1,1,1]",     10000,     10000.000
```
