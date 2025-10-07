package main

import (
	"fmt"
	"math"
	"time"

	"github.com/CodeStranger-Fred/assignemnt1/mcts"
	"github.com/CodeStranger-Fred/assignemnt1/mdp"
)



type GridTransition struct {
	Grid  [][]int
	GoalX int
	GoalY int
	MaxX  int
	MaxY  int
}

func (t GridTransition) Transition(s mdp.State, a mdp.Action) mdp.ProbabilityDistribution[mdp.State] {
	var x, y int
	fmt.Sscanf(string(s), "S%d_%d", &x, &y)
	nx, ny := x, y
	switch a {
	case "up":
		ny--
	case "down":
		ny++
	case "left":
		nx--
	case "right":
		nx++
	}
	if nx < 0 || ny < 0 || nx > t.MaxX || ny > t.MaxY || t.Grid[ny][nx] == 1 {
		nx, ny = x, y
	}
	next := mdp.State(fmt.Sprintf("S%d_%d", nx, ny))
	return mdp.DiscretePdf[mdp.State]{Map: map[mdp.State]mdp.Probability{next: 1.0}}
}

type GridReward struct {
	Grid  [][]int
	GoalX int
	GoalY int
}

func (r GridReward) Reward(s mdp.State, a mdp.Action, s1 mdp.State) mdp.ProbabilityDistribution[mdp.Reward] {
	var curX, curY, nx, ny int
	fmt.Sscanf(string(s), "S%d_%d", &curX, &curY)
	fmt.Sscanf(string(s1), "S%d_%d", &nx, &ny)
	reward := -0.01
	if nx == r.GoalX && ny == r.GoalY {
		reward = 1.0
	}
	return mdp.DiscretePdf[mdp.Reward]{Map: map[mdp.Reward]mdp.Probability{mdp.Reward(reward): 1.0}}
}

func reachedGoal(s mdp.State, goalX, goalY int) bool {
	var x, y int
	fmt.Sscanf(string(s), "S%d_%d", &x, &y)
	return x == goalX && y == goalY
}

func printGrid(grid [][]int, posX, posY, goalX, goalY int) {
	for y := 0; y < len(grid); y++ {
		for x := 0; x < len(grid[0]); x++ {
			if x == posX && y == posY {
				fmt.Print("🤖 ")
			} else if x == goalX && y == goalY {
				fmt.Print("🏁 ")
			} else if grid[y][x] == 1 {
				fmt.Print("⬛ ")
			} else {
				fmt.Print("⬜ ")
			}
		}
		fmt.Println()
	}
	fmt.Println()
}

func main() {
	grid := [][]int{
		{0, 0, 0, 0, 0},
		{0, 1, 1, 0, 0},
		{0, 0, 1, 0, 0},
		{0, 0, 1, 1, 0},
		{0, 0, 0, 0, 0},
	}
	goalX, goalY := 4, 4

	env := &mdp.MDP{
		TransitionFunction: GridTransition{Grid: grid, GoalX: goalX, GoalY: goalY, MaxX: 4, MaxY: 4},
		RewardFunction:     GridReward{Grid: grid, GoalX: goalX, GoalY: goalY},
		ActionSpace: mdp.DiscreteActionSpace{
			Action: []mdp.Action{"up", "down", "left", "right"},
		},
		InitialState: "S0_0",
	}

	model := &mcts.MDPAdapter{Env: env}
	opts := mcts.Options{
		C:               math.Sqrt2,
		MaxIterations:   3000,
		MaxRolloutDepth: 30,
	}

	trials := 1
	plannings := 1
	episodes := 5

	for trial := 0; trial < trials; trial++ {
		for plan := 0; plan < plannings; plan++ {
			for ep := 0; ep < episodes; ep++ {
				state := env.InitialState
				steps := 0

				fmt.Printf("\n=== Episode %d ===\n", ep)
				printGrid(grid, 0, 0, goalX, goalY)

				for !reachedGoal(state, goalX, goalY) && steps < 100 {
					best, _ := mcts.Search(model, state, opts)
					var x, y int
					fmt.Sscanf(string(state), "S%d_%d", &x, &y)
					s1 := env.TransitionFunction.(GridTransition).Transition(state, best).Choose()
					state = s1
					steps++

					fmt.Printf("Step %2d: move %-6s → %s\n", steps, best, state)
					var nx, ny int
					fmt.Sscanf(string(state), "S%d_%d", &nx, &ny)
					printGrid(grid, nx, ny, goalX, goalY)

					if reachedGoal(state, goalX, goalY) {
						break
					}
					time.Sleep(150 * time.Millisecond)
				}

				fmt.Printf("[trial: %d planning: %d episode #: %d] reached goal in %d steps\n", trial, plan, ep, steps)
				time.Sleep(500 * time.Millisecond)
			}
		}
	}
}
