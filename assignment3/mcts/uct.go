package mcts

import (
	"math"
)

func (t *tree) selectChildUCT(n *node) (interface{}, *node) {
	best := math.Inf(-1)
	var bestChild *node
	var bestAct interface{}
	lnN := math.Log(float64(max(1, n.N)))
	for a, ch := range n.children {
		explore := t.c * math.Sqrt(lnN/float64(1+ch.N))
		score := ch.Q + explore
		if score > best || (score == best && t.rng.Float64() < 0.5) {
			best = score
			bestChild = ch
			bestAct = a
		}
	}
	return bestAct, bestChild
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
