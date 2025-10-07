package newrep

import (
	"math"
)

func (t *tree) selectChildPUCT(n *node) (interface{}, *node) {
	var bestChild *node
	var bestAct interface{}
	best := math.Inf(-1)
	total := float64(max(1, n.N))
	for a, ch := range n.children {
		p := n.priors[a]
		q := ch.Q
		u := t.c * p * math.Sqrt(total) / float64(1+ch.N)
		score := q + u
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
