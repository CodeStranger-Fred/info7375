package newrep

import (
	"math"

	"github.com/CodeStranger-Fred/assignemnt1/mdp"
)

type node struct {
	state       mdp.State
	parent      *node
	parentAct   mdp.Action
	player      int
	N           int
	W           float64
	Q           float64
	terminal    bool
	children    map[mdp.Action]*node
	untriedActs []mdp.Action
	priors      map[mdp.Action]float64
}

func newNode(parent *node, parentAct mdp.Action, s mdp.State, player int) *node {
	return &node{
		state:     s,
		parent:    parent,
		parentAct: parentAct,
		player:    player,
		children:  map[mdp.Action]*node{},
	}
}

type tree struct {
	model    Model
	c        float64
	rp       Policy
	maxDepth int
	rng      interface {
		Float64() float64
		Intn(n int) int
	}
	prior PriorProvider
}

func (t *tree) selectPath(root *node) []*node {
	path := []*node{root}
	cur := root
	for {
		if cur.terminal {
			return path
		}
		if cur.untriedActs == nil {
			acts := t.model.Actions(cur.state)
			if len(acts) == 0 {
				cur.terminal = true
				return path
			}
			cur.untriedActs = append([]mdp.Action(nil), acts...)
			cur.priors = t.prior.Prior(cur.state, acts)
			return path
		}
		if len(cur.untriedActs) > 0 {
			return path
		}
		_, ch := t.selectChildPUCT(cur)
		path = append(path, ch)
		cur = ch
	}
}

func (t *tree) expand(leaf *node) *node {
	if leaf.untriedActs == nil {
		leaf.untriedActs = t.model.Actions(leaf.state)
		leaf.priors = t.prior.Prior(leaf.state, leaf.untriedActs)
	}
	if len(leaf.untriedActs) == 0 {
		leaf.terminal = true
		return nil
	}
	i := len(leaf.untriedActs) - 1
	a := leaf.untriedActs[i]
	leaf.untriedActs = leaf.untriedActs[:i]
	s1, _, term := t.model.Step(leaf.state, a)
	ch := newNode(leaf, a, s1, 1)
	ch.terminal = term
	leaf.children[a] = ch
	return ch
}

func (t *tree) rollout(start *node) float64 {
	G := 0.0
	s := start.state
	player := start.player
	term := start.terminal
	depth := 0
	for !term {
		if t.maxDepth > 0 && depth >= t.maxDepth {
			break
		}
		acts := t.model.Actions(s)
		if len(acts) == 0 {
			break
		}
		a := t.rp.Choose(s, acts)
		var r float64
		s, r, term = t.model.Step(s, a)
		G += float64(player) * r
		player = 1
		depth++
	}
	return G
}

func (t *tree) backprop(path []*node, G float64) {
	for i := len(path) - 1; i >= 0; i-- {
		n := path[i]
		n.N++
		n.W += G
		n.Q = n.W / float64(n.N)
		G = -G
	}
}

func (t *tree) bestAction(root *node) (mdp.Action, int, float64) {
	var bestA mdp.Action
	bestN := -1
	bestQ := math.Inf(-1)
	for a, ch := range root.children {
		if ch.N > bestN || (ch.N == bestN && ch.Q > bestQ) {
			bestA, bestN, bestQ = a, ch.N, ch.Q
		}
	}
	return bestA, bestN, bestQ
}
