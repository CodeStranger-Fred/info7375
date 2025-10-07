package newrep

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/CodeStranger-Fred/assignemnt1/mdp"
)

type Policy interface {
	Choose(s mdp.State, actions []mdp.Action) mdp.Action
}

type RandomPolicy struct{ R *rand.Rand }

func (rp RandomPolicy) Choose(_ mdp.State, actions []mdp.Action) mdp.Action {
	return actions[rp.R.Intn(len(actions))]
}

type PriorProvider interface {
	Prior(state mdp.State, actions []mdp.Action) map[mdp.Action]float64
}

type Options struct {
	C               float64
	MaxIterations   int
	TimeBudget      time.Duration
	MaxRolloutDepth int
	RolloutPolicy   Policy
	Rand            *rand.Rand
	Prior           PriorProvider
}

func fillDefaults(o Options) Options {
	if o.C == 0 {
		o.C = math.Sqrt2
	}
	if o.Rand == nil {
		o.Rand = rand.New(rand.NewSource(time.Now().UnixNano()))
	}
	if o.RolloutPolicy == nil {
		o.RolloutPolicy = RandomPolicy{R: o.Rand}
	}
	if o.MaxRolloutDepth <= 0 {
		o.MaxRolloutDepth = 30
	}
	if o.Prior == nil {
		o.Prior = UniformPrior{} 
	}
	return o
}

func Search(model Model, root mdp.State, opts Options) (mdp.Action, string) {
	cfg := fillDefaults(opts)
	r := newNode(nil, mdp.Action(""), root, 1)
	t := &tree{
		model:    model,
		c:        cfg.C,
		rp:       cfg.RolloutPolicy,
		maxDepth: cfg.MaxRolloutDepth,
		rng:      cfg.Rand,
		prior:    cfg.Prior,
	}
	deadline := time.Now().Add(cfg.TimeBudget)
	iters := 0
	for {
		if cfg.TimeBudget > 0 && time.Now().After(deadline) {
			break
		}
		if cfg.MaxIterations > 0 && iters >= cfg.MaxIterations {
			break
		}
		iters++
		path := t.selectPath(r)
		leaf := path[len(path)-1]
		var ex *node
		if !leaf.terminal {
			ex = t.expand(leaf)
			if ex != nil {
				path = append(path, ex)
			}
		}
		start := path[len(path)-1]
		G := t.rollout(start)
		t.backprop(path, G)
	}
	a, n, q := t.bestAction(r)
	return a, fmt.Sprintf("iter=%d, children=%d, bestN=%d, bestQ=%.6f", iters, len(r.children), n, q)
}
