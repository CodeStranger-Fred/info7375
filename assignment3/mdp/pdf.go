package mdp

import (
	"math"

	"math/rand"
)


type ProbabilityDistribution[Category comparable] interface {
	Choose() Category
}

type Probability float64

type DiscretePdf[Category comparable] struct {
	Map map[Category] Probability
}

func (p DiscretePdf[Category]) Choose() Category {
	p.Check()
	v := rand.Float64()
	cumulative := 0.0
	var last Category
	for st, prob := range p.Map {
		cumulative += float64(prob)
		if cumulative >= v {
			return st
		}
		last = st
	}
	return last
}

func (p DiscretePdf[Category]) Check() {
	sum := 0.0
	for _, prob := range p.Map {
		sum += float64(prob)
	}
	
	if math.Abs(sum-1) > .001 {
		panic("Probability dont sum to 1")
	}
}