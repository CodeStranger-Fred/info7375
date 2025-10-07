package mdp

import (
	"fmt"
	"math"
)

// Gradient bandit (softmax over action preferences)
type GradientBandit struct {
	Alpha         float64
	UseBaseline   bool
	Preferences   map[Action]float64
	AverageReward float64
}

func (g *GradientBandit) Name() string {
	return fmt.Sprintf("gradient (α=%.2f, baseline=%v)", g.Alpha, g.UseBaseline)
}

// Numerically stable softmax (subtract max preference)
func softmaxStable(prefs map[Action]float64) map[Action]float64 {
	maxH := math.Inf(-1)
	for _, h := range prefs {
		if h > maxH {
			maxH = h
		}
	}
	sum := 0.0
	pi := make(map[Action]float64, len(prefs))
	for a, h := range prefs {
		v := math.Exp(h - maxH)
		pi[a] = v
		sum += v
	}
	for a := range pi {
		pi[a] /= sum
	}
	return pi
}

// Use pointer receiver so updates to Preferences persist
func (g *GradientBandit) Act(agent *Agent, mdp *MDP, s0 State) ProbabilityDistribution[Action] {
	// Initialize preferences once
	if g.Preferences == nil {
		g.Preferences = make(map[Action]float64)
		das := mdp.ActionSpace.(DiscreteActionSpace) // your field is Action []Action
		for _, a := range das.Action {
			g.Preferences[a] = 0
		}
	}

	pi := softmaxStable(g.Preferences)
	pdf := DiscretePdf[Action]{Map: make(map[Action]Probability, len(pi))}
	for a, p := range pi {
		pdf.Map[a] = Probability(p)
	}
	return pdf
}

func (g *GradientBandit) Learn(agent *Agent, mdp *MDP, s0 State, a Action, r Reward, s1 State) {
	pi := softmaxStable(g.Preferences)

	// Baseline (EMA)
	baseline := 0.0
	if g.UseBaseline {
		g.AverageReward += 0.1 * (float64(r) - g.AverageReward)
		baseline = g.AverageReward
	}

	adv := float64(r) - baseline
	for a2 := range g.Preferences {
		if a2 == a {
			g.Preferences[a2] += g.Alpha * adv * (1 - pi[a2])
		} else {
			g.Preferences[a2] -= g.Alpha * adv * pi[a2]
		}
	}
}

