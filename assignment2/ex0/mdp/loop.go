package mdp
 
func Loop(env *MDP, agent *Agent, maxTimeSteps int) {
    state := env.InitialState

    for t := 0; t < maxTimeSteps; t++ {
        s0 := state

        actPdf := agent.Policy.Act(agent, env, state)
        a := actPdf.Choose()

        s1 := env.TransitionFunction.Transition(s0, a).Choose()
        r  := env.RewardFunction.Reward(s0, a, s1).Choose()

        if learner, ok := agent.Policy.(interface {
            Learn(*Agent, *MDP, State, Action, Reward, State)
        }); ok {
            learner.Learn(agent, env, s0, a, r, s1)
        }

        agent.Step(s0, a, s1, r)
        state = s1
    }
}